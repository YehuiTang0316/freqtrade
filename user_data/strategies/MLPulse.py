# MLPulse - FreqAI LightGBM 3-class classifier for intraday 5m trading
# Uses LightGBMClassifier to predict short-term price direction (up/down/neutral)
# with traditional TA confirmation signals for entry/exit
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

import logging
from datetime import datetime
from functools import reduce

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy, Trade

import talib.abstract as ta
from technical import qtpylib


logger = logging.getLogger(__name__)


class MLPulse(IStrategy):
    """
    MLPulse: FreqAI LightGBM 3-class classifier strategy on 5m timeframe.

    Model predicts price direction over next 1h (12 candles):
      - "up":      rolling mean return > +0.3%
      - "down":    rolling mean return < -0.3%
      - "neutral": otherwise

    Entry requires ML prediction + probability > 55% + RSI/volume confirmation.
    Exit uses ML signal flip + profit-tiered RSI + time stop + trailing stop.
    """

    INTERFACE_VERSION = 3

    can_short = True

    # --- ROI ---
    minimal_roi = {
        "0": 0.02,
        "60": 0.01,
        "180": 0.005,
        "360": 0,
    }

    # --- Stoploss ---
    stoploss = -0.02

    # --- Trailing stop ---
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # --- Timeframe ---
    timeframe = "5m"
    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # FreqAI needs enough candles for the longest indicator period (50) across
    # the longest informative timeframe (1h = 12x multiplier) + some buffer
    startup_candle_count: int = 100

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # -----------------------------------------------------------------------
    # FreqAI feature engineering
    # -----------------------------------------------------------------------

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Features expanded across all indicator_periods_candles, timeframes,
        shifted candles, and correlation pairs.
        """

        # Momentum
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)
        dataframe["%-willr-period"] = ta.WILLR(dataframe, timeperiod=period)

        # Trend
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["%-close-ema-ratio-period"] = dataframe["close"] / ta.EMA(
            dataframe, timeperiod=period
        )
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)

        # Volatility (Bollinger Bands)
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]
        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
            dataframe["close"] / dataframe["bb_lowerband-period"]
        )

        # Volume
        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Features expanded across timeframes, shifted candles, and correlation pairs
        but NOT across indicator periods.
        """

        # Price action
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        # MACD components
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["%-macd"] = macd["macd"]
        dataframe["%-macdsignal"] = macd["macdsignal"]
        dataframe["%-macdhist"] = macd["macdhist"]

        # OBV rate of change
        dataframe["%-obv"] = ta.OBV(dataframe)
        dataframe["%-obv_roc"] = dataframe["%-obv"].pct_change(5)

        # ATR as percentage of close
        dataframe["%-atr_pct"] = ta.ATR(dataframe, timeperiod=14) / dataframe["close"]

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Features computed once on the base timeframe only (no expansion).
        Good for time-based features that shouldn't be duplicated.
        """

        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek

        # Trading session block (0=Asia, 1=Europe, 2=US)
        hour = dataframe["date"].dt.hour
        dataframe["%-session_block"] = np.where(
            hour < 8, 0, np.where(hour < 16, 1, 2)
        )

        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        3-class classification target based on rolling mean return over
        the next label_period_candles (12 candles = 1 hour on 5m).
        """

        label_period = self.freqai_info["feature_parameters"]["label_period_candles"]

        # Rolling mean of future closes vs current close
        future_mean = (
            dataframe["close"]
            .shift(-label_period)
            .rolling(label_period)
            .mean()
        )
        pct_return = future_mean / dataframe["close"] - 1

        # Classify: up > +0.3%, down < -0.3%, else neutral
        dataframe["&s-direction"] = np.where(
            pct_return > 0.003,
            "up",
            np.where(pct_return < -0.003, "down", "neutral"),
        )

        return dataframe

    # -----------------------------------------------------------------------
    # Indicators (main timeframe, after FreqAI)
    # -----------------------------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Run FreqAI - this populates &s-direction, do_predict, up/down/neutral probs
        dataframe = self.freqai.start(dataframe, metadata, self)

        # TA confirmation indicators (computed on raw 5m data)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_sma_20"] = ta.SMA(dataframe["volume"], timeperiod=20)

        return dataframe

    # -----------------------------------------------------------------------
    # Entry signals
    # -----------------------------------------------------------------------

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Volume filter: above 50% of 20-period SMA
        vol_ok = (df["volume"] > df["volume_sma_20"] * 0.5) & (df["volume"] > 0)

        # ===== LONG =====
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&s-direction"] == "up",
            df["up"] > 0.55,
            df["rsi_14"] < 70,
            vol_ok,
        ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions),
                ["enter_long", "enter_tag"],
            ] = (1, "ml_long")

        # ===== SHORT =====
        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&s-direction"] == "down",
            df["down"] > 0.55,
            df["rsi_14"] > 30,
            vol_ok,
        ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions),
                ["enter_short", "enter_tag"],
            ] = (1, "ml_short")

        return df

    # -----------------------------------------------------------------------
    # Exit signals
    # -----------------------------------------------------------------------

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # ML signal flip: long position but model says "down"
        exit_long_conditions = [
            df["do_predict"] == 1,
            df["&s-direction"] == "down",
            df["down"] > 0.50,
        ]
        if exit_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_long_conditions),
                ["exit_long", "exit_tag"],
            ] = (1, "ml_exit_long")

        # ML signal flip: short position but model says "up"
        exit_short_conditions = [
            df["do_predict"] == 1,
            df["&s-direction"] == "up",
            df["up"] > 0.50,
        ]
        if exit_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_short_conditions),
                ["exit_short", "exit_tag"],
            ] = (1, "ml_exit_short")

        return df

    # -----------------------------------------------------------------------
    # Custom exit: time stop + profit-tiered RSI + uncertainty exit
    # -----------------------------------------------------------------------

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]
        trade_duration_h = (current_time - trade.open_date_utc).total_seconds() / 3600

        # 1. Time stop: close after 6h if not profitable
        if trade_duration_h > 6 and current_profit < 0.002:
            return "exit_time_6h"

        # 2. Profit-tiered RSI exit
        rsi = last_candle.get("rsi_14", 50)

        if not trade.is_short:
            # Long: take profit if RSI overbought and in profit
            if current_profit > 0.015 and rsi > 75:
                return "long_rsi_tp_high"
            if current_profit > 0.008 and rsi > 80:
                return "long_rsi_tp_extreme"
        else:
            # Short: take profit if RSI oversold and in profit
            if current_profit > 0.015 and rsi < 25:
                return "short_rsi_tp_high"
            if current_profit > 0.008 and rsi < 20:
                return "short_rsi_tp_extreme"

        # 3. Model uncertainty exit: close if model says don't predict
        if last_candle.get("do_predict", 1) == 0 and current_profit > 0:
            return "exit_model_uncertain"

        return None

    # -----------------------------------------------------------------------
    # Slippage protection
    # -----------------------------------------------------------------------

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        # 0.25% slippage protection
        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True

    # -----------------------------------------------------------------------
    # Fixed 2x leverage
    # -----------------------------------------------------------------------

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return 2.0
