# RawPulse - 20x Intraday ML Strategy with Raw Data Features
# Uses LightGBMClassifier on raw price/volume features (no standard TA indicators)
# High leverage day-trading: tight stops, 4h max holding, forced intraday close
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

import logging
from datetime import datetime
from functools import reduce

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy, Trade, DecimalParameter, IntParameter

import talib.abstract as ta


logger = logging.getLogger(__name__)


class RawPulse(IStrategy):
    """
    RawPulse: 20x leverage intraday FreqAI LightGBM 3-class classifier on 5m.

    Key differences from MLPulse:
      - Features derived from raw data (volume, OHLC structure, dollar volume)
        instead of standard TA indicators (RSI, MACD, BB, etc.)
      - 20x leverage with tight 0.3% stoploss (= 6% capital risk per trade)
      - 4-hour max holding period (forced intraday close)
      - 30-minute prediction window (6 candles on 5m)

    Model predicts price direction over next 30min (6 candles):
      - "up":      return > +0.3%
      - "down":    return < -0.3%
      - "neutral": otherwise

    Target: ~3-5 trades/day, 55%+ win rate, ~0.5% avg win / 0.3% avg loss.
    """

    INTERFACE_VERSION = 3

    can_short = True

    # --- ROI: leveraged returns (at 20x, 0.5% price move = 10% return) ---
    minimal_roi = {
        "0": 0.20,      # 20% return (= 1% price at 20x)
        "30": 0.12,     # 12% after 30min
        "60": 0.08,     # 8% after 1h
        "120": 0.04,    # 4% after 2h
        "240": 0,       # breakeven after 4h
    }

    # --- Stoploss: -10% return (= 0.5% adverse price move at 20x) ---
    stoploss = -0.10

    # --- Trailing stop ---
    trailing_stop = True
    trailing_stop_positive = 0.05     # 5% trail (= 0.25% price)
    trailing_stop_positive_offset = 0.10  # activate after 10% return (0.5% price)
    trailing_only_offset_is_reached = True

    # --- Timeframe ---
    timeframe = "5m"
    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 100

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,  # critical for 20x - must have exchange-side stop
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # --- Hyperopt-optimizable parameters ---
    buy_ml_prob = DecimalParameter(0.50, 0.85, default=0.65, decimals=2, space="buy", optimize=True)
    buy_vol_factor = DecimalParameter(0.3, 1.5, default=0.8, decimals=1, space="buy", optimize=True)
    sell_ml_prob = DecimalParameter(0.50, 0.85, default=0.65, decimals=2, space="sell", optimize=True)
    sell_vol_factor = DecimalParameter(0.3, 1.5, default=0.8, decimals=1, space="sell", optimize=True)

    # -----------------------------------------------------------------------
    # FreqAI feature engineering - RAW DATA FEATURES ONLY
    # -----------------------------------------------------------------------

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Group A (Volume Dynamics) + Group E (Range/Volatility)
        Expanded across all indicator_periods_candles, timeframes, shifts, corr pairs.
        """

        # --- Group A: Volume dynamics ---
        vol_ma = dataframe["volume"].rolling(period).mean()
        dataframe["%-rel_vol-period"] = dataframe["volume"] / (vol_ma + 1e-10)

        dataframe["%-vol_momentum-period"] = (
            dataframe["volume"].rolling(period).mean()
            / (dataframe["volume"].rolling(period * 2).mean() + 1e-10)
        )

        dataframe["%-vol_volatility-period"] = (
            dataframe["volume"].rolling(period).std()
            / (vol_ma + 1e-10)
        )

        # --- Group E: Range / Volatility ---
        price_range = dataframe["high"] - dataframe["low"]
        dataframe["%-range_pct-period"] = (
            price_range.rolling(period).mean()
            / (dataframe["close"] + 1e-10)
        )

        avg_range = price_range.rolling(period * 3).mean()
        dataframe["%-relative_range-period"] = (
            price_range / (avg_range + 1e-10)
        )

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Group B (Dollar Volume) + Group C (Candle Structure) + Group D (Volume-Price Interaction)
        Expanded across timeframes, shifts, corr pairs but NOT indicator periods.
        """

        # --- Group B: Dollar volume ---
        dollar_vol = dataframe["close"] * dataframe["volume"]
        dv_mean = dollar_vol.rolling(20).mean()
        dv_std = dollar_vol.rolling(20).std()
        dataframe["%-dollar_vol_zscore"] = (dollar_vol - dv_mean) / (dv_std + 1e-10)

        # --- Group C: Candle structure (from OHLC relative positions) ---
        body = abs(dataframe["close"] - dataframe["open"])
        full_range = dataframe["high"] - dataframe["low"]

        dataframe["%-body_ratio"] = body / (full_range + 1e-10)

        dataframe["%-close_location"] = (
            (dataframe["close"] - dataframe["low"]) / (full_range + 1e-10)
        )

        dataframe["%-upper_wick"] = (
            (dataframe["high"] - dataframe[["close", "open"]].max(axis=1))
            / (full_range + 1e-10)
        )

        dataframe["%-lower_wick"] = (
            (dataframe[["close", "open"]].min(axis=1) - dataframe["low"])
            / (full_range + 1e-10)
        )

        # --- Group D: Volume-Price interaction ---
        returns = dataframe["close"].pct_change()
        dataframe["%-price_vol_corr"] = (
            returns.rolling(20).corr(dataframe["volume"].pct_change())
        )

        vol_safe = dataframe["volume"].replace(0, np.nan)
        dataframe["%-return_per_vol"] = (
            returns / (vol_safe / vol_safe.rolling(20).mean())
        ).fillna(0)

        # Directional volume: positive returns * volume vs negative
        pos_vol = (dataframe["volume"] * (returns > 0).astype(float)).rolling(12).sum()
        neg_vol = (dataframe["volume"] * (returns < 0).astype(float)).rolling(12).sum()
        dataframe["%-directional_vol"] = (pos_vol - neg_vol) / (pos_vol + neg_vol + 1e-10)

        # --- Raw price change ---
        dataframe["%-pct_change"] = returns

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Group G: Time features (no expansion).
        """

        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek

        # Trading session (0=Asia 00-08, 1=Europe 08-16, 2=US 16-24)
        hour = dataframe["date"].dt.hour
        dataframe["%-session_block"] = np.where(
            hour < 8, 0, np.where(hour < 16, 1, 2)
        )

        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        3-class classification: predict direction over next 30min (6 candles on 5m).
        Threshold: +/-0.3% for up/down.
        """

        label_period = self.freqai_info["feature_parameters"]["label_period_candles"]

        # Rolling mean of future closes vs current close (smoother than single point)
        future_mean = (
            dataframe["close"]
            .shift(-label_period)
            .rolling(label_period)
            .mean()
        )
        pct_return = future_mean / dataframe["close"] - 1

        # Classify: up > +0.5%, down < -0.5%, else neutral
        dataframe["&s-direction"] = np.where(
            pct_return > 0.005,
            "up",
            np.where(pct_return < -0.005, "down", "neutral"),
        )

        return dataframe

    # -----------------------------------------------------------------------
    # Indicators (after FreqAI)
    # -----------------------------------------------------------------------

    def informative_pairs(self):
        return [("BTC/USDT:USDT", "1h")]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Run FreqAI
        dataframe = self.freqai.start(dataframe, metadata, self)

        # Simple volume SMA for entry filter (not a feature, just a filter)
        dataframe["volume_sma_20"] = dataframe["volume"].rolling(20).mean()

        # BTC 1h trend filter (uses TA only for BTC filter, not as model features)
        btc_1h = self.dp.get_pair_dataframe("BTC/USDT:USDT", "1h")
        if len(btc_1h) > 0:
            btc_1h["btc_ema_50"] = ta.EMA(btc_1h, timeperiod=50)
            btc_1h["btc_ema_200"] = ta.EMA(btc_1h, timeperiod=200)
            btc_1h["btc_rsi_14"] = ta.RSI(btc_1h, timeperiod=14)
            btc_1h["btc_trend"] = 0
            btc_1h.loc[btc_1h["btc_ema_50"] > btc_1h["btc_ema_200"], "btc_trend"] = 1
            btc_1h.loc[btc_1h["btc_ema_50"] < btc_1h["btc_ema_200"], "btc_trend"] = -1

            btc_1h = btc_1h[["date", "btc_trend", "btc_rsi_14"]].copy()
            dataframe = dataframe.merge(btc_1h, on="date", how="left")
            dataframe["btc_trend"] = dataframe["btc_trend"].ffill().fillna(0)
            dataframe["btc_rsi_14"] = dataframe["btc_rsi_14"].ffill().fillna(50)
        else:
            dataframe["btc_trend"] = 0
            dataframe["btc_rsi_14"] = 50

        return dataframe

    # -----------------------------------------------------------------------
    # Entry signals
    # -----------------------------------------------------------------------

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Volume filter
        vol_ok = (df["volume"] > df["volume_sma_20"] * self.buy_vol_factor.value) & (df["volume"] > 0)

        # ===== LONG =====
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&s-direction"] == "up",
            df["up"] > self.buy_ml_prob.value,
            df["btc_trend"] >= 0,
            df["btc_rsi_14"] > 40,
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
            df["down"] > self.sell_ml_prob.value,
            df["btc_trend"] <= 0,
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
        # ML signal flip: long but model says down
        exit_long_conditions = [
            df["do_predict"] == 1,
            df["&s-direction"] == "down",
            df["down"] > 0.45,
        ]
        if exit_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_long_conditions),
                ["exit_long", "exit_tag"],
            ] = (1, "ml_exit_long")

        # ML signal flip: short but model says up
        exit_short_conditions = [
            df["do_predict"] == 1,
            df["&s-direction"] == "up",
            df["up"] > 0.45,
        ]
        if exit_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_short_conditions),
                ["exit_short", "exit_tag"],
            ] = (1, "ml_exit_short")

        return df

    # -----------------------------------------------------------------------
    # Custom exit: 4h forced close + profit-tiered exit + uncertainty
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

        # current_profit is leveraged return: at 20x, 1% price = 20% return

        # 1. FORCED INTRADAY CLOSE: 4h max holding period
        if trade_duration_h > 4:
            return "exit_forced_4h"

        # 2. Big win take-profit: 25%+ return (1.25% price move at 20x)
        if current_profit > 0.25:
            return "exit_tp_high"

        # 3. Quick take-profit: 12%+ return after 30min
        if current_profit > 0.12 and trade_duration_h > 0.5:
            return "exit_tp_retreat"

        # 4. Time decay: close stagnant trades after 2h
        if trade_duration_h > 2 and current_profit < 0.03:
            return "exit_time_decay_2h"

        # 5. Model uncertainty exit
        if last_candle.get("do_predict", 1) == 0 and current_profit > -0.05:
            return "exit_model_uncertain"

        return None

    # -----------------------------------------------------------------------
    # Slippage protection (tighter for 20x)
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

        # 0.1% slippage protection (tight for 20x: 0.1% slip = 2% leveraged loss)
        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.001)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.001)):
                return False

        return True

    # -----------------------------------------------------------------------
    # 20x leverage
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
        return 20.0
