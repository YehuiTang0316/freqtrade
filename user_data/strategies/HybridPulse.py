import numpy as np
from datetime import datetime
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
    Trade,
    IntParameter,
    DecimalParameter,
    informative,
)
import talib.abstract as ta
from technical import qtpylib


class HybridPulse(IStrategy):
    """
    HybridPulse v1: Best of BreakPulse + TrendPulse combined.

    Two entry types:
    1. Squeeze breakout (from BreakPulse): BB/Keltner squeeze → breakout
    2. Trend pullback (from TrendPulse): Oversold/overbought in trend direction

    Same core framework: 1h base, 4h trend, BTC filter, 2x leverage.
    """

    INTERFACE_VERSION = 3

    can_short = True
    process_only_new_candles = True

    timeframe = "1h"
    startup_candle_count: int = 200

    # ROI from BreakPulse v3 hyperopt (proven optimal)
    minimal_roi = {
        "0": 0.187,
        "352": 0.14,
        "874": 0.048,
        "2222": 0,
    }

    stoploss = -0.087

    trailing_stop = True
    trailing_stop_positive = 0.021
    trailing_stop_positive_offset = 0.07
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # Hyperopt: breakout params
    buy_adx_min = IntParameter(15, 30, default=17, space="buy", optimize=True)
    buy_volume_mult = DecimalParameter(1.0, 3.0, default=1.7, decimals=1, space="buy", optimize=True)
    keltner_atr_mult = DecimalParameter(1.5, 3.0, default=2.1, decimals=1, space="buy", optimize=True)

    # Hyperopt: pullback params
    buy_rsi_low = IntParameter(20, 40, default=30, space="buy", optimize=True)
    sell_rsi_high = IntParameter(60, 80, default=70, space="sell", optimize=True)

    sell_adx_min = IntParameter(15, 30, default=15, space="sell", optimize=True)
    sell_volume_mult = DecimalParameter(0.5, 2.0, default=1.0, decimals=1, space="sell", optimize=True)

    # ----------------------------------------------------------------------
    # BTC 1h filter
    # ----------------------------------------------------------------------
    @informative("1h", "BTC/USDT:USDT")
    def populate_indicators_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    # ----------------------------------------------------------------------
    # 4h trend
    # ----------------------------------------------------------------------
    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)

        dataframe["trend"] = 0
        dataframe.loc[
            (dataframe["ema_50"] > dataframe["ema_200"]) & (dataframe["adx"] > 20),
            "trend",
        ] = 1
        dataframe.loc[
            (dataframe["ema_50"] < dataframe["ema_200"]) & (dataframe["adx"] > 20),
            "trend",
        ] = -1

        return dataframe

    # ----------------------------------------------------------------------
    # 1h indicators
    # ----------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === Squeeze detection (from BreakPulse) ===
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_upper"] = bollinger["upper"]
        dataframe["bb_mid"] = bollinger["mid"]
        dataframe["bb_width"] = (
            (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"]
        )

        dataframe["kc_mid"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["atr_20"] = ta.ATR(dataframe, timeperiod=20)

        dataframe["bb_width_sma"] = ta.SMA(dataframe["bb_width"], timeperiod=50)
        dataframe["bb_tight"] = dataframe["bb_width"] < dataframe["bb_width_sma"]

        # === Pullback detection (from TrendPulse) ===
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_3"] = ta.RSI(dataframe, timeperiod=3)
        dataframe["williams_r"] = ta.WILLR(dataframe, timeperiod=14)

        # EMAs
        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)

        # Momentum
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd_hist"] = macd["macdhist"]

        # Volume
        dataframe["volume_sma_20"] = ta.SMA(dataframe["volume"], timeperiod=20)

        return dataframe

    # ----------------------------------------------------------------------
    # Entry signals
    # ----------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # BTC filters
        btc_ok_long = (
            (dataframe["btc_usdt_ema_50_1h"] > dataframe["btc_usdt_ema_200_1h"])
            & (dataframe["btc_usdt_rsi_14_1h"] > 35)
        )
        btc_ok_short = dataframe["btc_usdt_rsi_14_1h"] < 70

        # Keltner channels (computed here for hyperopt compatibility)
        kc_upper = dataframe["kc_mid"] + dataframe["atr_20"] * self.keltner_atr_mult.value
        kc_lower = dataframe["kc_mid"] - dataframe["atr_20"] * self.keltner_atr_mult.value

        squeeze_on = (
            (dataframe["bb_lower"] > kc_lower)
            & (dataframe["bb_upper"] < kc_upper)
        )
        prev_squeeze = squeeze_on.shift(1)
        prev_squeeze = prev_squeeze.infer_objects(copy=False).fillna(False)

        # ========== TYPE 1: SQUEEZE BREAKOUT (from BreakPulse) ==========

        # Long squeeze breakout
        long_squeeze = (
            btc_ok_long
            & (dataframe["trend_4h"] >= 0)
            & (prev_squeeze | dataframe["bb_tight"])
            & (dataframe["close"] > dataframe["bb_upper"])
            & (dataframe["macd_hist"] > 0)
            & (dataframe["adx"] > self.buy_adx_min.value)
            & (dataframe["ema_9"] > dataframe["ema_21"])
            & (dataframe["volume"] > dataframe["volume_sma_20"] * self.buy_volume_mult.value)
            & (dataframe["volume"] > 0)
        )

        # Short squeeze breakout
        short_squeeze = (
            btc_ok_short
            & (dataframe["trend_4h"] <= 0)
            & (prev_squeeze | dataframe["bb_tight"])
            & (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["macd_hist"] < 0)
            & (dataframe["adx"] > self.sell_adx_min.value)
            & (dataframe["ema_9"] < dataframe["ema_21"])
            & (dataframe["volume"] > dataframe["volume_sma_20"] * self.sell_volume_mult.value)
            & (dataframe["volume"] > 0)
        )

        # Long momentum breakout
        long_momentum = (
            btc_ok_long
            & (dataframe["trend_4h"] == 1)
            & (dataframe["close"] > kc_upper)
            & (dataframe["rsi_14"] > 55)
            & (dataframe["rsi_14"] < 75)
            & (dataframe["adx"] > 25)
            & (dataframe["macd_hist"] > 0)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 1.2)
            & (dataframe["volume"] > 0)
        )

        # Short momentum breakout
        short_momentum = (
            btc_ok_short
            & (dataframe["trend_4h"] == -1)
            & (dataframe["close"] < kc_lower)
            & (dataframe["rsi_14"] < 45)
            & (dataframe["rsi_14"] > 25)
            & (dataframe["adx"] > 25)
            & (dataframe["macd_hist"] < 0)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 1.2)
            & (dataframe["volume"] > 0)
        )

        # ========== TYPE 2: TREND PULLBACK (from TrendPulse) ==========

        # Long pullback: oversold in uptrend
        long_pullback = (
            btc_ok_long
            & (dataframe["trend_4h"] == 1)
            & (dataframe["rsi_14"] < self.buy_rsi_low.value)
            & (dataframe["rsi_3"] > 3)
            & (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["williams_r"] < -90)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 0.5)
            & (dataframe["volume"] > 0)
        )

        # Short pullback: overbought in downtrend
        short_pullback = (
            btc_ok_short
            & (dataframe["trend_4h"] == -1)
            & (dataframe["rsi_14"] > self.sell_rsi_high.value)
            & (dataframe["rsi_3"] < 97)
            & (dataframe["close"] > dataframe["bb_upper"])
            & (dataframe["williams_r"] > -10)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 0.5)
            & (dataframe["volume"] > 0)
        )

        # Assign entries with priority: squeeze > momentum > pullback
        dataframe.loc[long_squeeze, ["enter_long", "enter_tag"]] = (1, "long_squeeze_break")
        dataframe.loc[long_momentum & ~long_squeeze, ["enter_long", "enter_tag"]] = (1, "long_momentum")
        dataframe.loc[long_pullback & ~long_squeeze & ~long_momentum, ["enter_long", "enter_tag"]] = (1, "long_pullback")

        dataframe.loc[short_squeeze, ["enter_short", "enter_tag"]] = (1, "short_squeeze_break")
        dataframe.loc[short_momentum & ~short_squeeze, ["enter_short", "enter_tag"]] = (1, "short_momentum")
        dataframe.loc[short_pullback & ~short_squeeze & ~short_momentum, ["enter_short", "enter_tag"]] = (1, "short_pullback")

        return dataframe

    # ----------------------------------------------------------------------
    # Exit signals
    # ----------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, ["exit_long", "exit_tag"]] = (0, "")
        dataframe.loc[:, ["exit_short", "exit_tag"]] = (0, "")
        return dataframe

    # ----------------------------------------------------------------------
    # Custom exit
    # ----------------------------------------------------------------------
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
        trade_duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600

        if trade_duration_hours > 48 and current_profit < 0.005:
            return "exit_time_48h"

        if not trade.is_short and current_profit > 0.005:
            if last_candle.get("trend_4h", 0) == -1:
                return "long_exit_trend_flip"

        if trade.is_short and current_profit > 0.005:
            if last_candle.get("trend_4h", 0) == 1:
                return "short_exit_trend_flip"

        if current_profit > 0.02:
            if not trade.is_short and last_candle.get("macd_hist", 0) < 0:
                return "long_exit_momentum_fade"
            if trade.is_short and last_candle.get("macd_hist", 0) > 0:
                return "short_exit_momentum_fade"

        return None

    # ----------------------------------------------------------------------
    # Leverage
    # ----------------------------------------------------------------------
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
