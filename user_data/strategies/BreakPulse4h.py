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


class BreakPulse4h(IStrategy):
    """
    BreakPulse4h v1: Same squeeze breakout logic as BreakPulse v3, but on 4h timeframe.

    Hypothesis: 4h squeeze breakouts may produce even cleaner signals with less noise.
    Uses 1d trend direction instead of 4h, BTC 4h filter.
    """

    INTERFACE_VERSION = 3

    can_short = True
    process_only_new_candles = True

    timeframe = "4h"
    startup_candle_count: int = 200

    # Wider ROI for 4h (trades last longer)
    minimal_roi = {
        "0": 0.30,
        "720": 0.15,
        "2880": 0.05,
        "5760": 0,
    }

    stoploss = -0.10

    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.08
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

    # Hyperopt parameters
    buy_adx_min = IntParameter(15, 30, default=17, space="buy", optimize=True)
    buy_volume_mult = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="buy", optimize=True)
    keltner_atr_mult = DecimalParameter(1.5, 3.0, default=2.1, decimals=1, space="buy", optimize=True)
    sell_adx_min = IntParameter(15, 30, default=15, space="sell", optimize=True)
    sell_volume_mult = DecimalParameter(0.5, 2.0, default=1.0, decimals=1, space="sell", optimize=True)

    # ----------------------------------------------------------------------
    # BTC 4h filter
    # ----------------------------------------------------------------------
    @informative("4h", "BTC/USDT:USDT")
    def populate_indicators_btc_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    # ----------------------------------------------------------------------
    # 4h indicators (main timeframe)
    # ----------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_upper"] = bollinger["upper"]
        dataframe["bb_mid"] = bollinger["mid"]
        dataframe["bb_width"] = (
            (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"]
        )

        # Keltner base components
        dataframe["kc_mid"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["atr_20"] = ta.ATR(dataframe, timeperiod=20)

        # BB width relative to history
        dataframe["bb_width_sma"] = ta.SMA(dataframe["bb_width"], timeperiod=50)
        dataframe["bb_tight"] = dataframe["bb_width"] < dataframe["bb_width_sma"]

        # Trend indicators
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        # Trend direction from own timeframe EMAs
        dataframe["trend"] = 0
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe.loc[
            (dataframe["ema_50"] > dataframe["ema_200"]) & (dataframe["adx"] > 20),
            "trend",
        ] = 1
        dataframe.loc[
            (dataframe["ema_50"] < dataframe["ema_200"]) & (dataframe["adx"] > 20),
            "trend",
        ] = -1

        # Momentum
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)

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
            (dataframe["btc_usdt_ema_50_4h"] > dataframe["btc_usdt_ema_200_4h"])
            & (dataframe["btc_usdt_rsi_14_4h"] > 35)
        )
        btc_ok_short = dataframe["btc_usdt_rsi_14_4h"] < 70

        # Keltner channels
        kc_upper = dataframe["kc_mid"] + dataframe["atr_20"] * self.keltner_atr_mult.value
        kc_lower = dataframe["kc_mid"] - dataframe["atr_20"] * self.keltner_atr_mult.value

        squeeze_on = (
            (dataframe["bb_lower"] > kc_lower)
            & (dataframe["bb_upper"] < kc_upper)
        )
        prev_squeeze = squeeze_on.shift(1)
        prev_squeeze = prev_squeeze.infer_objects(copy=False).fillna(False)

        # Long squeeze breakout
        long_squeeze = (
            btc_ok_long
            & (dataframe["trend"] >= 0)
            & (prev_squeeze | dataframe["bb_tight"])
            & (dataframe["close"] > dataframe["bb_upper"])
            & (dataframe["macd_hist"] > 0)
            & (dataframe["adx"] > self.buy_adx_min.value)
            & (dataframe["ema_9"] > dataframe["ema_21"])
            & (dataframe["volume"] > dataframe["volume_sma_20"] * self.buy_volume_mult.value)
            & (dataframe["volume"] > 0)
        )

        long_momentum = (
            btc_ok_long
            & (dataframe["trend"] == 1)
            & (dataframe["close"] > kc_upper)
            & (dataframe["rsi_14"] > 55)
            & (dataframe["rsi_14"] < 75)
            & (dataframe["adx"] > 25)
            & (dataframe["macd_hist"] > 0)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 1.2)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[long_squeeze, ["enter_long", "enter_tag"]] = (1, "long_squeeze_break")
        dataframe.loc[long_momentum & ~long_squeeze, ["enter_long", "enter_tag"]] = (1, "long_momentum")

        # Short squeeze breakout
        short_squeeze = (
            btc_ok_short
            & (dataframe["trend"] <= 0)
            & (prev_squeeze | dataframe["bb_tight"])
            & (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["macd_hist"] < 0)
            & (dataframe["adx"] > self.sell_adx_min.value)
            & (dataframe["ema_9"] < dataframe["ema_21"])
            & (dataframe["volume"] > dataframe["volume_sma_20"] * self.sell_volume_mult.value)
            & (dataframe["volume"] > 0)
        )

        short_momentum = (
            btc_ok_short
            & (dataframe["trend"] == -1)
            & (dataframe["close"] < kc_lower)
            & (dataframe["rsi_14"] < 45)
            & (dataframe["rsi_14"] > 25)
            & (dataframe["adx"] > 25)
            & (dataframe["macd_hist"] < 0)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 1.2)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[short_squeeze, ["enter_short", "enter_tag"]] = (1, "short_squeeze_break")
        dataframe.loc[short_momentum & ~short_squeeze, ["enter_short", "enter_tag"]] = (1, "short_momentum")

        return dataframe

    # Exit / Custom exit / Leverage same as BreakPulse
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, ["exit_long", "exit_tag"]] = (0, "")
        dataframe.loc[:, ["exit_short", "exit_tag"]] = (0, "")
        return dataframe

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return None
        last = dataframe.iloc[-1]
        hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        if hours > 96 and current_profit < 0.005:
            return "exit_time_96h"
        if not trade.is_short and current_profit > 0.005 and last.get("trend", 0) == -1:
            return "long_exit_trend_flip"
        if trade.is_short and current_profit > 0.005 and last.get("trend", 0) == 1:
            return "short_exit_trend_flip"
        if current_profit > 0.03:
            if not trade.is_short and last.get("macd_hist", 0) < 0:
                return "long_exit_momentum_fade"
            if trade.is_short and last.get("macd_hist", 0) > 0:
                return "short_exit_momentum_fade"
        return None

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        return 2.0
