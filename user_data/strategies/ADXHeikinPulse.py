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


class ADXHeikinPulse(IStrategy):
    """
    ADXSqueezePulse v1: ADX compression -> expansion breakout on 1h.

    Alternative compression detection to BreakPulse's BB/Keltner squeeze.
    When ADX drops below threshold (no trend) then rises sharply (trend starting),
    enter in the direction of the new trend.

    Hypothesis: ADX compression may catch different breakouts than BB/Keltner.
    """

    INTERFACE_VERSION = 3
    can_short = True
    process_only_new_candles = True
    timeframe = "1h"
    startup_candle_count: int = 200

    minimal_roi = {"0": 0.187, "352": 0.14, "874": 0.048, "2222": 0}
    stoploss = -0.087
    trailing_stop = True
    trailing_stop_positive = 0.021
    trailing_stop_positive_offset = 0.07
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {"entry": "limit", "exit": "limit", "stoploss": "market", "stoploss_on_exchange": False}
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # Hyperopt
    adx_low = IntParameter(10, 25, default=18, space="buy", optimize=True)
    adx_rising_min = DecimalParameter(1.0, 5.0, default=2.0, decimals=1, space="buy", optimize=True)
    buy_volume_mult = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="buy", optimize=True)
    sell_volume_mult = DecimalParameter(0.5, 2.0, default=1.0, decimals=1, space="sell", optimize=True)

    @informative("1h", "BTC/USDT:USDT")
    def populate_indicators_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["trend"] = 0
        dataframe.loc[(dataframe["ema_50"] > dataframe["ema_200"]) & (dataframe["adx"] > 20), "trend"] = 1
        dataframe.loc[(dataframe["ema_50"] < dataframe["ema_200"]) & (dataframe["adx"] > 20), "trend"] = -1
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ADX and directional indicators
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)

        # ADX recent minimum (was trend weak recently?)
        dataframe["adx_min_5"] = dataframe["adx"].rolling(window=5).min()
        # ADX change (is it rising?)
        dataframe["adx_change"] = dataframe["adx"] - dataframe["adx"].shift(1)

        # Heikin Ashi trend confirmation
        ha = qtpylib.heikinashi(dataframe)
        dataframe["ha_bull"] = (ha["close"] > ha["open"]).astype(int)
        dataframe["ha_bear"] = (ha["close"] < ha["open"]).astype(int)
        dataframe["ha_bull_3"] = dataframe["ha_bull"].rolling(window=3).sum() == 3
        dataframe["ha_bear_3"] = dataframe["ha_bear"].rolling(window=3).sum() == 3

        # Standard
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_upper"] = bollinger["upper"]
        dataframe["bb_lower"] = bollinger["lower"]

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd_hist"] = macd["macdhist"]

        dataframe["volume_sma_20"] = ta.SMA(dataframe["volume"], timeperiod=20)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        btc_ok_long = (
            (dataframe["btc_usdt_ema_50_1h"] > dataframe["btc_usdt_ema_200_1h"])
            & (dataframe["btc_usdt_rsi_14_1h"] > 35)
        )
        btc_ok_short = dataframe["btc_usdt_rsi_14_1h"] < 70

        # ADX squeeze detection: was low recently, now rising
        adx_was_low = dataframe["adx_min_5"] < self.adx_low.value
        adx_rising = dataframe["adx_change"] > self.adx_rising_min.value

        # Long: ADX squeeze break + bullish direction + HA confirmation
        long_adx = (
            btc_ok_long
            & (dataframe["trend_4h"] >= 0)
            & adx_was_low
            & adx_rising
            & dataframe["ha_bull_3"]
            & (dataframe["plus_di"] > dataframe["minus_di"])
            & (dataframe["close"] > dataframe["bb_upper"])
            & (dataframe["ema_9"] > dataframe["ema_21"])
            & (dataframe["macd_hist"] > 0)
            & (dataframe["rsi_14"] < 80)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * self.buy_volume_mult.value)
            & (dataframe["volume"] > 0)
        )

        long_momentum = (
            btc_ok_long
            & (dataframe["trend_4h"] == 1)
            & (dataframe["adx"] > 30)
            & (dataframe["plus_di"] > dataframe["minus_di"])
            & (dataframe["close"] > dataframe["ema_50"])
            & (dataframe["rsi_14"] > 50)
            & (dataframe["rsi_14"] < 75)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 1.2)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[long_adx, ["enter_long", "enter_tag"]] = (1, "long_adx_squeeze")
        dataframe.loc[long_momentum & ~long_adx, ["enter_long", "enter_tag"]] = (1, "long_adx_momentum")

        # Short: ADX squeeze break + bearish direction
        short_adx = (
            btc_ok_short
            & (dataframe["trend_4h"] <= 0)
            & adx_was_low
            & adx_rising
            & dataframe["ha_bear_3"]
            & (dataframe["minus_di"] > dataframe["plus_di"])
            & (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["ema_9"] < dataframe["ema_21"])
            & (dataframe["macd_hist"] < 0)
            & (dataframe["rsi_14"] > 20)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * self.sell_volume_mult.value)
            & (dataframe["volume"] > 0)
        )

        short_momentum = (
            btc_ok_short
            & (dataframe["trend_4h"] == -1)
            & (dataframe["adx"] > 30)
            & (dataframe["minus_di"] > dataframe["plus_di"])
            & (dataframe["close"] < dataframe["ema_50"])
            & (dataframe["rsi_14"] < 50)
            & (dataframe["rsi_14"] > 25)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 1.2)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[short_adx, ["enter_short", "enter_tag"]] = (1, "short_adx_squeeze")
        dataframe.loc[short_momentum & ~short_adx, ["enter_short", "enter_tag"]] = (1, "short_adx_momentum")

        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe.loc[:, ["exit_long", "exit_tag"]] = (0, "")
        dataframe.loc[:, ["exit_short", "exit_tag"]] = (0, "")
        return dataframe

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return None
        last = dataframe.iloc[-1]
        hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        if hours > 48 and current_profit < 0.005:
            return "exit_time_48h"
        if not trade.is_short and current_profit > 0.005 and last.get("trend_4h", 0) == -1:
            return "long_exit_trend_flip"
        if trade.is_short and current_profit > 0.005 and last.get("trend_4h", 0) == 1:
            return "short_exit_trend_flip"
        if current_profit > 0.02:
            if not trade.is_short and last.get("macd_hist", 0) < 0:
                return "long_exit_momentum_fade"
            if trade.is_short and last.get("macd_hist", 0) > 0:
                return "short_exit_momentum_fade"
        return None

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        return 2.0
