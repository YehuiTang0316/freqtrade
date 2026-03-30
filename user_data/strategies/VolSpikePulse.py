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


class VolSpikePulse(IStrategy):
    """
    VolSpikePulse v1: Volume spike breakout on 1h.

    Core idea: Unusually high volume candles signal institutional activity.
    Enter in the direction of the volume spike when confirmed by trend.
    Volume-first approach vs BreakPulse's price-first approach.
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
    vol_spike_mult = DecimalParameter(2.0, 5.0, default=3.0, decimals=1, space="buy", optimize=True)
    buy_adx_min = IntParameter(15, 30, default=20, space="buy", optimize=True)
    sell_adx_min = IntParameter(15, 30, default=15, space="sell", optimize=True)
    candle_body_pct = DecimalParameter(0.3, 0.8, default=0.5, decimals=1, space="buy", optimize=True)

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
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)

        # Volume analysis
        dataframe["volume_sma_20"] = ta.SMA(dataframe["volume"], timeperiod=20)
        dataframe["volume_sma_50"] = ta.SMA(dataframe["volume"], timeperiod=50)

        # Candle body analysis
        dataframe["body"] = abs(dataframe["close"] - dataframe["open"])
        dataframe["range"] = dataframe["high"] - dataframe["low"]
        dataframe["body_pct"] = dataframe["body"] / dataframe["range"].replace(0, np.nan)

        # Bullish/bearish candle
        dataframe["bullish"] = dataframe["close"] > dataframe["open"]
        dataframe["bearish"] = dataframe["close"] < dataframe["open"]

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd_hist"] = macd["macdhist"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        btc_ok_long = (
            (dataframe["btc_usdt_ema_50_1h"] > dataframe["btc_usdt_ema_200_1h"])
            & (dataframe["btc_usdt_rsi_14_1h"] > 35)
        )
        btc_ok_short = dataframe["btc_usdt_rsi_14_1h"] < 70

        # Volume spike detection
        vol_spike = dataframe["volume"] > dataframe["volume_sma_20"] * self.vol_spike_mult.value

        # Strong candle body (not doji)
        strong_body = dataframe["body_pct"] > self.candle_body_pct.value

        # Long: bullish volume spike in uptrend
        long_vol = (
            btc_ok_long
            & (dataframe["trend_4h"] >= 0)
            & vol_spike
            & dataframe["bullish"]
            & strong_body
            & (dataframe["adx"] > self.buy_adx_min.value)
            & (dataframe["ema_9"] > dataframe["ema_21"])
            & (dataframe["rsi_14"] < 75)
            & (dataframe["close"] > dataframe["ema_50"])
            & (dataframe["volume"] > 0)
        )

        # Long 2: volume spike with price closing above recent range
        long_range_break = (
            btc_ok_long
            & (dataframe["trend_4h"] == 1)
            & vol_spike
            & dataframe["bullish"]
            & (dataframe["close"] > dataframe["high"].shift(1))
            & (dataframe["macd_hist"] > 0)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[long_vol, ["enter_long", "enter_tag"]] = (1, "long_vol_spike")
        dataframe.loc[long_range_break & ~long_vol, ["enter_long", "enter_tag"]] = (1, "long_vol_range_break")

        # Short: bearish volume spike in downtrend
        short_vol = (
            btc_ok_short
            & (dataframe["trend_4h"] <= 0)
            & vol_spike
            & dataframe["bearish"]
            & strong_body
            & (dataframe["adx"] > self.sell_adx_min.value)
            & (dataframe["ema_9"] < dataframe["ema_21"])
            & (dataframe["rsi_14"] > 25)
            & (dataframe["close"] < dataframe["ema_50"])
            & (dataframe["volume"] > 0)
        )

        short_range_break = (
            btc_ok_short
            & (dataframe["trend_4h"] == -1)
            & vol_spike
            & dataframe["bearish"]
            & (dataframe["close"] < dataframe["low"].shift(1))
            & (dataframe["macd_hist"] < 0)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[short_vol, ["enter_short", "enter_tag"]] = (1, "short_vol_spike")
        dataframe.loc[short_range_break & ~short_vol, ["enter_short", "enter_tag"]] = (1, "short_vol_range_break")

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
        return None

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        return 2.0
