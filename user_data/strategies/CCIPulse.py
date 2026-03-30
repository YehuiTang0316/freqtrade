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


class CCIPulse(IStrategy):
    """
    CCIPulse v1: CCI momentum + Heikin Ashi + BB squeeze on 1h.

    CCI > +100 = statistically strong upward momentum
    CCI < -100 = statistically strong downward momentum
    Combined with HA trend confirmation and BB squeeze release.
    Fusion of best elements from HeikinPulse (HA) and BreakPulse (squeeze).
    """

    INTERFACE_VERSION = 3
    can_short = True
    process_only_new_candles = True
    timeframe = "1h"
    startup_candle_count: int = 200

    minimal_roi = {"0": 0.577, "324": 0.161, "1022": 0.062, "2011": 0}
    stoploss = -0.094
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.043
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    order_types = {"entry": "limit", "exit": "limit", "stoploss": "market", "stoploss_on_exchange": False}
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    cci_bull = IntParameter(80, 150, default=100, space="buy", optimize=True)
    cci_bear = IntParameter(-150, -80, default=-100, space="sell", optimize=True)
    keltner_atr_mult = DecimalParameter(1.5, 3.0, default=2.1, decimals=1, space="buy", optimize=True)
    buy_adx_min = IntParameter(15, 30, default=20, space="buy", optimize=True)
    sell_adx_min = IntParameter(10, 25, default=15, space="sell", optimize=True)
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
        # CCI
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=20)

        # Heikin Ashi
        ha = qtpylib.heikinashi(dataframe)
        dataframe["ha_bull"] = (ha["close"] > ha["open"]).astype(int)
        dataframe["ha_bear"] = (ha["close"] < ha["open"]).astype(int)
        dataframe["ha_bull_2"] = dataframe["ha_bull"].rolling(window=2).sum() == 2
        dataframe["ha_bear_2"] = dataframe["ha_bear"].rolling(window=2).sum() == 2

        # BB squeeze
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_upper"] = bollinger["upper"]
        dataframe["bb_mid"] = bollinger["mid"]
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"]
        dataframe["kc_mid"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["atr_20"] = ta.ATR(dataframe, timeperiod=20)
        dataframe["bb_width_sma"] = ta.SMA(dataframe["bb_width"], timeperiod=50)
        dataframe["bb_tight"] = dataframe["bb_width"] < dataframe["bb_width_sma"]

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)
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

        kc_upper = dataframe["kc_mid"] + dataframe["atr_20"] * self.keltner_atr_mult.value
        kc_lower = dataframe["kc_mid"] - dataframe["atr_20"] * self.keltner_atr_mult.value
        squeeze = (dataframe["bb_lower"] > kc_lower) & (dataframe["bb_upper"] < kc_upper)
        prev_squeeze = squeeze.shift(1).infer_objects(copy=False).fillna(False)

        # Long: CCI strong + HA bull + squeeze release
        long_1 = (
            btc_ok_long
            & (dataframe["trend_4h"] >= 0)
            & (prev_squeeze | dataframe["bb_tight"])
            & (dataframe["cci"] > self.cci_bull.value)
            & dataframe["ha_bull_2"]
            & (dataframe["close"] > dataframe["bb_upper"])
            & (dataframe["adx"] > self.buy_adx_min.value)
            & (dataframe["macd_hist"] > 0)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * self.buy_volume_mult.value)
            & (dataframe["volume"] > 0)
        )

        # Long 2: CCI extreme + HA + momentum
        long_2 = (
            btc_ok_long
            & (dataframe["trend_4h"] == 1)
            & (dataframe["cci"] > 150)
            & dataframe["ha_bull_2"]
            & (dataframe["ema_9"] > dataframe["ema_21"])
            & (dataframe["adx"] > 25)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 1.2)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[long_1, ["enter_long", "enter_tag"]] = (1, "long_cci_squeeze")
        dataframe.loc[long_2 & ~long_1, ["enter_long", "enter_tag"]] = (1, "long_cci_extreme")

        # Short
        short_1 = (
            btc_ok_short
            & (dataframe["trend_4h"] <= 0)
            & (prev_squeeze | dataframe["bb_tight"])
            & (dataframe["cci"] < self.cci_bear.value)
            & dataframe["ha_bear_2"]
            & (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["adx"] > self.sell_adx_min.value)
            & (dataframe["macd_hist"] < 0)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * self.sell_volume_mult.value)
            & (dataframe["volume"] > 0)
        )

        short_2 = (
            btc_ok_short
            & (dataframe["trend_4h"] == -1)
            & (dataframe["cci"] < -150)
            & dataframe["ha_bear_2"]
            & (dataframe["ema_9"] < dataframe["ema_21"])
            & (dataframe["adx"] > 25)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 1.0)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[short_1, ["enter_short", "enter_tag"]] = (1, "short_cci_squeeze")
        dataframe.loc[short_2 & ~short_1, ["enter_short", "enter_tag"]] = (1, "short_cci_extreme")

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
