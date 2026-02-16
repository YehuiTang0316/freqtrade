# TrendPulse v5 - Optimizable trend-following strategy
# v5: Hyperopt-ready with optimizable parameters + 1h timeframe for less noise
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    IntParameter,
    DecimalParameter,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
from technical import qtpylib


class TrendPulse(IStrategy):
    """
    TrendPulse v5: Optimizable trend-following strategy on 1h timeframe.

    Uses 4h for trend direction, 1h for entries.
    Hyperopt-ready for parameter optimization.
    """

    INTERFACE_VERSION = 3

    can_short: bool = True

    # ROI (optimized by hyperopt)
    minimal_roi = {
        "0": 0.365,
        "149": 0.232,
        "644": 0.073,
        "1260": 0,
    }

    # Stoploss (optimized by hyperopt)
    stoploss = -0.067

    # Trailing stop (optimized by hyperopt)
    trailing_stop = True
    trailing_stop_positive = 0.295
    trailing_stop_positive_offset = 0.321
    trailing_only_offset_is_reached = True

    # 1h base timeframe
    timeframe = "1h"
    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 200

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # Optimized parameters (hyperopt round 2)
    buy_rsi_low = IntParameter(15, 35, default=35, space="buy", optimize=True)
    buy_rsi_high_1h = IntParameter(30, 60, default=44, space="buy", optimize=True)
    buy_williams_r = IntParameter(-98, -80, default=-94, space="buy", optimize=True)

    sell_rsi_high = IntParameter(65, 85, default=65, space="sell", optimize=True)
    sell_rsi_low_1h = IntParameter(40, 70, default=52, space="sell", optimize=True)
    sell_williams_r = IntParameter(-20, -2, default=-3, space="sell", optimize=True)

    # -----------------------------------------------------------------------
    # Informative indicators - 4h (trend direction)
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # BTC filter
    # -----------------------------------------------------------------------
    @informative("1h", "BTC/USDT:USDT")
    def populate_indicators_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    # -----------------------------------------------------------------------
    # Main 1h indicators
    # -----------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_3"] = ta.RSI(dataframe, timeperiod=3)

        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_upper"] = bollinger["upper"]
        dataframe["bb_mid"] = bollinger["mid"]

        dataframe["williams_r"] = ta.WILLR(dataframe, timeperiod=14)
        dataframe["volume_sma_20"] = ta.SMA(dataframe["volume"], timeperiod=20)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd_hist"] = macd["macdhist"]

        return dataframe

    # -----------------------------------------------------------------------
    # Entry signals
    # -----------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        btc_ok = dataframe["btc_usdt_rsi_14_1h"] > 30

        # ===== LONG: Oversold in uptrend =====
        long_1 = (
            btc_ok
            & (dataframe["trend_4h"] == 1)
            & (dataframe["rsi_14"] < self.buy_rsi_low.value)
            & (dataframe["rsi_3"] > 3)
            & (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["williams_r"] < self.buy_williams_r.value)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 0.5)
            & (dataframe["volume"] > 0)
        )

        # Long 2: Oversold in neutral
        long_2 = (
            btc_ok
            & (dataframe["trend_4h"] >= 0)
            & (dataframe["rsi_14"] < 20)
            & (dataframe["rsi_3"] > 2)
            & (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["williams_r"] < -95)
            & (dataframe["volume"] > 0)
        )

        # Long 3: Pullback in uptrend (not extreme oversold)
        long_3 = (
            btc_ok
            & (dataframe["trend_4h"] == 1)
            & (dataframe["rsi_14"] < 40)
            & (dataframe["rsi_14"] > 20)
            & (dataframe["close"] > dataframe["ema_50"])
            & (dataframe["close"] < dataframe["bb_mid"])
            & (dataframe["macd_hist"] < 0)
            & (dataframe["rsi_14_4h"] < 65)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 0.5)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[long_1, ["enter_long", "enter_tag"]] = (1, "long_oversold")
        dataframe.loc[long_2 & ~long_1, ["enter_long", "enter_tag"]] = (1, "long_deep_oversold")
        dataframe.loc[long_3 & ~long_1 & ~long_2, ["enter_long", "enter_tag"]] = (1, "long_pullback")

        # ===== SHORT: Overbought in downtrend =====
        short_1 = (
            (dataframe["trend_4h"] == -1)
            & (dataframe["rsi_14"] > self.sell_rsi_high.value)
            & (dataframe["rsi_3"] < 97)
            & (dataframe["close"] > dataframe["bb_upper"])
            & (dataframe["williams_r"] > self.sell_williams_r.value)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 0.5)
            & (dataframe["volume"] > 0)
        )

        # Short 2: Overbought in neutral
        short_2 = (
            (dataframe["trend_4h"] <= 0)
            & (dataframe["rsi_14"] > 80)
            & (dataframe["rsi_3"] < 98)
            & (dataframe["close"] > dataframe["bb_upper"])
            & (dataframe["williams_r"] > -5)
            & (dataframe["volume"] > 0)
        )

        # Short 3: Rally failure in downtrend
        short_3 = (
            (dataframe["trend_4h"] == -1)
            & (dataframe["rsi_14"] > 55)
            & (dataframe["rsi_14"] < 75)
            & (dataframe["close"] < dataframe["ema_50"])
            & (dataframe["close"] > dataframe["bb_mid"])
            & (dataframe["macd_hist"] > 0)
            & (dataframe["rsi_14_4h"] > 35)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 0.5)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[short_1, ["enter_short", "enter_tag"]] = (1, "short_overbought")
        dataframe.loc[short_2 & ~short_1, ["enter_short", "enter_tag"]] = (1, "short_deep_overbought")
        dataframe.loc[short_3 & ~short_1 & ~short_2, ["enter_short", "enter_tag"]] = (1, "short_rally_fail")

        return dataframe

    # -----------------------------------------------------------------------
    # Exit signals
    # -----------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, ["exit_long", "exit_tag"]] = (0, "")
        dataframe.loc[:, ["exit_short", "exit_tag"]] = (0, "")
        return dataframe

    # -----------------------------------------------------------------------
    # Custom exit
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

        # Time exit: close after 48h if not profitable
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600
        if trade_duration > 48 and current_profit < 0.005:
            return "exit_time_48h"

        # Protect profits: trend flip
        if trade.is_short and current_profit > 0.005 and last_candle["trend_4h"] == 1:
            return "short_exit_trend_flip"
        if not trade.is_short and current_profit > 0.005 and last_candle["trend_4h"] == -1:
            return "long_exit_trend_flip"

        return None

    # -----------------------------------------------------------------------
    # Leverage
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
