# NFixSimplified - Simplified strategy based on NostalgiaForInfinityX7 core logic
# Core idea: Multi-timeframe oversold bounce capture + layered exit management
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
from technical import qtpylib


class NFixSimplified(IStrategy):
    """
    Simplified NostalgiaForInfinityX7 strategy.
    Multi-timeframe oversold bounce capture with layered RSI exits.
    Supports both long and short on futures.
    """

    INTERFACE_VERSION = 3

    can_short: bool = True
    position_adjustment_enable = True

    # ROI - high value as safety net, rely on custom_exit
    minimal_roi = {"0": 0.20}

    stoploss = -0.10
    trailing_stop = False

    timeframe = "5m"
    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Need 200 candles for EMA_200
    startup_candle_count: int = 200

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # -----------------------------------------------------------------------
    # Informative indicators - BTC market filter (1h)
    # -----------------------------------------------------------------------
    @informative("1h", "BTC/USDT:USDT")
    def populate_indicators_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi_3"] = ta.RSI(dataframe, timeperiod=3)
        return dataframe

    # -----------------------------------------------------------------------
    # Informative indicators - 15m
    # -----------------------------------------------------------------------
    @informative("15m")
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi_3"] = ta.RSI(dataframe, timeperiod=3)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=3, fastd_period=3)
        dataframe["stochrsi_k"] = stoch_rsi["fastk"]
        dataframe["stochrsi_d"] = stoch_rsi["fastd"]
        return dataframe

    # -----------------------------------------------------------------------
    # Informative indicators - 1h
    # -----------------------------------------------------------------------
    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi_3"] = ta.RSI(dataframe, timeperiod=3)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=3, fastd_period=3)
        dataframe["stochrsi_k"] = stoch_rsi["fastk"]
        dataframe["stochrsi_d"] = stoch_rsi["fastd"]
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        aroon = ta.AROON(dataframe, timeperiod=14)
        dataframe["aroon_up"] = aroon["aroonup"]
        return dataframe

    # -----------------------------------------------------------------------
    # Informative indicators - 4h
    # -----------------------------------------------------------------------
    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi_3"] = ta.RSI(dataframe, timeperiod=3)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=3, fastd_period=3)
        dataframe["stochrsi_k"] = stoch_rsi["fastk"]
        dataframe["stochrsi_d"] = stoch_rsi["fastd"]
        aroon = ta.AROON(dataframe, timeperiod=14)
        dataframe["aroon_up"] = aroon["aroonup"]
        return dataframe

    # -----------------------------------------------------------------------
    # Main 5m indicators
    # -----------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe["rsi_3"] = ta.RSI(dataframe, timeperiod=3)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)

        # EMA 200 - trend filter
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_upperband"] = bollinger["upper"]

        # Williams %R
        dataframe["williams_r"] = ta.WILLR(dataframe, timeperiod=14)

        # Close max 12 candles - for drawdown detection
        dataframe["close_max_12"] = dataframe["close"].rolling(12).max()

        return dataframe

    # -----------------------------------------------------------------------
    # Entry signals
    # -----------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ===== Global protection for longs =====
        global_long_protection = (
            (
                (dataframe["rsi_3"] > 2.0)
                | (dataframe["rsi_14_1h"] < 30.0)
                | (dataframe["rsi_14_4h"] < 30.0)
            )
            & (
                (dataframe["rsi_3_1h"] > 16.0)
                | (dataframe["rsi_14_1h"] < 40.0)
                | (dataframe["stochrsi_k_1h"] < 20.0)
            )
        )

        # ===== Long Entry 1: Standard oversold bounce =====
        long_entry_1 = (
            global_long_protection
            & (dataframe["rsi_3"] > 6.0)
            & (dataframe["rsi_14"] < 36.0)
            & (dataframe["rsi_3_15m"] > 6.0)
            & (dataframe["rsi_14_1h"] < 80.0)
            & (dataframe["rsi_14_4h"] < 80.0)
            & (dataframe["close"] < dataframe["bb_lowerband"])
            & (dataframe["btc_usdt_rsi_3_1h"] > 10.0)
            & (dataframe["volume"] > 0)
        )

        # ===== Long Entry 2: Deep oversold bounce =====
        long_entry_2 = (
            global_long_protection
            & (dataframe["rsi_3"] > 2.0)
            & (dataframe["rsi_14"] < 20.0)
            & (dataframe["rsi_3_1h"] > 5.0)
            & (dataframe["rsi_14_1h"] < 40.0)
            & (dataframe["close"] > dataframe["close_max_12"] * 0.92)
            & (dataframe["btc_usdt_rsi_3_1h"] > 10.0)
            & (dataframe["volume"] > 0)
        )

        # ===== Long Entry 3: Momentum recovery =====
        long_entry_3 = (
            global_long_protection
            & (dataframe["rsi_3"] > 12.0)
            & (dataframe["rsi_14"] < 46.0)
            & (dataframe["stochrsi_k_1h"] < 30.0)
            & (dataframe["aroon_up_4h"] < 50.0)
            & (dataframe["rsi_14_4h"] < 70.0)
            & (dataframe["btc_usdt_rsi_3_1h"] > 10.0)
            & (dataframe["volume"] > 0)
        )

        # Combine long entries with tags
        dataframe.loc[long_entry_1, ["enter_long", "enter_tag"]] = (1, "long_1_std_oversold")
        dataframe.loc[
            long_entry_2 & ~long_entry_1,
            ["enter_long", "enter_tag"],
        ] = (1, "long_2_deep_oversold")
        dataframe.loc[
            long_entry_3 & ~long_entry_1 & ~long_entry_2,
            ["enter_long", "enter_tag"],
        ] = (1, "long_3_momentum_recovery")

        # ===== Short Entry 1: Standard overbought pullback =====
        short_entry_1 = (
            (dataframe["rsi_3"] < 94.0)
            & (dataframe["rsi_14"] > 64.0)
            & (dataframe["rsi_14_1h"] > 20.0)
            & (dataframe["rsi_14_4h"] > 20.0)
            & (dataframe["close"] > dataframe["bb_upperband"])
            & (dataframe["volume"] > 0)
        )

        # ===== Short Entry 2: Deep overbought pullback =====
        short_entry_2 = (
            (dataframe["rsi_3"] < 98.0)
            & (dataframe["rsi_14"] > 80.0)
            & (dataframe["rsi_3_1h"] < 95.0)
            & (dataframe["rsi_14_1h"] > 60.0)
            & (dataframe["volume"] > 0)
        )

        # Combine short entries with tags
        dataframe.loc[short_entry_1, ["enter_short", "enter_tag"]] = (1, "short_1_std_overbought")
        dataframe.loc[
            short_entry_2 & ~short_entry_1,
            ["enter_short", "enter_tag"],
        ] = (1, "short_2_deep_overbought")

        return dataframe

    # -----------------------------------------------------------------------
    # Exit signals (placeholder - main logic in custom_exit)
    # -----------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, ["exit_long", "exit_tag"]] = (0, "")
        dataframe.loc[:, ["exit_short", "exit_tag"]] = (0, "")
        return dataframe

    # -----------------------------------------------------------------------
    # Custom exit - layered RSI profit-taking
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
        rsi_14 = last_candle["rsi_14"]
        williams_r = last_candle["williams_r"]
        ema_200 = last_candle["ema_200"]
        close = last_candle["close"]

        if trade.is_short:
            return self._exit_short(current_profit, rsi_14, williams_r, close, ema_200)
        else:
            return self._exit_long(current_profit, rsi_14, williams_r, close, ema_200)

    def _exit_long(
        self,
        current_profit: float,
        rsi_14: float,
        williams_r: float,
        close: float,
        ema_200: float,
    ) -> str | None:
        # Williams %R extreme exit
        if williams_r > -1.0 and current_profit > 0.01:
            return "long_exit_williams_r"

        above_ema = close > ema_200

        # Layered RSI exit based on profit tiers
        if above_ema:
            # Above EMA_200 - tighter exits (trend is favorable, take profit earlier on weakness)
            if current_profit > 0.20:
                if rsi_14 < 42.0:
                    return "long_exit_profit_20_above"
            elif current_profit > 0.10:
                if rsi_14 < 46.0:
                    return "long_exit_profit_10_above"
            elif current_profit > 0.06:
                if rsi_14 < 40.0:
                    return "long_exit_profit_6_above"
            elif current_profit > 0.04:
                if rsi_14 < 36.0:
                    return "long_exit_profit_4_above"
            elif current_profit > 0.02:
                if rsi_14 < 32.0:
                    return "long_exit_profit_2_above"
            elif current_profit > 0.005:
                if rsi_14 < 26.0:
                    return "long_exit_profit_05_above"
        else:
            # Below EMA_200 - slightly looser exits (counter-trend, give more room)
            if current_profit > 0.20:
                if rsi_14 < 44.0:
                    return "long_exit_profit_20_below"
            elif current_profit > 0.10:
                if rsi_14 < 48.0:
                    return "long_exit_profit_10_below"
            elif current_profit > 0.06:
                if rsi_14 < 42.0:
                    return "long_exit_profit_6_below"
            elif current_profit > 0.04:
                if rsi_14 < 38.0:
                    return "long_exit_profit_4_below"
            elif current_profit > 0.02:
                if rsi_14 < 34.0:
                    return "long_exit_profit_2_below"
            elif current_profit > 0.005:
                if rsi_14 < 28.0:
                    return "long_exit_profit_05_below"

        return None

    def _exit_short(
        self,
        current_profit: float,
        rsi_14: float,
        williams_r: float,
        close: float,
        ema_200: float,
    ) -> str | None:
        # Williams %R extreme exit (inverted for shorts)
        if williams_r < -99.0 and current_profit > 0.01:
            return "short_exit_williams_r"

        below_ema = close < ema_200

        # Layered RSI exit based on profit tiers (inverted thresholds for shorts)
        if below_ema:
            # Below EMA_200 - tighter exits for shorts (trend favorable for shorts)
            if current_profit > 0.20:
                if rsi_14 > 58.0:
                    return "short_exit_profit_20_below"
            elif current_profit > 0.10:
                if rsi_14 > 54.0:
                    return "short_exit_profit_10_below"
            elif current_profit > 0.06:
                if rsi_14 > 60.0:
                    return "short_exit_profit_6_below"
            elif current_profit > 0.04:
                if rsi_14 > 64.0:
                    return "short_exit_profit_4_below"
            elif current_profit > 0.02:
                if rsi_14 > 68.0:
                    return "short_exit_profit_2_below"
            elif current_profit > 0.005:
                if rsi_14 > 74.0:
                    return "short_exit_profit_05_below"
        else:
            # Above EMA_200 - looser exits for shorts (counter-trend)
            if current_profit > 0.20:
                if rsi_14 > 56.0:
                    return "short_exit_profit_20_above"
            elif current_profit > 0.10:
                if rsi_14 > 52.0:
                    return "short_exit_profit_10_above"
            elif current_profit > 0.06:
                if rsi_14 > 58.0:
                    return "short_exit_profit_6_above"
            elif current_profit > 0.04:
                if rsi_14 > 62.0:
                    return "short_exit_profit_4_above"
            elif current_profit > 0.02:
                if rsi_14 > 66.0:
                    return "short_exit_profit_2_above"
            elif current_profit > 0.005:
                if rsi_14 > 72.0:
                    return "short_exit_profit_05_above"

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
        return 3.0

    # -----------------------------------------------------------------------
    # DCA - Dollar Cost Averaging (long only)
    # -----------------------------------------------------------------------
    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:
        # Only DCA for longs
        if trade.is_short:
            return None

        # Only 1 additional entry allowed
        if trade.nr_of_successful_entries >= 2:
            return None

        # DCA when loss reaches -5%
        if current_profit > -0.05:
            return None

        # Add 0.5x of initial stake
        try:
            stake_amount = trade.stake_amount * 0.5
            if min_stake is not None and stake_amount < min_stake:
                stake_amount = min_stake
            if stake_amount > max_stake:
                return None
            return stake_amount, "dca_long_-5pct"
        except Exception:
            return None
