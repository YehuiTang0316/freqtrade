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


class DonchianPulse(IStrategy):
    """
    DonchianPulse v1: Donchian channel breakout on 1h timeframe.

    Classic turtle-trading inspired breakout system adapted for crypto futures.
    Uses Donchian channels (highest high / lowest low over N periods) for entries.
    4h trend filter + BTC health filter.
    """

    INTERFACE_VERSION = 3

    can_short = True
    process_only_new_candles = True

    timeframe = "1h"
    startup_candle_count: int = 200

    # ROI: wide for trend-following breakouts
    minimal_roi = {
        "0": 0.20,
        "240": 0.10,
        "720": 0.04,
        "1440": 0,
    }

    stoploss = -0.07

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.05
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
    donchian_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    buy_adx_min = IntParameter(15, 30, default=20, space="buy", optimize=True)
    buy_volume_mult = DecimalParameter(0.8, 2.5, default=1.2, decimals=1, space="buy", optimize=True)
    sell_adx_min = IntParameter(15, 30, default=20, space="sell", optimize=True)
    sell_volume_mult = DecimalParameter(0.8, 2.5, default=1.0, decimals=1, space="sell", optimize=True)

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
    # 4h trend direction
    # ----------------------------------------------------------------------
    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

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
    # 1h main indicators
    # ----------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Donchian channels: compute for multiple periods (hyperopt will select)
        for period in range(10, 31):
            dataframe[f"dc_upper_{period}"] = dataframe["high"].rolling(window=period).max()
            dataframe[f"dc_lower_{period}"] = dataframe["low"].rolling(window=period).min()
            dataframe[f"dc_mid_{period}"] = (
                dataframe[f"dc_upper_{period}"] + dataframe[f"dc_lower_{period}"]
            ) / 2

        # Previous candle values for breakout detection
        for period in range(10, 31):
            dataframe[f"dc_upper_prev_{period}"] = dataframe[f"dc_upper_{period}"].shift(1)
            dataframe[f"dc_lower_prev_{period}"] = dataframe[f"dc_lower_{period}"].shift(1)

        # Standard indicators
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

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

        period = self.donchian_period.value

        # Donchian breakout detection: close crosses above/below previous channel
        dc_upper = dataframe[f"dc_upper_prev_{period}"]
        dc_lower = dataframe[f"dc_lower_prev_{period}"]

        # ===== LONG: Price breaks above Donchian upper =====
        long_breakout = (
            btc_ok_long
            & (dataframe["trend_4h"] >= 0)
            & (dataframe["close"] > dc_upper)
            & (dataframe["adx"] > self.buy_adx_min.value)
            & (dataframe["ema_9"] > dataframe["ema_21"])
            & (dataframe["rsi_14"] < 80)  # not extreme overbought
            & (dataframe["volume"] > dataframe["volume_sma_20"] * self.buy_volume_mult.value)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[long_breakout, ["enter_long", "enter_tag"]] = (1, "long_donchian_break")

        # ===== SHORT: Price breaks below Donchian lower =====
        short_breakout = (
            btc_ok_short
            & (dataframe["trend_4h"] <= 0)
            & (dataframe["close"] < dc_lower)
            & (dataframe["adx"] > self.sell_adx_min.value)
            & (dataframe["ema_9"] < dataframe["ema_21"])
            & (dataframe["rsi_14"] > 20)  # not extreme oversold
            & (dataframe["volume"] > dataframe["volume_sma_20"] * self.sell_volume_mult.value)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[short_breakout, ["enter_short", "enter_tag"]] = (1, "short_donchian_break")

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

        # Time exit: 48h
        if trade_duration_hours > 48 and current_profit < 0.005:
            return "exit_time_48h"

        # Trend flip exit
        if not trade.is_short and current_profit > 0.005:
            if last_candle.get("trend_4h", 0) == -1:
                return "long_exit_trend_flip"

        if trade.is_short and current_profit > 0.005:
            if last_candle.get("trend_4h", 0) == 1:
                return "short_exit_trend_flip"

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
