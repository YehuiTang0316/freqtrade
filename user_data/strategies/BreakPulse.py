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


class BreakPulse(IStrategy):
    """
    BreakPulse v1: Volatility squeeze breakout on 1h timeframe.

    Core idea: When Bollinger Bands contract (low volatility), a big move is
    coming. Enter on the breakout in the direction confirmed by higher TF trend.

    Key differences from TrendPulse:
    - TrendPulse = mean-reversion in trend (buy dips/sell rallies)
    - BreakPulse = momentum breakout after squeeze (catch trend initiations)
    """

    INTERFACE_VERSION = 3

    can_short = True
    process_only_new_candles = True

    timeframe = "1h"
    startup_candle_count: int = 200

    # ROI: wide to let breakout trends run
    minimal_roi = {
        "0": 0.30,
        "120": 0.15,
        "480": 0.05,
        "1440": 0,
    }

    # Stoploss: moderate for 2x leverage
    stoploss = -0.05

    # Trailing stop: wide offset to avoid premature exit on breakout retest
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
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
    buy_adx_min = IntParameter(15, 30, default=20, space="buy", optimize=True)
    buy_squeeze_lookback = IntParameter(10, 30, default=20, space="buy", optimize=True)
    buy_volume_mult = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="buy", optimize=True)

    sell_adx_min = IntParameter(15, 30, default=20, space="sell", optimize=True)
    sell_volume_mult = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="sell", optimize=True)

    # Keltner channel ATR multiplier
    keltner_atr_mult = DecimalParameter(1.0, 2.5, default=1.5, decimals=1, space="buy", optimize=True)

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
        # Bollinger Bands (for squeeze detection)
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_upper"] = bollinger["upper"]
        dataframe["bb_mid"] = bollinger["mid"]
        dataframe["bb_width"] = (
            (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"]
        )

        # Keltner Channels base components (ATR mult applied in entry signals for hyperopt)
        dataframe["kc_mid"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["atr_20"] = ta.ATR(dataframe, timeperiod=20)

        # BB width percentile (how tight are bands vs recent history)
        dataframe["bb_width_sma"] = ta.SMA(dataframe["bb_width"], timeperiod=50)
        dataframe["bb_tight"] = dataframe["bb_width"] < dataframe["bb_width_sma"]

        # Momentum indicators
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
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

        # Compute Keltner channels using hyperopt parameter
        kc_upper = dataframe["kc_mid"] + dataframe["atr_20"] * self.keltner_atr_mult.value
        kc_lower = dataframe["kc_mid"] - dataframe["atr_20"] * self.keltner_atr_mult.value

        # Squeeze detection: BB inside Keltner = compression
        squeeze_on = (
            (dataframe["bb_lower"] > kc_lower)
            & (dataframe["bb_upper"] < kc_upper)
        )
        prev_squeeze = squeeze_on.shift(1).fillna(False)

        # ===== LONG: Upward breakout from squeeze =====
        long_squeeze_break = (
            btc_ok_long
            & (dataframe["trend_4h"] >= 0)  # 4h not bearish
            & (
                prev_squeeze                  # was in squeeze
                | dataframe["bb_tight"]       # or bands were tight
            )
            & (dataframe["close"] > dataframe["bb_upper"])  # breaking above BB
            & (dataframe["macd_hist"] > 0)  # MACD momentum positive
            & (dataframe["adx"] > self.buy_adx_min.value)  # trend strength
            & (dataframe["ema_9"] > dataframe["ema_21"])  # short-term momentum up
            & (dataframe["volume"] > dataframe["volume_sma_20"] * self.buy_volume_mult.value)
            & (dataframe["volume"] > 0)
        )

        # Long 2: Strong momentum breakout (no squeeze needed, just trend alignment)
        long_momentum = (
            btc_ok_long
            & (dataframe["trend_4h"] == 1)  # 4h confirmed uptrend
            & (dataframe["close"] > kc_upper)  # above Keltner
            & (dataframe["rsi_14"] > 55)
            & (dataframe["rsi_14"] < 75)  # not overbought
            & (dataframe["adx"] > 25)  # strong trend
            & (dataframe["macd_hist"] > 0)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 1.2)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[long_squeeze_break, ["enter_long", "enter_tag"]] = (1, "long_squeeze_break")
        dataframe.loc[long_momentum & ~long_squeeze_break, ["enter_long", "enter_tag"]] = (1, "long_momentum")

        # ===== SHORT: Downward breakout from squeeze =====
        short_squeeze_break = (
            btc_ok_short
            & (dataframe["trend_4h"] <= 0)  # 4h not bullish
            & (
                prev_squeeze
                | dataframe["bb_tight"]
            )
            & (dataframe["close"] < dataframe["bb_lower"])  # breaking below BB
            & (dataframe["macd_hist"] < 0)
            & (dataframe["adx"] > self.sell_adx_min.value)
            & (dataframe["ema_9"] < dataframe["ema_21"])
            & (dataframe["volume"] > dataframe["volume_sma_20"] * self.sell_volume_mult.value)
            & (dataframe["volume"] > 0)
        )

        short_momentum = (
            btc_ok_short
            & (dataframe["trend_4h"] == -1)
            & (dataframe["close"] < kc_lower)
            & (dataframe["rsi_14"] < 45)
            & (dataframe["rsi_14"] > 25)  # not oversold
            & (dataframe["adx"] > 25)
            & (dataframe["macd_hist"] < 0)
            & (dataframe["volume"] > dataframe["volume_sma_20"] * 1.2)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[short_squeeze_break, ["enter_short", "enter_tag"]] = (1, "short_squeeze_break")
        dataframe.loc[short_momentum & ~short_squeeze_break, ["enter_short", "enter_tag"]] = (1, "short_momentum")

        return dataframe

    # ----------------------------------------------------------------------
    # Exit signals
    # ----------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Rely primarily on ROI / trailing / custom_exit
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

        # Time exit: close after 48h if not profitable
        if trade_duration_hours > 48 and current_profit < 0.005:
            return "exit_time_48h"

        # Trend flip exit: 4h trend reverses while in profit
        if not trade.is_short and current_profit > 0.005:
            if last_candle.get("trend_4h", 0) == -1:
                return "long_exit_trend_flip"

        if trade.is_short and current_profit > 0.005:
            if last_candle.get("trend_4h", 0) == 1:
                return "short_exit_trend_flip"

        # Momentum fade: MACD histogram reverses while in decent profit
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
