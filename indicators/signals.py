"""
indicators/signals.py — Room 2: Technical Indicator Engine
============================================================
Responsibility: take a raw OHLCV DataFrame and return it enriched with every
indicator value the strategy needs.  No trading decisions are made here.

Why the `ta` library?
  `ta` wraps pandas-ta and provides a clean, Pythonic interface for 80+ indicators.
  All calculations are vectorised over the full DataFrame — no row-by-row loops.

Design principle — non-destructive:
  We always work on a copy of the input DataFrame and return the enriched copy.
  The caller's original DataFrame is never mutated.
"""

import pandas as pd
from loguru import logger
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

import config


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators required by the Triple Confirmation strategy
    and return an enriched DataFrame.

    New columns added:
        ema_fast      — Exponential Moving Average (fast period, default 9)
        ema_slow      — Exponential Moving Average (slow period, default 21)
        rsi           — Relative Strength Index (default 14)
        bb_upper      — Bollinger Band upper band
        bb_middle     — Bollinger Band middle band (SMA)
        bb_lower      — Bollinger Band lower band
        bb_width      — Band width normalised: (upper - lower) / middle
        atr           — Average True Range (default 14)
        macd          — MACD line (12/26 EMA difference)
        macd_signal   — MACD signal line (9-period EMA of MACD)
        volume_sma    — 20-period simple moving average of volume

    Args:
        df: Raw OHLCV DataFrame as returned by data/fetcher.py.
            Required columns: open, high, low, close, volume.

    Returns:
        A new DataFrame (copy of input) with all indicator columns appended.
        NaN rows at the beginning (warm-up period) are dropped.

    Raises:
        ValueError: If the DataFrame has too few rows to compute indicators reliably.
    """
    # --- Guard: minimum rows check ---
    # The slowest indicator (EMA 21) needs 21 bars minimum, but we require 50
    # to ensure all indicators have fully "warmed up" before we trust their values.
    if len(df) < config.CANDLES_REQUIRED:
        raise ValueError(
            f"DataFrame has only {len(df)} rows — need at least "
            f"{config.CANDLES_REQUIRED} (CANDLES_REQUIRED) to compute indicators."
        )

    # Work on a copy so we never mutate the caller's data.
    df = df.copy()

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # --- EMA Fast & Slow (Trend Filter) ---
    # EMA gives more weight to recent prices than a simple moving average.
    # Fast (9) reacts quickly to price changes; slow (21) filters out noise.
    df["ema_fast"] = EMAIndicator(
        close=close, window=config.EMA_FAST, fillna=False
    ).ema_indicator()

    df["ema_slow"] = EMAIndicator(
        close=close, window=config.EMA_SLOW, fillna=False
    ).ema_indicator()

    # --- RSI (Momentum Filter) ---
    # RSI oscillates 0-100.  Values above 50 mean buyers are in control.
    # Values above 70 typically indicate overbought conditions.
    df["rsi"] = RSIIndicator(
        close=close, window=config.RSI_PERIOD, fillna=False
    ).rsi()

    # --- Bollinger Bands (Volatility Filter) ---
    # Standard Bollinger setup: 20-period SMA ± 2 standard deviations.
    bb = BollingerBands(
        close=close,
        window=config.BB_PERIOD,
        window_dev=config.BB_STD,
        fillna=False,
    )
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"]  = bb.bollinger_lband()

    # Band Width normalised by the middle band.
    # This makes BBW comparable across different price levels and assets.
    # Expanding BBW → the market is breaking out of consolidation.
    # Guard against division by zero (bb_middle = 0 is theoretically possible
    # if all closes in the window were 0, though effectively impossible on BTC).
    bb_middle_safe = df["bb_middle"].replace(0, float("nan"))
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_middle_safe

    # --- ATR (Volatility Measure for Stop/TP Placement) ---
    # ATR measures the average range between high and low over N periods.
    # A higher ATR means more volatile conditions → wider stops are appropriate.
    df["atr"] = AverageTrueRange(
        high=high,
        low=low,
        close=close,
        window=config.ATR_PERIOD,
        fillna=False,
    ).average_true_range()

    # --- MACD (Supplementary Momentum Indicator) ---
    # Not used in the core signal logic (Phase 1), but logged for analysis
    # and available for Phase 2 feature engineering.
    macd_obj = MACD(close=close, fillna=False)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()

    # --- Volume SMA ---
    # Rolling 20-period average volume — useful for confirming breakouts.
    # High volume + price expansion = more reliable signal.
    df["volume_sma"] = df["volume"].rolling(window=20).mean()

    # --- Drop NaN rows (indicator warm-up period) ---
    # The first ~21 rows will have NaN for the slowest indicators.
    # We drop them to prevent the strategy from ever seeing incomplete data.
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_dropped = rows_before - len(df)

    if rows_dropped > 0:
        logger.debug(f"Dropped {rows_dropped} NaN warm-up rows after indicator computation.")

    # --- Sanity check: warn if any unexpected NaNs remain ---
    indicator_cols = [
        "ema_fast", "ema_slow", "rsi",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "atr", "macd", "macd_signal", "volume_sma",
    ]
    for col in indicator_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logger.warning(
                f"Unexpected NaN values in '{col}' after dropna: {nan_count} rows. "
                "Indicator computation may be unreliable."
            )

    logger.debug(
        f"Indicators computed. Rows available: {len(df)} | "
        f"Latest close: {df['close'].iloc[-1]:.2f} | "
        f"EMA9: {df['ema_fast'].iloc[-1]:.2f} | "
        f"EMA21: {df['ema_slow'].iloc[-1]:.2f} | "
        f"RSI: {df['rsi'].iloc[-1]:.1f} | "
        f"ATR: {df['atr'].iloc[-1]:.2f}"
    )

    return df


# ---------------------------------------------------------------------------
# Standalone test — run `python indicators/signals.py` from the project root
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from loguru import logger

    from data.fetcher import fetch_ohlcv

    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="DEBUG",
        format="{time:HH:mm:ss} | {level} | {message}",
    )

    logger.info("=== Signals standalone test ===")

    df_raw = fetch_ohlcv(limit=100)
    df_ind = compute_indicators(df_raw)

    last = df_ind.iloc[-1]
    print("\n--- Latest candle with all indicators ---")
    print(f"  Timestamp  : {last['timestamp']}")
    print(f"  Close      : ${last['close']:.2f}")
    print(f"  EMA Fast   : {last['ema_fast']:.2f}")
    print(f"  EMA Slow   : {last['ema_slow']:.2f}")
    print(f"  RSI        : {last['rsi']:.2f}")
    print(f"  BB Upper   : {last['bb_upper']:.2f}")
    print(f"  BB Middle  : {last['bb_middle']:.2f}")
    print(f"  BB Lower   : {last['bb_lower']:.2f}")
    print(f"  BB Width   : {last['bb_width']:.4f}")
    print(f"  ATR        : {last['atr']:.2f}")
    print(f"  MACD       : {last['macd']:.4f}")
    print(f"  MACD Sig   : {last['macd_signal']:.4f}")
    print(f"  Vol SMA    : {last['volume_sma']:.4f}")
    print(f"\n  Total rows after indicator computation: {len(df_ind)}")
