"""
strategy/engine.py — Room 3: Signal Decision Logic
====================================================
Responsibility: examine the latest indicator values and return a structured
trading signal.  No order placement happens here — this module only decides
what SHOULD happen, not how to execute it.

The Triple Confirmation Strategy:
  Signal 1 — EMA crossover (trend filter): EMA(9) > EMA(21)
  Signal 2 — RSI in range (momentum filter): 50 < RSI < 70
  Signal 3 — BB Width expanding (volatility filter): current BBW > previous BBW

ALL three must be true simultaneously to trigger a buy.
Exit when EMA(9) crosses back below EMA(21) — the same trend filter that let us in
  is the one that kicks us out.

Design note — "second-to-last row" rule:
  We always evaluate on the SECOND-TO-LAST candle (index -2), never the last (-1).
  The last candle is currently forming — its close/RSI/EMA values will change until
  the candle closes.  Evaluating on it would mean trading on incomplete data.
  The second-to-last candle is fully closed and its values are final.
"""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
del _sys, _pathlib

from datetime import datetime, timezone

from loguru import logger

import config


def evaluate_signal(df) -> dict:
    """
    Evaluate the Triple Confirmation strategy on the most recently CLOSED candle.

    Always reads from df.iloc[-2] (second-to-last row = last fully closed candle).
    The current forming candle at df.iloc[-1] is intentionally ignored.

    Args:
        df: Indicator-enriched DataFrame as returned by indicators/signals.py.
            Must contain: ema_fast, ema_slow, rsi, bb_width, atr, close, timestamp.
            Must have at least 2 rows (need both -1 and -2 for BBW comparison).

    Returns:
        dict with keys:
            action       — "buy" | "sell" | "hold"
            reason       — Human-readable explanation of the decision
            entry_price  — Close price of the evaluated candle
            stop_loss    — Entry - (STOP_ATR_MULTIPLIER × ATR)
            take_profit  — Entry + (TP_ATR_MULTIPLIER × ATR)
            atr          — Current ATR value (stop distance reference)
            confidence   — Placeholder: 1.0 (Phase 2 will replace with ML probability)
            timestamp    — ISO-8601 UTC string of the evaluated candle

    Raises:
        ValueError: If the DataFrame does not have enough rows to evaluate.
    """
    if len(df) < 3:
        raise ValueError(
            f"DataFrame must have at least 3 rows to evaluate a signal. Got {len(df)}."
        )

    # The "current" closed candle — this is what we trade on.
    row = df.iloc[-2]

    # Previous closed candle — needed for BBW expansion check.
    # len(df) >= 3 is guaranteed by the guard above, so this is always safe.
    prev = df.iloc[-3]

    # --- Extract indicator values ---
    ema_fast     = row["ema_fast"]
    ema_slow     = row["ema_slow"]
    rsi          = row["rsi"]
    bb_width_now = row["bb_width"]
    bb_width_prv = prev["bb_width"]
    atr          = row["atr"]
    close        = row["close"]
    ts           = str(row["timestamp"])

    # --- Compute stops and targets ---
    stop_loss   = close - (config.STOP_ATR_MULTIPLIER * atr)
    take_profit = close + (config.TP_ATR_MULTIPLIER * atr)

    # --- Triple Confirmation Logic ---

    # Signal 1: EMA crossover — short-term trend agrees with medium-term trend.
    ema_crossover = ema_fast > ema_slow

    # Signal 2: RSI in the "momentum confirmed, not overbought" zone.
    rsi_in_range = config.RSI_LOWER < rsi < config.RSI_UPPER

    # Signal 3: Bollinger Band Width is expanding — the market is in motion, not
    # consolidating.  Buying during expansion avoids whipsaws in tight ranges.
    bb_expanding = bb_width_now > bb_width_prv

    # --- Build reason string for logging and transparency ---
    conditions = {
        f"EMA({config.EMA_FAST}) > EMA({config.EMA_SLOW})": ema_crossover,
        f"{config.RSI_LOWER} < RSI < {config.RSI_UPPER}":   rsi_in_range,
        "BBW expanding":                                      bb_expanding,
    }
    passed = [name for name, ok in conditions.items() if ok]
    failed = [name for name, ok in conditions.items() if not ok]

    # --- Determine action ---
    if ema_crossover and rsi_in_range and bb_expanding:
        # All three conditions met → enter long.
        action = "buy"
        reason = f"Triple confirmation: {', '.join(passed)}"

    elif not ema_crossover:
        # EMA crossed back below — trend is reversing.
        # This is both a "don't enter" and an "exit open position" signal.
        action = "sell"
        reason = (
            f"EMA crossover reversed: EMA({config.EMA_FAST})={ema_fast:.2f} "
            f"< EMA({config.EMA_SLOW})={ema_slow:.2f}"
        )

    else:
        # Trend is up but one or more confirmation signals missing.
        action = "hold"
        reason = f"Conditions not met. Failed: {', '.join(failed) if failed else 'none'}"

    # --- Log every evaluation (DEBUG level — verbose but useful for debugging) ---
    logger.debug(
        f"SIGNAL: {action.upper():4s} | "
        f"ts={ts} | close={close:.2f} | "
        f"EMA9={ema_fast:.2f} EMA21={ema_slow:.2f} | "
        f"RSI={rsi:.1f} | BBW={bb_width_now:.4f}(prev={bb_width_prv:.4f}) | "
        f"ATR={atr:.2f} | SL={stop_loss:.2f} | TP={take_profit:.2f}"
    )

    if action in ("buy", "sell"):
        logger.info(
            f"SIGNAL: {action.upper()} | {reason} | "
            f"entry={close:.2f} SL={stop_loss:.2f} TP={take_profit:.2f} ATR={atr:.2f}"
        )

    return {
        "action":       action,
        "reason":       reason,
        "entry_price":  float(close),
        "stop_loss":    float(stop_loss),
        "take_profit":  float(take_profit),
        "atr":          float(atr),
        "confidence":   1.0,   # Phase 2: replace with XGBoost win-probability
        "timestamp":    ts,
    }


def check_exit_condition(df, open_trade: dict) -> bool:
    """
    Check whether an open position should be closed due to EMA signal reversal.

    This is the "Signal Reversal Exit" — the third exit condition after stop-loss
    and take-profit.  The OCO order handles stop/TP automatically; this function
    handles the manual close when the trend reverses before either is hit.

    We check df.iloc[-2] (latest closed candle) to avoid acting on forming data.

    Args:
        df:         Indicator-enriched DataFrame (same as evaluate_signal receives).
        open_trade: Dict describing the open position, must contain at minimum:
                    {"entry_price": float, "side": "buy", ...}

    Returns:
        True  — close the position now (EMA reversed against us).
        False — hold, no reversal yet.
    """
    if len(df) < 2:
        logger.warning("Not enough data rows to check exit condition — holding.")
        return False

    row = df.iloc[-2]
    ema_fast = row["ema_fast"]
    ema_slow = row["ema_slow"]

    # For a long position: exit when the fast EMA drops below the slow EMA.
    # This means the short-term momentum has turned bearish — the trend that
    # justified our entry no longer exists.
    if open_trade.get("side", "buy") == "buy":
        reversed_ = ema_fast < ema_slow
        if reversed_:
            logger.info(
                f"EXIT SIGNAL: EMA reversal detected. "
                f"EMA({config.EMA_FAST})={ema_fast:.2f} < "
                f"EMA({config.EMA_SLOW})={ema_slow:.2f}. "
                f"Closing long position entered at {open_trade.get('entry_price', '?')}"
            )
        return reversed_

    # Phase 1 is long-only.  This branch exists for future short support.
    return False


# ---------------------------------------------------------------------------
# Standalone test — run `python strategy/engine.py` from the project root
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from loguru import logger

    from data.fetcher import fetch_ohlcv
    from indicators.signals import compute_indicators

    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="DEBUG",
        format="{time:HH:mm:ss} | {level} | {message}",
    )

    logger.info("=== Strategy engine standalone test ===")

    df = fetch_ohlcv(limit=100)
    df = compute_indicators(df)
    signal = evaluate_signal(df)

    print("\n--- Current Signal ---")
    for key, value in signal.items():
        print(f"  {key:15s}: {value}")

    # Simulate an open position to test the exit checker.
    fake_trade = {"side": "buy", "entry_price": signal["entry_price"]}
    should_exit = check_exit_condition(df, fake_trade)
    print(f"\n  check_exit_condition: {should_exit}")
