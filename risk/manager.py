"""
risk/manager.py — Position Sizing, Trade Approval, and Circuit Breakers
=========================================================================
Responsibility: be the final gatekeeper between a strategy signal and actual
order placement.  This module can veto ANY trade regardless of what the strategy
says — and it tracks the portfolio's health over time.

Key principles:
  1. Never risk more than MAX_RISK_PER_TRADE_PCT (2%) of account on a single trade.
  2. Daily circuit breaker: halt if today's loss exceeds MAX_DAILY_LOSS_PCT (5%).
  3. Drawdown circuit breaker: halt if we fall MAX_DRAWDOWN_PCT (15%) from peak.
  4. Streak circuit breaker: halt after MAX_CONSECUTIVE_LOSSES (5) in a row.
  5. Phase 1 is single-position only: one open trade at a time, ever.

Position sizing math (Fixed Fractional / partial Kelly):
  risk_amount      = account_balance × 0.02          (= $200 on $10k)
  stop_distance    = |entry_price - stop_loss_price|  (in USD per BTC)
  position_size    = risk_amount / stop_distance       (in BTC)

  Example: ATR=$800, stop=1.5×ATR=$1,200
    position_size = $200 / $1,200 = 0.167 BTC

  This is the professional standard for volatile assets.  Full Kelly is
  mathematically optimal but practically leads to ruin in crypto markets.
"""

from copy import deepcopy

from loguru import logger

import config


# ---------------------------------------------------------------------------
# Portfolio state type alias (for documentation clarity)
# ---------------------------------------------------------------------------
# The portfolio_state dict is the single mutable object that tracks everything.
# It is initialised in main.py and passed by reference into every risk function.
#
# Schema:
#   balance           (float)      — Current USDT account balance
#   peak_balance      (float)      — Highest balance ever reached (drawdown reference)
#   daily_pnl         (float)      — PnL realised today (resets at UTC midnight)
#   total_pnl         (float)      — Cumulative PnL since bot started
#   consecutive_losses(int)        — Unbroken streak of losing trades
#   trades_today      (int)        — Number of completed trades today
#   open_position     (dict|None)  — Details of the current open trade, or None
#   trade_history     (list)       — List of completed trade result dicts
#   halted            (bool)       — True if a circuit breaker has fired
#   halt_reason       (str)        — Why the bot was halted


def make_initial_portfolio_state(starting_balance: float = config.ACCOUNT_BALANCE) -> dict:
    """
    Create and return a fresh portfolio state dictionary.

    Call this once at startup in main.py, then pass the same dict (by reference)
    to all risk functions throughout the bot's lifetime.

    Args:
        starting_balance: Initial account balance in USD.

    Returns:
        dict: Initialised portfolio state with all fields set to safe defaults.
    """
    return {
        "balance":              starting_balance,
        "peak_balance":         starting_balance,
        "start_of_day_balance": starting_balance,  # reference for daily loss limit
        "daily_pnl":            0.0,
        "total_pnl":            0.0,
        "consecutive_losses":   0,
        "trades_today":         0,
        "open_position":        None,
        "trade_history":        [],
        "halted":               False,
        "halt_reason":          "",
    }


# ---------------------------------------------------------------------------
# Function 1: Position Sizing
# ---------------------------------------------------------------------------

def calculate_position_size(
    account_balance: float,
    entry_price: float,
    stop_loss_price: float,
) -> float:
    """
    Calculate how many BTC to buy such that the maximum loss if stopped out
    equals exactly MAX_RISK_PER_TRADE_PCT of the account balance.

    Args:
        account_balance:  Current USDT balance.
        entry_price:      Planned trade entry price (USD per BTC).
        stop_loss_price:  Planned stop-loss price (USD per BTC).

    Returns:
        Position size in BTC.  Returns 0.0 if the signal is invalid
        (stop distance zero or negative — this should never happen in practice
        but guards against edge cases like ATR=0 or misconfigured levels).
    """
    # How much money we are allowed to lose on this trade.
    risk_amount: float = account_balance * config.MAX_RISK_PER_TRADE_PCT

    # Distance from entry to stop, in USD per BTC.
    stop_distance: float = abs(entry_price - stop_loss_price)

    if stop_distance <= 0:
        logger.error(
            f"Invalid stop distance: {stop_distance:.2f}. "
            "Entry and stop-loss prices are too close or inverted. Returning 0."
        )
        return 0.0

    # Core position sizing formula.
    position_size_btc: float = risk_amount / stop_distance

    # Convert to USD to check against the position-size cap.
    position_size_usd: float = position_size_btc * entry_price
    max_position_usd: float  = account_balance * config.MAX_POSITION_SIZE_PCT

    # Cap: never put more than MAX_POSITION_SIZE_PCT of account into a single trade.
    if position_size_usd > max_position_usd:
        logger.warning(
            f"Position size ${position_size_usd:.2f} exceeds cap "
            f"${max_position_usd:.2f} ({config.MAX_POSITION_SIZE_PCT:.0%} of balance). "
            "Capping position."
        )
        position_size_btc = max_position_usd / entry_price

    logger.info(
        f"Position sizing: balance=${account_balance:.2f} | "
        f"risk={config.MAX_RISK_PER_TRADE_PCT:.0%} (${risk_amount:.2f}) | "
        f"stop_distance=${stop_distance:.2f} | "
        f"size={position_size_btc:.5f} BTC (${position_size_btc * entry_price:.2f})"
    )

    return position_size_btc


# ---------------------------------------------------------------------------
# Function 2: Trade Approval (the gatekeeper)
# ---------------------------------------------------------------------------

def approve_trade(signal: dict, portfolio_state: dict) -> tuple[bool, str]:
    """
    Run every circuit breaker and sanity check before allowing a trade to execute.

    This is the last line of defence between a signal and real (or simulated) money.
    Even if the strategy fires a "buy", this function can block it.

    Checks performed (in order — first failure blocks the trade):
      1. Bot is not halted by a circuit breaker.
      2. No existing open position (Phase 1 is single-position only).
      3. Signal action is actually "buy" (only buy signals need approval).
      4. Daily loss has not exceeded MAX_DAILY_LOSS_PCT.
      5. Total drawdown from peak has not exceeded MAX_DRAWDOWN_PCT.
      6. Consecutive losses have not exceeded MAX_CONSECUTIVE_LOSSES.
      7. Calculated position size is above the minimum order size.

    Args:
        signal:          Signal dict from strategy/engine.py (evaluate_signal).
        portfolio_state: Live portfolio state dict from make_initial_portfolio_state.

    Returns:
        (True, "approved")            — Trade is cleared to execute.
        (False, "reason for denial")  — Trade is blocked; reason explains why.
    """
    # --- Check 1: Circuit breaker already active ---
    if portfolio_state["halted"]:
        reason = f"CIRCUIT BREAKER ACTIVE: {portfolio_state['halt_reason']}"
        logger.warning(f"Trade blocked: {reason}")
        return False, reason

    # --- Check 2: Only one open position at a time ---
    if portfolio_state["open_position"] is not None:
        reason = "Existing open position — Phase 1 is single-position only."
        logger.debug(f"Trade blocked: {reason}")
        return False, reason

    # --- Check 3: Only process buy signals ---
    if signal.get("action") != "buy":
        reason = f"Signal action is '{signal.get('action')}', not 'buy'."
        logger.debug(f"Trade blocked: {reason}")
        return False, reason

    balance           = portfolio_state["balance"]
    daily_pnl         = portfolio_state["daily_pnl"]
    peak_balance      = portfolio_state["peak_balance"]
    start_of_day_bal  = portfolio_state.get("start_of_day_balance", balance)

    # --- Check 4: Daily loss circuit breaker ---
    # Compare against start-of-day balance, not current balance.
    # Using current balance would raise the threshold whenever we profit mid-day,
    # making the circuit breaker progressively weaker throughout a winning session.
    max_daily_loss = start_of_day_bal * config.MAX_DAILY_LOSS_PCT
    if daily_pnl < -max_daily_loss:
        reason = (
            f"Daily loss ${abs(daily_pnl):.2f} exceeds limit "
            f"${max_daily_loss:.2f} ({config.MAX_DAILY_LOSS_PCT:.0%} of balance)."
        )
        _trigger_circuit_breaker(portfolio_state, reason)
        return False, reason

    # --- Check 5: Total drawdown circuit breaker ---
    drawdown = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
    if drawdown >= config.MAX_DRAWDOWN_PCT:
        reason = (
            f"Max drawdown {drawdown:.1%} reached limit {config.MAX_DRAWDOWN_PCT:.0%}. "
            f"Peak: ${peak_balance:.2f}, Current: ${balance:.2f}."
        )
        _trigger_circuit_breaker(portfolio_state, reason)
        return False, reason

    # --- Check 6: Consecutive loss streak ---
    if portfolio_state["consecutive_losses"] >= config.MAX_CONSECUTIVE_LOSSES:
        reason = (
            f"Consecutive loss streak ({portfolio_state['consecutive_losses']}) "
            f"reached limit ({config.MAX_CONSECUTIVE_LOSSES}). Human review needed."
        )
        _trigger_circuit_breaker(portfolio_state, reason)
        return False, reason

    # --- Check 7: Minimum position size ---
    position_btc = calculate_position_size(
        account_balance=balance,
        entry_price=signal["entry_price"],
        stop_loss_price=signal["stop_loss"],
    )
    position_usd = position_btc * signal["entry_price"]

    if position_btc <= 0 or position_usd < config.MIN_ORDER_SIZE_USD:
        reason = (
            f"Position size ${position_usd:.2f} is below minimum "
            f"${config.MIN_ORDER_SIZE_USD:.2f}. Trade too small to execute."
        )
        logger.warning(f"Trade blocked: {reason}")
        return False, reason

    logger.info(
        f"Trade APPROVED: {signal['action'].upper()} {position_btc:.5f} BTC "
        f"@ ${signal['entry_price']:.2f} | "
        f"SL=${signal['stop_loss']:.2f} TP=${signal['take_profit']:.2f}"
    )
    return True, "approved"


# ---------------------------------------------------------------------------
# Function 3: Portfolio State Update (after a trade closes)
# ---------------------------------------------------------------------------

def update_portfolio_state(trade_result: dict, portfolio_state: dict) -> dict:
    """
    Update the portfolio state after a completed trade.

    Call this immediately after a trade closes (stop hit, TP hit, or manual exit).
    Mutates and returns the portfolio_state dict.

    Args:
        trade_result: Dict describing the closed trade outcome:
            {
                "pnl_usd":     float,    # Profit or loss in USD (negative = loss)
                "pnl_pct":     float,    # PnL as fraction of entry value
                "exit_reason": str,      # "stop_loss" | "take_profit" | "signal_exit"
                "entry_price": float,
                "exit_price":  float,
                "quantity":    float,    # BTC traded
                "entry_time":  str,
                "exit_time":   str,
            }
        portfolio_state: Current portfolio state dict (mutated in place).

    Returns:
        The updated portfolio_state dict.
    """
    pnl = trade_result["pnl_usd"]

    # Update balances.
    portfolio_state["balance"]    += pnl
    portfolio_state["daily_pnl"]  += pnl
    portfolio_state["total_pnl"]  += pnl

    # Update peak balance (high-water mark for drawdown calculation).
    if portfolio_state["balance"] > portfolio_state["peak_balance"]:
        portfolio_state["peak_balance"] = portfolio_state["balance"]

    # Track consecutive losses.
    if pnl < 0:
        portfolio_state["consecutive_losses"] += 1
    else:
        # Any winning trade resets the streak — even a scratch/breakeven.
        portfolio_state["consecutive_losses"] = 0

    portfolio_state["trades_today"] += 1
    portfolio_state["open_position"] = None

    # Append to permanent history for analysis.
    portfolio_state["trade_history"].append(deepcopy(trade_result))

    # Log trade outcome.
    outcome = "WIN" if pnl >= 0 else "LOSS"
    logger.info(
        f"TRADE {outcome}: PnL=${pnl:+.2f} ({trade_result.get('pnl_pct', 0):.2%}) | "
        f"exit={trade_result.get('exit_reason', '?')} | "
        f"balance=${portfolio_state['balance']:.2f} | "
        f"streak={portfolio_state['consecutive_losses']} losses | "
        f"daily_pnl=${portfolio_state['daily_pnl']:+.2f}"
    )

    # Post-trade circuit breaker checks.
    # Run these even after updating state so the NEXT trade gets blocked early.
    _check_post_trade_circuit_breakers(portfolio_state)

    return portfolio_state


# ---------------------------------------------------------------------------
# Function 4: Daily State Reset (call at UTC midnight)
# ---------------------------------------------------------------------------

def reset_daily_state(portfolio_state: dict) -> dict:
    """
    Reset daily counters at UTC midnight.

    Daily PnL and trade count reset each day.  The halted flag is NOT automatically
    cleared — a human must manually clear it by restarting the bot after reviewing
    the situation.  This prevents automatic resumption after a bad day.

    Args:
        portfolio_state: Current portfolio state dict (mutated in place).

    Returns:
        Updated portfolio_state with daily counters zeroed.
    """
    old_daily = portfolio_state["daily_pnl"]
    portfolio_state["daily_pnl"]              = 0.0
    portfolio_state["trades_today"]           = 0
    # Reset the daily reference balance so the new day's 5% limit is correct.
    portfolio_state["start_of_day_balance"]   = portfolio_state["balance"]

    logger.info(
        f"Daily state reset at UTC midnight. "
        f"Yesterday's PnL: ${old_daily:+.2f}. "
        f"Bot halted: {portfolio_state['halted']}."
    )
    return portfolio_state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trigger_circuit_breaker(portfolio_state: dict, reason: str) -> None:
    """
    Activate the circuit breaker, log a CRITICAL message, and set the halted flag.

    Args:
        portfolio_state: Mutated in place — sets halted=True and records reason.
        reason:          Human-readable explanation of why trading was halted.
    """
    portfolio_state["halted"]      = True
    portfolio_state["halt_reason"] = reason

    logger.critical(
        f"CIRCUIT BREAKER TRIGGERED — Bot halted. Reason: {reason} | "
        f"Balance: ${portfolio_state['balance']:.2f} | "
        f"Daily PnL: ${portfolio_state['daily_pnl']:+.2f} | "
        f"Drawdown from peak: "
        f"{(portfolio_state['peak_balance'] - portfolio_state['balance']) / portfolio_state['peak_balance']:.1%}"
    )


def _check_post_trade_circuit_breakers(portfolio_state: dict) -> None:
    """
    Re-run circuit breaker logic after a trade closes.

    This catches cases where a single large losing trade pushes us through
    a threshold, ensuring the NEXT trade doesn't execute before we check.
    """
    if portfolio_state["halted"]:
        return  # Already halted.

    balance           = portfolio_state["balance"]
    peak              = portfolio_state["peak_balance"]
    daily_pnl         = portfolio_state["daily_pnl"]
    streak            = portfolio_state["consecutive_losses"]
    start_of_day_bal  = portfolio_state.get("start_of_day_balance", balance)

    # Drawdown check.
    drawdown = (peak - balance) / peak if peak > 0 else 0.0
    if drawdown >= config.MAX_DRAWDOWN_PCT:
        _trigger_circuit_breaker(
            portfolio_state,
            f"Post-trade drawdown {drawdown:.1%} ≥ limit {config.MAX_DRAWDOWN_PCT:.0%}.",
        )
        return

    # Daily loss check — use start-of-day balance as the reference point.
    max_daily = start_of_day_bal * config.MAX_DAILY_LOSS_PCT
    if daily_pnl < -max_daily:
        _trigger_circuit_breaker(
            portfolio_state,
            f"Post-trade daily PnL ${daily_pnl:.2f} below limit -${max_daily:.2f}.",
        )
        return

    # Consecutive loss streak check.
    if streak >= config.MAX_CONSECUTIVE_LOSSES:
        _trigger_circuit_breaker(
            portfolio_state,
            f"Consecutive losses ({streak}) reached limit ({config.MAX_CONSECUTIVE_LOSSES}).",
        )
