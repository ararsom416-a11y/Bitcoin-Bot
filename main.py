"""
main.py — Entry Point: The Main Loop
======================================
Responsibility: tie every other module together into a running bot.

Startup sequence:
  1. Load config and .env
  2. Configure loguru (file + console)
  3. Print startup banner
  4. Initialise portfolio state
  5. Test Binance connection
  6. Fetch initial candles and verify data quality
  7. Enter the main trading loop

Main loop (every LOOP_INTERVAL_SECONDS):
  — Fetch fresh candles
  — Compute indicators
  — If position open: check for exit (signal reversal)
  — If no position: evaluate signal and risk-approve potential entry
  — Execute approved orders
  — Refresh the live status panel
  — Sleep until next cycle

Exit:
  Ctrl-C triggers a graceful shutdown — all open orders are cancelled before exit.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

import config
from data.fetcher import fetch_ohlcv, test_connection
from indicators.signals import compute_indicators
from strategy.engine import evaluate_signal, check_exit_condition
from risk.manager import (
    make_initial_portfolio_state,
    calculate_position_size,
    approve_trade,
    update_portfolio_state,
    reset_daily_state,
)
from execution.trader import (
    place_market_order,
    place_oco_order,
    cancel_all_orders,
    get_open_orders,
    get_account_balance,
    round_quantity,
)

# ---------------------------------------------------------------------------
# Rich console — used for the live status panel (separate from loguru logs)
# ---------------------------------------------------------------------------
console = Console()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """
    Configure loguru with two sinks:
      1. Rotating log file at config.LOG_FILE (INFO and above).
      2. Console stderr (WARNING and above, so we don't pollute the Rich panel).
    """
    Path("logs").mkdir(exist_ok=True)

    # Remove the default loguru handler (it would duplicate console output).
    logger.remove()

    # File sink: verbose, rotating, retained for 30 days.
    logger.add(
        config.LOG_FILE,
        level=config.LOG_LEVEL,
        rotation=config.LOG_ROTATION,
        retention=config.LOG_RETENTION,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
            "{name}:{function}:{line} | {message}"
        ),
        enqueue=True,   # Thread-safe async write.
        backtrace=True, # Full stack traces on ERROR+.
        diagnose=True,
    )

    # Console sink: WARNING and above only (INFO goes to file, not screen).
    # Rich handles the pretty terminal output; loguru handles persistent logs.
    logger.add(
        sys.stderr,
        level="WARNING",
        format="{time:HH:mm:ss} | <level>{level:<8}</level> | {message}",
        colorize=True,
    )


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

def print_banner(portfolio_state: dict) -> None:
    """Print a Rich-styled startup banner with bot configuration."""
    mode = "[bold red]LIVE[/bold red]" if not config.PAPER_TRADING else "[bold green]PAPER TRADING[/bold green]"
    console.print(
        Panel(
            f"""[bold cyan]Bitcoin Trading Bot[/bold cyan] — Phase 1: Triple Confirmation Trend

  Symbol       : [yellow]{config.SYMBOL}[/yellow]
  Timeframe    : [yellow]{config.TIMEFRAME}[/yellow]
  Mode         : {mode}
  Balance      : [green]${portfolio_state['balance']:,.2f}[/green]
  Max Risk/Trade: {config.MAX_RISK_PER_TRADE_PCT:.0%} (${portfolio_state['balance'] * config.MAX_RISK_PER_TRADE_PCT:,.2f})
  Daily Limit  : {config.MAX_DAILY_LOSS_PCT:.0%} max loss
  Strategy     : EMA({config.EMA_FAST}/{config.EMA_SLOW}) + RSI({config.RSI_PERIOD}) + BBW expansion
  Stops        : ATR×{config.STOP_ATR_MULTIPLIER} SL / ATR×{config.TP_ATR_MULTIPLIER} TP""",
            title="[bold white]Startup[/bold white]",
            border_style="cyan",
        )
    )


# ---------------------------------------------------------------------------
# Live status panel
# ---------------------------------------------------------------------------

def log_portfolio_status(
    portfolio_state: dict,
    current_price: float,
    last_signal: dict | None,
) -> None:
    """
    Print a Rich live status panel to the console on each loop iteration.

    Shows: current price, open position P&L, last signal, account health.

    Args:
        portfolio_state: Current portfolio state dict.
        current_price:   Latest BTC close price.
        last_signal:     Signal dict from the most recent evaluate_signal() call.
    """
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Key",   style="cyan",  no_wrap=True)
    table.add_column("Value", style="white", no_wrap=True)

    # --- Price ---
    table.add_row("BTC Price", f"${current_price:,.2f}")

    # --- Open position ---
    pos = portfolio_state.get("open_position")
    if pos:
        unrealised_pnl = (current_price - pos["entry_price"]) * pos["quantity"]
        pnl_color = "green" if unrealised_pnl >= 0 else "red"
        table.add_row("Position",   f"{pos['quantity']:.5f} BTC long")
        table.add_row("Entry",      f"${pos['entry_price']:,.2f}")
        table.add_row("Unrealised", f"[{pnl_color}]${unrealised_pnl:+,.2f}[/{pnl_color}]")
        table.add_row("Stop / TP",  f"${pos['stop_loss']:,.2f} / ${pos['take_profit']:,.2f}")
    else:
        table.add_row("Position", "[dim]None[/dim]")

    # --- Last signal ---
    if last_signal:
        action = last_signal.get("action", "?")
        colour = {"buy": "green", "sell": "red", "hold": "dim"}.get(action, "white")
        table.add_row("Last Signal", f"[{colour}]{action.upper()}[/{colour}]")
        table.add_row("Reason",      last_signal.get("reason", "")[:60])

    # --- Account health ---
    balance    = portfolio_state["balance"]
    daily_pnl  = portfolio_state["daily_pnl"]
    total_pnl  = portfolio_state["total_pnl"]
    peak       = portfolio_state["peak_balance"]
    drawdown   = (peak - balance) / peak * 100 if peak > 0 else 0.0
    d_colour   = "green" if daily_pnl >= 0 else "red"
    t_colour   = "green" if total_pnl >= 0 else "red"

    table.add_row("Balance",    f"${balance:,.2f}")
    table.add_row("Daily P&L",  f"[{d_colour}]${daily_pnl:+,.2f}[/{d_colour}]")
    table.add_row("Total P&L",  f"[{t_colour}]${total_pnl:+,.2f}[/{t_colour}]")
    table.add_row("Drawdown",   f"{drawdown:.2f}%")
    table.add_row("Trades",     str(portfolio_state["trades_today"]) + " today")

    # --- Circuit breaker ---
    if portfolio_state.get("halted"):
        status = f"[bold red]HALTED — {portfolio_state['halt_reason'][:50]}[/bold red]"
    else:
        status = "[green]ACTIVE[/green]"
    table.add_row("Bot Status", status)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    console.print(Panel(table, title=f"[bold]{ts}[/bold]", border_style="blue"))


# ---------------------------------------------------------------------------
# Daily reset checker
# ---------------------------------------------------------------------------

_last_reset_date: str = ""


def check_daily_reset(portfolio_state: dict) -> None:
    """
    Reset daily PnL and trade count at UTC midnight.

    Called once per main loop iteration.  Compares today's UTC date string
    against the last recorded reset date; resets if they differ.

    Args:
        portfolio_state: Mutated in place by reset_daily_state() if needed.
    """
    global _last_reset_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today != _last_reset_date:
        if _last_reset_date:  # Skip logging on first run (no "yesterday").
            reset_daily_state(portfolio_state)
        _last_reset_date = today


# ---------------------------------------------------------------------------
# Position close helper
# ---------------------------------------------------------------------------

def close_position(
    portfolio_state: dict,
    current_price: float,
    reason: str,
) -> None:
    """
    Cancel OCO orders and place a market sell to close the open position.

    Updates portfolio state via update_portfolio_state() after the fill.

    Args:
        portfolio_state: Current portfolio state (mutated in place).
        current_price:   Current market price (used for PnL estimate before fill).
        reason:          Why we're closing ("signal_exit", "stop_loss", "take_profit").
    """
    pos = portfolio_state.get("open_position")
    if not pos:
        logger.warning("close_position called but no open position found.")
        return

    logger.info(f"Closing position: reason={reason} | entry={pos['entry_price']:.2f} | current={current_price:.2f}")

    # Step 1: Cancel the existing OCO bracket so it doesn't conflict.
    cancel_all_orders(config.SYMBOL)

    # Step 2: Place market sell.
    try:
        order = place_market_order(config.SYMBOL, "sell", pos["quantity"])
        exit_price = float(order.get("average") or order.get("price") or current_price)
    except Exception as exc:
        logger.error(f"Failed to place market sell during exit: {exc}. Using current_price as estimate.")
        exit_price = current_price

    # Step 3: Compute realised PnL.
    pnl_usd = (exit_price - pos["entry_price"]) * pos["quantity"]
    pnl_pct = pnl_usd / (pos["entry_price"] * pos["quantity"]) if pos["quantity"] > 0 else 0.0

    trade_result = {
        "pnl_usd":     pnl_usd,
        "pnl_pct":     pnl_pct,
        "exit_reason": reason,
        "entry_price": pos["entry_price"],
        "exit_price":  exit_price,
        "quantity":    pos["quantity"],
        "entry_time":  pos.get("entry_time", ""),
        "exit_time":   datetime.now(timezone.utc).isoformat(),
    }

    update_portfolio_state(trade_result, portfolio_state)
    logger.info(
        f"Position closed: exit=${exit_price:.2f} | PnL=${pnl_usd:+.2f} ({pnl_pct:.2%}) | "
        f"reason={reason}"
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point.  Sets up the bot and enters the trading loop.

    Run from the project root:
        python main.py

    Stop cleanly with Ctrl-C.
    """
    # --- Startup ---
    setup_logging()
    logger.info("=" * 60)
    logger.info("Bitcoin Trading Bot — starting up")
    logger.info("=" * 60)

    portfolio_state = make_initial_portfolio_state(config.ACCOUNT_BALANCE)
    print_banner(portfolio_state)

    # --- Connection test ---
    logger.info("Testing Binance connection...")
    if not test_connection():
        console.print("[bold red]ERROR: Cannot connect to Binance. Check .env API keys and network.[/bold red]")
        sys.exit(1)

    # --- Sync starting balance from exchange ---
    # This ensures our internal accounting matches actual funds on the exchange,
    # catching discrepancies from previous sessions or manual transfers.
    exchange_balance = get_account_balance("USDT")
    if exchange_balance > 0:
        portfolio_state["balance"]      = exchange_balance
        portfolio_state["peak_balance"] = exchange_balance
        logger.info(f"Balance synced from exchange: ${exchange_balance:,.2f} USDT")
    else:
        logger.warning(
            f"Could not read exchange balance (got ${exchange_balance:.2f}). "
            "Using config.ACCOUNT_BALANCE as starting balance."
        )

    # --- Initial data quality check ---
    logger.info("Fetching initial candles for data quality check...")
    try:
        df_init = fetch_ohlcv(config.SYMBOL, config.TIMEFRAME, limit=config.FETCH_LIMIT)
        df_init = compute_indicators(df_init)
        logger.info(
            f"Initial data OK: {len(df_init)} candles | "
            f"latest close=${df_init['close'].iloc[-1]:.2f}"
        )
    except Exception as exc:
        console.print(f"[bold red]ERROR: Initial data fetch failed: {exc}[/bold red]")
        sys.exit(1)

    # Track the last signal for display purposes.
    last_signal: dict | None = None

    logger.info(f"Entering main loop (interval={config.LOOP_INTERVAL_SECONDS}s)...")
    console.print(f"\n[green]Bot is running.[/green] Press [bold]Ctrl-C[/bold] to stop cleanly.\n")

    # ---------------------------------------------------------------------------
    # Main trading loop
    # ---------------------------------------------------------------------------
    while True:
        try:
            # Daily reset check — resets at UTC midnight.
            check_daily_reset(portfolio_state)

            # --- Step 1: Fetch latest candles ---
            df = fetch_ohlcv(config.SYMBOL, config.TIMEFRAME, limit=config.FETCH_LIMIT)

            # --- Step 2: Compute indicators ---
            df = compute_indicators(df)

            current_price = float(df["close"].iloc[-1])

            # --- Step 3: Open position — check for exit ---
            if portfolio_state["open_position"] is not None:
                pos = portfolio_state["open_position"]

                # --- OCO fill detection ---
                # The OCO bracket (stop-loss + take-profit) lives on the exchange.
                # If one leg filled between loops, Binance closes both legs and we
                # have no BTC left.  Without this check, the bot would try to sell
                # BTC it doesn't own, causing an InsufficientFunds error and leaving
                # the portfolio_state stuck showing an open position forever.
                open_orders = get_open_orders(config.SYMBOL)
                if not open_orders:
                    # No open orders → OCO was filled while we weren't watching.
                    # Determine which leg filled and at what price.
                    actual_balance = get_account_balance("USDT")
                    pnl_usd = actual_balance - portfolio_state["balance"]
                    exit_price = pos["entry_price"] + (pnl_usd / pos["quantity"]) if pos["quantity"] > 0 else current_price

                    # Infer exit reason from the price level that was hit.
                    if exit_price >= pos["take_profit"] * 0.995:
                        exit_reason = "take_profit"
                    elif exit_price <= pos["stop_loss"] * 1.005:
                        exit_reason = "stop_loss"
                    else:
                        exit_reason = "oco_filled"  # filled somewhere in between

                    logger.info(
                        f"OCO bracket detected as filled (no open orders). "
                        f"exit_price≈${exit_price:.2f} | reason={exit_reason} | "
                        f"PnL≈${pnl_usd:+.2f}"
                    )

                    pnl_pct = pnl_usd / (pos["entry_price"] * pos["quantity"]) if pos["quantity"] > 0 else 0.0
                    trade_result = {
                        "pnl_usd":     pnl_usd,
                        "pnl_pct":     pnl_pct,
                        "exit_reason": exit_reason,
                        "entry_price": pos["entry_price"],
                        "exit_price":  exit_price,
                        "quantity":    pos["quantity"],
                        "entry_time":  pos.get("entry_time", ""),
                        "exit_time":   datetime.now(timezone.utc).isoformat(),
                    }
                    update_portfolio_state(trade_result, portfolio_state)

                # Check if the signal has reversed (EMA crossover exit condition).
                elif check_exit_condition(df, pos):
                    close_position(portfolio_state, current_price, "signal_exit")
                else:
                    logger.debug(
                        f"Position open — no exit signal. "
                        f"P&L estimate: ${(current_price - pos['entry_price']) * pos['quantity']:+.2f}"
                    )

            # --- Step 4: No open position — look for new entry ---
            else:
                signal = evaluate_signal(df)
                last_signal = signal

                if signal["action"] == "buy":
                    # --- Step 5: Risk approval ---
                    approved, reason = approve_trade(signal, portfolio_state)

                    if approved:
                        # --- Step 6: Calculate position size ---
                        qty = calculate_position_size(
                            account_balance=portfolio_state["balance"],
                            entry_price=signal["entry_price"],
                            stop_loss_price=signal["stop_loss"],
                        )
                        qty = round_quantity(qty)

                        if qty <= 0:
                            logger.warning("Calculated position size is zero — skipping trade.")
                        else:
                            # --- Place market entry order ---
                            entry_order = place_market_order(config.SYMBOL, "buy", qty)
                            actual_entry = float(
                                entry_order.get("average")
                                or entry_order.get("price")
                                or signal["entry_price"]
                            )

                            # Recompute SL/TP from actual fill price.
                            actual_sl = actual_entry - (config.STOP_ATR_MULTIPLIER * signal["atr"])
                            actual_tp = actual_entry + (config.TP_ATR_MULTIPLIER * signal["atr"])

                            # --- Place OCO bracket (stop-loss + take-profit) ---
                            try:
                                place_oco_order(
                                    symbol=config.SYMBOL,
                                    quantity=qty,
                                    take_profit_price=actual_tp,
                                    stop_loss_price=actual_sl,
                                )
                            except Exception as oco_exc:
                                logger.error(
                                    f"OCO order failed: {oco_exc}. "
                                    "Position is UNPROTECTED — manual intervention required."
                                )

                            # Record the open position in portfolio state.
                            portfolio_state["open_position"] = {
                                "side":        "buy",
                                "quantity":    qty,
                                "entry_price": actual_entry,
                                "stop_loss":   actual_sl,
                                "take_profit": actual_tp,
                                "entry_time":  datetime.now(timezone.utc).isoformat(),
                                "signal":      signal,
                            }

                            logger.info(
                                f"TRADE OPENED: BUY {qty:.5f} BTC @ ${actual_entry:.2f} | "
                                f"SL=${actual_sl:.2f} TP=${actual_tp:.2f} | "
                                f"reason: {signal['reason']}"
                            )

                    else:
                        logger.info(f"Trade blocked by risk manager: {reason}")

            # --- Step 7: Display live status panel ---
            log_portfolio_status(portfolio_state, current_price, last_signal)

            # --- Step 8: Sleep until next cycle ---
            logger.debug(f"Sleeping {config.LOOP_INTERVAL_SECONDS}s until next check.")
            time.sleep(config.LOOP_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            # ---------------------------------------------------------------
            # Graceful shutdown — cancel orders before exiting.
            # ---------------------------------------------------------------
            console.print("\n[yellow]Shutdown signal received (Ctrl-C).[/yellow]")
            logger.info("Bot stopping — cancelling all open orders...")

            try:
                cancel_all_orders(config.SYMBOL)
                logger.info("All orders cancelled successfully.")
            except Exception as exc:
                logger.error(f"Error cancelling orders during shutdown: {exc}")

            # Final status display.
            try:
                df_final = fetch_ohlcv(config.SYMBOL, config.TIMEFRAME, limit=10)
                final_price = float(df_final["close"].iloc[-1])
            except Exception:
                final_price = 0.0

            console.print(
                Panel(
                    f"  Bot stopped cleanly.\n"
                    f"  Session P&L : [{'green' if portfolio_state['total_pnl'] >= 0 else 'red'}]"
                    f"${portfolio_state['total_pnl']:+,.2f}[/]\n"
                    f"  Final Balance: ${portfolio_state['balance']:,.2f}\n"
                    f"  Trades today : {portfolio_state['trades_today']}",
                    title="[bold]Session Summary[/bold]",
                    border_style="yellow",
                )
            )
            logger.info(
                f"Session ended. PnL=${portfolio_state['total_pnl']:+.2f} | "
                f"balance=${portfolio_state['balance']:.2f}"
            )
            break

        except Exception as exc:
            # Catch-all for unexpected errors — log and wait before retrying.
            # We never want the bot to die silently.
            logger.error(
                f"Unexpected error in main loop: {exc}",
                exc_info=True,
            )
            console.print(f"[red]Loop error: {exc}. Waiting 60s before retry...[/red]")
            time.sleep(60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
