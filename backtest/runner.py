"""
backtest/runner.py — Historical Simulation and Walk-Forward Validation
========================================================================
Responsibility: simulate the Triple Confirmation strategy on historical data
using vectorbt and report detailed performance metrics.

Two modes:
  1. Full backtest   — single pass over the entire date range.
  2. Walk-forward    — rolling-window validation across 5 chronological folds.
     This is far more reliable than a single backtest because it tests the
     strategy on data it has never "seen" during indicator calculation.

Why vectorbt?
  — Vectorised operations: backtests complete in milliseconds, not minutes.
  — Built-in portfolio simulation: handles compounding, commission, slippage.
  — Professional-grade statistics: Sharpe, Sortino, Calmar, win rate, etc.

Why walk-forward validation?
  A strategy that looks great on a single backtest can be "lucky" — the chosen
  date range happened to suit its parameters.  Walk-forward testing exposes this
  by repeatedly training on one period and testing on the next unseen period.
  A robust strategy shows consistent Sharpe ratio across ALL folds (low std dev).
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import vectorbt as vbt
from loguru import logger

import config
from data.fetcher import fetch_ohlcv
from indicators.signals import compute_indicators


# Suppress vectorbt's own FutureWarnings only — leave other libraries' warnings visible.
warnings.filterwarnings("ignore", category=FutureWarning, module="vectorbt")
warnings.filterwarnings("ignore", category=FutureWarning, module="numba")


# ---------------------------------------------------------------------------
# Benchmark thresholds (from the build spec)
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "win_rate_pct":     {"minimum": 45.0, "good": 52.0, "excellent": 58.0},
    "sharpe_ratio":     {"minimum": 1.0,  "good": 1.5,  "excellent": 2.0},
    "max_drawdown_pct": {"minimum": 15.0, "good": 10.0, "excellent": 7.0},  # lower is better
    "profit_factor":    {"minimum": 1.3,  "good": 1.6,  "excellent": 2.0},
}


# ---------------------------------------------------------------------------
# Signal generation (vectorised over the full DataFrame)
# ---------------------------------------------------------------------------

def _generate_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Apply Triple Confirmation strategy logic to every row of the DataFrame
    and return boolean entry and exit signal Series.

    This mirrors the logic in strategy/engine.py but works row-wise over
    the entire historical DataFrame — vectorbt needs arrays, not single values.

    Args:
        df: Indicator-enriched DataFrame (output of compute_indicators).

    Returns:
        (entries, exits): Boolean pd.Series aligned with df index.
            entries[i] = True  → go long at close of bar i
            exits[i]   = True  → close long at close of bar i
    """
    # Shift bb_width by 1 to get the PREVIOUS bar's value without look-ahead.
    bb_width_prev = df["bb_width"].shift(1)

    # Triple confirmation — all three must be True simultaneously.
    entries: pd.Series = (
        (df["ema_fast"] > df["ema_slow"])                                    # Signal 1
        & (df["rsi"] > config.RSI_LOWER) & (df["rsi"] < config.RSI_UPPER)   # Signal 2
        & (df["bb_width"] > bb_width_prev)                                   # Signal 3
    )

    # Exit: EMA reversal (fast crosses below slow).
    exits: pd.Series = df["ema_fast"] < df["ema_slow"]

    # Ensure no NaN rows generate signals.
    entries = entries.fillna(False)
    exits   = exits.fillna(False)

    logger.debug(
        f"Signal generation complete: {entries.sum()} entries, {exits.sum()} exits "
        f"over {len(df)} bars."
    )
    return entries, exits


# ---------------------------------------------------------------------------
# ATR-based stop-loss and take-profit arrays
# ---------------------------------------------------------------------------

def _compute_sl_tp_arrays(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Compute ATR-based stop-loss and take-profit price arrays for every bar.

    vectorbt supports dynamic (per-bar) SL/TP via sl_stop and tp_stop parameters.
    We express them as fractional distances from the entry price rather than
    absolute prices — vectorbt's from_signals() handles the math internally.

    Args:
        df: Indicator-enriched DataFrame (must contain 'atr' and 'close').

    Returns:
        (sl_fraction, tp_fraction): Float Series representing the fractional
        stop-loss and take-profit distances.
            sl_stop = ATR × 1.5 / close
            tp_stop = ATR × 2.5 / close
    """
    sl_fraction = (df["atr"] * config.STOP_ATR_MULTIPLIER) / df["close"]
    tp_fraction = (df["atr"] * config.TP_ATR_MULTIPLIER)   / df["close"]
    return sl_fraction, tp_fraction


# ---------------------------------------------------------------------------
# Core backtest function
# ---------------------------------------------------------------------------

def run_backtest(
    symbol: str    = config.SYMBOL,
    timeframe: str = config.TIMEFRAME,
    start_date: str | None = None,
    end_date: str | None   = None,
    df: pd.DataFrame | None = None,
) -> dict:
    """
    Run a full backtest of the Triple Confirmation strategy over a date range.

    Fetches historical data (or uses a provided DataFrame), computes indicators,
    generates entry/exit signals, and passes everything to vectorbt for simulation.

    Args:
        symbol:     Trading pair in ccxt format.
        timeframe:  Candle width (e.g. "1h").
        start_date: ISO date string, e.g. "2024-01-01".  If None, uses all available data.
        end_date:   ISO date string, e.g. "2024-12-31".  If None, uses all available data.
        df:         Pre-computed indicator DataFrame.  If provided, skips fetch + compute.
                    Useful for walk-forward validation to avoid redundant API calls.

    Returns:
        dict with performance metrics:
            total_return_pct, sharpe_ratio, sortino_ratio, max_drawdown_pct,
            win_rate_pct, profit_factor, total_trades,
            avg_trade_duration_hours, best_trade_pct, worst_trade_pct
    """
    logger.info(
        f"Starting backtest: {symbol} {timeframe} | "
        f"{start_date or 'earliest'} → {end_date or 'latest'}"
    )

    # --- Step 1: Fetch data (if not provided) ---
    if df is None:
        # Fetch as many candles as the exchange will provide.
        df_raw = fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=1000)

        # Apply date filters if specified.
        if start_date:
            df_raw = df_raw[df_raw["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            df_raw = df_raw[df_raw["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]

        df_raw = df_raw.reset_index(drop=True)

        # --- Step 2: Compute indicators ---
        df = compute_indicators(df_raw)

    if len(df) < config.CANDLES_REQUIRED:
        raise ValueError(
            f"Insufficient data for backtest: {len(df)} rows after filtering. "
            f"Need at least {config.CANDLES_REQUIRED}."
        )

    # --- Step 3: Generate signals ---
    entries, exits = _generate_signals(df)
    sl_frac, tp_frac = _compute_sl_tp_arrays(df)

    # --- Step 3b: Compute per-bar position sizes (ATR-based fixed-fraction) ---
    # This replicates the live bot's risk math for each historical bar:
    #   risk_amount    = balance × MAX_RISK_PCT           (e.g. 2% of equity)
    #   stop_distance  = ATR × STOP_MULTIPLIER            (e.g. 1.5 × ATR)
    #   position_usd   = risk_amount / stop_distance × close
    #   position_frac  = position_usd / balance  =  MAX_RISK_PCT / sl_frac
    # sl_frac = (ATR × 1.5) / close, so:
    #   position_frac  = MAX_RISK_PCT / sl_frac
    # Clamp to MAX_POSITION_SIZE_PCT to avoid runaway sizes when ATR is tiny.
    size_array = (config.MAX_RISK_PER_TRADE_PCT / sl_frac).clip(
        upper=config.MAX_POSITION_SIZE_PCT
    )
    # Replace any inf/NaN (e.g. if ATR is 0) with the minimum viable fraction.
    size_array = size_array.replace([np.inf, -np.inf], config.MAX_POSITION_SIZE_PCT)
    size_array = size_array.fillna(config.MAX_RISK_PER_TRADE_PCT)

    # --- Step 4: Run vectorbt simulation ---
    close_series = df["close"]

    # vectorbt.Portfolio.from_signals() simulates a full trading history.
    # init_cash:    starting capital.
    # fees:         commission per trade (0.1% = 0.001).
    # sl_stop / tp_stop: fractional distances for auto stop/TP management.
    # size:         per-bar target allocation as fraction of equity (0.0–1.0).
    # size_type:    "targetpercent" — size is fraction of current equity value.
    # Upon conflict (both entry and exit on same bar), vectorbt uses the exit.
    portfolio = vbt.Portfolio.from_signals(
        close=close_series,
        entries=entries,
        exits=exits,
        sl_stop=sl_frac,                       # Stop-loss: ATR × 1.5 / close
        tp_stop=tp_frac,                       # Take-profit: ATR × 2.5 / close
        init_cash=config.ACCOUNT_BALANCE,
        fees=config.BACKTEST_COMMISSION,
        freq=timeframe,
        size=size_array,                       # ATR-derived fractional position sizing
        size_type="targetpercent",             # size is fraction of current equity
    )

    # --- Step 5: Extract statistics ---
    stats = portfolio.stats()

    # vectorbt stat names vary slightly by version — extract with fallbacks.
    def _get_stat(key: str, fallback: float = 0.0) -> float:
        """Safely extract a stat from the vectorbt stats Series."""
        try:
            val = stats.get(key, fallback)
            return float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else fallback
        except Exception:
            return fallback

    total_return_pct = _get_stat("Total Return [%]")
    sharpe_ratio     = _get_stat("Sharpe Ratio")
    sortino_ratio    = _get_stat("Sortino Ratio")
    max_drawdown_pct = abs(_get_stat("Max Drawdown [%]"))
    win_rate_pct     = _get_stat("Win Rate [%]")
    total_trades     = int(_get_stat("Total Trades"))

    # Profit factor = gross profit / gross loss.
    # vectorbt may call this "Profit Factor" or compute it from trade stats.
    try:
        trades = portfolio.trades.records_readable
        if len(trades) > 0:
            gross_profit = trades[trades["PnL"] > 0]["PnL"].sum()
            gross_loss   = abs(trades[trades["PnL"] < 0]["PnL"].sum())
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

            # Average trade duration in hours.
            if "Duration" in trades.columns:
                avg_duration_hours = float(
                    trades["Duration"].dt.total_seconds().mean() / 3600
                )
            else:
                avg_duration_hours = 0.0

            best_trade_pct  = float(trades["Return [%]"].max()) if "Return [%]" in trades.columns else 0.0
            worst_trade_pct = float(trades["Return [%]"].min()) if "Return [%]" in trades.columns else 0.0
            # Per-trade fractional returns (e.g. 0.015 = +1.5%) for Monte Carlo.
            trade_returns_list = (
                (trades["Return [%]"] / 100.0).tolist()
                if "Return [%]" in trades.columns else []
            )
        else:
            profit_factor      = 0.0
            avg_duration_hours = 0.0
            best_trade_pct     = 0.0
            worst_trade_pct    = 0.0
            trade_returns_list = []
    except Exception as exc:
        logger.warning(f"Could not compute detailed trade stats: {exc}")
        profit_factor      = 0.0
        avg_duration_hours = 0.0
        best_trade_pct     = 0.0
        worst_trade_pct    = 0.0
        trade_returns_list = []

    results = {
        "total_return_pct":         total_return_pct,
        "sharpe_ratio":             sharpe_ratio,
        "sortino_ratio":            sortino_ratio,
        "max_drawdown_pct":         max_drawdown_pct,
        "win_rate_pct":             win_rate_pct,
        "profit_factor":            profit_factor,
        "total_trades":             total_trades,
        "avg_trade_duration_hours": avg_duration_hours,
        "best_trade_pct":           best_trade_pct,
        "worst_trade_pct":          worst_trade_pct,
        # Per-trade returns list — passed directly to run_monte_carlo().
        "trade_returns":            trade_returns_list,
    }

    logger.info(
        f"Backtest complete: {total_trades} trades | "
        f"return={total_return_pct:.2f}% | Sharpe={sharpe_ratio:.2f} | "
        f"MDD={max_drawdown_pct:.2f}% | WR={win_rate_pct:.2f}%"
    )

    return results


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def run_walk_forward_validation(
    symbol: str    = config.SYMBOL,
    timeframe: str = config.TIMEFRAME,
    n_folds: int   = config.WALKFORWARD_FOLDS,
) -> dict:
    """
    Chronological walk-forward validation across N folds.

    Methodology:
      1. Fetch all available data and split into N equal chronological folds.
      2. For each fold i (starting from fold 2):
           Train on folds 1..i-1 (not directly used — our strategy has no training,
           but the concept applies — we just test on the held-out fold).
           Test on fold i.
           Apply an embargo period between train and test to prevent look-ahead leakage.
      3. Report the mean and std of Sharpe ratio across all test folds.

    Robustness criteria (from build spec):
      ROBUST if: mean Sharpe > 1.0 AND std Sharpe < 0.5

    Args:
        symbol:    Trading pair.
        timeframe: Candle width.
        n_folds:   Number of chronological folds (default 5).

    Returns:
        dict with:
            fold_results      — list of per-fold backtest result dicts
            mean_sharpe       — float
            std_sharpe        — float
            is_robust         — bool
            robustness_reason — str
    """
    logger.info(f"Starting walk-forward validation: {n_folds} folds for {symbol} {timeframe}")

    # Fetch all available data.
    df_raw = fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=1000)
    df_all = compute_indicators(df_raw)

    if len(df_all) < n_folds * config.CANDLES_REQUIRED:
        raise ValueError(
            f"Not enough data for {n_folds}-fold walk-forward validation. "
            f"Have {len(df_all)} rows, need at least {n_folds * config.CANDLES_REQUIRED}."
        )

    # Split into N equal folds.
    fold_size  = len(df_all) // n_folds
    embargo_size = max(1, int(fold_size * config.WALKFORWARD_EMBARGO_FRAC))

    fold_sharpes = []
    fold_results = []

    for fold_idx in range(1, n_folds):
        # Test fold is fold_idx (0-indexed).
        test_start = fold_idx * fold_size + embargo_size
        test_end   = (fold_idx + 1) * fold_size

        if test_end > len(df_all):
            test_end = len(df_all)

        df_test = df_all.iloc[test_start:test_end].reset_index(drop=True)

        if len(df_test) < config.CANDLES_REQUIRED:
            logger.warning(f"Fold {fold_idx+1}: insufficient test rows ({len(df_test)}), skipping.")
            continue

        logger.info(
            f"Walk-forward fold {fold_idx}/{n_folds-1}: "
            f"testing on rows {test_start}–{test_end} "
            f"({df_test['timestamp'].iloc[0]} → {df_test['timestamp'].iloc[-1]})"
        )

        try:
            result = run_backtest(df=df_test)
            result["fold"] = fold_idx
            fold_results.append(result)
            fold_sharpes.append(result["sharpe_ratio"])

            logger.info(
                f"  Fold {fold_idx} result: Sharpe={result['sharpe_ratio']:.2f} | "
                f"WR={result['win_rate_pct']:.1f}% | "
                f"Trades={result['total_trades']}"
            )

        except Exception as exc:
            logger.error(f"Walk-forward fold {fold_idx} failed: {exc}")

    if not fold_sharpes:
        return {
            "fold_results": [],
            "mean_sharpe":  0.0,
            "std_sharpe":   float("inf"),
            "is_robust":    False,
            "robustness_reason": "No valid folds completed.",
        }

    mean_sharpe = float(np.mean(fold_sharpes))
    std_sharpe  = float(np.std(fold_sharpes))

    is_robust = mean_sharpe > 1.0 and std_sharpe < 0.5
    robustness_reason = (
        f"Mean Sharpe={mean_sharpe:.2f} ({'≥' if mean_sharpe > 1.0 else '<'} 1.0), "
        f"Std Sharpe={std_sharpe:.2f} ({'<' if std_sharpe < 0.5 else '≥'} 0.5)."
    )

    logger.info(
        f"Walk-forward complete: mean_sharpe={mean_sharpe:.2f} ± {std_sharpe:.2f} | "
        f"robust={'YES' if is_robust else 'NO'}"
    )

    return {
        "fold_results":       fold_results,
        "mean_sharpe":        mean_sharpe,
        "std_sharpe":         std_sharpe,
        "is_robust":          is_robust,
        "robustness_reason":  robustness_reason,
    }


# ---------------------------------------------------------------------------
# Benchmark reporting
# ---------------------------------------------------------------------------

def print_benchmark_report(results: dict) -> bool:
    """
    Print a formatted PASS/FAIL benchmark report for a backtest result.

    Args:
        results: Dict from run_backtest().

    Returns:
        True if ALL minimum benchmarks are met, False otherwise.
    """
    print("\n" + "=" * 60)
    print("  BACKTEST BENCHMARK REPORT")
    print("=" * 60)
    print(f"  Total Trades         : {results.get('total_trades', 0)}")
    print(f"  Total Return         : {results.get('total_return_pct', 0):.2f}%")
    print(f"  Avg Trade Duration   : {results.get('avg_trade_duration_hours', 0):.1f}h")
    print(f"  Best Trade           : {results.get('best_trade_pct', 0):.2f}%")
    print(f"  Worst Trade          : {results.get('worst_trade_pct', 0):.2f}%")
    print(f"  Sortino Ratio        : {results.get('sortino_ratio', 0):.2f}")
    print("-" * 60)
    print(f"  {'Metric':<25} {'Value':>10}  {'Min':>8}  {'Good':>8}  {'Status'}")
    print("-" * 60)

    all_pass = True

    check_metrics = [
        ("win_rate_pct",     "Win Rate (%)",      False),  # higher is better
        ("sharpe_ratio",     "Sharpe Ratio",      False),
        ("max_drawdown_pct", "Max Drawdown (%)",  True),   # lower is better
        ("profit_factor",    "Profit Factor",     False),
    ]

    for key, label, lower_is_better in check_metrics:
        value   = results.get(key, 0.0)
        minimum = BENCHMARKS[key]["minimum"]
        good    = BENCHMARKS[key]["good"]
        exc     = BENCHMARKS[key]["excellent"]

        if lower_is_better:
            passed_min = value <= minimum
            quality = (
                "EXCELLENT" if value <= exc
                else "GOOD" if value <= good
                else "PASS" if passed_min
                else "FAIL"
            )
        else:
            passed_min = value >= minimum
            quality = (
                "EXCELLENT" if value >= exc
                else "GOOD" if value >= good
                else "PASS" if passed_min
                else "FAIL"
            )

        if not passed_min:
            all_pass = False

        status_sym = "✓" if passed_min else "✗"
        print(
            f"  {label:<25} {value:>10.2f}  {minimum:>8.1f}  {good:>8.1f}  "
            f"{status_sym} {quality}"
        )

    print("=" * 60)

    if all_pass:
        print("  RESULT: ALL MINIMUM BENCHMARKS MET — strategy is viable.")
    else:
        print("  RESULT: *** STRATEGY DOES NOT MEET MINIMUM THRESHOLDS ***")
        print("          *** DO NOT RUN LIVE UNTIL BENCHMARKS ARE MET   ***")

    print("=" * 60 + "\n")
    return all_pass


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def run_monte_carlo(
    trade_returns: list[float],
    n_simulations: int = config.MONTE_CARLO_SIMULATIONS,
    initial_capital: float = config.ACCOUNT_BALANCE,
    ruin_threshold: float = config.MAX_DRAWDOWN_PCT,
    seed: int = 42,
) -> dict:
    """
    Bootstrap Monte Carlo simulation — reshuffle historical trade returns N times.

    Takes the actual sequence of per-trade returns from the backtest and randomly
    shuffles their ORDER (not their values) N times.  Each shuffle produces a
    different equity curve from the same set of trades.

    This answers the critical question:
        "Is this strategy profitable because it has a genuine edge,
         or because the trades happened to arrive in a lucky order?"

    A robust strategy should show:
        — Median return > 0% (profitable in most orderings)
        — Probability of profit  ≥ 60%
        — Probability of ruin   ≤ 10%
        — Actual result ranking < 90th percentile (not relying on lucky ordering)

    Args:
        trade_returns:   Per-trade fractional returns (e.g. 0.015 = +1.5%).
                         Pass results["trade_returns"] from run_backtest().
        n_simulations:   Number of random shuffles to run.
        initial_capital: Starting equity for the simulation.
        ruin_threshold:  Max drawdown fraction defined as "ruin" (default 15%).
        seed:            RNG seed for reproducibility.  Pass None for random.

    Returns:
        dict with:
            n_simulations      — number of simulations run
            n_trades           — number of trades in the sequence
            median_return_pct  — median final return across all sims (%)
            mean_return_pct    — mean final return (%)
            pct5_return        — 5th-percentile return: bad-luck scenario (%)
            pct95_return       — 95th-percentile return: good-luck scenario (%)
            min_return         — worst single simulation return (%)
            max_return         — best single simulation return (%)
            median_max_dd_pct  — median max drawdown (%)
            pct95_max_dd_pct   — 95th-percentile max drawdown: worst-case (%)
            prob_profit_pct    — % of sims that ended with positive return
            prob_ruin_pct      — % of sims where drawdown hit ruin_threshold
            all_returns        — list of all sim final returns (for histogram)
            all_max_drawdowns  — list of all sim max drawdowns
    """
    if len(trade_returns) < 2:
        logger.warning(
            f"Monte Carlo requires at least 2 completed trades. "
            f"Got {len(trade_returns)}. Run the strategy longer before simulating."
        )
        return {
            "n_simulations": 0,
            "n_trades":      len(trade_returns),
        }

    rng       = np.random.default_rng(seed=seed)
    trade_arr = np.array(trade_returns, dtype=np.float64)

    sim_final_returns: list[float] = []
    sim_max_drawdowns: list[float] = []

    for _ in range(n_simulations):
        # Shuffle without replacement — same trades, different ordering.
        shuffled = rng.permutation(trade_arr)

        # Build compounded equity curve: equity[i] = equity[i-1] × (1 + return[i]).
        # np.cumprod is vectorised and fast for 1000-trade sequences.
        equity_curve = initial_capital * np.cumprod(1.0 + shuffled)

        # Final return as a fraction.
        final_return = (equity_curve[-1] - initial_capital) / initial_capital
        sim_final_returns.append(float(final_return))

        # Max drawdown from the equity curve.
        # Prepend initial_capital so the peak starts at the correct value.
        equity_with_start = np.concatenate([[initial_capital], equity_curve])
        running_peak      = np.maximum.accumulate(equity_with_start)
        drawdowns         = (running_peak - equity_with_start) / running_peak
        sim_max_drawdowns.append(float(np.max(drawdowns)))

    returns_arr   = np.array(sim_final_returns)
    drawdowns_arr = np.array(sim_max_drawdowns)

    prob_profit = float(np.mean(returns_arr > 0)) * 100.0
    prob_ruin   = float(np.mean(drawdowns_arr >= ruin_threshold)) * 100.0

    logger.info(
        f"Monte Carlo ({n_simulations:,} simulations, {len(trade_returns)} trades): "
        f"median={np.median(returns_arr)*100:.2f}% | "
        f"prob_profit={prob_profit:.1f}% | prob_ruin={prob_ruin:.1f}%"
    )

    return {
        "n_simulations":     n_simulations,
        "n_trades":          len(trade_returns),
        "median_return_pct": float(np.median(returns_arr)) * 100.0,
        "mean_return_pct":   float(np.mean(returns_arr))   * 100.0,
        "pct5_return":       float(np.percentile(returns_arr, 5))  * 100.0,
        "pct95_return":      float(np.percentile(returns_arr, 95)) * 100.0,
        "min_return":        float(returns_arr.min()) * 100.0,
        "max_return":        float(returns_arr.max()) * 100.0,
        "median_max_dd_pct": float(np.median(drawdowns_arr))        * 100.0,
        "pct95_max_dd_pct":  float(np.percentile(drawdowns_arr, 95)) * 100.0,
        "prob_profit_pct":   prob_profit,
        "prob_ruin_pct":     prob_ruin,
        "all_returns":       sim_final_returns,
        "all_max_drawdowns": sim_max_drawdowns,
    }


def print_monte_carlo_report(
    mc: dict,
    actual_return_pct: float,
) -> None:
    """
    Print a formatted Monte Carlo results report.

    Shows the distribution of outcomes, probability statistics, and compares
    the actual backtest result against the simulated distribution to give a
    "luck score" — how much of the result depended on trade ordering.

    Args:
        mc:                Dict returned by run_monte_carlo().
        actual_return_pct: Total return % from the actual backtest (for comparison).
    """
    if mc.get("n_simulations", 0) == 0:
        print("\n  Monte Carlo: not enough trades to simulate (need ≥ 2).\n")
        return

    n = mc["n_simulations"]

    print("\n" + "=" * 62)
    print(f"  MONTE CARLO  ({n:,} shuffled trade sequences, {mc['n_trades']} trades)")
    print("=" * 62)

    print(f"\n  {'Actual backtest return':<28}: {actual_return_pct:+.2f}%")
    print()

    print(f"  Return distribution across all simulations:")
    print(f"    {'Worst simulation':<24}: {mc['min_return']:+.2f}%")
    print(f"    {'5th  percentile':<24}: {mc['pct5_return']:+.2f}%  (bad-luck scenario)")
    print(f"    {'Median':<24}: {mc['median_return_pct']:+.2f}%")
    print(f"    {'Mean':<24}: {mc['mean_return_pct']:+.2f}%")
    print(f"    {'95th percentile':<24}: {mc['pct95_return']:+.2f}%  (good-luck scenario)")
    print(f"    {'Best simulation':<24}: {mc['max_return']:+.2f}%")
    print()

    print(f"  Max drawdown distribution:")
    print(f"    {'Median max DD':<24}: {mc['median_max_dd_pct']:.2f}%")
    print(f"    {'95th pct max DD':<24}: {mc['pct95_max_dd_pct']:.2f}%  (worst-case scenario)")
    print()

    # Luck score — where does the actual result rank in the distribution?
    all_ret = mc.get("all_returns", [])
    if all_ret:
        rank_pct = float(np.mean(np.array(all_ret) * 100 < actual_return_pct)) * 100
        print(f"  Actual result ranks at    : {rank_pct:.0f}th percentile")

        if rank_pct > 90:
            luck_label = "LUCK-DEPENDENT  — result relies heavily on favorable trade ordering"
            luck_sym   = "!"
        elif rank_pct >= 50:
            luck_label = "ABOVE MEDIAN    — moderate luck contribution"
            luck_sym   = "~"
        else:
            luck_label = "BELOW MEDIAN    — result does not rely on lucky ordering"
            luck_sym   = "✓"

        print(f"  Luck assessment           : {luck_sym} {luck_label}")
    print()

    # Probability statistics with PASS/FAIL marks.
    prob_profit = mc["prob_profit_pct"]
    prob_ruin   = mc["prob_ruin_pct"]

    profit_sym = "✓" if prob_profit >= 60 else ("~" if prob_profit >= 40 else "✗")
    ruin_sym   = "✓" if prob_ruin   <= 10 else ("~" if prob_ruin   <= 25 else "✗")

    print(f"  {profit_sym} Probability of profit    : {prob_profit:.1f}%  (target ≥ 60%)")
    print(f"  {ruin_sym} Probability of ruin      : {prob_ruin:.1f}%  (target ≤ 10%)")

    # ASCII histogram of the final return distribution.
    print()
    print("  Final Return Distribution")
    _print_ascii_histogram(
        values=[r * 100 for r in all_ret],
        n_bins=20,
        bar_width=36,
        zero_marker=True,
        unit="%",
    )

    print("=" * 62 + "\n")


def _print_ascii_histogram(
    values: list[float],
    n_bins: int = 20,
    bar_width: int = 36,
    zero_marker: bool = True,
    unit: str = "",
) -> None:
    """
    Print a horizontal ASCII bar-chart histogram to stdout.

    Args:
        values:      List of float values to bucket.
        n_bins:      Number of equal-width bins.
        bar_width:   Maximum width of the longest bar in characters.
        zero_marker: If True, annotate the bin that contains zero.
        unit:        Unit label appended to bin labels (e.g. "%").
    """
    arr = np.array(values)
    if arr.size == 0:
        return

    lo, hi = arr.min(), arr.max()
    if lo == hi:
        print(f"  All values equal: {lo:.2f}{unit}")
        return

    bins   = np.linspace(lo, hi, n_bins + 1)
    counts, edges = np.histogram(arr, bins=bins)
    max_count = counts.max()

    for i, count in enumerate(counts):
        bar_len   = int(round(count / max_count * bar_width)) if max_count > 0 else 0
        bar       = "#" * bar_len
        bin_lo    = edges[i]
        bin_hi    = edges[i + 1]
        zero_note = " ← break-even" if (zero_marker and bin_lo <= 0.0 < bin_hi) else ""
        print(f"  {bin_lo:+7.1f}{unit} │{bar:<{bar_width}}{zero_note}")

    # Bottom axis.
    print(f"  {'':>8}  └{'─' * bar_width}")
    print(f"  {'':>9}  0{' ' * (bar_width - 2)}{max_count}")


# ---------------------------------------------------------------------------
# Standalone test — run `python backtest/runner.py` from the project root
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from loguru import logger
    import sys
    sys.path.insert(0, ".")

    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}",
    )

    print("\n========================================")
    print("  Bitcoin Bot — Backtest Runner")
    print("========================================")

    # --- Full backtest ---
    print("\n[1/3] Running full backtest...")
    try:
        results = run_backtest(symbol=config.SYMBOL, timeframe=config.TIMEFRAME)
        all_pass = print_benchmark_report(results)
    except Exception as exc:
        logger.error(f"Full backtest failed: {exc}")
        results  = {}
        all_pass = False

    # --- Monte Carlo simulation ---
    print(f"\n[2/3] Running Monte Carlo ({config.MONTE_CARLO_SIMULATIONS:,} simulations)...")
    try:
        trade_returns = results.get("trade_returns", [])
        mc = run_monte_carlo(trade_returns=trade_returns)
        print_monte_carlo_report(mc, actual_return_pct=results.get("total_return_pct", 0.0))
    except Exception as exc:
        logger.error(f"Monte Carlo simulation failed: {exc}")

    # --- Walk-forward validation ---
    print("\n[3/3] Running walk-forward validation ({} folds)...".format(config.WALKFORWARD_FOLDS))
    try:
        wf = run_walk_forward_validation(
            symbol=config.SYMBOL,
            timeframe=config.TIMEFRAME,
            n_folds=config.WALKFORWARD_FOLDS,
        )

        print("\n--- Walk-Forward Validation Results ---")
        for fold in wf["fold_results"]:
            print(
                f"  Fold {fold['fold']}: "
                f"Sharpe={fold['sharpe_ratio']:.2f} | "
                f"WR={fold['win_rate_pct']:.1f}% | "
                f"Trades={fold['total_trades']}"
            )

        print(f"\n  Mean Sharpe : {wf['mean_sharpe']:.2f}")
        print(f"  Std Sharpe  : {wf['std_sharpe']:.2f}")
        print(f"  Robust?     : {'YES' if wf['is_robust'] else 'NO'}")
        print(f"  Reason      : {wf['robustness_reason']}")

        if not wf["is_robust"]:
            print(
                "\n  WARNING: Strategy is NOT robust across walk-forward folds. "
                "Do not trade live."
            )
    except Exception as exc:
        logger.error(f"Walk-forward validation failed: {exc}")

    sys.exit(0 if all_pass else 1)
