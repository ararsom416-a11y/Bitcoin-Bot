"""
config.py — The Control Panel
==============================
Single source of truth for every tunable parameter in the bot.
If you need to change a strategy setting, risk threshold, or timing value,
change it HERE — and only here.  No magic numbers anywhere else in the codebase.

Load order:
  1. Python constants defined below (safe defaults / strategy parameters)
  2. .env secrets loaded via python-dotenv (API keys, never committed)

Usage:
  from config import SYMBOL, EMA_FAST, ...  (import what you need)
  import config                              (access as config.SYMBOL, etc.)
"""

import os
from dotenv import load_dotenv

# Load .env file from the project root.
# This populates os.environ so getenv calls below work correctly.
load_dotenv()

# ---------------------------------------------------------------------------
# Exchange / connectivity
# ---------------------------------------------------------------------------

# The trading pair.  ccxt uses "BTC/USDT" format with a slash.
SYMBOL: str = "BTC/USDT"

# Candle timeframe.  "1h" = 1-hour candles.
# Changing this to "4h" or "15m" would shift the entire strategy's tempo.
TIMEFRAME: str = "1h"

# When True  → connect to Binance Testnet (fake money, real market data).
# When False → connect to live Binance (REAL MONEY — be certain before flipping).
PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "true").lower() == "true"

# Binance API credentials — read from .env, never hard-coded.
BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")

# ---------------------------------------------------------------------------
# Strategy parameters — Triple Confirmation Trend
# ---------------------------------------------------------------------------

# --- Signal 1: EMA Crossover (Trend Filter) ---
# Fast EMA catches recent price momentum.
# Slow EMA represents the medium-term trend direction.
# When fast > slow → the short-term trend agrees with the medium-term trend → bullish.
EMA_FAST: int = 9    # periods
EMA_SLOW: int = 21   # periods

# --- Signal 2: RSI (Momentum Filter) ---
# RSI period: standard 14-period Wilder smoothing.
RSI_PERIOD: int = 14

# Buy zone: RSI must be between these bounds.
# > 50 confirms bullish momentum is dominant.
# < 70 means we are NOT overbought — there is room left for the move to continue.
# Entries above 70 historically have poor risk/reward (buying exhausted moves).
RSI_LOWER: float = 50.0
RSI_UPPER: float = 70.0

# --- Signal 3: Bollinger Band Width (Volatility Expansion Filter) ---
# Period and standard-deviation multiplier for the Bollinger Band calculation.
BB_PERIOD: int = 20
BB_STD: float = 2.0

# --- ATR — used for stop-loss and take-profit placement ---
# ATR adapts automatically to market volatility:
#   volatile market → wider ATR → wider stop (avoids getting shaken out)
#   calm market    → tighter ATR → tighter stop (locks in gains sooner)
ATR_PERIOD: int = 14

# Stop-loss is placed this many ATRs below the entry price.
# 1.5× is tight enough to cut losers quickly but wide enough to survive normal noise.
STOP_ATR_MULTIPLIER: float = 1.5

# Take-profit is placed this many ATRs above the entry price.
# 2.5× gives a ~1:1.67 risk/reward ratio.
# At that ratio the strategy breaks even at a 43% win rate — everything above is profit.
TP_ATR_MULTIPLIER: float = 2.5

# ---------------------------------------------------------------------------
# Risk management — these are circuit breakers, not suggestions
# ---------------------------------------------------------------------------

# Starting paper-trading capital.
ACCOUNT_BALANCE: float = 10_000.0  # USD

# Maximum percentage of account to risk on a single trade.
# 2% is the professional standard for volatile assets.
# Risking more than this leads to ruin curves even with a positive-expectancy strategy.
MAX_RISK_PER_TRADE_PCT: float = 0.02   # 2%

# Daily circuit breaker: halt trading if today's losses exceed this fraction of balance.
# Prevents a bad day from becoming a catastrophic day.
MAX_DAILY_LOSS_PCT: float = 0.05       # 5%

# Drawdown circuit breaker: halt if we have fallen this far from our peak balance ever.
# 15% is the absolute floor — beyond this the strategy is likely broken, not just unlucky.
MAX_DRAWDOWN_PCT: float = 0.15         # 15%

# Consecutive-loss circuit breaker: halt after this many losses in a row.
# 5 straight losses is statistically unusual for a positive-expectancy strategy —
# it suggests either a regime change or a bug that needs human review.
MAX_CONSECUTIVE_LOSSES: int = 5

# Maximum position size as a fraction of account (in USD value).
# Prevents the math from producing absurdly large positions when ATR is tiny.
MAX_POSITION_SIZE_PCT: float = 0.20    # 20%

# Minimum order size in USD equivalent.  Binance rejects orders below ~$10.
MIN_ORDER_SIZE_USD: float = 10.0

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

# Minimum number of candles required in the DataFrame before we compute indicators.
# The slowest indicator (BB/EMA 21) needs ~21 bars, so 50 gives comfortable headroom.
CANDLES_REQUIRED: int = 50

# How often the main loop wakes up to check for new signals (in seconds).
# 300s = 5 minutes.  We trade 1h candles so checking every 5 min is more than adequate.
LOOP_INTERVAL_SECONDS: int = 300

# Number of historical candles to fetch on each loop iteration.
FETCH_LIMIT: int = 100

# Number of retry attempts for API calls before giving up.
API_RETRY_ATTEMPTS: int = 3

# Base sleep time (seconds) for exponential backoff on API retries.
API_RETRY_BASE_SLEEP: float = 2.0

# BTC quantity precision — Binance lot-size step for BTC/USDT spot.
BTC_STEP_SIZE: float = 0.00001  # 5 decimal places

# ---------------------------------------------------------------------------
# Backtest settings
# ---------------------------------------------------------------------------

# Commission per trade (round-trip = 2×).  Binance standard taker fee.
BACKTEST_COMMISSION: float = 0.001    # 0.1%

# Number of folds for walk-forward validation.
WALKFORWARD_FOLDS: int = 5

# Embargo period (fraction of a fold to skip between train and test).
# Prevents look-ahead leakage caused by indicators spanning the fold boundary.
WALKFORWARD_EMBARGO_FRAC: float = 0.1

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# loguru log level for the rotating file handler.
LOG_LEVEL: str = "INFO"

# Path to the log file (relative to project root).
# The logs/ directory is created automatically by main.py on startup.
LOG_FILE: str = "logs/trading_bot.log"

# Maximum size of a single log file before it rotates.
LOG_ROTATION: str = "10 MB"

# How many rotated log files to keep.
LOG_RETENTION: str = "30 days"
