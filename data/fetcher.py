"""
data/fetcher.py — Room 1: Market Data
=======================================
Responsibility: fetch OHLCV candle data from Binance and return a clean DataFrame.
Nothing else.  This module does not compute indicators, make trading decisions,
or touch any order management code.

Why ccxt?
  ccxt is a unified exchange library that works with 100+ exchanges using the same API.
  If we ever switch from Binance to Bybit or Kraken, we change one line here — nothing else.

Why testnet?
  Binance Testnet (https://testnet.binance.vision) provides real market data feeds
  but uses virtual funds.  We can run the full system without risking real money.
  Controlled by PAPER_TRADING in config.py.
"""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
del _sys, _pathlib

import time
from datetime import datetime, timezone

import ccxt
import pandas as pd
from loguru import logger

import config


def _build_exchange() -> ccxt.binance:
    """
    Construct and return a configured ccxt Binance exchange object.

    When PAPER_TRADING is True, set_sandbox_mode(True) redirects all API calls
    to https://testnet.binance.vision automatically.  No manual URL wrangling needed.

    Returns:
        ccxt.binance: Ready-to-use exchange instance.
    """
    exchange = ccxt.binance(
        {
            "apiKey": config.BINANCE_API_KEY,
            "secret": config.BINANCE_SECRET_KEY,
            "options": {
                # Use the spot market (as opposed to futures/margin).
                "defaultType": "spot",
            },
            # ccxt will raise ccxt.RequestTimeout after this many milliseconds.
            "timeout": 30_000,
            # Automatically handle Binance rate-limit headers.
            "enableRateLimit": True,
        }
    )

    if config.PAPER_TRADING:
        # Redirect all requests to the Binance Testnet endpoints.
        exchange.set_sandbox_mode(True)
        logger.debug("Exchange configured in SANDBOX (testnet) mode.")
    else:
        logger.warning(
            "Exchange configured in LIVE mode.  Real money at risk!"
        )

    return exchange


# Module-level exchange singleton — built once, reused on every fetch call.
# This avoids the overhead of re-initialising the exchange object on each loop tick.
_exchange: ccxt.binance = _build_exchange()


def fetch_ohlcv(
    symbol: str = config.SYMBOL,
    timeframe: str = config.TIMEFRAME,
    limit: int = config.FETCH_LIMIT,
) -> pd.DataFrame:
    """
    Fetch OHLCV (Open/High/Low/Close/Volume) candle data from Binance.

    Retries up to API_RETRY_ATTEMPTS times with exponential backoff on transient
    errors (network timeouts, rate limits, exchange unavailability).

    Args:
        symbol:    Trading pair in ccxt slash-format, e.g. "BTC/USDT".
        timeframe: Candle width, e.g. "1h", "15m", "4h".
        limit:     Number of candles to fetch (most recent N candles).

    Returns:
        pd.DataFrame with columns:
            timestamp (datetime, UTC) | open | high | low | close | volume
        Rows are sorted oldest-first (standard time-series convention).

    Raises:
        RuntimeError: If all retry attempts are exhausted.
    """
    last_exception: Exception | None = None

    for attempt in range(1, config.API_RETRY_ATTEMPTS + 1):
        try:
            logger.debug(
                f"Fetching {limit} candles for {symbol} [{timeframe}] "
                f"(attempt {attempt}/{config.API_RETRY_ATTEMPTS})"
            )

            # ccxt returns a list of lists: [[timestamp_ms, O, H, L, C, V], ...]
            raw: list[list] = _exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )

            if not raw:
                raise ValueError(
                    f"Received empty response from exchange for {symbol} {timeframe}."
                )

            df = _parse_ohlcv(raw)

            logger.info(
                f"Fetched {len(df)} candles for {symbol} [{timeframe}] | "
                f"range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}"
            )
            return df

        except ccxt.RateLimitExceeded as exc:
            # The exchange told us we're sending too many requests.
            # Back off with the same schedule as other transient errors.
            # attempt is 1-indexed: attempt=1 → 2s, attempt=2 → 4s, attempt=3 → 8s.
            wait = config.API_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            logger.warning(
                f"Rate limit exceeded (attempt {attempt}). "
                f"Sleeping {wait}s before retry. Detail: {exc}"
            )
            last_exception = exc
            time.sleep(wait)

        except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
            wait = config.API_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            logger.warning(
                f"Network error on fetch attempt {attempt}: {exc}. "
                f"Retrying in {wait}s."
            )
            last_exception = exc
            time.sleep(wait)

        except ccxt.ExchangeNotAvailable as exc:
            wait = config.API_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            logger.warning(
                f"Exchange not available (attempt {attempt}). "
                f"Sleeping {wait}s. Detail: {exc}"
            )
            last_exception = exc
            time.sleep(wait)

        except ccxt.BaseError as exc:
            # Any other ccxt-specific error (auth failure, bad symbol, etc.)
            # These are usually not retryable, so we raise immediately.
            logger.error(f"Non-retryable exchange error: {exc}")
            raise

        except ValueError as exc:
            # Empty or malformed response — worth retrying.
            wait = config.API_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            logger.warning(
                f"Empty response (attempt {attempt}): {exc}. Retrying in {wait}s."
            )
            last_exception = exc
            time.sleep(wait)

    # All attempts exhausted.
    raise RuntimeError(
        f"Failed to fetch OHLCV data after {config.API_RETRY_ATTEMPTS} attempts. "
        f"Last error: {last_exception}"
    )


def _parse_ohlcv(raw: list[list]) -> pd.DataFrame:
    """
    Convert the raw ccxt list-of-lists into a clean, typed DataFrame.

    Args:
        raw: List of [timestamp_ms, open, high, low, close, volume] rows from ccxt.

    Returns:
        pd.DataFrame with proper column names and UTC datetime timestamps.
    """
    df = pd.DataFrame(
        raw,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    # ccxt returns timestamps in milliseconds since Unix epoch.
    # Convert to proper UTC datetime objects for readability and time-arithmetic.
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # Ensure all price/volume columns are float64 (ccxt may return strings on some
    # exchanges or in edge cases).
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort oldest → newest (defensive — ccxt normally returns this order).
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def test_connection() -> bool:
    """
    Ping the exchange to verify connectivity before the main loop starts.

    Returns:
        True if the exchange responds successfully, False otherwise.
    """
    try:
        _exchange.load_markets()
        logger.info("Binance connection test: OK")
        return True
    except ccxt.AuthenticationError as exc:
        logger.error(
            f"Authentication failed — check your API key and secret in .env: {exc}"
        )
        return False
    except ccxt.NetworkError as exc:
        logger.error(f"Network error during connection test: {exc}")
        return False
    except ccxt.BaseError as exc:
        logger.error(f"Exchange error during connection test: {exc}")
        return False


# ---------------------------------------------------------------------------
# Standalone test — run `python data/fetcher.py` from the project root to verify
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, pathlib
    # Ensure the project root is on the path so `import config` works when
    # this file is run directly as `python data/fetcher.py`.
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    # Reconfigure the already-imported loguru logger for console-only standalone output.
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="DEBUG",
        format="{time:HH:mm:ss} | {level} | {message}",
    )

    logger.info("=== Fetcher standalone test ===")

    ok = test_connection()
    if not ok:
        logger.error("Connection failed — check .env credentials and network.")
        raise SystemExit(1)

    df = fetch_ohlcv(limit=100)

    print("\n--- Last 5 candles ---")
    pd.set_option("display.float_format", "{:.2f}".format)
    print(df.tail(5).to_string(index=False))
    print(f"\nTotal rows fetched: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"dtypes:\n{df.dtypes}")
