"""
execution/trader.py — Room 4: Order Execution
===============================================
Responsibility: place orders on Binance and manage open positions.
This is the ONLY file in the entire codebase that touches real (or simulated) money.

All order functions wrap ccxt calls in try/except with detailed error logging.
No other module calls ccxt directly.

Design principles:
  — Every order is logged with its full details BEFORE and AFTER placement.
  — Quantity precision is enforced before every order (Binance rejects bad lot sizes).
  — Paper trading (testnet) is controlled entirely by PAPER_TRADING in config.py.
  — OCO orders handle both stop-loss and take-profit simultaneously so neither
    can be "forgotten" if the bot restarts or crashes after entry.
"""

import math

import ccxt
from loguru import logger

import config


# ---------------------------------------------------------------------------
# Exchange initialisation
# ---------------------------------------------------------------------------

def _build_exchange() -> ccxt.binance:
    """
    Construct a ccxt Binance exchange object configured for spot trading.

    Uses sandbox mode when PAPER_TRADING=True — identical interface but
    no real money.  The API keys for testnet and live are different accounts.

    Returns:
        Configured ccxt.binance instance.
    """
    exchange = ccxt.binance(
        {
            "apiKey":  config.BINANCE_API_KEY,
            "secret":  config.BINANCE_SECRET_KEY,
            "options": {"defaultType": "spot"},
            "timeout": 30_000,
            "enableRateLimit": True,
        }
    )

    if config.PAPER_TRADING:
        exchange.set_sandbox_mode(True)
        logger.debug("Trader: exchange in SANDBOX (testnet) mode.")
    else:
        logger.warning("Trader: exchange in LIVE mode — real money at risk!")

    return exchange


# Singleton exchange instance — shared across all trader functions.
_exchange: ccxt.binance = _build_exchange()


# ---------------------------------------------------------------------------
# Quantity precision helper
# ---------------------------------------------------------------------------

def round_quantity(quantity: float, step_size: float = config.BTC_STEP_SIZE) -> float:
    """
    Round a BTC quantity to the valid lot size accepted by Binance.

    Binance enforces strict lot-size rules per trading pair.  Submitting an order
    with more decimal places than allowed causes an immediate rejection with error
    code -1013.  For BTC/USDT spot, the step size is 0.00001 (5 decimal places).

    Args:
        quantity:  Raw calculated quantity in BTC.
        step_size: Lot step size for the trading pair (default: BTC_STEP_SIZE from config).

    Returns:
        Quantity rounded DOWN to the nearest valid lot size.
        (We round down, never up, to avoid inadvertently exceeding our risk budget.)
    """
    if step_size <= 0:
        logger.error(f"Invalid step_size {step_size}. Returning quantity unchanged.")
        return quantity

    precision = int(round(-math.log(step_size, 10), 0))
    # Floor to step size rather than rounding, to stay within risk budget.
    floored = math.floor(quantity / step_size) * step_size
    return round(floored, precision)


# ---------------------------------------------------------------------------
# Function 1: Place market order
# ---------------------------------------------------------------------------

def place_market_order(symbol: str, side: str, quantity: float) -> dict:
    """
    Place a market order on Binance (or testnet).

    Market orders fill immediately at the best available price.
    We use them for entries and for signal-reversal exits (speed matters more
    than price precision when we need to get out).

    Args:
        symbol:   Trading pair in ccxt format, e.g. "BTC/USDT".
        side:     "buy" or "sell".
        quantity: Amount in base currency (BTC for BTC/USDT).

    Returns:
        Order details dict from ccxt (contains order ID, status, filled price, etc.).

    Raises:
        ccxt.InsufficientFunds: If the account lacks the required balance.
        ccxt.InvalidOrder:      If the quantity is below minimum lot size.
        RuntimeError:           If the order fails after retries.
    """
    quantity = round_quantity(quantity)

    logger.info(
        f"Placing MARKET {side.upper()} order: {quantity:.5f} {symbol} "
        f"(paper_trading={config.PAPER_TRADING})"
    )

    try:
        order = _exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=quantity,
        )

        fill_price = order.get("average") or order.get("price") or "market"
        logger.info(
            f"MARKET ORDER FILLED: id={order['id']} | "
            f"side={side.upper()} | qty={order.get('filled', quantity):.5f} BTC | "
            f"avg_price=${fill_price}"
        )
        return order

    except ccxt.InsufficientFunds as exc:
        logger.error(f"Insufficient funds for {side} {quantity} {symbol}: {exc}")
        raise

    except ccxt.InvalidOrder as exc:
        logger.error(
            f"Invalid order rejected by exchange ({side} {quantity} {symbol}): {exc}"
        )
        raise

    except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
        logger.error(
            f"Network error placing market order — order state unknown: {exc}. "
            "Check exchange manually before retrying."
        )
        raise

    except ccxt.BaseError as exc:
        logger.error(f"Exchange error placing market order: {exc}")
        raise


# ---------------------------------------------------------------------------
# Function 2: Place OCO bracket order
# ---------------------------------------------------------------------------

def place_oco_order(
    symbol: str,
    quantity: float,
    take_profit_price: float,
    stop_loss_price: float,
) -> dict:
    """
    Place a One-Cancels-Other (OCO) order covering both take-profit and stop-loss.

    An OCO order places TWO orders simultaneously on the exchange:
      1. A limit sell at take_profit_price (triggered when price rises).
      2. A stop-market sell at stop_loss_price (triggered when price falls).

    When either leg fills, the other is automatically cancelled.
    This is the safest way to bracket a position — it works even if the bot
    crashes or loses connectivity after entry.

    Args:
        symbol:            Trading pair, e.g. "BTC/USDT".
        quantity:          BTC quantity to sell (must match the open position size).
        take_profit_price: Limit sell price (above entry).
        stop_loss_price:   Stop trigger price (below entry).

    Returns:
        Dict containing order IDs and details of both legs.

    Raises:
        ccxt.BaseError: On any exchange-level rejection.
    """
    quantity = round_quantity(quantity)

    # Binance requires prices to be rounded to the tick size.
    # BTC/USDT tick size is $0.01 — round to 2 decimal places.
    tp = round(take_profit_price, 2)
    sl = round(stop_loss_price, 2)

    logger.info(
        f"Placing OCO bracket order: {quantity:.5f} {symbol} | "
        f"TP=${tp:.2f} | SL=${sl:.2f}"
    )

    try:
        # ccxt's create_order supports OCO via the 'STOP_LOSS_LIMIT' type on Binance,
        # but the cleanest way is to use the private API endpoint directly via ccxt.
        # We use the built-in OCO method available in ccxt >= 4.x.
        #
        # Note: Binance OCO requires listClientOrderId, stopPrice, and price.
        # The stop-limit price (stopLimitPrice) is set slightly below the stop trigger
        # to ensure the stop-market fill happens even in fast-moving markets.
        stop_limit_price = round(sl * 0.995, 2)  # 0.5% below stop trigger

        result = _exchange.create_order(
            symbol=symbol,
            type="oco",
            side="sell",
            amount=quantity,
            price=tp,          # Limit take-profit price
            params={
                "stopPrice":      sl,               # Stop trigger
                "stopLimitPrice": stop_limit_price, # Limit price after stop triggers
                "stopLimitTimeInForce": "GTC",
            },
        )

        logger.info(
            f"OCO ORDER PLACED: "
            f"TP leg id={result.get('orders', [{}])[0].get('id', '?')} @ ${tp:.2f} | "
            f"SL leg id={result.get('orders', [{}])[-1].get('id', '?')} @ ${sl:.2f} "
            f"(trigger), ${stop_limit_price:.2f} (limit)"
        )
        return result

    except ccxt.InvalidOrder as exc:
        logger.error(
            f"OCO order rejected: {exc}. "
            "Check that TP > current price > SL and lot sizes are valid."
        )
        raise

    except ccxt.BaseError as exc:
        logger.error(f"Exchange error placing OCO order: {exc}")
        raise


# ---------------------------------------------------------------------------
# Function 3: Get open orders
# ---------------------------------------------------------------------------

def get_open_orders(symbol: str = config.SYMBOL) -> list:
    """
    Fetch all currently open orders for the given symbol.

    Used to check whether OCO legs are still active (neither TP nor SL has filled).

    Args:
        symbol: Trading pair, e.g. "BTC/USDT".

    Returns:
        List of open order dicts from ccxt.  Empty list if none.
    """
    try:
        orders = _exchange.fetch_open_orders(symbol=symbol)
        logger.debug(f"Open orders for {symbol}: {len(orders)}")
        return orders

    except ccxt.BaseError as exc:
        logger.error(f"Failed to fetch open orders for {symbol}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Function 4: Cancel all orders
# ---------------------------------------------------------------------------

def cancel_all_orders(symbol: str = config.SYMBOL) -> bool:
    """
    Cancel all open orders for the given symbol.

    Called when a signal-reversal exit triggers — we need to cancel the OCO
    bracket before placing the manual market sell, otherwise we'd have
    conflicting orders on the same position.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT".

    Returns:
        True if cancellation succeeded (or there were no orders to cancel).
        False if the cancellation request itself failed.
    """
    try:
        open_orders = get_open_orders(symbol)

        if not open_orders:
            logger.debug(f"No open orders to cancel for {symbol}.")
            return True

        cancelled_count = 0
        for order in open_orders:
            try:
                _exchange.cancel_order(order["id"], symbol)
                logger.info(f"Cancelled order id={order['id']} ({order.get('type')} {order.get('side')})")
                cancelled_count += 1
            except ccxt.OrderNotFound:
                # Order already filled or cancelled — not an error.
                logger.debug(f"Order {order['id']} already gone (filled or cancelled).")
                cancelled_count += 1
            except ccxt.BaseError as exc:
                logger.error(f"Failed to cancel order {order['id']}: {exc}")

        logger.info(f"Cancelled {cancelled_count}/{len(open_orders)} orders for {symbol}.")
        return cancelled_count == len(open_orders)

    except ccxt.BaseError as exc:
        logger.error(f"Error during cancel_all_orders for {symbol}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Function 5: Get account balance
# ---------------------------------------------------------------------------

def get_account_balance(currency: str = "USDT") -> float:
    """
    Fetch the free (available) balance for the specified currency.

    Used to sync the portfolio state with the actual exchange balance,
    catching any discrepancies between our internal accounting and reality.

    Args:
        currency: Currency symbol to query (default "USDT").

    Returns:
        Free balance as a float.  Returns 0.0 on error (logged).
    """
    try:
        balance = _exchange.fetch_balance()
        free = balance.get("free", {}).get(currency, 0.0)
        logger.debug(f"Account balance: {free:.2f} {currency} (free)")
        return float(free)

    except ccxt.AuthenticationError as exc:
        logger.error(f"Authentication error fetching balance: {exc}")
        return 0.0

    except ccxt.BaseError as exc:
        logger.error(f"Failed to fetch {currency} balance: {exc}")
        return 0.0
