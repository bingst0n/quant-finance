import os
import sys
import time
import logging

# Set matplotlib to non-interactive backend BEFORE any other imports
# (makepreds.py calls .plot() at module level)
import matplotlib
matplotlib.use('Agg')

# Ensure working directory is the project root so relative paths in makepreds work
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# Add neuralnetwork dir to path so makepreds can be imported
sys.path.insert(0, os.path.join(PROJECT_DIR, 'neuralnetwork'))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_DIR, '.env'))

import urllib.request
import json

import robin_stocks.robinhood as rh
import pandas as pd
import yfinance as yf

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=os.path.join(PROJECT_DIR, 'weekly_run.log'),
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
# Set ALLOCATION_AMOUNT in .env to fix the dollar amount managed by this strategy.
# Leave it unset (or set to "portfolio") to use your total portfolio equity.
ALLOCATION_ENV = os.getenv('ALLOCATION_AMOUNT', 'portfolio')

MIN_ORDER_DOLLARS = 1.0   # ignore deltas smaller than this to avoid tiny orders
SLEEP_BETWEEN_ORDERS = 2  # seconds to wait between API calls
RESERVE_PCT = 0.05        # reserve 5% of portfolio for taxes and stability

NOTIFICATION_WEBHOOK_URL = os.getenv('NOTIFICATION_WEBHOOK_URL', '')


def send_notification(title: str, message: str) -> None:
    """POST a notification to the configured webhook URL.

    Works with Pushcut  (https://api.pushcut.io/<key>/notifications/<name>)
    and ntfy.sh         (https://ntfy.sh/<topic>).
    Set NOTIFICATION_WEBHOOK_URL in .env to enable.
    """
    if not NOTIFICATION_WEBHOOK_URL:
        return
    try:
        payload = json.dumps({"title": title, "text": message}).encode()
        req = urllib.request.Request(
            NOTIFICATION_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
        log.info(f"Notification sent: {title}")
    except Exception as exc:
        log.warning(f"Notification failed (non-fatal): {exc}")


def login():
    username = os.environ['ROBINHOOD_USERNAME']
    password = os.environ['ROBINHOOD_PASSWORD']
    # store_session=True caches the auth token so MFA is only needed on first run
    rh.login(username, password, store_session=True)
    log.info("Logged in to Robinhood")


def get_predictions():
    import importlib
    import makepreds
    importlib.reload(makepreds)          # force a fresh run each time
    return makepreds.predictions.copy()


def get_allocation_amount():
    if ALLOCATION_ENV.lower() == 'portfolio':
        profile = rh.profiles.load_portfolio_profile()
        amount = float(profile['equity']) * (1 - RESERVE_PCT) - 10.00
    else:
        amount = float(ALLOCATION_ENV)
    log.info(f"Allocation amount: ${amount:.2f}")
    return amount


def get_current_holdings(tickers):
    """Return dict of ticker -> current dollar value for tickers in our universe."""
    holdings = rh.account.build_holdings()
    current = {}
    for ticker in tickers:
        if ticker in holdings:
            current[ticker] = float(holdings[ticker]['equity'])
        else:
            current[ticker] = 0.0
    return current


def rebalance(predictions, allocation_amount):
    tickers = predictions.index.tolist()

    # Floor negative predictions to 0
    pred = predictions['predicted_return_pct'].astype(float).clip(lower=0)
    total_pct = pred.sum()

    if total_pct == 0:
        log.info("All predictions <= 0 — no trades placed this week.")
        print("All predictions are negative or zero. Holding cash, no trades placed.")
        return

    # Target dollar allocation per ticker
    target = (pred / total_pct) * allocation_amount

    current = get_current_holdings(tickers)

    # Delta = how much to buy (positive) or sell (negative) per ticker
    deltas = {t: target[t] - current[t] for t in tickers}

    log.info(f"Target:  {target.to_dict()}")
    log.info(f"Current: {current}")
    log.info(f"Deltas:  {deltas}")

    print("\n── Target allocation ──────────────────────")
    for t in tickers:
        print(f"  {t:5s}  target ${target[t]:8.2f}  current ${current[t]:8.2f}  delta ${deltas[t]:+.2f}")

    # Sell first to free up cash
    print("\n── Sells ───────────────────────────────────")
    for ticker, delta in deltas.items():
        if delta < -MIN_ORDER_DOLLARS:
            amount = abs(delta)
            print(f"  SELL ${amount:.2f} of {ticker}")
            log.info(f"SELL ${amount:.2f} of {ticker}")
            rh.order_sell_fractional_by_price(ticker, amount, timeInForce='gfd', extendedHours=False)
            time.sleep(SLEEP_BETWEEN_ORDERS)

    # Brief pause before buying
    time.sleep(5)

    # Buy
    print("\n── Buys ────────────────────────────────────")
    for ticker, delta in deltas.items():
        if delta > MIN_ORDER_DOLLARS:
            print(f"  BUY  ${delta:.2f} of {ticker}")
            log.info(f"BUY ${delta:.2f} of {ticker}")
            rh.order_buy_fractional_by_price(ticker, delta, timeInForce='gfd', extendedHours=False)
            time.sleep(SLEEP_BETWEEN_ORDERS)

    print("\nRebalance complete.")
    log.info("Rebalance complete.")


LOCKFILE = os.path.join(PROJECT_DIR, '.last_run_date')

def already_ran_today() -> bool:
    today = str(pd.Timestamp.now().date())
    if os.path.exists(LOCKFILE):
        with open(LOCKFILE) as f:
            return f.read().strip() == today
    return False

def mark_ran_today() -> None:
    with open(LOCKFILE, 'w') as f:
        f.write(str(pd.Timestamp.now().date()))


def main(force: bool = False):
    log.info("=" * 60)
    log.info("Weekly rebalance starting")
    print("Weekly rebalance starting...")

    if already_ran_today() and not force:
        msg = "Rebalance already ran today — skipping to avoid day trades. Use --force to override."
        log.warning(msg)
        print(msg)
        return

    try:
        login()
        predictions = get_predictions()
        log.info(f"Predictions:\n{predictions}")
        print(f"\nPredictions:\n{predictions}")

        allocation_amount = get_allocation_amount()
        print(f"\nManaging ${allocation_amount:.2f}")

        rebalance(predictions, allocation_amount)
        mark_ran_today()
        send_notification(
            "Portfolio rebalanced",
            f"Weekly rebalance complete. Managing ${allocation_amount:.2f}.",
        )

    except Exception as e:
        log.error(f"Rebalance failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        send_notification("Portfolio rebalance FAILED", str(e))
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Override the day trade lockfile')
    args = parser.parse_args()
    main(force=args.force)
