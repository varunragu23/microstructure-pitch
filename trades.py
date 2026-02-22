"""
Fetch pre-event trades for selected Kalshi event tickers and plot dollar-amount distributions.

Usage:
    python fetch_trades.py
"""

import requests
import csv
import sys
import os
from datetime import datetime, timezone
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib and numpy are required. Install with: pip install matplotlib numpy")
    sys.exit(1)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Target event tickers
TARGET_TICKERS = [
    "KXTRUMPMENTION-25DEC31",
    "KXMAMDANIMENTION-25SEP06",
    "KXEARNINGSMENTIONCBRL-25SEP17",
    "KXSOUTHPARKMENTION-25SEP24",
]


def load_milestone_start_times(csv_path="data/milestones.csv"):
    """Load the earliest start time for each target event ticker from milestones CSV."""
    start_times = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickers = row["primary_event_ticker"].split(";")
            for ticker in tickers:
                ticker = ticker.strip()
                if ticker in TARGET_TICKERS:
                    dt = datetime.fromisoformat(row["start_date"].replace("Z", "+00:00"))
                    if ticker not in start_times or dt < start_times[ticker]:
                        start_times[ticker] = dt
    return start_times


def get_markets_for_event(event_ticker):
    """Get all market tickers for a given event ticker."""
    url = f"{BASE_URL}/markets"
    all_markets = []
    cursor = None

    while True:
        params = {"limit": 200, "event_ticker": event_ticker}
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        markets = data.get("markets", [])
        all_markets.extend(markets)

        cursor = data.get("cursor")
        if not cursor or not markets:
            break

    return all_markets


def fetch_trades(ticker, max_ts=None):
    """Fetch all trades for a market ticker, optionally before max_ts (epoch seconds)."""
    all_trades = []
    cursor = None

    while True:
        params = {"limit": 1000, "ticker": ticker}
        if cursor:
            params["cursor"] = cursor
        if max_ts is not None:
            params["max_ts"] = int(max_ts)

        resp = requests.get(f"{BASE_URL}/markets/trades", params=params)
        resp.raise_for_status()
        data = resp.json()

        trades = data.get("trades", [])
        all_trades.extend(trades)

        cursor = data.get("cursor")
        if not cursor or not trades:
            break

    return all_trades


def write_trades_csv(all_trades, output_path="data/trades.csv"):
    """Write trades to CSV."""
    if not all_trades:
        print("No trades to write.")
        return

    fieldnames = [
        "event_ticker",
        "ticker",
        "trade_id",
        "count",
        "yes_price",
        "no_price",
        "taker_side",
        "created_time",
        "dollar_amount",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for trade in all_trades:
            writer.writerow(trade)

    print(f"\nWrote {len(all_trades)} trades to {output_path}")


def create_plots(all_trades):
    """Create distribution plots of dollar amount per trade for each event ticker."""
    trades_by_event = defaultdict(list)
    for trade in all_trades:
        trades_by_event[trade["event_ticker"]].append(trade)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, event_ticker in enumerate(TARGET_TICKERS):
        ax = axes[idx]
        trades = trades_by_event.get(event_ticker, [])

        if not trades:
            ax.set_title(f"{event_ticker}\n(no trades found)")
            ax.set_xlabel("Dollar Amount ($)")
            ax.set_ylabel("Frequency")
            continue

        dollar_amounts = [t["dollar_amount"] for t in trades]

        # Choose bins: use a reasonable range
        max_amount = max(dollar_amounts)
        min_amount = min(dollar_amounts)

        if max_amount <= 10:
            bin_edges = np.arange(0, max_amount + 1, 0.5)
        elif max_amount <= 100:
            bin_edges = np.arange(0, max_amount + 5, 5)
        elif max_amount <= 500:
            bin_edges = np.arange(0, max_amount + 10, 10)
        else:
            bin_edges = np.arange(0, max_amount + 50, 50)

        if len(bin_edges) < 3:
            bin_edges = 20  # fallback to auto bins

        ax.hist(dollar_amounts, bins=bin_edges, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_xlabel("Dollar Amount ($)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{event_ticker}\n({len(trades)} trades)")

        # Add summary stats
        median_val = np.median(dollar_amounts)
        mean_val = np.mean(dollar_amounts)
        ax.axvline(median_val, color="red", linestyle="--", linewidth=1, label=f"Median: ${median_val:.2f}")
        ax.axvline(mean_val, color="orange", linestyle="--", linewidth=1, label=f"Mean: ${mean_val:.2f}")
        ax.legend(fontsize=8)

    plt.suptitle("Pre-Event Trade Distribution by Dollar Amount", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = "data/trade_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.show()


def main():
    # Step 1: Load milestone start times
    start_times = load_milestone_start_times()
    print("Milestone start times (earliest per event ticker):")
    for t in TARGET_TICKERS:
        if t in start_times:
            print(f"  {t}: {start_times[t].isoformat()}")
        else:
            print(f"  {t}: NOT FOUND in milestones.csv")

    all_trades = []

    for event_ticker in TARGET_TICKERS:
        if event_ticker not in start_times:
            print(f"\nWARNING: No start time found for {event_ticker}, skipping.")
            continue

        max_dt = start_times[event_ticker]
        max_ts = max_dt.timestamp()

        # Step 2: Get markets for this event
        print(f"\nFetching markets for event {event_ticker}...")
        try:
            markets = get_markets_for_event(event_ticker)
            market_tickers = [m["ticker"] for m in markets]
            print(f"  Found {len(market_tickers)} markets: {market_tickers}")
        except Exception as e:
            print(f"  Error fetching markets: {e}")
            continue

        # Step 3: Fetch trades for each market
        for market_ticker in market_tickers:
            print(f"  Fetching trades for {market_ticker} (before {max_dt.isoformat()})...")
            try:
                trades = fetch_trades(market_ticker, max_ts=max_ts)
                for trade in trades:
                    trade["event_ticker"] = event_ticker
                    # Compute dollar amount: count * yes_price / 100
                    trade["dollar_amount"] = trade.get("count", 0) * trade.get("yes_price", 0) / 100.0
                all_trades.extend(trades)
                print(f"    Retrieved {len(trades)} trades")
            except Exception as e:
                print(f"    Error: {e}")

    print(f"\nTotal trades collected: {len(all_trades)}")

    if not all_trades:
        print("No trades found. Exiting.")
        sys.exit(1)

    # Step 4: Write CSV
    write_trades_csv(all_trades)

    # Step 5: Create plots
    create_plots(all_trades)


if __name__ == "__main__":
    main()
