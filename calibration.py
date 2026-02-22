"""
Calibration analysis for high-dollar-volume trades (>= 90th percentile per event).

- Fetches market outcomes from the Kalshi API
- Computes taker-side implied probability and win/loss
- Plots a calibration chart and prints performance metrics

Usage:
    python calibration.py
"""

import csv
import sys
import requests
import time
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib and numpy are required. Install with: pip install matplotlib numpy")
    sys.exit(1)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

TARGET_TICKERS = [
    "KXTRUMPMENTION-25DEC31",
    "KXMAMDANIMENTION-25SEP06",
    "KXEARNINGSMENTIONCBRL-25SEP17",
    "KXSOUTHPARKMENTION-25SEP24",
]

FRIENDLY_NAMES = {
    "KXTRUMPMENTION-25DEC31": "Trump Mention (Army Anniversary)",
    "KXMAMDANIMENTION-25SEP06": "Mamdani Mention (Oligarchy Tour)",
    "KXEARNINGSMENTIONCBRL-25SEP17": "CBRL Earnings Mention",
    "KXSOUTHPARKMENTION-25SEP24": "South Park Mention",
}


def load_trades(csv_path="data/trades.csv"):
    """Load trades from CSV, returning list of dicts."""
    trades = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["count"] = int(row["count"])
            row["yes_price"] = int(row["yes_price"])
            row["no_price"] = int(row["no_price"])
            row["dollar_amount"] = float(row["dollar_amount"])
            trades.append(row)
    return trades


def fetch_market_result(ticker):
    """Fetch the resolution result for a market ticker."""
    resp = requests.get(f"{BASE_URL}/markets/{ticker}")
    resp.raise_for_status()
    market = resp.json().get("market", {})
    return market.get("result")  # "yes" or "no"


def fetch_all_market_results(tickers):
    """Fetch results for all market tickers with rate-limit handling."""
    results = {}
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        try:
            result = fetch_market_result(ticker)
            results[ticker] = result
            if (i + 1) % 20 == 0:
                print(f"  Fetched {i+1}/{total} market results...")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"  Rate limited, sleeping 2s...")
                time.sleep(2)
                result = fetch_market_result(ticker)
                results[ticker] = result
            else:
                print(f"  Error fetching {ticker}: {e}")
                results[ticker] = None
    return results


def compute_taker_fields(trade, market_result):
    """Compute taker-side probability and whether the taker won."""
    taker_side = trade["taker_side"]

    # Taker-side price (cents -> probability as fraction)
    if taker_side == "yes":
        taker_price = trade["yes_price"] / 100.0
    else:
        taker_price = trade["no_price"] / 100.0

    # Did the taker win?
    taker_won = 1 if (market_result == taker_side) else 0

    return taker_price, taker_won


def create_calibration_plot(trades_by_event, market_results):
    """Create calibration chart for 90th-percentile trades across all 4 events."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    # Calibration bins: 10 bins from 0 to 1
    bin_edges = np.linspace(0, 1, 11)  # [0, 0.1, 0.2, ..., 1.0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    all_metrics = {}

    for idx, event_ticker in enumerate(TARGET_TICKERS):
        ax = axes[idx]
        trades = trades_by_event.get(event_ticker, [])

        if not trades:
            ax.set_title(f"{FRIENDLY_NAMES.get(event_ticker, event_ticker)}\n(no trades)")
            continue

        # Compute dollar volume 90th percentile for this event
        dollar_amounts = [t["dollar_amount"] for t in trades]
        p90 = np.percentile(dollar_amounts, 90)

        # Filter to >= 90th percentile
        high_vol_trades = [t for t in trades if t["dollar_amount"] >= p90]

        # Compute taker price and outcome for each trade
        taker_prices = []
        taker_outcomes = []
        taker_edges = []

        for t in high_vol_trades:
            result = market_results.get(t["ticker"])
            if result is None:
                continue
            taker_price, taker_won = compute_taker_fields(t, result)
            taker_prices.append(taker_price)
            taker_outcomes.append(taker_won)
            # Edge: actual outcome - predicted probability
            taker_edges.append(taker_won - taker_price)

        taker_prices = np.array(taker_prices)
        taker_outcomes = np.array(taker_outcomes)
        taker_edges = np.array(taker_edges)

        if len(taker_prices) == 0:
            ax.set_title(f"{FRIENDLY_NAMES.get(event_ticker, event_ticker)}\n(no resolved trades)")
            continue

        # Bin trades by taker-side implied probability
        bin_indices = np.digitize(taker_prices, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)

        bin_actual = []
        bin_predicted = []
        bin_counts = []

        for b in range(len(bin_centers)):
            mask = bin_indices == b
            count = mask.sum()
            bin_counts.append(count)
            if count > 0:
                bin_actual.append(taker_outcomes[mask].mean())
                bin_predicted.append(taker_prices[mask].mean())
            else:
                bin_actual.append(np.nan)
                bin_predicted.append(np.nan)

        bin_actual = np.array(bin_actual)
        bin_predicted = np.array(bin_predicted)
        bin_counts = np.array(bin_counts)

        # Plot calibration curve
        valid = bin_counts > 0
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.scatter(
            bin_predicted[valid], bin_actual[valid],
            s=bin_counts[valid] * 15, alpha=0.7, color="steelblue",
            edgecolors="black", linewidths=0.5, zorder=5,
        )
        ax.plot(bin_predicted[valid], bin_actual[valid], "o-", color="steelblue", alpha=0.5, markersize=0)

        # Annotate counts
        for b in range(len(bin_centers)):
            if bin_counts[b] > 0:
                ax.annotate(
                    f"n={bin_counts[b]}",
                    (bin_predicted[b], bin_actual[b]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=6, ha="center", color="gray",
                )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Taker-Side Implied Probability")
        ax.set_ylabel("Actual Win Rate")
        ax.set_title(
            f"{FRIENDLY_NAMES.get(event_ticker, event_ticker)}\n"
            f"(p90 ≥ ${p90:.2f}, n={len(taker_prices)} trades)",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="upper left")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Compute metrics
        win_rate = taker_outcomes.mean()
        avg_taker_price = taker_prices.mean()
        avg_edge = taker_edges.mean()
        brier_score = np.mean((taker_prices - taker_outcomes) ** 2)

        # ROI: if taker wins, they get $1 per contract, paid taker_price
        # profit per dollar = (payout - cost) / cost
        # payout = taker_won * count, cost = taker_price * count
        total_cost = 0
        total_payout = 0
        for t, tp, tw in zip(high_vol_trades, taker_prices, taker_outcomes):
            result = market_results.get(t["ticker"])
            if result is None:
                continue
            cost = tp * t["count"]
            payout = tw * t["count"]
            total_cost += cost
            total_payout += payout

        roi = (total_payout - total_cost) / total_cost if total_cost > 0 else 0

        all_metrics[event_ticker] = {
            "n_trades_total": len(trades),
            "p90_threshold": p90,
            "n_high_vol": len(taker_prices),
            "win_rate": win_rate,
            "avg_implied_prob": avg_taker_price,
            "avg_edge": avg_edge,
            "brier_score": brier_score,
            "roi": roi,
            "total_cost": total_cost,
            "total_payout": total_payout,
        }

    plt.suptitle(
        "Calibration: 90th Percentile Dollar Volume Trades\n(Taker-Side Implied Probability vs. Actual Outcome)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    output_path = "data/calibration_chart.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved calibration chart to {output_path}")
    plt.close()

    return all_metrics


def print_metrics(all_metrics):
    """Print a formatted summary of calibration metrics."""
    print("\n" + "=" * 90)
    print("CALIBRATION METRICS — 90th Percentile Dollar Volume Trades (Taker Side)")
    print("=" * 90)

    header = f"{'Event':<35} {'# Trades':>8} {'p90 $':>8} {'Win%':>7} {'Impl%':>7} {'Edge':>8} {'Brier':>7} {'ROI':>8}"
    print(header)
    print("-" * 90)

    for event_ticker in TARGET_TICKERS:
        m = all_metrics.get(event_ticker)
        if m is None:
            continue
        name = FRIENDLY_NAMES.get(event_ticker, event_ticker)[:34]
        print(
            f"{name:<35} "
            f"{m['n_high_vol']:>8d} "
            f"{m['p90_threshold']:>8.2f} "
            f"{m['win_rate']:>6.1%} "
            f"{m['avg_implied_prob']:>6.1%} "
            f"{m['avg_edge']:>+7.1%} "
            f"{m['brier_score']:>7.4f} "
            f"{m['roi']:>+7.1%}"
        )

    print("-" * 90)
    print("\nMetric definitions:")
    print("  Win%       = Fraction of trades where the taker's side was the market outcome")
    print("  Impl%      = Average taker-side price (implied probability of winning)")
    print("  Edge       = Win% - Impl% (positive = taker outperformed market price)")
    print("  Brier      = Mean squared error of implied prob vs. outcome (lower = better calibrated)")
    print("  ROI        = (total payout - total cost) / total cost")
    print("  p90 $      = 90th percentile dollar volume threshold")

    # Also print cost/payout details
    print("\n" + "-" * 90)
    print(f"{'Event':<35} {'Total Cost':>12} {'Total Payout':>14} {'Net P&L':>12}")
    print("-" * 90)
    for event_ticker in TARGET_TICKERS:
        m = all_metrics.get(event_ticker)
        if m is None:
            continue
        name = FRIENDLY_NAMES.get(event_ticker, event_ticker)[:34]
        net = m["total_payout"] - m["total_cost"]
        print(
            f"{name:<35} "
            f"${m['total_cost']:>10.2f} "
            f"${m['total_payout']:>12.2f} "
            f"${net:>+10.2f}"
        )
    print("=" * 90)


def main():
    # Step 1: Load trades
    print("Loading trades from data/trades.csv...")
    trades = load_trades()
    print(f"  Loaded {len(trades)} trades")

    # Group by event
    trades_by_event = defaultdict(list)
    for t in trades:
        trades_by_event[t["event_ticker"]].append(t)

    for et in TARGET_TICKERS:
        print(f"  {et}: {len(trades_by_event[et])} trades")

    # Step 2: Get unique market tickers and fetch results
    unique_tickers = set(t["ticker"] for t in trades)
    print(f"\nFetching outcomes for {len(unique_tickers)} market tickers...")
    market_results = fetch_all_market_results(sorted(unique_tickers))
    resolved = sum(1 for v in market_results.values() if v is not None)
    print(f"  Resolved: {resolved}/{len(unique_tickers)}")

    # Step 3: Create calibration plot and compute metrics
    all_metrics = create_calibration_plot(trades_by_event, market_results)

    # Step 4: Print metrics
    print_metrics(all_metrics)


if __name__ == "__main__":
    main()
