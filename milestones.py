import requests
import csv
import sys

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2/milestones"

def fetch_all_milestones():
    """Fetch all milestones with category 'Mentions', paginating through all results."""
    all_milestones = []
    cursor = None

    while True:
        params = {
            "limit": 500,
            "category": "Mentions",
        }
        if cursor:
            params["cursor"] = cursor

        print(f"Fetching milestones (cursor={cursor})...")
        resp = requests.get(BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        milestones = data.get("milestones", [])
        all_milestones.extend(milestones)
        print(f"  Retrieved {len(milestones)} milestones (total so far: {len(all_milestones)})")

        cursor = data.get("cursor")
        if not cursor or not milestones:
            break

    return all_milestones


def write_csv(milestones, output_path="milestones_mentions.csv"):
    """Write milestones to CSV with start_date and primary_event_ticker columns."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "start_date", "primary_event_ticker"])

        for m in milestones:
            primary_tickers = m.get("primary_event_tickers") or []
            ticker_str = ";".join(primary_tickers) if primary_tickers else ""
            writer.writerow([
                m.get("id", ""),
                m.get("title", ""),
                m.get("start_date", ""),
                ticker_str,
            ])

    print(f"\nWrote {len(milestones)} rows to {output_path}")


def main():
    milestones = fetch_all_milestones()
    if not milestones:
        print("No milestones found for category 'Mentions'.")
        sys.exit(0)

    write_csv(milestones)


if __name__ == "__main__":
    main()
