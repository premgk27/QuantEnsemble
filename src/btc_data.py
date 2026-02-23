"""Kraken BTC/USD daily OHLCV fetcher for live prediction.

NOTE: Kraken's public OHLC endpoint returns at most ~720 recent daily bars (~2 years).
This is sufficient for computing forward_4 features (warmup = ~25 bars) but NOT
enough for walk-forward training (needs 1250+ bars).

For training, use the local TradingView CSV (KRAKEN_BTCUSD, 1D.csv).
For live prediction, this module is the data source.

Usage:
    from btc_data import load_kraken_daily
    df = load_kraken_daily()                 # uses cache if fresh (< 20h old)
    df = load_kraken_daily(max_age_hours=0)  # always re-fetch

    # As a script:
    uv run python src/btc_data.py
"""

import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "kraken_btcusd_daily.csv"

# BTC on Kraken started ~2013-04-17
_KRAKEN_BTC_LAUNCH_TS = 1366156800


def fetch_all_kraken_daily() -> pd.DataFrame:
    """Fetch complete BTC/USD daily OHLCV history from Kraken via paginated requests.

    Kraken returns up to 720 candles per request. For ~4,500 daily bars we need
    ~7 pages. The `last` field in each response is the `since` value for the next page.

    Returns DataFrame with DatetimeIndex (UTC) and columns: open, high, low, close, volume.
    """
    print("  Fetching full Kraken BTCUSD daily history (paginated)...")
    since = _KRAKEN_BTC_LAUNCH_TS
    all_rows = []
    page = 0

    while True:
        page += 1
        resp = requests.get(
            KRAKEN_OHLC_URL,
            params={"pair": "XBTUSD", "interval": 1440, "since": since},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("error"):
            raise RuntimeError(f"Kraken API error: {data['error']}")

        result_key = next(k for k in data["result"] if k != "last")
        rows = data["result"][result_key]
        last_ts = int(data["result"]["last"])

        print(f"    Page {page}: {len(rows)} rows  (last ts → {last_ts})")
        all_rows.extend(rows)

        if len(rows) < 720:
            break  # received a partial page — no more history

        since = last_ts
        time.sleep(1.0)  # polite rate limiting

    df = pd.DataFrame(
        all_rows,
        columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
    )
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    df = df.set_index("time").sort_index()

    # Drop duplicates that can appear at page boundaries
    df = df[~df.index.duplicated(keep="last")]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["open", "high", "low", "close", "volume"]]
    print(f"  Total: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


def load_kraken_daily(
    cache_path: Path = CACHE_PATH,
    max_age_hours: float = 20.0,
) -> pd.DataFrame:
    """Load BTC daily OHLCV, using local cache when fresh enough.

    Args:
        cache_path: Where to read/write the cached CSV.
        max_age_hours: Re-fetch if cache is older than this many hours.
                       Default 20h means once-per-day fetches stay cached.
                       Set to 0 to always re-fetch.

    Returns:
        DataFrame with DatetimeIndex (UTC), columns: open, high, low, close, volume.
    """
    if cache_path.exists() and max_age_hours > 0:
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
        if age_hours < max_age_hours:
            print(f"  Loading cached Kraken data ({age_hours:.1f}h old) from {cache_path.name}...")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            print(f"  Cached: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
            return df
        else:
            print(f"  Cache is {age_hours:.1f}h old (> {max_age_hours}h), refreshing...")

    df = fetch_all_kraken_daily()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"  Saved to {cache_path}")
    return df


if __name__ == "__main__":
    import sys
    force = "--refresh" in sys.argv
    df = load_kraken_daily(max_age_hours=0 if force else 20)
    print(f"\nSample (last 5 rows):")
    print(df.tail().to_string())
    print(f"\nNaN counts:\n{df.isnull().sum().to_string()}")
