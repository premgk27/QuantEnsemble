"""On-chain, derivatives, and sentiment data loader.

Sources (all free, no API key required):
  1. Binance Futures   — funding rate (8h, aggregated to daily)
  2. Alternative.me    — Fear & Greed Index (daily, from 2018-02)
  3. Blockchain.com    — hash rate, transaction count (daily, from ~2009)

Data is fetched once and cached to data/onchain/*.csv.
To refresh a source: delete the corresponding cache file and re-run.

Usage:
    from onchain_loader import load_all_onchain
    df = load_all_onchain()   # returns daily DataFrame with DatetimeIndex (UTC)
"""

import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

ONCHAIN_DIR = Path(__file__).resolve().parent.parent / "data" / "onchain"
ONCHAIN_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_BASE = "https://fapi.binance.com"
ALT_ME_BASE = "https://api.alternative.me"
BLOCKCHAIN_BASE = "https://api.blockchain.info"

REQUEST_TIMEOUT = 30
RETRY_DELAY = 5  # seconds between retries


def _get(url: str, params: dict = None, retries: int = 3) -> dict | list:
    """GET request with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Retry {attempt + 1}/{retries - 1} after error: {e}")
                time.sleep(RETRY_DELAY)
            else:
                raise


# ---------------------------------------------------------------------------
# Source 1: Binance Futures — Funding Rate
# ---------------------------------------------------------------------------
FUNDING_CACHE = ONCHAIN_DIR / "funding_rate.csv"


def fetch_funding_rate(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Fetch all historical 8h funding rates from Binance and aggregate to daily.

    Returns DataFrame with columns: funding_rate_daily, funding_rate_7d
    """
    print(f"  Fetching Binance funding rates for {symbol}...")
    url = f"{BINANCE_BASE}/fapi/v1/fundingRate"

    all_records = []
    # BTC perpetuals launched Sept 2019 — start from there
    start_ms = int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)

    while True:
        data = _get(url, params={"symbol": symbol, "startTime": start_ms, "limit": 1000})
        if not data:
            break
        all_records.extend(data)
        last_ts = data[-1]["fundingTime"]
        if len(data) < 1000:
            break
        start_ms = last_ts + 1
        time.sleep(0.2)  # polite rate limiting

    print(f"    Got {len(all_records)} funding rate records")

    df = pd.DataFrame(all_records)
    df["dt"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df.set_index("dt").sort_index()

    # Aggregate 8h rates to daily
    daily = df["funding_rate"].resample("D").mean().rename("funding_rate_daily")
    daily = daily.to_frame()
    daily["funding_rate_7d"] = daily["funding_rate_daily"].rolling(7).mean()
    # Annualized funding rate (3 payments/day * 365)
    daily["funding_rate_ann"] = daily["funding_rate_daily"] * 3 * 365

    return daily


def load_funding_rate(refresh: bool = False) -> pd.DataFrame:
    if FUNDING_CACHE.exists() and not refresh:
        df = pd.read_csv(FUNDING_CACHE, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    df = fetch_funding_rate()
    df.to_csv(FUNDING_CACHE)
    return df


# ---------------------------------------------------------------------------
# Source 2: Alternative.me — Fear & Greed Index
# ---------------------------------------------------------------------------
FEAR_GREED_CACHE = ONCHAIN_DIR / "fear_greed.csv"


def fetch_fear_greed() -> pd.DataFrame:
    """Fetch all historical Fear & Greed Index from alternative.me.

    Returns DataFrame with columns: fear_greed, fear_greed_7d
    """
    print("  Fetching Fear & Greed Index...")
    url = f"{ALT_ME_BASE}/fng/"
    data = _get(url, params={"limit": 0, "format": "json"})

    records = data["data"]
    print(f"    Got {len(records)} Fear & Greed records")

    df = pd.DataFrame(records)
    df["dt"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["fear_greed"] = df["value"].astype(float)
    df = df.set_index("dt")[["fear_greed"]].sort_index()

    df["fear_greed_7d"] = df["fear_greed"].rolling(7).mean()
    # Normalize to [-1, 1] for model consumption
    df["fear_greed_norm"] = (df["fear_greed"] - 50) / 50

    return df


def load_fear_greed(refresh: bool = False) -> pd.DataFrame:
    if FEAR_GREED_CACHE.exists() and not refresh:
        df = pd.read_csv(FEAR_GREED_CACHE, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    df = fetch_fear_greed()
    df.to_csv(FEAR_GREED_CACHE)
    return df


# ---------------------------------------------------------------------------
# Source 3: Blockchain.com — Hash Rate + Transaction Count
# ---------------------------------------------------------------------------
BLOCKCHAIN_CACHE = ONCHAIN_DIR / "blockchain_metrics.csv"


def fetch_blockchain_metrics() -> pd.DataFrame:
    """Fetch hash rate and transaction count from blockchain.com Charts API.

    Returns DataFrame with columns:
        hash_rate, hash_rate_ratio_30d, tx_count, tx_count_ratio_30d
    """
    print("  Fetching Blockchain.com metrics...")

    def get_chart(chart_name: str) -> pd.Series:
        url = f"{BLOCKCHAIN_BASE}/charts/{chart_name}"
        data = _get(url, params={
            "timespan": "all",
            "format": "json",
            "sampled": "false",
            "metadata": "false",
            "cors": "true",
        })
        values = data["values"]
        s = pd.Series(
            {pd.Timestamp(v["x"], unit="s", tz="UTC"): v["y"] for v in values},
            name=chart_name,
        )
        return s.sort_index()

    hash_rate = get_chart("hash-rate")
    print(f"    Hash rate: {len(hash_rate)} records")
    time.sleep(1)

    tx_count = get_chart("n-transactions")
    print(f"    Tx count: {len(tx_count)} records")

    df = pd.DataFrame({"hash_rate": hash_rate, "tx_count": tx_count})
    df = df.resample("D").last()  # ensure daily

    # Regime ratios: current vs 30d rolling mean
    df["hash_rate_ratio_30d"] = df["hash_rate"] / df["hash_rate"].rolling(30).mean()
    df["tx_count_ratio_30d"] = df["tx_count"] / df["tx_count"].rolling(30).mean()

    # Log of hash rate (more stationary)
    df["log_hash_rate"] = np.log(df["hash_rate"].replace(0, np.nan))

    return df


def load_blockchain_metrics(refresh: bool = False) -> pd.DataFrame:
    if BLOCKCHAIN_CACHE.exists() and not refresh:
        df = pd.read_csv(BLOCKCHAIN_CACHE, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    df = fetch_blockchain_metrics()
    df.to_csv(BLOCKCHAIN_CACHE)
    return df


# ---------------------------------------------------------------------------
# Main: Load and merge all sources
# ---------------------------------------------------------------------------
def load_all_onchain(refresh: bool = False) -> pd.DataFrame:
    """Load all on-chain/sentiment data, merge to daily DatetimeIndex (UTC).

    Args:
        refresh: If True, re-fetch all sources even if cache exists.

    Returns:
        DataFrame with daily rows and columns:
            funding_rate_daily, funding_rate_7d, funding_rate_ann,
            fear_greed, fear_greed_7d, fear_greed_norm,
            hash_rate, hash_rate_ratio_30d, tx_count, tx_count_ratio_30d, log_hash_rate
    """
    print("\nLoading on-chain / sentiment data...")
    parts = []

    try:
        parts.append(load_funding_rate(refresh=refresh))
    except Exception as e:
        print(f"  WARNING: Funding rate fetch failed: {e}")

    try:
        parts.append(load_fear_greed(refresh=refresh))
    except Exception as e:
        print(f"  WARNING: Fear & Greed fetch failed: {e}")

    try:
        parts.append(load_blockchain_metrics(refresh=refresh))
    except Exception as e:
        print(f"  WARNING: Blockchain metrics fetch failed: {e}")

    if not parts:
        raise RuntimeError("All on-chain data sources failed")

    merged = pd.concat(parts, axis=1).sort_index()

    # Normalize index to UTC date (strip time component for merge with daily OHLCV)
    merged.index = merged.index.normalize()

    print(f"  On-chain data: {len(merged)} rows, {len(merged.columns)} columns")
    print(f"  Date range: {merged.index[0]} to {merged.index[-1]}")
    print(f"  Columns: {list(merged.columns)}")
    return merged


if __name__ == "__main__":
    df = load_all_onchain(refresh=True)
    print("\nSample (last 5 rows):")
    print(df.tail().to_string())
    print(f"\nNaN counts:\n{df.isnull().sum().to_string()}")
