"""Data loader for TradingView CSV exports.

Reads OHLCV data, drops precomputed indicators, parses unix timestamps.
"""

import pandas as pd
from pathlib import Path

# Default data paths
DATA_DIR = Path("/Users/prgk/Downloads/charts")
SPY_4H = DATA_DIR / "BATS_SPY, 240, ma, full data.csv"
SPY_DAILY = DATA_DIR / "BATS_SPY, 1D.csv"

OHLCV_COLS = ["time", "open", "high", "low", "close", "volume"]


def load_ohlcv(filepath: str | Path) -> pd.DataFrame:
    """Load a TradingView CSV export, keeping only OHLCV columns.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume.
    """
    df = pd.read_csv(filepath)

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()

    # Keep only OHLCV
    df = df[OHLCV_COLS].copy()

    # Parse unix timestamps
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any fully-null rows
    df = df.dropna(how="all")

    return df


def load_spy_4h() -> pd.DataFrame:
    return load_ohlcv(SPY_4H)


def load_spy_daily() -> pd.DataFrame:
    return load_ohlcv(SPY_DAILY)


if __name__ == "__main__":
    for name, loader in [("4h", load_spy_4h), ("daily", load_spy_daily)]:
        df = loader()
        print(f"\n=== SPY {name} ===")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Null counts:\n{df.isnull().sum()}")
        print(df.head())
