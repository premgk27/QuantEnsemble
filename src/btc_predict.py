"""Generate daily BTC trading signal using the trained RF model.

Usage:
    uv run python src/btc_predict.py              # print signal
    uv run python src/btc_predict.py --log        # print + append to outputs/btc_signals.jsonl
    uv run python src/btc_predict.py --model models/btc_daily_rf.pkl

Fetches last ~90 days of BTC daily OHLCV from Kraken public API (no API key needed),
computes the 4 features the model was trained on, and outputs a directional signal.

Signal values:
    +1  LONG  — model predicts next-day return > +1.0%
    -1  SHORT — model predicts next-day return < -1.0%
     0  FLAT  — model is in neutral zone (hold / no new position)

Run this script once daily after UTC midnight (after yesterday's daily close finalises).
The signal applies to the next full day, entered at the next available price.

Paper trading workflow:
    1. Run after each UTC midnight close
    2. Compare signal to current position
    3. If signal changed: execute trade at market open
    4. Log actual fill price in btc_signals.jsonl manually or via broker API
    5. After 30+ days, compare paper vs backtest metrics
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests

from features import build_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "btc_daily_rf.pkl"
SIGNALS_LOG = Path(__file__).resolve().parent.parent / "outputs" / "btc_signals.jsonl"

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
WARMUP_BARS = 100  # fetched from API; need >=25 valid rows after warmup dropna

LABEL = {1: "LONG", -1: "SHORT", 0: "FLAT"}


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------
def fetch_kraken_daily(n_bars: int = WARMUP_BARS) -> pd.DataFrame:
    """Fetch recent BTC/USD daily OHLCV from Kraken public API.

    Returns DataFrame with DatetimeIndex (UTC) and columns: open, high, low, close, volume.
    Kraken may return an in-progress candle for the current UTC day; the caller
    should be aware of this and use the appropriate row.
    """
    since_ts = int((datetime.now(timezone.utc) - timedelta(days=n_bars + 5)).timestamp())
    params = {
        "pair": "XBTUSD",
        "interval": 1440,   # 1440 minutes = 1 day
        "since": since_ts,
    }

    print(f"  Fetching Kraken OHLC (daily, last ~{n_bars} bars)...")
    resp = requests.get(KRAKEN_OHLC_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("error"):
        raise RuntimeError(f"Kraken API error: {data['error']}")

    # Result key is the Kraken pair name (XXBTZUSD or similar)
    result_key = next(k for k in data["result"] if k != "last")
    rows = data["result"][result_key]

    df = pd.DataFrame(
        rows,
        columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
    )
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    df = df.set_index("time").sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def get_signal(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Load model, fetch data, compute features for latest bar, return prediction."""

    # 1. Load model artifact
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run 'uv run python src/btc_train.py' first."
        )
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    feature_names = artifact["feature_names"]
    print(f"  Model trained:   {artifact['trained_at'][:10]}")
    print(f"  Last train date: {artifact['last_train_date']}")
    print(f"  Features:        {feature_names}")

    # 2. Fetch recent OHLCV
    print()
    df = fetch_kraken_daily(n_bars=WARMUP_BARS)

    # The last Kraken bar may be an in-progress candle if fetched mid-day.
    # For daily crypto (UTC midnight close), running after close means the last
    # bar is the most recently completed day.  We use it as-is.
    last_close_date = df.index[-1].date()
    last_close_price = float(df["close"].iloc[-1])
    print(f"  Rows returned:   {len(df)}")
    print(f"  Date range:      {df.index[0].date()} to {last_close_date}")
    print(f"  Latest close:    ${last_close_price:,.2f}")

    if len(df) < 25:
        raise RuntimeError(
            f"Only {len(df)} bars returned from Kraken — not enough for feature warmup."
        )

    # 3. Build features (prediction mode: don't drop last row for missing target)
    features = build_features(df, timeframe="daily", drop_target_nans=False)

    if len(features) == 0:
        raise RuntimeError("No valid feature rows after build_features — insufficient data.")

    # The last row corresponds to the latest completed bar
    last_row = features[feature_names].iloc[-1]
    signal_date = features.index[-1]

    # Sanity check: any NaN in the 4 features?
    if last_row.isnull().any():
        bad = last_row[last_row.isnull()].index.tolist()
        raise RuntimeError(f"NaN features for {signal_date.date()}: {bad}")

    # 4. Predict
    X_pred = last_row.values.reshape(1, -1)
    pred = int(model.predict(X_pred)[0])

    # Probabilities
    proba = model.predict_proba(X_pred)[0]
    classes = [int(c) for c in model.classes_]
    prob_dict = dict(zip(classes, [float(p) for p in proba]))

    # 5. Print result
    print(f"\n{'='*52}")
    print(f"  BTC DAILY SIGNAL — {signal_date.date()}")
    print(f"{'='*52}")
    print(f"  Close:    ${last_close_price:,.2f}")
    print(f"  Signal:   {LABEL[pred]} ({pred:+d})")
    print(f"  Probs:    Down={prob_dict.get(-1,0):.1%}  "
          f"Neutral={prob_dict.get(0,0):.1%}  Up={prob_dict.get(1,0):.1%}")
    print(f"\n  Features used:")
    for name, val in last_row.items():
        print(f"    {name:20s} = {val:.6f}")
    print(f"{'='*52}")

    return {
        "date": str(signal_date.date()),
        "close": last_close_price,
        "signal": pred,
        "signal_label": LABEL[pred],
        "prob_down": prob_dict.get(-1, 0.0),
        "prob_neutral": prob_dict.get(0, 0.0),
        "prob_up": prob_dict.get(1, 0.0),
        "features": {k: float(v) for k, v in last_row.items()},
        "model_trained_at": artifact["trained_at"],
        "predicted_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    model_path = DEFAULT_MODEL_PATH
    do_log = "--log" in sys.argv

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--model" and i + 1 < len(args):
            model_path = Path(args[i + 1])
        elif not arg.startswith("--") and i == 0:
            model_path = Path(arg)

    print(f"\nBTC Daily Signal Generator")
    print(f"{'='*52}")

    result = get_signal(model_path)

    if do_log:
        SIGNALS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SIGNALS_LOG, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"\nSignal logged → {SIGNALS_LOG}")

    return result


if __name__ == "__main__":
    main()
