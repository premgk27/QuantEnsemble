"""Train BTC daily model on all available data and save to disk.

Usage:
    uv run python src/btc_train.py
    uv run python src/btc_train.py --output models/btc_daily_rf.pkl
    uv run python src/btc_train.py --upload-s3 my-bucket-name

Trains RF + StandardScaler pipeline on ALL available BTC daily data using the
forward_4 feature set (ema_ratio_12, atr_14, log_ret_20d, mfi_14).

Model artifact saved includes model, feature names, training metadata.
Run this once per week or after downloading fresh TradingView data.

With --upload-s3: also uploads the model to S3 for use by the prediction Lambda.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_ohlcv, DATA_DIR
from features import build_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BTC_DAILY = DATA_DIR / "KRAKEN_BTCUSD, 1D.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "btc_daily_rf.pkl"

FORWARD_4 = ["ema_ratio_12", "atr_14", "log_ret_20d", "mfi_14"]


def make_rf_clf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=500, max_depth=5, max_features="sqrt",
            min_samples_leaf=50, random_state=42, n_jobs=-1,
        )),
    ])


def train(data_path: Path = BTC_DAILY, output_path: Path = DEFAULT_MODEL_PATH):
    print(f"\nTraining BTC daily model on all available data...")
    print(f"  Data:  {data_path}")
    print(f"  Model: {output_path}")

    raw = load_ohlcv(data_path)
    print(f"  Rows: {len(raw)},  {raw.index[0].date()} to {raw.index[-1].date()}")

    data = build_features(raw, timeframe="daily")

    # ±1.0% bucket target (confirmed winner from threshold sweep)
    data["target_bucket_10"] = pd.cut(
        data["target_ret"],
        bins=[-np.inf, -0.01, 0.01, np.inf],
        labels=[-1, 0, 1],
    ).astype(float)

    X = data[FORWARD_4].values
    y = data["target_bucket_10"].values

    dist = pd.Series(y).value_counts(normalize=True).sort_index()
    print(f"\n  Features ({len(FORWARD_4)}): {FORWARD_4}")
    print(f"  Training samples: {len(X)}")
    print(f"  Class distribution:")
    print(f"    Down (< -1%):  {dist.get(-1.0, 0):.1%}")
    print(f"    Neutral:       {dist.get(0.0, 0):.1%}")
    print(f"    Up (> +1%):    {dist.get(1.0, 0):.1%}")

    print(f"\n  Fitting RF...")
    model = make_rf_clf()
    model.fit(X, y)
    print(f"  Done.")

    artifact = {
        "model": model,
        "feature_names": FORWARD_4,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_on_rows": len(X),
        "last_train_date": str(data.index[-1].date()),
        "last_train_close": float(raw.loc[data.index[-1], "close"]),
        "bucket_threshold": 0.01,  # ±1% bucket
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nModel saved → {output_path}")
    print(f"  Trained at:      {artifact['trained_at']}")
    print(f"  Last train date: {artifact['last_train_date']}")
    print(f"  Last close:      ${artifact['last_train_close']:,.0f}")
    return artifact


def main():
    output_path = DEFAULT_MODEL_PATH
    data_path = BTC_DAILY
    s3_bucket = None
    s3_key = "models/btc_daily_rf.pkl"

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--output" and i + 1 < len(args):
            output_path = Path(args[i + 1]); i += 2
        elif args[i] == "--data" and i + 1 < len(args):
            data_path = Path(args[i + 1]); i += 2
        elif args[i] == "--upload-s3" and i + 1 < len(args):
            s3_bucket = args[i + 1]; i += 2
        elif not args[i].startswith("--") and i == 0:
            output_path = Path(args[i]); i += 1
        else:
            i += 1

    artifact = train(data_path=data_path, output_path=output_path)

    if s3_bucket:
        import boto3
        s3 = boto3.client("s3")
        print(f"\nUploading model to s3://{s3_bucket}/{s3_key} ...")
        s3.upload_file(str(output_path), s3_bucket, s3_key)
        print(f"  Uploaded. Lambda can now use this model.")


if __name__ == "__main__":
    main()
