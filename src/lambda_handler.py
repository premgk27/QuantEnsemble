"""AWS Lambda handler — BTC daily signal prediction (+ optional trade execution).

Triggered daily by EventBridge (recommended: 00:05 UTC, after BTC midnight close).

Flow:
    1. Download trained model from S3
    2. Fetch last ~100 BTC daily bars from Kraken API
    3. Compute forward_4 features
    4. Predict direction (LONG / SHORT / FLAT)
    5. [Optional] Execute trade on Kraken (if ENABLE_TRADING=true)
    6. Append signal + trade result to S3 log
    7. Publish notification to SNS (→ email)

Required environment variables:
    S3_BUCKET           — S3 bucket for model and signal log  e.g. "my-btc-signals"

Optional environment variables:
    S3_MODEL_KEY          — S3 key for model file         default: "models/btc_daily_rf.pkl"
    S3_SIGNALS_KEY        — S3 key for signal log         default: "outputs/btc_signals.jsonl"
    SNS_TOPIC_ARN         — SNS topic ARN for notifications (email)
    ENABLE_TRADING        — "true" to execute trades on Kraken Futures (default: "false")
    KRAKEN_SECRET_NAME    — Secrets Manager secret name with Kraken credentials
                            Required when ENABLE_TRADING=true
                            Secret format: {"api_key": "...", "api_secret": "..."}
    KRAKEN_FUTURES_ENV    — "demo" (default) or "live"
                            demo → demo-futures.kraken.com  (paper money, no risk)
                            live → futures.kraken.com       (real money)
    POSITION_SIZE_USD     — USD notional per trade         default: "1000"

EventBridge rule (run daily at 00:05 UTC):
    cron(5 0 * * ? *)
"""

import json
import os
import pickle
import tempfile
from pathlib import Path

import boto3

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = os.environ.get("S3_MODEL_KEY", "models/btc_daily_rf.pkl")
S3_SIGNALS_KEY = os.environ.get("S3_SIGNALS_KEY", "outputs/btc_signals.jsonl")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
ENABLE_TRADING = os.environ.get("ENABLE_TRADING", "false").lower() == "true"
KRAKEN_SECRET_NAME = os.environ.get("KRAKEN_SECRET_NAME", "")
POSITION_SIZE_USD = float(os.environ.get("POSITION_SIZE_USD", "1000"))

s3_client = boto3.client("s3")
sns_client = boto3.client("sns") if SNS_TOPIC_ARN else None

LABEL = {1: "LONG", -1: "SHORT", 0: "FLAT"}


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------
def handler(event, context):
    """Main Lambda handler — download model, predict, notify."""
    print(f"Event: {json.dumps(event)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "btc_daily_rf.pkl"

        # 1. Download model from S3
        print(f"Downloading model: s3://{S3_BUCKET}/{S3_MODEL_KEY}")
        s3_client.download_file(S3_BUCKET, S3_MODEL_KEY, str(model_path))

        # 2. Predict signal
        from btc_predict import get_signal
        result = get_signal(model_path=model_path)

    # 3. Execute trade (optional — only when ENABLE_TRADING=true)
    trade_result = None
    if ENABLE_TRADING:
        env_label = os.environ.get("KRAKEN_FUTURES_ENV", "demo").upper()
        print(f"\nTrading enabled [{env_label}] — executing on Kraken Futures...")
        from btc_trade import execute_trade, load_credentials
        api_key, api_secret = load_credentials(secret_name=KRAKEN_SECRET_NAME or None)
        trade_result = execute_trade(
            signal=result["signal"],
            api_key=api_key,
            api_secret=api_secret,
            position_size_usd=POSITION_SIZE_USD,
            dry_run=False,
        )
        result["trade"] = trade_result
        print(f"Trade result: {trade_result.get('action')}")
    else:
        print(f"\nTrading disabled (ENABLE_TRADING=false) — signal only.")

    # 4. Append signal (+ trade if executed) to S3 log
    _append_to_s3_log(result)

    # 5. Publish SNS notification
    if SNS_TOPIC_ARN:
        _publish_sns(result, trade_result)

    print(f"Done. Signal: {result['signal_label']} ({result['signal']:+d})")
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _append_to_s3_log(result: dict):
    """Read existing log from S3, append new entry, write back."""
    line = json.dumps(result) + "\n"

    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_SIGNALS_KEY)
        existing = obj["Body"].read().decode("utf-8")
    except s3_client.exceptions.NoSuchKey:
        existing = ""
    except Exception as e:
        # Key might not exist yet on first run
        if "NoSuchKey" in str(e) or "404" in str(e):
            existing = ""
        else:
            raise

    content = existing + line
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=S3_SIGNALS_KEY,
        Body=content.encode("utf-8"),
        ContentType="application/x-ndjson",
    )
    print(f"Signal logged → s3://{S3_BUCKET}/{S3_SIGNALS_KEY}")


def _publish_sns(result: dict, trade_result: dict = None):
    """Publish a human-readable signal (+ optional trade) notification to SNS."""
    label = result["signal_label"]
    date = result["date"]
    close = result["close"]

    subject = f"BTC {label} — {date} @ ${close:,.0f}"

    lines = [
        f"BTC Daily Signal",
        f"{'='*40}",
        f"Date:    {date}",
        f"Close:   ${close:,.2f}",
        f"Signal:  {label} ({result['signal']:+d})",
        f"",
        f"Probabilities:",
        f"  Down (< -1%):  {result['prob_down']:.1%}",
        f"  Neutral:       {result['prob_neutral']:.1%}",
        f"  Up   (> +1%):  {result['prob_up']:.1%}",
        f"",
        f"Features:",
    ]
    for name, val in result["features"].items():
        lines.append(f"  {name:20s} = {val:.6f}")

    if trade_result:
        action = trade_result.get("action", "unknown")
        env = trade_result.get("environment", "")
        lines += ["", f"Trade [{env}]:", f"  {action.upper()}"]
        if trade_result.get("action") not in ("hold",):
            for order in trade_result.get("orders", []):
                lines.append(
                    f"  {order['side'].upper():4s}  {order['size']:>6} contracts"
                    + (f"  →  {order['order_id']}" if order.get("order_id") else "")
                )
        if trade_result.get("btc_price"):
            lines.append(f"  BTC price: ${trade_result['btc_price']:,.2f}")
    else:
        lines += ["", "Trading: disabled (ENABLE_TRADING=false)"]

    lines += ["", f"Model trained: {result['model_trained_at'][:10]}"]
    message = "\n".join(lines)

    sns_client.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=subject,
        Message=message,
    )
    print(f"SNS published: {subject}")
