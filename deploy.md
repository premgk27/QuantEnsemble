# Deployment Guide

## Prerequisites

- AWS CLI configured (`aws configure`) with permissions for Lambda, ECR, S3, SNS, EventBridge, IAM
- Docker running (for building the container image)
- Python + uv installed locally

---

## One-Time Setup

Run the setup script once to create all AWS infrastructure:

```bash
AWS_REGION=us-east-1 S3_BUCKET=my-btc-signals ./deploy/setup.sh
```

This creates:
1. **S3 bucket** — stores model and signal log
2. **SNS topic** — sends email notifications
3. **IAM role** — Lambda execution permissions (S3 read/write + SNS publish)
4. **ECR repository + Docker image** — containerised Lambda (~250MB)
5. **Lambda function** — `btc-signal-predict` (512MB, 120s timeout)
6. **EventBridge rule** — runs daily at 00:05 UTC

After setup, subscribe your email to receive signals:

```bash
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:123456789:btc-signal-notify \
  --protocol email \
  --notification-endpoint you@example.com
```

Check your inbox and click the confirmation link.

---

## Initial Model Upload

Train on all available local data and upload to S3:

```bash
# Download fresh BTC daily data from TradingView first, then:
uv run python src/btc_train.py --upload-s3 my-btc-signals
```

This uses the local TradingView CSV (`/Users/prgk/Downloads/charts/KRAKEN_BTCUSD, 1D.csv`).

---

## Test Before Going Live

Invoke the Lambda manually and check the response:

```bash
aws lambda invoke \
  --function-name btc-signal-predict \
  --region us-east-1 \
  --payload '{}' \
  /tmp/response.json

cat /tmp/response.json
```

Expected response (paper mode, `ENABLE_TRADING=false`):
```json
{
  "date": "2026-02-23",
  "signal": 1,
  "signal_label": "LONG",
  "close": 66191.10,
  "prob_down": 0.345,
  "prob_neutral": 0.301,
  "prob_up": 0.354,
  "features": {
    "ema_ratio_12": 0.970913,
    "atr_14": 3089.988808,
    "log_ret_20d": -0.133730,
    "mfi_14": 28.227326
  }
}
```

Expected response (live mode, `ENABLE_TRADING=true`):
```json
{
  "date": "2026-02-23",
  "signal": 1,
  "signal_label": "LONG",
  "close": 66191.10,
  "prob_down": 0.345,
  "prob_neutral": 0.301,
  "prob_up": 0.354,
  "features": { "ema_ratio_12": 0.970913, "...": "..." },
  "trade": {
    "action": "buy",
    "order_id": "ABCDEF-12345-GHIJK",
    "volume_btc": 0.014906,
    "estimated_price_usd": 66191.10,
    "estimated_total_usd": 986.90,
    "btc_before": 0.0,
    "usd_before": 1000.0,
    "executed_at": "2026-02-23T00:05:12+00:00"
  }
}
```

---

## Ongoing Maintenance

### Weekly (or after downloading new TradingView data)

Retrain the model with latest price history:

```bash
uv run python src/btc_train.py --upload-s3 my-btc-signals
```

### Deploy code changes

After modifying any `src/` file, rebuild and push the container:

```bash
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1
ECR_URI="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/btc-signal"

# Login, build, push
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build --platform linux/amd64 -t btc-signal:latest -f deploy/Dockerfile .
docker tag btc-signal:latest ${ECR_URI}:latest
docker push ${ECR_URI}:latest

# Update Lambda to use new image
aws lambda update-function-code \
  --function-name btc-signal-predict \
  --image-uri ${ECR_URI}:latest \
  --region $AWS_REGION
```

### Check signal log

```bash
aws s3 cp s3://my-btc-signals/outputs/btc_signals.jsonl - | tail -5
```

---

## Local Development

Run locally without Lambda:

```bash
# Train model
uv run python src/btc_train.py

# Get today's signal
uv run python src/btc_predict.py

# Get today's signal + append to local log
uv run python src/btc_predict.py --log
```

---

## Paper Trading Workflow

1. Signal arrives by email at ~00:05 UTC each day
2. Log whether you acted on it and at what price
3. After 30+ days, compare paper P&L to the backtest expectation:
   - Backtest Sharpe: 1.287 (net of 5bps)
   - Expected win rate: ~52.5%
   - Expected max losing streak: ~7–10 days (based on yearly breakdown)

---

## Enabling Trading (Futures Demo → Live)

The system uses **Kraken Futures** (`pi_xbtusd` perpetual) for execution — not spot.
This gives true LONG / SHORT / FLAT support. Paper → live is a single env var change.

---

### Phase 2: Kraken Futures Demo (paper trading)

**Step 1 — Create demo API keys**

Go to `https://demo-futures.kraken.com` → Settings → API Keys → Create Key.
Permissions needed: **General** + **Trade**.

**Step 2 — Store credentials in AWS Secrets Manager**

```bash
aws secretsmanager create-secret \
  --name "btc-signal/kraken-demo" \
  --secret-string '{"api_key": "YOUR_DEMO_KEY", "api_secret": "YOUR_DEMO_SECRET"}'
```

**Step 3 — Give Lambda permission to read the secret**

Add to the Lambda IAM role inline policy:
```json
{
  "Effect": "Allow",
  "Action": "secretsmanager:GetSecretValue",
  "Resource": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT:secret:btc-signal/*"
}
```

**Step 4 — Enable demo trading on Lambda**

```bash
aws lambda update-function-configuration \
  --function-name btc-signal-predict \
  --environment "Variables={
    S3_BUCKET=my-btc-signals,
    SNS_TOPIC_ARN=arn:aws:sns:...,
    ENABLE_TRADING=true,
    KRAKEN_FUTURES_ENV=demo,
    KRAKEN_SECRET_NAME=btc-signal/kraken-demo,
    POSITION_SIZE_USD=1000
  }"
```

**Step 5 — Test locally first**

```bash
export KRAKEN_API_KEY=your_demo_key
export KRAKEN_API_SECRET=your_demo_secret
export KRAKEN_FUTURES_ENV=demo

# Dry-run: queries demo account, prints what would happen, no order placed
uv run python src/btc_trade.py --signal 1 --dry-run

# Execute on demo (paper money — safe)
uv run python src/btc_trade.py --signal 1
```

**What the email looks like in demo mode:**

```
Subject: BTC LONG — 2026-02-24 @ $67,012

Trade [DEMO]:
  flat → long
  BUY   1000 contracts  →  order_id=ABCDEF-12345
  BTC price: $67,012.00
```

---

### Phase 3: Go Live (one env var change)

After 30+ days of demo trading validate the execution matches expectations:

**Step 1 — Create live Kraken Futures API key**

Go to `https://futures.kraken.com` → Settings → API Keys → Create Key.
Same permissions: General + Trade. Do NOT enable Withdrawal.

**Step 2 — Store live credentials**

```bash
aws secretsmanager create-secret \
  --name "btc-signal/kraken-live" \
  --secret-string '{"api_key": "YOUR_LIVE_KEY", "api_secret": "YOUR_LIVE_SECRET"}'
```

**Step 3 — Switch Lambda to live**

```bash
aws lambda update-function-configuration \
  --function-name btc-signal-predict \
  --environment "Variables={
    S3_BUCKET=my-btc-signals,
    SNS_TOPIC_ARN=arn:aws:sns:...,
    ENABLE_TRADING=true,
    KRAKEN_FUTURES_ENV=live,
    KRAKEN_SECRET_NAME=btc-signal/kraken-live,
    POSITION_SIZE_USD=1000
  }"
```

That's it. Same code, same Lambda, same Docker image. Only the env vars change.

---

## Architecture Diagram

See `architecture.md`.
