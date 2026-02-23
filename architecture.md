# Architecture

## System Overview

```
Your laptop (weekly, after downloading fresh TradingView data)
  uv run python src/btc_train.py --upload-s3 my-btc-signals
  └── trains RF on ~4,463 days of OHLCV → saves model.pkl → uploads to S3

EventBridge  cron(5 0 * * ? *)   [00:05 UTC daily]
  └── triggers Lambda (512MB, ~5s, ~$0.00/month)
       ├── 1. download model.pkl from S3
       ├── 2. fetch last ~105 daily bars from Kraken PUBLIC API  (no auth)
       ├── 3. compute features: ema_ratio_12, atr_14, log_ret_20d, mfi_14
       ├── 4. predict → LONG / SHORT / FLAT
       │
       ├── [ENABLE_TRADING=false]  paper mode
       │    └── skip trade execution
       │
       ├── [ENABLE_TRADING=true]   live mode
       │    ├── load API key from AWS Secrets Manager
       │    ├── GET /0/private/Balance  → current BTC + USD holdings
       │    ├── compare current position vs new signal
       │    ├── if changed: POST /0/private/AddOrder  (market order)
       │    └── attach order result to response
       │
       ├── 5. append signal + trade to btc_signals.jsonl in S3
       └── 6. publish to SNS → email notification
```

## Signal → Trade Logic

Execution uses **Kraken Futures perpetual** (`pi_xbtusd`), not spot.
This supports all three signals including true short positions.

```
Signal  Current pos   Action
──────  ───────────   ──────────────────────────────────────────────────────
 +1     flat        → BUY  $N contracts  (open long)
 +1     short       → BUY  existing_size + $N contracts  (close short, open long)
 +1     long        → HOLD (already positioned correctly)
  0     long        → SELL existing_size contracts  (close long, go flat)
  0     short       → BUY  existing_size contracts  (close short, go flat)
  0     flat        → HOLD
 -1     flat        → SELL $N contracts  (open short)
 -1     long        → SELL existing_size + $N contracts  (close long, open short)
 -1     short       → HOLD
```

`$N` = `POSITION_SIZE_USD` env var (default $1,000). 1 contract = $1 notional on `pi_xbtusd`.

## Source Files

| File | Purpose |
|------|---------|
| `src/data_loader.py` | Load TradingView CSV exports (OHLCV only) |
| `src/features.py` | All feature computation — `build_features(df, timeframe, drop_target_nans)` |
| `src/btc_data.py` | Kraken OHLCV fetcher for live prediction (last ~720 bars, cached) |
| `src/onchain_loader.py` | Free on-chain data (Binance funding, Fear&Greed, blockchain.com) |
| `src/btc_feature_selection.py` | Greedy walk-forward forward selection of features |
| `src/btc_model.py` | Full backtest — walk-forward CV, all feature sets, regime analysis |
| `src/btc_train.py` | Train RF on all history, save model; `--upload-s3 <bucket>` |
| `src/btc_predict.py` | Generate today's signal from saved model + Kraken API |
| `src/btc_trade.py` | Kraken order execution — BUY / SELL / HOLD based on signal |
| `src/lambda_handler.py` | AWS Lambda entry point — predict + optional trade + SNS notify |
| `src/multi_asset_trend.py` | Equity multi-asset model (SPY, QQQ, IWM, DIA, TLT, GLD) |

## Model

**Algorithm:** Random Forest Classifier + StandardScaler pipeline
**Features (4):** `ema_ratio_12`, `atr_14`, `log_ret_20d`, `mfi_14`
**Target:** 3-class bucket (Down if < −1%, Up if > +1%, Neutral otherwise)
**Validation:** Walk-forward expanding window (INITIAL_TRAIN=1250, STEP_SIZE=20)
**Transaction cost:** 5 bps one-way

### Why these 4 features

| Feature | What it captures |
|---------|-----------------|
| `ema_ratio_12` | Short-term trend direction (price above/below 12-bar EMA) |
| `atr_14` | Volatility regime (high vol = different market dynamics) |
| `log_ret_20d` | Monthly momentum (crypto trends persist strongly at daily timescale) |
| `mfi_14` | Money Flow Index — volume-weighted price action (smart money signal) |

Selected by greedy forward selection maximising OOS Sharpe. 4 features beats 31 features
(1.287 vs 0.658 Sharpe) — less features = less noise in low-SNR financial data.

### Backtest Performance (OOS 2017–2026)

| Metric | Model | Buy & Hold |
|--------|-------|-----------|
| Sharpe | 1.287 | 0.522 |
| Ann. Return | 111% | 35.5% |
| Max Drawdown | 56.4% | 84.6% |
| Win Rate | 52.5% | — |
| Calmar Ratio | ~2.0 | ~0.4 |

Profitable in all regimes: Bull (2.130), Bear (0.725), Sideways (0.561).

## Data Flow

```
TradingView CSV (local)
        │
        ▼
  btc_train.py  ──→  models/btc_daily_rf.pkl  ──→  S3 (models/btc_daily_rf.pkl)
  (weekly, manual)


EventBridge (daily 00:05 UTC)
        │
        ▼
  lambda_handler.py
        │
        ├─ S3 download ──────────────────────── btc_daily_rf.pkl
        │
        ├─ Kraken PUBLIC API ─────────────────── last 105 daily OHLCV bars
        │   (no auth required)
        │
        ├─ btc_predict.py
        │   features.py  →  ema_ratio_12, atr_14, log_ret_20d, mfi_14
        │   RF.predict() →  signal: +1 / 0 / -1
        │
        ├─ btc_trade.py  [ENABLE_TRADING=true only]
        │   Secrets Manager  →  Kraken API key
        │   Kraken PRIVATE API:
        │     GET  /0/private/Balance     →  current holdings
        │     POST /0/private/AddOrder    →  market order (buy or sell)
        │
        ├─ S3 append  ────────────────────────── btc_signals.jsonl
        │
        └─ SNS publish  ──────────────────────── email notification
```

## AWS Infrastructure

| Resource | Purpose | Cost |
|----------|---------|------|
| S3 bucket | Model storage + signal log | ~$0.001/month |
| ECR repository | Container image (~250MB) | ~$0.03/month |
| Lambda function | Daily prediction + trade (512MB, ~5s) | $0 (free tier) |
| EventBridge rule | Daily schedule (00:05 UTC) | $0 (free tier) |
| SNS topic | Email notification | $0 (first 1,000/month free) |
| Secrets Manager | Kraken API credentials (live trading only) | ~$0.40/month per secret |
| **Total (paper)** | | **~$0.03/month** |
| **Total (live)** | | **~$0.43/month** |

## Deployment Progression

```
Phase 1  ENABLE_TRADING=false              Signal email only, no execution.
                                           Run until you trust the signal cadence.

Phase 2  ENABLE_TRADING=true              Kraken Futures DEMO — paper money.
         KRAKEN_FUTURES_ENV=demo           Full LONG/SHORT/FLAT, zero real risk.
         KRAKEN_SECRET_NAME=...            Get demo keys: demo-futures.kraken.com
                                           Run for 30+ days to validate execution.

Phase 3  KRAKEN_FUTURES_ENV=live          Kraken Futures LIVE — real money.
                                           Only env var changes. Same keys format,
                                           same code, different endpoint.
```

## Position Logic

On neutral prediction (0): **go flat** (close any open futures position).
In futures, holding through a neutral prediction carries unnecessary overnight risk.
