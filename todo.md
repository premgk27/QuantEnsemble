# TODO

## Trading System — Go Live

### Phase 1: Signal emails (no money, no risk)
- [ ] Run `./deploy/setup.sh` to create AWS infrastructure
- [ ] Train model and upload: `uv run python src/btc_train.py --upload-s3 my-btc-signals`
- [ ] Subscribe email to SNS: `aws sns subscribe ...`
- [ ] Invoke Lambda manually to test: `aws lambda invoke ...`
- [ ] Wait for first automated email at 00:05 UTC

### Phase 2: Demo futures trading (paper money)
- [ ] Create demo API keys at demo-futures.kraken.com
- [ ] Store in AWS Secrets Manager
- [ ] Enable trading on Lambda (`ENABLE_TRADING=true`, `KRAKEN_FUTURES_ENV=demo`)
- [ ] Run 30+ days — validate execution matches signal cadence

### Phase 3: Live (when confident)
- [ ] Create live keys at futures.kraken.com (no Withdrawal permission)
- [ ] Store separately in Secrets Manager
- [ ] Change one env var: `KRAKEN_FUTURES_ENV=live`

### Ongoing
- [ ] Retrain weekly after downloading new TradingView CSV
- [ ] Compare paper P&L to backtest expectation after 30+ days
  - Expected Sharpe: ~1.287, Win rate: ~52.5%, Max losing streak: 7-10 days

---

## Research Backlog

### High priority
- [ ] **Premium on-chain data (Phase 5)** — Glassnode (hashrate, realized value, UTXO age bands),
      Binance/Bybit funding rates + open interest. Academic studies show this can push
      directional accuracy from 52% to 82%+. Biggest potential alpha in the project.

### Lower priority
- [ ] **Equity fundamental model (Phase 6)** — 3-6 month horizon, cross-sectional stock ranking.
      Ridge/logistic, P/E + revenue growth + momentum. SimFin or Sharadar for data.
- [ ] **Portfolio optimizer** — combine BTC signal + equity signals into single capital allocation.
      Rough idea: Kelly fraction or equal-risk weighting across the strategies.
