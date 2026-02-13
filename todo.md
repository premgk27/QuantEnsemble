# QuantEnsemble — Phase 1 TODO

## Step 1: Data ✅
- [x] Get daily OHLCV data (`/Users/prgk/Downloads/charts/BATS_SPY, 1D.csv` — ~8,316 rows)
- [x] Get 4h OHLCV data (`/Users/prgk/Downloads/charts/BATS_SPY, 240, ma, full data.csv` — ~11,000 rows)
- [x] Combine historical splits into single files
- [x] Decide ticker — starting with SPY
- [x] Get VIX data (CBOE:VIX from TradingView)
- [x] Get sector ETF data (XLK, XLE, XLF, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC)
- [ ] Verify data quality (missing bars, gaps, splits/dividends adjusted)

## Step 2: Research & Feature Selection ✅
- [x] Research optimal feature count for XGBoost (~25-30 for our data size)
- [x] Research best-performing features for XGBoost stock prediction
- [x] Compile top 30 features (see CLAUDE.md for full list)
- [x] Decide timeframes: 4h (primary) + daily (secondary). No 1h needed.
- [x] Confirm sample-to-feature ratio is safe (4h: 366x, daily: 83x)
- [x] Update CLAUDE.md with all decisions

## Step 3: Feature Engineering ← YOU ARE HERE
- [ ] Set up project structure (src/, data/, notebooks/)
- [ ] Write data loader — read CSVs, keep only time/OHLCV, parse timestamps
- [ ] Compute features A: Returns & Momentum (8 features)
  - [ ] Log returns (1d, 2d, 5d, 10d, 20d)
  - [ ] Lagged returns (t-1, t-2, t-3)
  - [ ] ROC (10d)
  - [ ] Momentum divergence (return_5d - return_20d)
- [ ] Compute features B: Trend (5 features)
  - [ ] EMA ratios (close/EMA_12, close/EMA_26, close/EMA_50)
  - [ ] MACD histogram
  - [ ] ADX-14
- [ ] Compute features C: Mean Reversion (5 features)
  - [ ] RSI-14
  - [ ] Stochastic %K (14)
  - [ ] Bollinger Band %B (20, 2std)
  - [ ] Bollinger Band Width
  - [ ] CCI-20
- [ ] Compute features D: Volatility (5 features)
  - [ ] ATR-14
  - [ ] Garman-Klass volatility
  - [ ] Vol ratio (5d/20d realized vol)
  - [ ] Rolling std of returns (20d)
  - [ ] Intraday range ratio (high-low)/close
- [ ] Compute features E: Volume (4 features)
  - [ ] OBV
  - [ ] Volume ratio (volume / 20d avg)
  - [ ] MFI-14
  - [ ] ADI (Accumulation/Distribution)
- [ ] Compute features F: Price Structure (3 features)
  - [ ] Close position in range (close-low)/(high-low)
  - [ ] Gap (open - prev_close) / prev_close
  - [ ] Rolling mean ratio (close / SMA_20)
- [ ] Compute features G: Regime Detection (3 features)
  - [ ] Rolling autocorrelation (20-period, lag-1)
  - [ ] Rolling skewness (20d returns)
  - [ ] Rolling kurtosis (20d returns)
- [ ] Compute features H: External Context (3 features)
  - [ ] VIX close (align timestamps with SPY data)
  - [ ] Sector ETF return (XLK for SPY)
  - [ ] Day of week (sin/cos encoded)
- [ ] Create target variable: next-period return (or sign)
- [ ] Trim warmup rows (leading NaNs from rolling calculations)
- [ ] Sanity checks: no future leakage, all features stationary, no raw prices
- [ ] Correlation matrix — drop one of any pair >0.9
- [ ] Final feature count after pruning: aim for ~25-30

## Step 4: Base Models
- [ ] Train Ridge/Lasso regression (walk-forward CV)
- [ ] Train XGBoost (walk-forward CV, with regularization + early stopping)
- [ ] Evaluate each independently: Sharpe, accuracy, hit rate, profit factor
- [ ] Compare daily vs 4h performance

## Step 5: Simple Ensemble
- [ ] `ensemble_signal = 0.5 * ridge_pred + 0.5 * xgboost_pred`
- [ ] Compare ensemble vs individual models (OOS Sharpe)

## Step 6: Meta-Model (Stacking)
- [ ] Generate OOS predictions from base models via walk-forward CV
- [ ] Train Ridge meta-learner on stacked OOS predictions
- [ ] Evaluate improvement over simple averaging
- [ ] **No leakage** — meta-learner only sees genuinely OOS predictions

## Step 7: Backtesting & Evaluation
- [ ] Simulate trading with transaction costs
- [ ] Metrics: Sharpe ratio, max drawdown, turnover, profit factor
- [ ] Stress test across market regimes (bull, bear, sideways, high vol)

---

### Pitfalls Checklist (review before each step)
- [ ] No raw prices as features (non-stationary)
- [ ] No future-derived features (data leakage)
- [ ] Normalize using only past data (rolling mean/std, not full dataset)
- [ ] Walk-forward CV only — no standard k-fold
- [ ] Transaction costs in all evaluations
- [ ] For 4h bars: translate day-based periods to bar counts (14-day RSI → 91-bar RSI)
