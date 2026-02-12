# QuantEnsemble — Phase 1 TODO

## Step 1: Data ✅
- [x] Get daily OHLCV data
- [x] Get 4h OHLCV data
- [ ] Decide ticker(s) — start with SPY or AAPL, expand later
- [ ] Verify data quality (missing bars, gaps, splits/dividends adjusted)

## Step 2: Feature Engineering ← YOU ARE HERE

### Top 30 Features (Combined & Prioritized)

**A. Returns & Momentum (8 features)**
| # | Feature | Notes |
|---|---------|-------|
| 1 | Log returns (1d, 2d, 5d, 10d, 20d) | Multiple lookbacks = multi-scale momentum. Count as 5 features. |
| 2 | Lagged returns (t-1, t-2, t-3) | Gives XGBoost temporal context. NOT lagged prices (non-stationary). |
| 3 | ROC (Rate of Change, 10d) | Pure momentum magnitude |
| 4 | Momentum divergence (return_5d - return_20d) | Acceleration/deceleration signal |

**B. Trend (5 features)**
| # | Feature | Notes |
|---|---------|-------|
| 5 | EMA ratios: close/EMA_12, close/EMA_26, close/EMA_50 | Price distance from trend at 3 scales. Not raw EMAs. |
| 6 | MACD histogram | EMA_12 - EMA_26 minus signal line. Momentum acceleration. |
| 7 | ADX-14 | Trend strength (not direction). >25 = trending, <20 = range-bound. |

**C. Mean Reversion / Oscillators (5 features)**
| # | Feature | Notes |
|---|---------|-------|
| 8 | RSI-14 | Top-3 feature in almost every XGBoost study |
| 9 | Stochastic %K (14) | Complements RSI — based on price range |
| 10 | Bollinger Band %B (20, 2std) | Price position within bands (0-1 scale) |
| 11 | Bollinger Band Width | Volatility squeeze detection |
| 12 | CCI-20 | Unbounded — extreme values (>200, <-200) are powerful |

**D. Volatility (5 features)**
| # | Feature | Notes |
|---|---------|-------|
| 13 | ATR-14 | Core volatility measure |
| 14 | Garman-Klass volatility | Uses full OHLC — 5-8x more efficient than close-to-close |
| 15 | Vol ratio (5d/20d realized vol) | Is short-term vol expanding or compressing? |
| 16 | Rolling std of returns (20d) | Simple historical volatility |
| 17 | Intraday range ratio: (high-low)/close | Bar-level volatility/conviction |

**E. Volume (4 features)**
| # | Feature | Notes |
|---|---------|-------|
| 18 | OBV (On-Balance Volume) | Tree models excel at detecting OBV-price divergences |
| 19 | Volume ratio (volume / 20d avg volume) | Abnormal activity detection |
| 20 | MFI-14 (Money Flow Index) | Volume-weighted RSI — quality of buying pressure |
| 21 | ADI (Accumulation/Distribution) | Complements OBV with price-position weighting |

**F. Price Structure (3 features)**
| # | Feature | Notes |
|---|---------|-------|
| 22 | Close position in range: (close-low)/(high-low) | Where price closed within the bar (0=low, 1=high) |
| 23 | Gap: (open - prev_close) / prev_close | Overnight information flow |
| 24 | Rolling mean ratio: close / SMA_20 | Price relative to short-term average |

**G. Regime Detection (3 features)**
| # | Feature | Notes |
|---|---------|-------|
| 25 | Rolling autocorrelation (20-period, lag-1) | Positive=trending, negative=mean-reverting |
| 26 | Rolling skewness (20d returns) | Asymmetry — negative skew precedes drawdowns |
| 27 | Rolling kurtosis (20d returns) | Fat tails — market producing extreme moves |

**H. External Context (3 features — optional, needs extra data)**
| # | Feature | Notes |
|---|---------|-------|
| 28 | VIX close | Fear gauge, negatively correlated with equities |
| 29 | Sector ETF return | Stock rarely moves against its sector |
| 30 | Day of week (sin/cos encoded) | Weak but real calendar effect. Use cyclical encoding. |

### Feature Engineering Tasks
- [ ] Load and clean data (handle NaNs, verify date sorting)
- [ ] Compute features 1-27 (core, OHLCV-derived only)
- [ ] Compute features 28-30 if external data is available
- [ ] Create target variable: next-period return (or sign)
- [ ] Trim warmup rows (leading NaNs from rolling calculations)
- [ ] Sanity checks: no future leakage, all features stationary, no raw prices
- [ ] Run correlation matrix — drop one of any pair >0.9 correlated
- [ ] Final feature count after pruning: aim for ~25-30

## Step 3: Base Models
- [ ] Train Ridge/Lasso regression (walk-forward CV)
- [ ] Train XGBoost (walk-forward CV, with regularization + early stopping)
- [ ] Evaluate each independently: Sharpe, accuracy, hit rate, profit factor
- [ ] Compare daily vs 4h performance

## Step 4: Simple Ensemble
- [ ] `ensemble_signal = 0.5 * ridge_pred + 0.5 * xgboost_pred`
- [ ] Compare ensemble vs individual models (OOS Sharpe)

## Step 5: Meta-Model (Stacking)
- [ ] Generate OOS predictions from base models via walk-forward CV
- [ ] Train Ridge meta-learner on stacked OOS predictions
- [ ] Evaluate improvement over simple averaging
- [ ] **No leakage** — meta-learner only sees genuinely OOS predictions

## Step 6: Backtesting & Evaluation
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
