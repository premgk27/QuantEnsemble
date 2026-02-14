# Quant ML Trading Model — Project QuantEnsemble Guide

## Core Principles

### Different Horizons = Different Problems
- The drivers of price movement are completely different at different timescales.
- Features, model complexity, and validation strategy must all change with the prediction horizon.
- A model that works for daily predictions will NOT work for monthly or longer — treat each horizon as a separate research project.

### Horizon-Specific Feature Guidance
- **1-5 days:** Recent price action, technical indicators, news sentiment, abnormal volume, short-term mean reversion signals, earnings surprise reactions.
- **1-3 months:** Medium-term momentum (3-12 month returns), earnings revision trends, fund flows, cross-asset signals, relative strength vs peers.
- **6-12 months:** Valuation ratios (P/E, EV/EBITDA), balance sheet quality, macro regime indicators (yield curve, credit cycle), quality factors.

### Key Modeling Principles by Horizon

**Cross-sectional models help a lot at longer horizons.**
Instead of predicting "will Apple go up next month," predict "which of these 500 stocks will outperform the others."
This multiplies your effective sample size because each month you have 500 relative predictions rather than one absolute prediction.

**Simpler models for longer horizons.**
Counterintuitive but critical. For daily predictions, XGBoost with many features can work because you have enough data to support model complexity.
For monthly or quarterly predictions, you're often better off with a simple linear model using 3-5 well-understood factors, because you simply don't have enough data to reliably train anything more complex without overfitting.

**Domain knowledge becomes more critical at longer horizons.**
With daily models you can somewhat brute-force feature discovery — throw in many features and let the model sort it out.
With monthly models you need to be much more deliberate about feature selection based on economic reasoning, because the data won't save you from bad choices.

---

## Data

### Timeframes
- **4h bars (primary):** ~11,000 rows, 2004–2026. Enough for 30 features (366x sample-to-feature ratio).
- **Daily bars:** Secondary timeframe for comparison and multi-timeframe features.
- **1h bars: NOT needed.** Adds noise, not signal. Revisit only if 4h+daily underperform.

### Data Sources
- **OHLCV (4h):** `/Users/prgk/Downloads/charts/BATS_SPY, 240, ma, full data.csv`
- **OHLCV (daily):** `/Users/prgk/Downloads/charts/BATS_SPY, 1D.csv`
- **VIX:** CBOE:VIX from TradingView (daily + 4h)
- **Sector ETFs:** XLK, XLE, XLF, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC

### Data Rules
- Combine all historical splits into single files — walk-forward CV handles train/test splitting.
- Drop precomputed indicators from TradingView exports — compute all features from raw OHLCV ourselves.
- Keep only: time, open, high, low, close, volume.

### Sample-to-Feature Ratio
- Rule of thumb: need 50-100x more samples than features (financial data is noisy).
- 4h (~11,000 rows) with 30 features = 366x ratio — very comfortable.
- Daily (~2,500 rows) with 30 features = 83x ratio — tight but OK.

---

## Feature Set (Top 30)

### A. Returns & Momentum (8 features)
| # | Feature | Notes |
|---|---------|-------|
| 1 | Log returns (1d, 2d, 5d, 10d, 20d) | Multi-scale momentum. 5 features. |
| 2 | Lagged returns (t-1, t-2, t-3) | Temporal context for XGBoost. NOT lagged prices (non-stationary). |
| 3 | ROC (Rate of Change, 10d) | Pure momentum magnitude |
| 4 | Momentum divergence (return_5d - return_20d) | Acceleration/deceleration signal |

### B. Trend (5 features)
| # | Feature | Notes |
|---|---------|-------|
| 5 | EMA ratios: close/EMA_12, close/EMA_26, close/EMA_50 | Price distance from trend at 3 scales |
| 6 | MACD histogram | Momentum acceleration |
| 7 | ADX-14 | Trend strength. >25=trending, <20=range-bound |

### C. Mean Reversion / Oscillators (5 features)
| # | Feature | Notes |
|---|---------|-------|
| 8 | RSI-14 | Top feature in most XGBoost studies |
| 9 | Stochastic %K (14) | Complements RSI — based on price range |
| 10 | Bollinger Band %B (20, 2std) | Price position within bands (0-1) |
| 11 | Bollinger Band Width | Volatility squeeze detection |
| 12 | CCI-20 | Unbounded — extremes (>200, <-200) are powerful |

### D. Volatility (5 features)
| # | Feature | Notes |
|---|---------|-------|
| 13 | ATR-14 | Core volatility measure |
| 14 | Garman-Klass volatility | Uses full OHLC, 5-8x more efficient than close-to-close |
| 15 | Vol ratio (5d/20d realized vol) | Short-term vol expanding or compressing? |
| 16 | Rolling std of returns (20d) | Historical volatility |
| 17 | Intraday range ratio: (high-low)/close | Bar-level volatility |

### E. Volume (4 features)
| # | Feature | Notes |
|---|---------|-------|
| 18 | OBV (On-Balance Volume) | Tree models excel at OBV-price divergences |
| 19 | Volume ratio (volume / 20d avg) | Abnormal activity detection |
| 20 | MFI-14 (Money Flow Index) | Volume-weighted RSI |
| 21 | ADI (Accumulation/Distribution) | Complements OBV |

### F. Price Structure (3 features)
| # | Feature | Notes |
|---|---------|-------|
| 22 | Close position in range: (close-low)/(high-low) | Where price closed in bar (0=low, 1=high) |
| 23 | Gap: (open - prev_close) / prev_close | Overnight information flow |
| 24 | Rolling mean ratio: close / SMA_20 | Price relative to short-term average |

### G. Regime Detection (3 features)
| # | Feature | Notes |
|---|---------|-------|
| 25 | Rolling autocorrelation (20-period, lag-1) | Positive=trending, negative=mean-reverting |
| 26 | Rolling skewness (20d returns) | Negative skew precedes drawdowns |
| 27 | Rolling kurtosis (20d returns) | Fat tails detection |

### H. External Context (3 features — optional, needs extra data)
| # | Feature | Notes |
|---|---------|-------|
| 28 | VIX close | Fear gauge, negatively correlated with equities |
| 29 | Sector ETF return | Stock rarely moves against its sector |
| 30 | Day of week (sin/cos encoded) | Weak signal — first to prune if needed. 2 features (sin+cos). |

### Feature Rules
- All features must be stationary (returns and ratios, never raw prices).
- Normalize using only past data (rolling windows, not full dataset).
- After computing, check correlation matrix — drop one of any pair >0.9.
- For 4h bars, translate day-based periods to bar counts (e.g., 14-day RSI → 91-bar RSI).
- Target variable: next-period return (or sign of next-period return).

### Feature Engineering Results

**Pipeline output (zero nulls after warmup trimming):**
- Daily: 8,295 rows × 37 features + 2 targets (1993-03 to 2026-02)
- 4h: 10,827 rows × 37 features + 2 targets (2004-07 to 2026-02)

**Target columns:**
- `target_ret` — log return of next bar: `ln(close[t+1] / close[t])`
- `target_sign` — sign of next-bar return: +1 (up), -1 (down), 0 (flat)

**Correlated pairs to prune (>0.9) — drop left, keep right:**
| Drop | Keep | Reason | Corr |
|------|------|--------|------|
| `roc_10d` | `log_ret_10d` | Log returns are mathematically cleaner (additive) | 0.999 |
| `cci_20` | `bb_pctb` | BB %B is bounded (0-1), more interpretable | 0.983 |
| `adi` | `obv` | OBV is simpler and more widely used | 0.977 |
| `close_sma20_ratio` | `ema_ratio_26` | EMA is more responsive, already have 3 EMA ratios | 0.963 |
| `ret_std_20d` | `gk_vol` | Garman-Klass uses full OHLC (5-8x more efficient) | 0.950 |
| `ema_ratio_50` | `ema_ratio_26` | 26 and 50 are too similar; keep 12 and 26 for two scales | 0.948 |

**After pruning: ~31 features** — within the 25-30 target range (comfortable sample-to-feature ratios: 4h=349x, daily=268x).

**Code:** `src/data_loader.py` (CSV loading), `src/features.py` (all feature computation).

### Base Model Results (Walk-Forward CV, Expanding Window)

**Config:** initial_train=1,250, step=20, target=`target_ret` (regression), 5bps transaction cost.

**Daily (8,295 samples, 31 features, ~7,045 OOS predictions):**

| Metric | Ridge | XGBoost | RF |
|--------|-------|---------|-----|
| Sharpe (gross) | 0.191 | -0.006 | **0.437** |
| Sharpe (net, 5bps) | -0.137 | -0.392 | **0.242** |
| Dir Accuracy | 51.1% | 49.1% | **53.4%** |
| Profit Factor (gross) | 1.04 | 1.00 | **1.09** |
| Max DD (gross) | 0.625 | 1.369 | **0.511** |
| Avg Turnover | 0.506 | 0.596 | **0.301** |

**4h (10,827 samples, 31 features, ~9,577 OOS predictions):**

| Metric | Ridge | XGBoost | RF |
|--------|-------|---------|-----|
| Sharpe (gross) | 0.277 | 0.051 | **0.435** |
| Sharpe (net, 5bps) | -0.303 | -0.444 | **0.143** |
| Dir Accuracy | 50.7% | 49.2% | **53.3%** |
| Profit Factor (gross) | 1.06 | 1.01 | **1.09** |
| Max DD (gross) | 0.351 | 0.671 | **0.340** |
| Avg Turnover | 0.605 | 0.517 | **0.304** |

**Key findings:**
- RF dominates on both timeframes — best Sharpe, lowest turnover, only net-positive model.
- XGBoost overfits despite regularization (low SNR in financial data).
- Ridge has a thin edge wiped out by high turnover (~50-60%).
- RF's low turnover (0.30) is the key advantage — it naturally produces smoother predictions.
- Daily RF slightly better net Sharpe (0.242 vs 0.143) due to lower per-bar cost impact.
- No leakage red flags (all Sharpe < 1).

**Code:** `src/train.py` (walk-forward CV, all 3 models). Usage: `uv run python src/train.py [ridge,xgboost,rf] [daily|4h]`

### Turnover Filter Results (Step 4.5)

**Best filter: EMA alpha=0.5 + dead-zone 50th percentile**

| Config | Sharpe(gross) | Sharpe(net) | Turnover | PF(net) |
|--------|---------------|-------------|----------|---------|
| RF raw (no filter) | 0.437 | 0.242 | 0.301 | 1.05 |
| **RF + ema=0.5+dz=50pct** | **0.340** | **0.313** | **0.041** | **1.06** |

- Turnover reduced 86% (0.301→0.041). Net Sharpe improved 0.242→0.313.
- Gross Sharpe drops (fewer trades = less exposure) but net improves because cost savings dominate.
- Filters are post-model — they don't change training, just how we trade on predictions.

### Ensemble Results (Step 5)

**Ensembles don't help — XGBoost (49.1% accuracy) is actively harmful.**

| Model | Net Sharpe (raw) | Net Sharpe (filtered) |
|-------|------------------|-----------------------|
| RF alone | 0.242 | **0.313** |
| Equal weight (3 models) | -0.373 | 0.233 |
| RF-heavy (0.6/0.2/0.2) | -0.377 | 0.139 |

- Including XGBoost in any ensemble degrades performance.
- Ensemble concept requires base models with positive edge; revisit when XGBoost improves.
- **Current best strategy: RF + ema=0.5+dz=50pct filter (net Sharpe 0.313).**

**Code:** `src/train.py` (includes `apply_filters()`, `sweep_filters()`, `build_ensembles()`). Saved to `outputs/ensemble_results_daily.pkl`.

---

## Ensemble & Meta-Model Strategy

### Why Ensemble?
- Different model types capture different patterns: linear models find stable proportional relationships; tree models (XGBoost) find nonlinear interactions and conditional effects.
- Combining models with uncorrelated errors produces a more robust signal than any single model.
- Correlation of errors matters more than individual accuracy — a weaker model that makes *different* mistakes is more valuable than a second strong model making the same mistakes.

### Base Models
1. **Linear Regression (Ridge/Lasso/Elastic Net)**
   - Captures stable, persistent linear relationships (momentum, value, mean reversion).
   - Very resistant to overfitting.
   - Interpretable and fast.
2. **XGBoost (Gradient Boosted Trees)**
   - Captures nonlinear interactions (e.g., "momentum works when volatility is low AND volume is rising").
   - Strong on tabular features.
   - Requires more care to avoid overfitting (regularization, early stopping, feature selection).

### Ensemble Methods (Progressive Complexity)

**Level 1 — Simple Averaging (START HERE)**
```
final_signal = 0.5 * linear_pred + 0.5 * xgboost_pred
```
Surprisingly effective. Errors are somewhat uncorrelated, so averaging smooths things out.

**Level 2 — Weighted Averaging**
Use out-of-sample performance to determine weights.
Optimize weights on a validation set. Adjust over time if one model performs better in recent regimes.

**Level 3 — Stacking (Meta-Learner)**
- Train base models (linear, XGBoost) on features.
- Generate **out-of-sample predictions** from each using walk-forward CV.
- Train a simple meta-model (Ridge regression or logistic regression) on those predictions.
- The meta-model learns *when* to trust which base model.
- **Critical rule:** The meta-model must ONLY see out-of-sample predictions from base models. Otherwise you will massively overfit.
- Keep the meta-learner simple (Ridge regression is usually enough).

**Level 4 — Regime-Based Blending (Advanced)**
Weight models differently depending on detected market regime (trending vs mean-reverting, high vs low volatility).

### Expanding the Ensemble (Future)
Once the 2-model ensemble works, consider adding:
- Random Forest (uncorrelated tree-based errors)
- Simple neural network (different inductive bias)
- Naive momentum/mean-reversion signal (simple but often uncorrelated)

---

## Research: MA Trend + Pullback Entry Strategy

**Concept:** Two-stage approach — (1) RF predicts MA direction, (2) combine with price position relative to MA for entry timing. Trade pullbacks in the direction of the trend.

**Signal logic:**
- MA rising + price below MA → strong long (+1.0, buying the pullback)
- MA falling + price above MA → strong short (-1.0, selling the bounce)
- MA rising + price above MA → weak long (+0.5, trend priced in)
- MA falling + price below MA → weak short (-0.5, trend priced in)

**SPY daily results (initial test):**
- MA direction accuracy: 88.5% — but misleading due to class imbalance (SPY SMA_20 rises 66% of the time)
- All variants net Sharpe negative (-0.11 to -0.17)
- Underperforms base RF (target_ret, net Sharpe 0.313 filtered)

**Why it didn't work on SPY:**
- SMA_20 direction is too slow/easy — 88.5% accuracy mostly reflects the base rate
- By the time SMA_20 changes direction, the move is largely over
- Entry timing (pullback logic) did help with drawdown reduction

**Future improvements to try:**
- Faster MA target (EMA_12 direction instead of SMA_20)
- Predict `target_ret_5` (5-bar forward return) — smoother than 1-bar, more actionable than MA direction
- Test on trending/more balanced assets (TLT, GLD) where MA direction isn't 66/34 skewed
- Use MA direction as a **regime filter on top of base RF** rather than standalone strategy

**New target columns added to features.py:**
- `target_ret_5`: Forward 5-bar cumulative log return
- `target_sign_5`: Sign of forward 5-bar return
- `target_ma_dir`: SMA_20 direction over next 5 bars (+1=rising, -1=falling)

**Code:** `src/strategy_ma_trend.py` (standalone, does not modify train.py)

---

## Project Roadmap

### Phase 1: Daily Model (1-5 Day Horizon) ← CURRENT
**Objective:** Build the full pipeline and establish a working baseline.

**Step 1 — Feature Engineering**
- Compute all 30 features from raw OHLCV (see Feature Set above)
- Both 4h and daily timeframes
- Target: sign or magnitude of next-period return

**Step 2 — Base Models**
- Train Ridge/Lasso regression on clean features
- Train XGBoost on same or expanded feature set
- Use walk-forward (expanding or rolling window) cross-validation
- Evaluate each model independently: Sharpe ratio, accuracy, hit rate, profit factor

**Step 3 — Simple Ensemble (0.5 * each)**
```
ensemble_signal = 0.5 * ridge_pred + 0.5 * xgboost_pred
```
- Compare ensemble Sharpe/accuracy vs individual models
- Confirm that combining actually improves out-of-sample performance

**Step 4 — Meta-Model (Stacking)**
- Generate OOS predictions from both base models via walk-forward CV
- Train Ridge meta-learner on stacked predictions
- Evaluate improvement over simple averaging
- Add additional base learners if stacking shows promise

**Step 5 — Backtesting & Evaluation**
- Simulate realistic trading with transaction costs
- Evaluate: Sharpe ratio, max drawdown, turnover, profit factor
- Stress test across different market regimes (bull, bear, sideways, high vol)

### Phase 2: Medium-Term Model (1-4 Weeks) — FUTURE
- Shift features to momentum, earnings revisions, fund flows
- Reduce model complexity (fewer features, stronger regularization)
- Handle overlapping returns carefully
- Consider cross-sectional ranking approach

### Phase 3: Multi-Horizon Blending — FUTURE
- Combine short-term and medium-term signals
- Portfolio optimizer on top to size positions
- Risk management layer (max drawdown, sector limits, leverage constraints)

---

## Critical Warning: Walk-Forward CV for Stacking

**This is the trickiest and most important part of the stacking pipeline.**

When generating out-of-sample predictions from base models for the meta-learner (Phase 1, Step 4), the walk-forward cross-validation must be implemented correctly. Getting this wrong — accidentally leaking future information — is the single most common mistake in quantitative ML. It will make your backtest results look incredible and then fail completely in live trading.

### What Can Go Wrong
- If base models see ANY future data during training, their "out-of-sample" predictions are contaminated.
- If the meta-learner trains on predictions that were generated using data from the same period it's trying to predict, you have leakage.
- Even subtle leakage (e.g., using a feature normalized with the full dataset mean instead of the rolling mean up to that point) can inflate results dramatically.

### How to Do It Correctly
1. Define a rolling or expanding training window for base models.
2. At each step, train base models ONLY on data up to time T.
3. Generate predictions for time T+1 (or T+1 to T+k) — these are genuinely out-of-sample.
4. Slide the window forward and repeat.
5. Collect all OOS predictions across the full timeline.
6. ONLY THEN train the meta-learner on these collected OOS predictions.
7. The meta-learner itself should also be validated walk-forward — never train and test it on the same period.

### Red Flags That You Have Leakage
- Backtest Sharpe ratio above 3-4 (almost certainly too good to be true for daily equity models).
- Massive performance drop when you add even one day of delay to predictions.
- Model performs nearly perfectly in-sample and dramatically worse in the most recent out-of-sample period.
- Accuracy is suspiciously consistent across all time periods and market regimes.

---

## Validation Rules (Non-Negotiable)
1. **Never leak future data.** Always use walk-forward or time-series split. No standard k-fold CV.
2. **Account for transaction costs** in all backtest evaluations.
3. **Track out-of-sample performance only.** In-sample accuracy is meaningless.
4. **Statistical significance matters.** A short backtest proving "it works" is likely luck. Need enough trades for confidence.
5. **Beware survivorship bias** in stock universe — include delisted stocks if possible.
6. **Regime-test everything.** A strategy that only works in bull markets is not a strategy.
