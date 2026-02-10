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

## Project Roadmap

### Phase 1: Daily Model (1-5 Day Horizon) ← CURRENT
**Objective:** Build the full pipeline and establish a working baseline.

**Step 1 — Feature Engineering**
- Price-derived: returns (1d, 5d, 10d, 20d), RSI, moving average crossovers, Bollinger Band position, ATR, abnormal volume
- Volatility: realized vol, vol ratio (short/long), intraday range
- Sentiment: news-based (if available)
- Target: sign or magnitude of next-day (or next 1-5 day) return

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