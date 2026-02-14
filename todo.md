# QuantEnsemble — Phase 1 TODO

## Step 1: Data ✅
## Step 2: Research & Feature Selection ✅
## Step 3: Feature Engineering ✅
- [x] Project structure, data loader, all 37 features computed (groups A-H)
- [x] Correlation analysis done — 6 features to prune (see CLAUDE.md)
- [ ] Verify data quality (missing bars, gaps, splits/dividends adjusted)
- [ ] Prune correlated features in code
- [ ] Sanity checks: no future leakage, all features stationary

## Step 4: Base Models ✅
- [x] Train Ridge regression (walk-forward CV) — Sharpe 0.19 gross
- [x] Train XGBoost (walk-forward CV, with regularization + early stopping) — Sharpe ~0 gross
- [x] Evaluate each independently: Sharpe, accuracy, hit rate, profit factor
- [ ] Compare daily vs 4h performance

## Step 4.5: Reduce Turnover ✅
- [x] Dead-zone filter: hold previous position when |pred| < percentile threshold
- [x] EMA smoothing: smooth raw predictions before taking sign
- [x] Combined filter sweep (3 EMA × 3 DZ = 9 combos)
- **Winner: ema=0.5 + dz=50pct** — turnover 0.301→0.041 (86% reduction), net Sharpe 0.242→0.313
- All combos improve net Sharpe over raw; combined filter dominates

## Step 5: Simple Ensemble ✅
- [x] Equal weight: (ridge + xgboost + rf) / 3
- [x] RF-heavy: 0.2*ridge + 0.2*xgboost + 0.6*rf
- **Result: ensembles DON'T help.** XGBoost (49.1% accuracy) is actively harmful — drags down all ensembles
- RF alone + filter (net Sharpe 0.313) beats all ensembles, raw and filtered
- Ensemble idea requires base models with positive edge; revisit if XGBoost improves

## Step 6: Multi-Asset Testing ← NEXT (Track A)
Test all 3 models (Ridge, XGBoost, RF) on different ETFs to find which assets suit which models.
Same pipeline as SPY — walk-forward CV, turnover filters, compare net Sharpe.

**High priority:**
- [ ] QQQ (tech, more volatile than SPY)
- [ ] TLT (20+ yr Treasury bonds, strong trends, macro-driven)
- [ ] GLD (gold, momentum/mean-reversion hybrid)

**Worth trying:**
- [ ] EEM (emerging markets, less efficient)
- [ ] XLE (energy sector, cyclical)
- [ ] HYG (high yield bonds, credit cycle)

**Need:** Daily OHLCV CSVs from TradingView in same format as SPY. Make data loader generic for any ticker.

## Step 7: MA Trend + Pullback Strategy ← NEXT (Track B)
Iterate on the two-stage strategy (predict trend direction → trade pullbacks). Runs in parallel with Track A.

**First test on SPY:** 88.5% MA accuracy but net Sharpe negative (class imbalance problem — SMA_20 rises 66% of time on SPY).

**Improvements to try:**
- [ ] Faster MA target: EMA_12 direction instead of SMA_20 (more timely signals)
- [ ] Test `target_ret_5` (5-bar forward return) as middle-ground target
- [ ] Use MA direction as a **regime filter on top of base RF** (not standalone)
- [ ] Test on TLT/GLD where MA direction is more balanced
- [ ] Confidence-weighted positions: full size only when RF probability > 0.7

**Code:** `src/strategy_ma_trend.py` (standalone, doesn't touch train.py)

## Step 8: Paper Trading / Live Validation
- [ ] Pick best model+asset combos from Step 6 & 7
- [ ] Build paper trading signal generator
- [ ] Forward-test on unseen data
- [ ] Monitor: Sharpe, turnover, drawdown in real-time

## Step 9: Ensemble & Meta-Model (revisit)
- [ ] Revisit ensemble once we have 2+ models with positive edge (possibly different assets)
- [ ] Meta-model stacking
- [ ] Stress test across market regimes (bull, bear, sideways, high vol)

---

### Pitfalls Checklist (review before each step)
- [ ] No raw prices as features (non-stationary)
- [ ] No future-derived features (data leakage)
- [ ] Normalize using only past data (rolling mean/std, not full dataset)
- [ ] Walk-forward CV only — no standard k-fold
- [ ] Transaction costs in all evaluations
- [ ] For 4h bars: translate day-based periods to bar counts (14-day RSI → 91-bar RSI)
