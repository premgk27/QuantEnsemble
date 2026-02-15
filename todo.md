# QuantEnsemble — TODO

## Phase 1: Single-Asset Pipeline (SPY) ✅

### Step 1-3: Data & Features ✅
- [x] Data loader, 37 features, correlation pruning → 31 features
- [x] Targets: 1-bar return, 5-bar return, vol-normalized, trend persistence, MA direction, 3-class bucketed, risk-adjusted

### Step 4: Base Models ✅
- [x] Ridge, XGBoost, RF — walk-forward CV on SPY daily
- [x] RF dominates (Sharpe 0.44 gross, 0.24 net, 53.4% accuracy)

### Step 4.5: Turnover Filters ✅
- [x] EMA smoothing + dead-zone filter sweep
- [x] Winner: ema=0.5+dz=50pct → turnover 0.30→0.04, net Sharpe 0.24→0.31

### Step 5: Ensemble ✅
- [x] Simple ensembles don't help — XGBoost drags performance down
- [x] RF alone + filter (0.313) beats all ensembles

### Step 5.5: Feature Reduction & Target Experiment ✅
- [x] Permutation importance (unreliable on low-SNR data, built-in importance more useful)
- [x] Curated 14-feature set (domain knowledge, minimal redundancy)
- [x] 40 configurations tested (2 models × 5 targets × 4 feature sets)
- **Winner: RF + target_bucket + curated_14 → 0.607 net Sharpe** (2x benchmark)
- XGBoost now viable with heavy reg + bucket target (0.520)
- `target_ret5_volnorm` best for trend following (0.426 filtered, turnover 0.007)

**Quick wins tested:**
- [x] Vol-adjusted bucket (0.432) — worse than fixed (0.607). Wide dynamic bands miss big directional moves.
- [x] Ensemble RF+XGBoost on bucketed target — marginal (0.612 vs 0.607). Not worth complexity.

**Code:** `src/experiment_feature_reduction.py`, `src/experiment_quick_wins.py`

---

## Phase 2: Multi-Asset Portfolio

### Phase A: Pooled Training Experiment ✅
- [x] Pooled training (SPY+QQQ+IWM+DIA stacked) vs individual-asset models
- **Result: Pooling hurts.** Best pooled: 0.455 vs single-asset models all better.

### Phase A.5: Independent Multi-Asset Portfolio ✅
- [x] Run RF + target_bucket + curated_14 independently on 6 assets (SPY, QQQ, IWM, DIA, TLT, GLD)
- [x] Portfolio-level metrics: combined Sharpe, diversification benefit, correlation
- [x] Equal-weight portfolio: 0.656 Sharpe, 1.13x diversification ratio
- [x] TLT/GLD uncorrelated with equities (0.01-0.09 correlation)
- **QQQ alone (0.918) beats equal-weight portfolio (0.656)**

### Phase A.6: Per-Asset Feature Selection ✅
- [x] Tested top-14 by RF Gini importance (50% data) vs curated_14 per asset
- **Result: Curated_14 wins on 4/6 assets.** Domain knowledge > data-driven selection.
- DIA improved (+0.175), GLD marginal (+0.037), rest worse (TLT collapsed: -0.440)
- Gini importance misleading — ranks by split ease, not predictive power

### Phase A.7: Class Imbalance Fix ← NEXT
- [ ] Test `class_weight="balanced"` in RF on all 6 assets
  - SPY has 23.3% strong-down vs 28.7% strong-up (model develops long bias)
  - During bear markets, "strong up" is still most common class (42.8%) due to violent rallies
  - Balanced weighting forces equal importance on all 3 classes
- [ ] Compare: balanced vs default RF on all assets, especially crash years (2000-01, 2008, 2022)

### Phase A.8: Hybrid Per-Asset Features (Future)
- [ ] Start with curated_14 base, swap 1-2 features per asset by domain reasoning
  - TLT: add `autocorr_20` (bonds trend), maybe drop `rsi_14` (low importance)
  - GLD: add `obv` (high importance for gold), drop `dow_sin`
  - DIA: add `stoch_k_14`, `bb_pctb` (both top-5 for DIA)
- [ ] Test each swap individually to isolate impact

### Phase A.9: Portfolio Weight Optimization (Future)
- [ ] Optimize weights (overweight QQQ/GLD, underweight IWM/DIA)
- [ ] Vol-weighted allocation
- [ ] Test: optimized portfolio vs QQQ alone

### Phase B: Mean Reversion Model (Thesis 1)
- [ ] Target: z-score compression over N days
- [ ] Only fire on extreme deviations (built-in selectivity like target_bucket)
- [ ] Features: RSI, BB %B, vol ratio, close position in range

### Phase C: Allocation Layer
- [ ] Rule-based router: which thesis applies to each asset on each day?
- [ ] Regime detection: ADX, autocorrelation, vol level

### Phase D: Expand Asset Universe
- [ ] More assets: EEM, XLE, HYG, sector ETFs
- [ ] Forex (separate test — different market structure)

---

## Phase 3: Live Validation
- [ ] Paper trading on best model(s)
- [ ] Forward-test on unseen data
- [ ] Monitor: Sharpe, turnover, drawdown in real-time

---

### Pitfalls Checklist (review before each step)
- [ ] No raw prices as features (non-stationary)
- [ ] No future-derived features (data leakage)
- [ ] Normalize using only past data (rolling mean/std, not full dataset)
- [ ] Walk-forward CV only — no standard k-fold
- [ ] Transaction costs in all evaluations
- [ ] For 5-day targets: hold positions for 5 bars, don't flip every bar
- [ ] When pooling assets: all features must be vol-normalized / scale-invariant
