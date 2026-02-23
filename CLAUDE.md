# Quant ML Trading Model — Project QuantEnsemble Guide

## Core Principles

### Different Horizons = Different Problems
- The drivers of price movement are completely different at different timescales.
- Features, model complexity, and validation strategy must all change with the prediction horizon.
- A model that works for daily predictions will NOT work for monthly or longer — treat each horizon as a separate research project.

### Different Timeframes = Different Feature Sets
- **This is empirically confirmed, not theoretical.** Daily BTC forward selection picked trend+momentum features (ema_ratio_12, log_ret_20d). 4h forward selection picked mean reversion features (lag_ret_2, vol_ratio_20d).
- Period calibration matters: RSI-14 on 4h bars = 56 hours lookback. Academic research (Hafid et al. 2024) found RSI-30 and Momentum-30 outperform RSI-14 on shorter timeframes.
- 4h bars are closer to efficient (weaker predictability). Academic finding: "Bitcoin prices were weakly efficient at the hourly frequency, but technical analysis became statistically significantly dominant at the daily horizon" (Gradojevic et al. 2023).

### Horizon-Specific Feature Guidance
- **4h bars:** Short-term mean reversion, intraday session effects, hour-of-day patterns, shorter-period oscillators (RSI-30, Stoch-200), volume spikes vs session average.
- **Daily bars:** Momentum and trend signals dominate. EMA ratios, 20-day returns, volume-weighted indicators (MFI). This is the sweet spot for BTC — confirmed by research and experiments.
- **1-3 months:** Medium-term momentum (3-12 month returns), earnings revision trends, fund flows, cross-asset signals, relative strength vs peers.
- **6-12 months:** Valuation ratios (P/E, EV/EBITDA), balance sheet quality, macro regime indicators (yield curve, credit cycle), quality factors.

### Key Modeling Principles by Horizon

**Cross-sectional models help a lot at longer horizons.**
Instead of predicting "will Apple go up next month," predict "which of these 500 stocks will outperform the others."
This multiplies your effective sample size because each month you have 500 relative predictions rather than one absolute prediction.

**Simpler models for longer horizons.**
Counterintuitive but critical. For daily predictions, XGBoost with many features can work because you have enough data to support model complexity.
For monthly or quarterly predictions, you're often better off with a simple linear model using 3-5 well-understood factors, because you simply don't have enough data to reliably train anything more complex without overfitting.

**Less features = more Sharpe (empirically confirmed on BTC).**
BTC daily: 4 features → 1.287 Sharpe, 31 features → 0.658 Sharpe. Adding features only adds noise.
The same pattern held for equities: curated_14 beat full_31 consistently.

**Domain knowledge becomes more critical at longer horizons.**
With daily models you can somewhat brute-force feature discovery — throw in many features and let the model sort it out.
With monthly models you need to be much more deliberate about feature selection based on economic reasoning, because the data won't save you from bad choices.

---

## Data

### Equity Data Sources
- **SPY 4h:** `/Users/prgk/Downloads/charts/BATS_SPY, 240, ma, full data.csv`
- **SPY daily:** `/Users/prgk/Downloads/charts/BATS_SPY, 1D.csv`
- **QQQ daily:** `/Users/prgk/Downloads/charts/BATS_QQQ, 1D.csv`
- **IWM daily:** `/Users/prgk/Downloads/charts/BATS_IWM, 1D.csv`
- **DIA daily:** `/Users/prgk/Downloads/charts/BATS_DIA, 1D.csv`
- **TLT daily:** `/Users/prgk/Downloads/charts/BATS_TLT, 1D.csv`
- **GLD daily:** `/Users/prgk/Downloads/charts/BATS_GLD, 1D.csv`

### BTC Data Sources (Kraken — has volume)
- **BTC daily:** `/Users/prgk/Downloads/charts/KRAKEN_BTCUSD, 1D.csv` — 4,505 rows (2013-10 to 2026-02)
- **BTC 4h:** `/Users/prgk/Downloads/charts/KRAKEN_BTCUSD, 240.csv` — 22,180 rows (2016-01 to 2026-02)
- **Important:** Use Kraken, NOT TradingView CRYPTO source — TradingView has no volume data.

### Data Rules
- Combine all historical splits into single files — walk-forward CV handles train/test splitting.
- Drop precomputed indicators from TradingView exports — compute all features from raw OHLCV ourselves.
- Keep only: time, open, high, low, close, volume.

### Sample-to-Feature Ratio
- Rule of thumb: need 50-100x more samples than features (financial data is noisy).
- BTC 4h (22k rows) with 31 features = 710x ratio — very comfortable.
- BTC daily (4.5k rows) with 4 features = 1,125x ratio — extremely comfortable.

---

## Feature Set — Equities (Daily, curated_14)

Selected by domain knowledge + forward selection validation. Beats all data-driven methods.

```
log_ret_5d, log_ret_20d, mom_divergence,
ema_ratio_12, ema_ratio_26, macd_hist,
rsi_14, bb_width, gk_vol, vol_ratio,
vol_ratio_20d, close_pos_range, gap, dow_sin
```

**Correlated pairs dropped (>0.9):**
| Drop | Keep | Corr |
|------|------|------|
| `roc_10d` | `log_ret_10d` | 0.999 |
| `cci_20` | `bb_pctb` | 0.983 |
| `adi` | `obv` | 0.977 |
| `close_sma20_ratio` | `ema_ratio_26` | 0.963 |
| `ret_std_20d` | `gk_vol` | 0.950 |
| `ema_ratio_50` | `ema_ratio_26` | 0.948 |

**Code:** `src/data_loader.py` (CSV loading), `src/features.py` (all feature computation).

---

## Feature Set — BTC Daily (forward_4, selected by greedy walk-forward Sharpe)

**Winning 4 features: `ema_ratio_12, atr_14, log_ret_20d, mfi_14`**

Why they work:
- `ema_ratio_12` — short-term trend direction (price above/below 12-bar EMA?)
- `atr_14` — volatility regime (high vol = different market dynamics)
- `log_ret_20d` — monthly momentum (crypto trends persist strongly)
- `mfi_14` — money flow index (volume-weighted price action = smart money signal)

Forward selection trace:
```
Step 1: +ema_ratio_12    Sharpe=1.072
Step 2: +atr_14          Sharpe=1.092  (+0.020)
Step 3: +log_ret_20d     Sharpe=1.172  (+0.080)
Step 4: +mfi_14          Sharpe=1.287  (+0.115)  ← PEAK
Step 5: +log_ret_1d      Sharpe=1.252  (-0.034)
Step 6: +vol_ratio       Sharpe=1.170  (-0.082)  → stopped
```

---

## Feature Set — BTC 4h (CLOSED — daily is the better BTC timeframe)

**Status: 4h investigated exhaustively. Decision: daily only.**

**Round 1 — standard 31 daily features:**
```
Step 1: +lag_ret_2         Sharpe=0.244  ← mean reversion signal
Step 2: +vol_ratio_20d     Sharpe=0.426  ← PEAK
Steps 3+: all negative → stopped
Best: 0.426 Sharpe (below 0.5 decision gate)
```

**Round 2 — expanded 41 features (added intraday-calibrated indicators):**
Added RSI-30, Stoch-200/30, Momentum-30, Williams %R, Disparity-7, hour_sin/cos, bar_body_ratio per academic research (Hafid et al. 2024).
```
Step 1: +vol_ratio         Sharpe=0.244
Step 2: +disparity_7       Sharpe=0.329
Step 3: +log_ret_2d        Sharpe=0.380
Step 4: +adx_14            Sharpe=0.431  ← PEAK
Steps 5+: negative → stopped
Best: 0.431 Sharpe — barely improved, still below 0.5 gate
```

**Conclusion: 4h closed.** Even with purpose-built intraday features, best 4h Sharpe = 0.431 vs buy-and-hold 0.244. Not worth trading costs and infrastructure complexity. The gap between daily (1.287) and 4h (0.431) is structural — academically confirmed that BTC is weakly efficient at hourly/sub-hourly frequencies.

**Only useful new 4h feature:** `disparity_7` (close / SMA_7 deviation). Others added minimal value.

**Why 4h failed:**
- 4h market is more mean-reverting, not trending — different physics than daily
- ±0.5% bucket on 4h: 50% neutral class, model barely trades
- Even with period-calibrated features (RSI-30, Stoch-200), only marginal improvement
- Structural efficiency limit — not fixable by feature engineering alone

---

## Academic Research Findings (2023-2025)

Key papers on crypto ML prediction:

**Feature importance:**
- Hafid et al. (arXiv:2410.06935, 2024): Top 8 features by χ² test = RSI-30, MACD, Momentum-30, Stochastic %K at 200 and 30 periods, RSI-14. Chi-squared test on XGBoost classifier for BTC direction.
- Ardia et al. (ScienceDirect, 2025): Technical indicators > blockchain metrics > Google Trends > price lags. Hashrate and block size are top on-chain features.
- Omole & Enke (Financial Innovation, 2024): 25 on-chain features (from 196 Glassnode metrics) using Boruta selection. Boruta + CNN-LSTM achieved 82.44% directional accuracy.

**Timeframe efficiency:**
- Gradojevic et al. (Int. Journal of Forecasting, 2023): BTC is weakly efficient at hourly frequency. Technical analysis becomes statistically significant at daily horizon. **Validates our finding that daily > 4h for BTC.**

**Model choice:**
- XGBoost outperforms deep learning in many studies for tabular features.
- Temporal Fusion Transformer (TFT) is best for sequential patterns.
- GRU, LightGBM competitive depending on asset.

**On-chain data experiment (free sources — TESTED, didn't improve forward_4):**
Built `src/onchain_loader.py` with Binance funding rates, Alternative.me Fear & Greed, blockchain.com hash rate.
forward_4 + hash_rate_ratio = 1.064 Sharpe (WORSE than forward_4 = 1.287).
Root cause: coverage gaps (47%/34% zeros pre-2019/2018) + hash_rate_ratio redundant with full forward_4.
hash_rate_ratio does help when paired with ema_ratio_12 alone (1.072 → 1.117).

**Future data sources (premium quality may help — free didn't):**
- Glassnode API: hashrate, block size, realized value, UTXO metrics (25 Boruta-selected features, full history from 2014+)
- Binance/Bybit API: open interest, liquidation levels (funding rate free data already tested — patchy)
- Google Trends: "bitcoin", "blockchain" search volume
- Premium data has full coverage vs patchy free data; studies using premium → 82%+ accuracy

---

## Equity Model Results Summary

### Baseline (SPY daily, regression target)
RF dominates: 0.437 gross Sharpe, 0.242 net Sharpe, 53.4% accuracy.

### Best Equity Config: RF + target_bucket + curated_14
**SPY: 0.607 net Sharpe** after turnover filter (EMA 0.5 + dead-zone 50th pct).

### Multi-Asset Results (6 assets, RF + target_bucket + curated_14)

| Asset | Net Sharpe | WinRate | Ann Ret | Turnover | MaxDD |
|-------|-----------|---------|---------|----------|-------|
| **QQQ** | **0.918** | 55.4% | 21.8% | 0.110 | 0.41 |
| SPY | 0.607 | 53.5% | 12.5% | 0.061 | 0.52 |
| GLD | 0.561 | 53.0% | 9.6% | 0.038 | 0.63 |
| TLT | 0.472 | 50.4% | 7.0% | 0.048 | 0.45 |
| DIA | 0.470 | 53.1% | 8.7% | 0.077 | 0.39 |
| IWM | 0.458 | 52.7% | 11.8% | 0.147 | 0.54 |

Equal-weight portfolio (6 assets): Sharpe 0.656, 1.13x diversification ratio.
**QQQ alone (0.918) beats the portfolio (0.656)** — equal weighting dilutes QQQ's edge.

### Key Equity Lessons
- **Pooled training hurts** — train per-asset independently.
- **Target design > model tuning.** target_bucket (3-class ±0.5%) doubles Sharpe vs regression target.
- **class_weight="balanced" fails globally** — hurts bull performance by 3-5x more than it helps crashes. Long bias is correct most of the time.
- **Probability thresholds don't help** — RF probabilities too clustered, model rarely confident >0.50. target_bucket neutral class already acts as confidence filter.
- **Curated_14 > data-driven selection** on 4/6 assets. Gini importance misleads.

**Code:** `src/multi_asset_trend.py`. Results: `outputs/multi_asset_trend_daily.pkl`.

---

## BTC Crypto Model Results

### Data
- **Source:** Kraken BTCUSD (has volume data, unlike TradingView CRYPTO source)
- **Daily:** 4,505 rows (2013-10 to 2026-02)
- **4h:** 22,180 rows (2016-01 to 2026-02)

### Bucket Threshold Sweep (BTC Daily)

| Threshold | Sharpe | Turnover | Down% | Neutral% | Up% |
|-----------|--------|----------|-------|----------|-----|
| ±0.5% | 0.500 | 0.514 | 37.1% | 20.7% | 42.2% |
| **±1.0%** | **0.658** | **0.162** | **29.2%** | **36.9%** | **33.9%** |
| ±1.5% | 0.422 | 0.018 | 23.4% | 49.4% | 27.2% |
| ±2.0% | 0.502 | 0.003 | 18.9% | 58.9% | 22.2% |

**±1.0% wins** — balanced class distribution with reasonable turnover.

### Bucket Threshold Sweep (BTC 4h)

| Threshold | Sharpe | Turnover | Neutral% |
|-----------|--------|----------|----------|
| ±0.1% | -0.028 | 0.743 | 12.9% |
| ±0.2% | 0.074 | 0.551 | 24.2% |
| ±0.3% | 0.196 | 0.213 | 34.6% |
| ±0.4% | 0.176 | 0.050 | 43.0% |
| **±0.5%** | **0.253** | **0.023** | **50.1%** |

Best 4h threshold is ±0.5% but all results weak. Root cause: wrong feature set for 4h, not threshold.

### BTC Daily Model Results (all feature sets, RF + target_bucket_10)

| Feature Set | N | Sharpe | WinRate | AnnRet | TotRet | Turn | MaxDD | PF |
|-------------|---|--------|---------|--------|--------|------|-------|-----|
| **forward_4** | **4** | **1.287** | **52.5%** | **111.1%** | **9.52** | **0.244** | **0.8317** | **1.28** |
| forward_6 | 6 | 1.170 | 52.4% | 97.2% | 8.66 | 0.273 | 1.0679 | 1.25 |
| gini_8 | 8 | 0.855 | 51.8% | 64.4% | 6.34 | 0.100 | 1.1144 | 1.18 |
| equity_curated | 14 | 0.797 | 52.1% | 59.0% | 5.91 | 0.176 | 1.3181 | 1.16 |
| all_features | 31 | 0.658 | 51.5% | 46.6% | 4.88 | 0.162 | 1.4400 | 1.13 |
| **Buy & Hold** | — | **0.522** | — | **35.5%** | **3.87** | — | **1.8705** | — |

**forward_4 beats buy-and-hold by 2.5x Sharpe with half the drawdown.**

### BTC Daily Yearly Breakdown (forward_4)

| Year | Return | $(x) | Sharpe | Mo WinR |
|------|--------|------|--------|---------|
| 2017 | +2.682 | +1361% | 3.29 | 77.8% |
| 2018 | +1.497 | +347% | 1.48 | 75.0% |
| 2019 | +1.821 | +518% | 2.20 | 75.0% |
| 2020 | +1.164 | +220% | 1.19 | 75.0% |
| 2021 | +1.591 | +391% | 1.64 | 75.0% |
| 2022 | -0.613 | -46% | -0.79 | 33.3% |
| 2023 | +0.353 | +42% | 0.67 | 50.0% |
| 2024 | +0.741 | +110% | 1.16 | 58.3% |
| 2025 | +0.533 | +70% | 1.06 | 66.7% |

### BTC Daily Regime Analysis (forward_4)

| Regime | Days | % of OOS | Sharpe | WinRate |
|--------|------|----------|--------|---------|
| Bull (trail 60d > +10%) | 1,358 | 42.3% | **2.130** | 53.5% |
| Bear (trail 60d < -10%) | 926 | 28.8% | **0.725** | 52.6% |
| Sideways | 929 | 28.9% | **0.561** | 50.9% |

**Model profitable in ALL regimes.** 63.8% long, 36.2% short.

### MaxDD Units Clarification
MaxDD column is in **log return units**, NOT percentage. Convert: `actual_pct_dd = 1 - exp(-MaxDD_log)`.
- forward_4 MaxDD: 0.8317 log → **56.4% actual percentage drawdown**
- Buy & hold MaxDD: 1.8705 log → **84.6% actual percentage drawdown**
- Calmar ratio (AnnRet / MaxDD%): 111% / 56.4% ≈ **2.0** (excellent, target > 1.0)

### Flat-on-Neutral vs Hold-on-Neutral (forward_4)

| Mode | Sharpe | WinRate | AnnRet | Turnover | MaxDD | PF | FlatTime |
|------|--------|---------|--------|----------|-------|-----|----------|
| **Hold neutral** | **1.287** | 52.5% | 111.1% | 0.244 | 0.8317 | 1.28 | 0% |
| Flat neutral | 1.278 | 53.2% | 100.4% | 0.302 | 0.8317 | 1.33 | 30.3% |

**Conclusion: hold neutral marginally better overall.** Flat neutral improves win rate (53.2%) and PF (1.33) but reduces annual return (100% vs 111%) and increases turnover (+24%). MaxDD identical. Use flat neutral if conservative live trading is preferred and win rate matters for psychology.

### BTC 4h Results (with standard daily-designed features — all weak)

| Feature Set | N | Sharpe | AnnRet | Turn |
|-------------|---|--------|--------|------|
| forward_4 | 4 | 0.209 | 4.8% | 0.007 |
| gini_8 | 8 | 0.281 | 6.6% | 0.004 |
| all_features | 31 | 0.216 | 5.0% | 0.001 |
| **Buy & Hold** | — | **0.244** | **5.9%** | — |

Model barely beats buy-and-hold on 4h. Root cause: wrong features + period mismatch.

**Code:** `src/btc_feature_selection.py` (feature selection), `src/btc_model.py` (full model).
Results: `outputs/btc_model_daily.pkl`.

---

## Feature Selection Methods — Empirical Findings

**Method 1 — RF Gini Importance:** Ranks by tree split ease, not predictive power. Volatile/high-variance features (bb_width, intraday_range, gk_vol) rank high because they're easy to split on. Consistently suboptimal — beats all-features but loses to forward selection.

**Method 2 — Permutation Importance:** Useless. All values near zero on both equities and BTC. Does not work for low-SNR financial data. Confirmed on every dataset tested.

**Method 3 — Forward Selection (greedy walk-forward Sharpe):** Clear winner. Finds features that complement each other, not just individually strong features. Picks the right set reliably. Slow on large datasets (22k rows × 31 candidates = hours). Speed fix: use n_estimators=100 for selection, 500 for final model.
**Bug fixed (important):** Original code stopped on first negative step after step 4, and returned the final degraded set instead of peak. Fixed: (1) track global best set (`best_ever_selected`), (2) stop only after **2 consecutive** steps with no improvement. Always returns peak Sharpe set.

**Method 4 — Domain Knowledge (curated sets):** Competitive with forward selection on equities (0.607 Sharpe vs 0.570 for data-driven). Faster and more interpretable. For crypto, domain curated features underperform forward selection (0.797 vs 1.287) — forward selection found a better set.

**Rule:** For new assets/timeframes, always run forward selection first. Use domain knowledge to narrow the candidate pool but let data confirm the final set.

---

## Ensemble & Meta-Model Strategy

### When Ensembles Work
- Need base models with **positive edge independently**. A bad model drags down a good one.
- Confirmed: including XGBoost (negative Sharpe) in ensembles with RF always hurt performance.
- Once both RF and XGBoost had positive edge (bucket target + heavy regularization), ensemble marginal benefit: +0.005 to +0.011 Sharpe.

### Meta-Learner Rules
- Meta-learner must ONLY see out-of-sample predictions from base models.
- Keep meta-learner simple (Ridge regression).
- Walk-forward validate the meta-learner itself — never train and test on same period.

---

## Critical Warning: Walk-Forward CV for Stacking

When generating out-of-sample predictions from base models for the meta-learner, the walk-forward cross-validation must be implemented correctly. Getting this wrong — accidentally leaking future information — is the single most common mistake in quantitative ML.

### How to Do It Correctly
1. Define expanding training window for base models.
2. At each step, train base models ONLY on data up to time T.
3. Generate predictions for time T+1 — these are genuinely out-of-sample.
4. Slide window forward and repeat.
5. ONLY THEN train the meta-learner on collected OOS predictions.
6. Walk-forward validate the meta-learner itself.

### Red Flags That You Have Leakage
- Backtest Sharpe above 3-4 (too good for daily equity models).
- Massive performance drop when adding one day of delay.
- Suspiciously consistent accuracy across all market regimes.

---

## Project Roadmap

### Phase 1: Equity Daily Model ✅ COMPLETE
Best result: RF + target_bucket + curated_14 = 0.607 net Sharpe (SPY), 0.918 (QQQ).

### Phase 2: Multi-Asset Equity Portfolio ✅ COMPLETE
Equal-weight portfolio (6 assets): 0.656 Sharpe. QQQ alone (0.918) beats portfolio.

### Phase 3: BTC Daily Model ✅ COMPLETE
RF + target_bucket_10 + forward_4: **1.287 net Sharpe, 111% annualized, profitable in all regimes.**

### Phase 4: BTC 4h Model ✅ CLOSED (daily is better)
- [x] Standard features fail on 4h (max 0.281 Sharpe)
- [x] Intraday-calibrated features (RSI-30, Stoch-200, disparity_7, etc.) tested → best 0.431 Sharpe
- [x] Decision: 4h is structurally weaker, not worth infrastructure complexity vs daily 1.287 Sharpe

### Phase 4.5: BTC Paper Trading ← NEXT
- [ ] `src/btc_train.py`: train RF on all available data (forward_4), save model to disk
- [ ] `src/btc_predict.py`: fetch last N days from Kraken API, compute features, output LONG/SHORT/FLAT
- [ ] Daily cron job: run predict after UTC close, log signal + price
- [ ] Track paper trades for 30+ days before live deployment

### Phase 5: On-Chain + Derivatives Data (Future)
Free on-chain data (Binance funding rates, blockchain.com) tested — didn't improve forward_4.
Premium data still worth trying:
- Glassnode API: hashrate, block size, realized value, UTXO metrics (full history from 2014+)
- Binance/Bybit: open interest, liquidation levels (higher quality funding rate data)
- Google Trends: "bitcoin", "blockchain" search volume

### Phase 6: Equity Fundamental Model (Future)
3-6 month horizon, cross-sectional stock ranking.
- Inputs: P/E, EV/EBITDA, revenue growth, ROE, earnings revisions, 6-month momentum
- Model: Ridge or logistic regression (data scarce at quarterly frequency)
- Data: Koyfin (web scraping needed — no API), SimFin, Sharadar

---

## Validation Rules (Non-Negotiable)
1. **Never leak future data.** Always use walk-forward or time-series split. No standard k-fold CV.
2. **Account for transaction costs** in all backtest evaluations.
3. **Track out-of-sample performance only.** In-sample accuracy is meaningless.
4. **Statistical significance matters.** A short backtest proving "it works" is likely luck. Need enough trades for confidence.
5. **Beware survivorship bias** in stock universe — include delisted stocks if possible.
6. **Regime-test everything.** A strategy that only works in bull markets is not a strategy.
7. **Match bucket threshold to timeframe.** Too narrow = too much noise. Too wide = model never trades. Sweet spot: 30-40% neutral class.
8. **Different timeframes need different features.** Period lengths AND indicator types must match the timescale.
