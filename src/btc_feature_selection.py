"""BTC Feature Selection: Compare 3 methods to find the best features.

Methods:
  1. RF Gini importance — rank by tree split importance
  2. Permutation importance — shuffle each feature, measure accuracy drop
  3. Forward selection — add features one-by-one, keep if improves walk-forward Sharpe

Uses first 50% of data for feature ranking, full data for walk-forward validation.
Data source: Kraken BTCUSD (has volume data).

BTC-specific adjustments:
  - 24/7 market: 6 bars/day for 4h (not 6.5 like equities)
  - Wider bucket thresholds tested (±0.5% to ±2.0%) — crypto is more volatile
  - All volume features included (OBV, vol_ratio_20d, MFI, ADI)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from data_loader import load_ohlcv, DATA_DIR
from features import build_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BTC_DAILY = DATA_DIR / "KRAKEN_BTCUSD, 1D.csv"
BTC_4H = DATA_DIR / "KRAKEN_BTCUSD, 240.csv"

INITIAL_TRAIN = 1250
STEP_SIZE = 20
COST_BPS = 5

# Correlated pairs to drop (from equity analysis — re-verify on BTC in correlation check)
DROP_COLS = [
    "roc_10d", "cci_20", "adi", "close_sma20_ratio",
    "ret_std_20d", "ema_ratio_50",
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


# ---------------------------------------------------------------------------
# BTC bucket targets with multiple thresholds
# ---------------------------------------------------------------------------
def add_btc_buckets(data, timeframe="daily"):
    """Add bucket targets appropriate for the timeframe.

    Daily: ±0.5%, ±1.0%, ±1.5%, ±2.0% (BTC daily moves are big)
    4h:    ±0.1%, ±0.2%, ±0.3%, ±0.4%, ±0.5% (4h bars are ~1/6 of daily)
    """
    ret = data["target_ret"]
    if timeframe == "4h":
        thresholds = [1, 2, 3, 4, 5]  # ±0.1% to ±0.5%
    else:
        thresholds = [5, 10, 15, 20]  # ±0.5% to ±2.0%
    for pct in thresholds:
        thresh = pct / 1000  # 5 -> 0.005, 2 -> 0.002
        col = f"target_bucket_{pct:02d}"
        data[col] = pd.cut(
            ret, bins=[-np.inf, -thresh, thresh, np.inf], labels=[-1, 0, 1]
        ).astype(float)
    return data


# ---------------------------------------------------------------------------
# Model & evaluation (same as multi_asset_trend.py)
# ---------------------------------------------------------------------------
def make_rf_clf(n_estimators=500):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=5, max_features="sqrt",
            min_samples_leaf=50, random_state=42, n_jobs=-1,
        )),
    ])


def preds_to_positions(preds):
    """Bucketed: neutral holds previous position."""
    positions = np.zeros(len(preds))
    positions[0] = preds[0]
    for i in range(1, len(preds)):
        if preds[i] != 0:
            positions[i] = preds[i]
        else:
            positions[i] = positions[i - 1]
    return positions


def evaluate(positions, actuals, cost_bps=COST_BPS):
    gross_returns = positions * actuals
    turnover = np.abs(np.diff(positions, prepend=0))
    costs = turnover * (cost_bps / 10_000)
    net_returns = gross_returns - costs

    mean_r = np.mean(net_returns)
    std_r = np.std(net_returns, ddof=1)
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0

    cum = np.cumsum(net_returns)
    running_max = np.maximum.accumulate(cum)
    max_dd = np.max(running_max - cum) if len(cum) > 0 else 0.0

    active = positions != 0
    win_rate = np.mean(net_returns[active] > 0) if active.sum() > 0 else 0.0

    n_years = len(net_returns) / 252
    total_ret = cum[-1] if len(cum) > 0 else 0.0
    if n_years > 0 and total_ret > -1:
        ann_ret = (1 + (np.exp(total_ret) - 1)) ** (1 / n_years) - 1
    else:
        ann_ret = 0.0

    return {
        "sharpe_net": sharpe,
        "win_rate": win_rate,
        "annual_return_net": ann_ret,
        "total_return_net": total_ret,
        "avg_turnover": np.mean(turnover),
        "max_dd_net": max_dd,
    }


def walk_forward_sharpe(X, y, y_pnl, initial_train=INITIAL_TRAIN, step=STEP_SIZE,
                        n_estimators=500):
    """Run walk-forward CV and return net Sharpe + full metrics."""
    all_preds = []
    all_actuals = []
    t = initial_train
    while t < len(X):
        test_end = min(t + step, len(X))
        model = make_rf_clf(n_estimators=n_estimators)
        model.fit(X[:t], y[:t])
        preds = model.predict(X[t:test_end])
        for i, (p, a) in enumerate(zip(preds, y_pnl[t:test_end])):
            all_preds.append(p)
            all_actuals.append(a)
        t = test_end

    preds_arr = np.array(all_preds)
    actuals_arr = np.array(all_actuals)
    positions = preds_to_positions(preds_arr)
    return evaluate(positions, actuals_arr)


# ---------------------------------------------------------------------------
# Method 1: RF Gini Importance
# ---------------------------------------------------------------------------
def gini_importance(X_train, y_train, feature_names):
    print("\n  Training RF for Gini importance...")
    model = make_rf_clf()
    model.fit(X_train, y_train)
    importances = model.named_steps["rf"].feature_importances_
    ranking = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    return ranking


# ---------------------------------------------------------------------------
# Method 2: Permutation Importance
# ---------------------------------------------------------------------------
def perm_importance(X_train, y_train, X_val, y_val, feature_names):
    print("\n  Training RF for permutation importance...")
    model = make_rf_clf()
    model.fit(X_train, y_train)
    result = permutation_importance(
        model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1,
        scoring="accuracy",
    )
    ranking = sorted(
        zip(feature_names, result.importances_mean, result.importances_std),
        key=lambda x: -x[1],
    )
    return ranking


# ---------------------------------------------------------------------------
# Method 3: Forward Selection (greedy, walk-forward Sharpe)
# ---------------------------------------------------------------------------
def forward_selection(X_all, y_all, y_pnl_all, feature_names, max_features=15,
                      n_estimators=500, candidate_indices=None):
    """Greedy forward selection.

    candidate_indices: optional list of feature indices to consider (pre-filtered).
    Stops after 2 consecutive steps with no improvement (< 0.01 delta).
    Returns the globally best feature set seen, not the final set.
    """
    print(f"\n  Running forward selection (n_estimators={n_estimators})...")
    selected = []
    remaining = list(candidate_indices) if candidate_indices is not None else list(range(len(feature_names)))
    best_sharpe = -np.inf
    best_ever_sharpe = -np.inf
    best_ever_selected = []
    consecutive_no_improve = 0
    history = []

    for step_num in range(max_features):
        best_feat = None
        best_step_sharpe = -np.inf

        for feat_idx in remaining:
            trial = selected + [feat_idx]
            X_trial = X_all[:, trial]
            m = walk_forward_sharpe(X_trial, y_all, y_pnl_all, n_estimators=n_estimators)
            if m["sharpe_net"] > best_step_sharpe:
                best_step_sharpe = m["sharpe_net"]
                best_feat = feat_idx

        if best_feat is not None:
            selected.append(best_feat)
            remaining.remove(best_feat)
            improvement = best_step_sharpe - best_sharpe
            best_sharpe = best_step_sharpe
            history.append({
                "step": step_num + 1,
                "feature": feature_names[best_feat],
                "sharpe": best_step_sharpe,
                "improvement": improvement,
            })
            print(f"    Step {step_num + 1}: +{feature_names[best_feat]:25s} "
                  f"Sharpe={best_step_sharpe:.3f}  (delta={improvement:+.3f})")

            # Track global best
            if best_step_sharpe > best_ever_sharpe:
                best_ever_sharpe = best_step_sharpe
                best_ever_selected = selected.copy()
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1

            # Stop after 2 consecutive steps with no improvement
            if consecutive_no_improve >= 2:
                print(f"    Stopping — 2 consecutive steps with no improvement")
                break

    print(f"    Global best: {len(best_ever_selected)} features, Sharpe={best_ever_sharpe:.3f}")
    return best_ever_selected, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "daily"
    if timeframe == "4":
        timeframe = "4h"
    include_onchain = "--onchain" in sys.argv

    print(f"\n{'#'*70}")
    print(f"  BTC FEATURE SELECTION ({timeframe}{'  +onchain' if include_onchain else ''})")
    print(f"{'#'*70}\n")

    # Load data
    path = BTC_4H if timeframe == "4h" else BTC_DAILY
    print(f"Loading BTC {timeframe} from {path}...")
    raw = load_ohlcv(path)
    print(f"  Raw: {len(raw)} rows, {raw.index[0]} to {raw.index[-1]}")
    print(f"  Price range: ${raw['close'].min():.2f} - ${raw['close'].max():.2f}")
    print(f"  Volume: min={raw['volume'].min():.2f}, max={raw['volume'].max():.2f}")

    # Load on-chain data if requested
    onchain_df = None
    if include_onchain:
        from onchain_loader import load_all_onchain
        onchain_df = load_all_onchain()

    # Build features — include intraday-specific features for 4h
    print("\nBuilding features...")
    include_intraday = (timeframe == "4h")
    data = build_features(raw, timeframe=timeframe, include_intraday=include_intraday,
                          onchain_df=onchain_df)
    data = add_btc_buckets(data, timeframe=timeframe)
    print(f"  Features built: {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")

    # Get feature columns (drop correlated + targets)
    feature_cols = [c for c in data.columns
                    if not c.startswith("target_") and c not in DROP_COLS]
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    # Determine which buckets we have
    bucket_cols = sorted([c for c in data.columns if c.startswith("target_bucket_")
                          and c != "target_bucket" and c != "target_bucket_vol"])

    # Class distribution for each bucket threshold
    print(f"\n{'='*70}")
    print(f"  Class Distribution by Bucket Threshold")
    print(f"{'='*70}")
    for bucket in bucket_cols:
        dist = data[bucket].value_counts(normalize=True).sort_index()
        thresh = bucket.split("_")[-1]
        print(f"  +/-{int(thresh)/10:.1f}%: ", end="")
        for cls in [-1.0, 0.0, 1.0]:
            label = {-1.0: "Down", 0.0: "Neutral", 1.0: "Up"}[cls]
            pct = dist.get(cls, 0)
            print(f"{label}={pct:.1%}  ", end="")
        print()

    # Correlation check on BTC
    print(f"\n{'='*70}")
    print(f"  BTC Correlation Check (>0.85)")
    print(f"{'='*70}")
    corr = data[feature_cols].corr().abs()
    corr_vals = corr.to_numpy().copy()
    np.fill_diagonal(corr_vals, 0)
    corr_df = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)
    high_corr = []
    for i in range(len(corr_df)):
        for j in range(i + 1, len(corr_df)):
            if corr_df.iloc[i, j] > 0.85:
                high_corr.append((corr_df.index[i], corr_df.columns[j], corr_df.iloc[i, j]))
    if high_corr:
        for a, b, c in sorted(high_corr, key=lambda x: -x[2]):
            print(f"  {a:25s} <-> {b:25s}: {c:.3f}")
    else:
        print("  No pairs above 0.85")

    # Prepare data
    X_all = data[feature_cols].values
    y_pnl = data["target_ret"].values
    split = len(data) // 2
    print(f"\n  Total: {len(data)} rows  |  Selection (1st half): {split}  |  Validation (2nd half): {len(data) - split}")

    # ======================================================================
    # BUCKET THRESHOLD SWEEP
    # ======================================================================
    print(f"\n{'='*70}")
    print(f"  Bucket Threshold Sweep (all features, full walk-forward)")
    print(f"{'='*70}")

    # Use fewer trees for sweep speed on large 4h dataset
    sweep_n_est = 100 if timeframe == "4h" else 500

    best_bucket = None
    best_bucket_sharpe = -np.inf
    for bucket in bucket_cols:
        y_buck = data[bucket].values
        m = walk_forward_sharpe(X_all, y_buck, y_pnl, n_estimators=sweep_n_est)
        thresh = bucket.split("_")[-1]
        marker = ""
        if m["sharpe_net"] > best_bucket_sharpe:
            best_bucket_sharpe = m["sharpe_net"]
            best_bucket = bucket
            marker = " <-- BEST"
        print(f"  +/-{int(thresh)/10:.1f}%:  S(n)={m['sharpe_net']:.3f}  WinRate={m['win_rate']:.1%}  "
              f"AnnRet={m['annual_return_net']:.1%}  Turn={m['avg_turnover']:.3f}  "
              f"MaxDD={m['max_dd_net']:.4f}{marker}")

    print(f"\n  --> Best bucket: {best_bucket} (Sharpe={best_bucket_sharpe:.3f})")
    target_col = best_bucket
    y_all = data[target_col].values
    y_select = y_all[:split]
    y_val_target = y_all[split:]
    X_select = X_all[:split]
    X_val = X_all[split:]

    # ======================================================================
    # METHOD 1: Gini Importance
    # ======================================================================
    print(f"\n{'='*70}")
    print(f"  METHOD 1: RF Gini Importance (trained on first 50%)")
    print(f"{'='*70}")

    gini_rank = gini_importance(X_select, y_select, feature_cols)
    print(f"\n  {'Rank':<6} {'Feature':25s} {'Importance':>12}")
    print(f"  {'-'*45}")
    for i, (feat, imp) in enumerate(gini_rank):
        print(f"  {i+1:<6} {feat:25s} {imp:>12.4f}")

    # Test top-N subsets
    eval_n_est = 100 if timeframe == "4h" else 500
    print(f"\n  Walk-forward Sharpe by top-N Gini features (n_estimators={eval_n_est}):")
    print(f"  {'N':>4} {'S(n)':>8} {'WinR':>8} {'Turn':>8} {'MaxDD':>8}")
    gini_results = []
    for n in [5, 8, 10, 12, 14, len(feature_cols)]:
        n = min(n, len(feature_cols))
        top_n = [f for f, _ in gini_rank[:n]]
        idxs = [feature_cols.index(f) for f in top_n]
        m = walk_forward_sharpe(X_all[:, idxs], y_all, y_pnl, n_estimators=eval_n_est)
        gini_results.append((n, top_n, m))
        print(f"  {n:>4} {m['sharpe_net']:>8.3f} {m['win_rate']:>8.1%} "
              f"{m['avg_turnover']:>8.3f} {m['max_dd_net']:>8.4f}")

    # ======================================================================
    # METHOD 2: Permutation Importance
    # ======================================================================
    print(f"\n{'='*70}")
    print(f"  METHOD 2: Permutation Importance (trained 1st half, evaluated 2nd half)")
    print(f"{'='*70}")

    perm_rank = perm_importance(X_select, y_select, X_val, y_val_target, feature_cols)
    print(f"\n  {'Rank':<6} {'Feature':25s} {'Importance':>12} {'Std':>10}")
    print(f"  {'-'*55}")
    for i, (feat, imp, std) in enumerate(perm_rank):
        print(f"  {i+1:<6} {feat:25s} {imp:>12.4f} {std:>10.4f}")

    # Test top-N subsets
    print(f"\n  Walk-forward Sharpe by top-N permutation features (n_estimators={eval_n_est}):")
    print(f"  {'N':>4} {'S(n)':>8} {'WinR':>8} {'Turn':>8} {'MaxDD':>8}")
    perm_results = []
    for n in [5, 8, 10, 12, 14, len(feature_cols)]:
        n = min(n, len(feature_cols))
        top_n = [f for f, _, _ in perm_rank[:n]]
        idxs = [feature_cols.index(f) for f in top_n]
        m = walk_forward_sharpe(X_all[:, idxs], y_all, y_pnl, n_estimators=eval_n_est)
        perm_results.append((n, top_n, m))
        print(f"  {n:>4} {m['sharpe_net']:>8.3f} {m['win_rate']:>8.1%} "
              f"{m['avg_turnover']:>8.3f} {m['max_dd_net']:>8.4f}")

    # ======================================================================
    # METHOD 3: Forward Selection
    # ======================================================================
    print(f"\n{'='*70}")
    print(f"  METHOD 3: Forward Selection (greedy, maximizes walk-forward Sharpe)")
    print(f"{'='*70}")

    # Speed optimization for 4h (22k rows): use 100 trees + top-20 Gini candidates
    # Cuts runtime from ~10 hours to ~20-30 minutes
    if timeframe == "4h":
        fwd_n_estimators = 100
        top20_gini = [feature_cols.index(f) for f, _ in gini_rank[:20]]
        print(f"  4h mode: n_estimators={fwd_n_estimators}, candidates=top-20 Gini")
        print(f"  Candidates: {[feature_cols[i] for i in top20_gini]}")
        fwd_selected, fwd_history = forward_selection(
            X_all, y_all, y_pnl, feature_cols,
            n_estimators=fwd_n_estimators, candidate_indices=top20_gini,
        )
        # Re-score final set with full 500 trees
        fwd_m = walk_forward_sharpe(X_all[:, fwd_selected], y_all, y_pnl, n_estimators=500)
    else:
        fwd_selected, fwd_history = forward_selection(X_all, y_all, y_pnl, feature_cols)
        fwd_m = walk_forward_sharpe(X_all[:, fwd_selected], y_all, y_pnl)
    fwd_features = [feature_cols[i] for i in fwd_selected]
    print(f"\n  Final forward-selected features ({len(fwd_features)}):")
    print(f"  {fwd_features}")
    print(f"  Sharpe={fwd_m['sharpe_net']:.3f}  WinRate={fwd_m['win_rate']:.1%}  "
          f"Turn={fwd_m['avg_turnover']:.3f}")

    # ======================================================================
    # COMPARISON
    # ======================================================================
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON — Best from each method")
    print(f"{'='*70}")

    best_gini = max(gini_results, key=lambda x: x[2]["sharpe_net"])
    best_perm = max(perm_results, key=lambda x: x[2]["sharpe_net"])
    # Final comparison always uses 500 trees for fair scoring
    all_feat_m = walk_forward_sharpe(X_all, y_all, y_pnl, n_estimators=500)

    # Equity curated_14 (adapted — includes vol_ratio_20d since we have volume)
    equity_curated = [
        "log_ret_5d", "log_ret_20d", "mom_divergence",
        "ema_ratio_12", "ema_ratio_26", "macd_hist",
        "rsi_14", "bb_width", "gk_vol", "vol_ratio",
        "vol_ratio_20d", "close_pos_range", "gap", "dow_sin",
    ]
    eq_avail = [f for f in equity_curated if f in feature_cols]
    eq_idxs = [feature_cols.index(f) for f in eq_avail]
    eq_m = walk_forward_sharpe(X_all[:, eq_idxs], y_all, y_pnl)

    print(f"\n  {'Method':25s} {'N':>4} {'S(n)':>8} {'WinR':>8} {'AnnRet':>8} {'Turn':>8} {'MaxDD':>8}")
    print(f"  {'-'*80}")
    for label, n, m in [
        (f"Gini top-{best_gini[0]}", best_gini[0], best_gini[2]),
        (f"Perm top-{best_perm[0]}", best_perm[0], best_perm[2]),
        ("Forward selection", len(fwd_selected), fwd_m),
        ("Equity curated_14", len(eq_avail), eq_m),
        ("All features", len(feature_cols), all_feat_m),
    ]:
        print(f"  {label:25s} {n:>4} {m['sharpe_net']:>8.3f} {m['win_rate']:>8.1%} "
              f"{m['annual_return_net']:>8.1%} {m['avg_turnover']:>8.3f} {m['max_dd_net']:>8.4f}")

    # Print the winning feature set
    winner = max(
        [("Gini", best_gini[1], best_gini[2]),
         ("Perm", [f for f, _, _ in perm_rank[:best_perm[0]]], best_perm[2]),
         ("Forward", fwd_features, fwd_m),
         ("Equity curated", eq_avail, eq_m),
         ("All", feature_cols, all_feat_m)],
        key=lambda x: x[2]["sharpe_net"],
    )
    print(f"\n  WINNER: {winner[0]} ({len(winner[1])} features, Sharpe={winner[2]['sharpe_net']:.3f})")
    print(f"  Features: {winner[1]}")

    print(f"\n{'#'*70}")
    print(f"  DONE")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
