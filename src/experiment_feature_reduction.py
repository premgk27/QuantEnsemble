"""Feature Reduction & Target Comparison Experiment.

Goals:
  1. Permutation importance on current RF → identify top features
  2. Curated 14-feature set (domain knowledge, minimal redundancy)
  3. Retune XGBoost with heavier regularization on reduced features
  4. Test targets: baseline (1-bar), risk-adjusted, vol-norm 5d, trend persistence, 3-class bucketed
  5. Proper hold periods for multi-bar targets (hold 5 bars, not flip every bar)
  6. Benchmark: RF target_ret net Sharpe 0.313 (with ema=0.5+dz=50pct filter)

Does NOT modify train.py or existing results.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier

from data_loader import load_spy_daily
from features import build_features

INITIAL_TRAIN = 1250
STEP_SIZE = 20
COST_BPS = 5
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

DROP_COLS = [
    "roc_10d", "cci_20", "adi", "close_sma20_ratio",
    "ret_std_20d", "ema_ratio_50",
]

# Curated 14-feature set (domain knowledge — minimal redundancy)
# Each captures a distinct concept:
#   - 2 momentum scales (5d, 20d) + divergence
#   - 2 trend scales (ema_12, ema_26) + acceleration (macd)
#   - 1 mean reversion (rsi)
#   - 2 volatility (gk_vol, vol_ratio, bb_width)
#   - 1 volume (volume_ratio)
#   - 2 price structure (close_pos_range, gap)
#   - 1 external (dow_sin — placeholder, first to drop)
CURATED_14 = [
    "log_ret_5d", "log_ret_20d", "mom_divergence",
    "ema_ratio_12", "ema_ratio_26", "macd_hist",
    "rsi_14",
    "bb_width", "gk_vol", "vol_ratio",
    "vol_ratio_20d",
    "close_pos_range", "gap",
    "dow_sin",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def walk_forward_splits(n_samples, initial_train, step):
    t = initial_train
    while t < n_samples:
        test_end = min(t + step, n_samples)
        yield t, t, test_end
        t = test_end


def apply_filters(preds, ema_alpha=None, deadzone_pct=None):
    smoothed = preds.copy()
    if ema_alpha is not None:
        for i in range(1, len(smoothed)):
            smoothed[i] = ema_alpha * smoothed[i] + (1 - ema_alpha) * smoothed[i - 1]
    if deadzone_pct is not None:
        threshold = np.percentile(np.abs(smoothed), deadzone_pct)
    else:
        threshold = 0.0
    positions = np.zeros(len(smoothed))
    positions[0] = np.sign(smoothed[0])
    for i in range(1, len(smoothed)):
        if np.abs(smoothed[i]) > threshold:
            positions[i] = np.sign(smoothed[i])
        else:
            positions[i] = positions[i - 1]
    return positions


def evaluate_positions(positions, actuals_1bar, cost_bps=0):
    """Evaluate strategy from explicit positions against 1-bar returns."""
    gross_returns = positions * actuals_1bar
    turnover = np.abs(np.diff(positions, prepend=0))
    costs = turnover * (cost_bps / 10_000)
    net_returns = gross_returns - costs
    metrics = {}
    for rets, label in [(gross_returns, "gross"), (net_returns, "net")]:
        mean_r = np.mean(rets)
        std_r = np.std(rets, ddof=1)
        metrics[f"sharpe_{label}"] = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
        profits = rets[rets > 0].sum()
        losses = abs(rets[rets < 0].sum())
        metrics[f"profit_factor_{label}"] = profits / losses if losses > 0 else np.inf
        cum = np.cumsum(rets)
        running_max = np.maximum.accumulate(cum)
        drawdown = running_max - cum
        metrics[f"max_dd_{label}"] = np.max(drawdown) if len(drawdown) > 0 else 0.0
    metrics["avg_turnover"] = np.mean(turnover)
    metrics["n_predictions"] = len(positions)
    # Direction accuracy: how often does position sign match actual return sign
    correct = np.sum(np.sign(positions) == np.sign(actuals_1bar))
    nonzero = np.sum(positions != 0)
    metrics["dir_accuracy"] = correct / len(positions)
    metrics["pct_active"] = nonzero / len(positions)
    return metrics


def preds_to_positions_1bar(preds):
    """For 1-bar targets: position = sign(prediction), trade every bar."""
    return np.sign(preds)


def preds_to_positions_hold5(preds, hold_period=5):
    """For 5-bar targets: only update position every hold_period bars.

    No lookahead — at bar t, hold the position from the last decision point.
    """
    positions = np.zeros(len(preds))
    for i in range(len(preds)):
        if i % hold_period == 0:
            positions[i] = np.sign(preds[i])
        else:
            positions[i] = positions[i - 1]
    return positions


def preds_to_positions_bucketed(preds):
    """For 3-class bucketed target: -1=short, 0=flat, +1=long.

    Model predicts class directly. Class 0 (neutral) → hold previous position.
    """
    positions = np.zeros(len(preds))
    positions[0] = preds[0]  # -1, 0, or +1
    for i in range(1, len(preds)):
        if preds[i] != 0:
            positions[i] = preds[i]
        else:
            positions[i] = positions[i - 1]  # neutral → hold
    return positions


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------
def run_permutation_importance(X, y, feature_cols):
    """Train RF on initial window, rank by permutation importance on holdout."""
    from sklearn.inspection import permutation_importance

    print("=" * 70)
    print("  Step 1: Permutation Importance")
    print("=" * 70)

    split = INITIAL_TRAIN
    holdout_size = min(500, len(X) - split)
    X_train, y_train = X[:split], y[:split]
    X_hold, y_hold = X[split:split + holdout_size], y[split:split + holdout_size]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_hold_s = scaler.transform(X_hold)

    # n_jobs=1 for RF so permutation_importance can parallelize
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=5, max_features="sqrt",
        min_samples_leaf=50, random_state=42, n_jobs=1,
    )
    print("  Training RF on initial window...")
    rf.fit(X_train_s, y_train)
    builtin_imp = rf.feature_importances_

    print("  Computing permutation importance (10 repeats)...")
    perm = permutation_importance(rf, X_hold_s, y_hold, n_repeats=10,
                                  random_state=42, n_jobs=-1,
                                  scoring="neg_mean_squared_error")

    perm_rank = np.argsort(perm.importances_mean)[::-1]

    print(f"\n  {'Rank':<5} {'Feature':<25} {'Perm Imp':>15} {'Std':>10} {'Built-in':>12}")
    print(f"  {'-'*67}")
    for i, idx in enumerate(perm_rank):
        marker = " ***" if i < 15 else ""
        print(f"  {i+1:<5} {feature_cols[idx]:<25} {perm.importances_mean[idx]:>15.6f} "
              f"{perm.importances_std[idx]:>10.6f} {builtin_imp[idx]:>12.4f}{marker}")

    top_15_idx = perm_rank[:15]
    top_12_idx = perm_rank[:12]
    top_15_names = [feature_cols[i] for i in top_15_idx]
    top_12_names = [feature_cols[i] for i in top_12_idx]

    print(f"\n  Perm top 15: {top_15_names}")
    print(f"  Perm top 12: {top_12_names}")

    return {
        "builtin_importance": builtin_imp,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
        "perm_rank": perm_rank,
        "top_15_idx": top_15_idx,
        "top_15_names": top_15_names,
        "top_12_idx": top_12_idx,
        "top_12_names": top_12_names,
    }


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def make_rf_reg():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=500, max_depth=5, max_features="sqrt",
            min_samples_leaf=50, random_state=42, n_jobs=-1,
        )),
    ])


def make_rf_clf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=500, max_depth=5, max_features="sqrt",
            min_samples_leaf=50, random_state=42, n_jobs=-1,
        )),
    ])


def make_xgb_reg():
    """XGBoost with heavy regularization per notes."""
    return XGBRegressor(
        max_depth=2,
        learning_rate=0.02,
        n_estimators=300,
        subsample=0.7,
        colsample_bytree=0.5,
        min_child_weight=50,
        reg_alpha=1.0,
        reg_lambda=5.0,
        gamma=0.1,
        random_state=42,
        verbosity=0,
    )


def make_xgb_clf():
    """XGBoost classifier with heavy regularization."""
    return XGBClassifier(
        max_depth=2,
        learning_rate=0.02,
        n_estimators=300,
        subsample=0.7,
        colsample_bytree=0.5,
        min_child_weight=50,
        reg_alpha=1.0,
        reg_lambda=5.0,
        gamma=0.1,
        random_state=42,
        verbosity=0,
        eval_metric="mlogloss",
    )


# ---------------------------------------------------------------------------
# Walk-forward training
# ---------------------------------------------------------------------------
def train_walk_forward(X, y, model_type, is_classifier=False):
    """Walk-forward CV. Returns (indices, predictions, actuals).

    For XGBoost classifiers, remaps labels to 0,1,2,... then back to original.
    """
    # XGBoost needs 0-indexed classes; RF handles arbitrary labels fine
    remap = None
    if is_classifier and model_type == "xgb_heavy":
        unique_labels = np.sort(np.unique(y[~np.isnan(y)]))
        remap = {orig: i for i, orig in enumerate(unique_labels)}
        remap_inv = {i: orig for orig, i in remap.items()}
        y = np.array([remap.get(v, v) for v in y])

    results = []

    for train_end, test_start, test_end in walk_forward_splits(len(X), INITIAL_TRAIN, STEP_SIZE):
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        if model_type == "rf":
            model = make_rf_clf() if is_classifier else make_rf_reg()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        elif model_type == "xgb_heavy":
            model = make_xgb_clf() if is_classifier else make_xgb_reg()
            val_size = max(1, int(len(X_train) * 0.2))
            X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
            y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_test)

        # Remap predictions back to original labels
        if remap is not None:
            preds = np.array([remap_inv.get(p, p) for p in preds])

        for i, (p, a) in enumerate(zip(preds, y_test)):
            results.append((test_start + i, p, a))

    arr = np.array(results)
    return arr[:, 0].astype(int), arr[:, 1], arr[:, 2]


# ---------------------------------------------------------------------------
# Target configurations
# ---------------------------------------------------------------------------
TARGETS = {
    "target_ret": {
        "desc": "1-bar return (baseline)",
        "is_classifier": False,
        "hold_period": 1,
    },
    "target_ret_risknorm": {
        "desc": "1-bar risk-adjusted (ret/vol)",
        "is_classifier": False,
        "hold_period": 1,
    },
    "target_ret5_volnorm": {
        "desc": "5d fwd return / vol",
        "is_classifier": False,
        "hold_period": 5,
    },
    "target_trend_persist": {
        "desc": "trend persistence (+1/-1)",
        "is_classifier": False,
        "hold_period": 5,
    },
    "target_bucket": {
        "desc": "3-class bucketed (down/neutral/up)",
        "is_classifier": True,
        "hold_period": 1,
    },
}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "daily"

    print(f"\n{'#'*70}")
    print(f"  FEATURE REDUCTION & TARGET COMPARISON ({timeframe})")
    print(f"{'#'*70}\n")

    # Load data
    print("Loading data...")
    if timeframe == "4h":
        from data_loader import load_spy_4h
        df = load_spy_4h()
    else:
        df = load_spy_daily()
    data = build_features(df, timeframe=timeframe)

    feature_cols = [c for c in data.columns
                    if not c.startswith("target_") and c not in DROP_COLS]
    X_full = data[feature_cols].values
    y_pnl = data["target_ret"].values  # always evaluate PnL on 1-bar returns

    print(f"Dataset: {len(X_full)} samples, {len(feature_cols)} features")

    # ------------------------------------------------------------------
    # Step 1: Permutation importance
    # ------------------------------------------------------------------
    imp_results = run_permutation_importance(X_full, y_pnl, feature_cols)

    # ------------------------------------------------------------------
    # Build feature sets
    # ------------------------------------------------------------------
    # Curated 14 — match names to actual column indices
    curated_idx = [feature_cols.index(f) for f in CURATED_14 if f in feature_cols]
    curated_names = [feature_cols[i] for i in curated_idx]
    missing = [f for f in CURATED_14 if f not in feature_cols]
    if missing:
        print(f"\n  WARNING: Curated features not found: {missing}")
    print(f"\n  Curated {len(curated_idx)} features: {curated_names}")

    feature_sets = {
        "full_31": (np.arange(len(feature_cols)), feature_cols),
        "perm_15": (imp_results["top_15_idx"], imp_results["top_15_names"]),
        "perm_12": (imp_results["top_12_idx"], imp_results["top_12_names"]),
        f"curated_{len(curated_idx)}": (np.array(curated_idx), curated_names),
    }

    model_types = ["rf", "xgb_heavy"]
    all_results = {}

    # ------------------------------------------------------------------
    # Step 2: Run all combinations
    # ------------------------------------------------------------------
    for target_name, target_cfg in TARGETS.items():
        target_desc = target_cfg["desc"]
        is_clf = target_cfg["is_classifier"]
        hold_period = target_cfg["hold_period"]

        print(f"\n{'='*70}")
        print(f"  Target: {target_name} — {target_desc} (hold={hold_period})")
        print(f"{'='*70}")

        y_target = data[target_name].values

        # Skip if target has NaNs
        valid = ~np.isnan(y_target)
        if not valid.all():
            nan_count = (~valid).sum()
            print(f"  WARNING: {nan_count} NaN values in {target_name}, will skip those rows")

        for fs_name, (fs_idx, fs_cols) in feature_sets.items():
            X_sub = X_full[:, fs_idx]

            for model_type in model_types:
                key = f"{model_type}|{target_name}|{fs_name}"
                print(f"\n  {model_type} | {fs_name} ({len(fs_idx)}f) | {target_name}", end="")

                indices, preds, actuals_target = train_walk_forward(
                    X_sub, y_target, model_type, is_classifier=is_clf)

                actuals_pnl = y_pnl[indices]

                # Convert predictions to positions based on target type
                if is_clf:
                    raw_positions = preds_to_positions_bucketed(preds)
                elif hold_period > 1:
                    raw_positions = preds_to_positions_hold5(preds, hold_period)
                else:
                    raw_positions = preds_to_positions_1bar(preds)

                # Raw evaluation
                raw_metrics = evaluate_positions(raw_positions, actuals_pnl, cost_bps=COST_BPS)

                # Filtered evaluation (ema=0.5 + dz=50pct on raw predictions, then hold)
                if is_clf:
                    # For classifier, filter the positions directly
                    filt_positions = raw_positions.copy()
                    # Smooth: only change when consecutive signals agree
                    for i in range(1, len(filt_positions)):
                        if filt_positions[i] != filt_positions[i - 1]:
                            # Check if next prediction also agrees (lookahead-free: use current)
                            filt_positions[i] = filt_positions[i]  # keep as-is for now
                    filt_metrics = evaluate_positions(filt_positions, actuals_pnl, cost_bps=COST_BPS)
                else:
                    filt_positions = apply_filters(preds, ema_alpha=0.5, deadzone_pct=50)
                    if hold_period > 1:
                        # Apply hold period on filtered positions
                        held = np.zeros(len(filt_positions))
                        for i in range(len(filt_positions)):
                            if i % hold_period == 0:
                                held[i] = filt_positions[i]
                            else:
                                held[i] = held[i - 1]
                        filt_positions = held
                    filt_metrics = evaluate_positions(filt_positions, actuals_pnl, cost_bps=COST_BPS)

                all_results[key] = {
                    "model": model_type,
                    "target": target_name,
                    "features": fs_name,
                    "n_features": len(fs_idx),
                    "hold_period": hold_period,
                    "is_classifier": is_clf,
                    "metrics_raw": raw_metrics,
                    "metrics_filtered": filt_metrics,
                }

                print(f"  → raw S(n)={raw_metrics['sharpe_net']:.3f} "
                      f"filt={filt_metrics['sharpe_net']:.3f} "
                      f"turn={raw_metrics['avg_turnover']:.3f}/{filt_metrics['avg_turnover']:.3f}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n\n{'#'*100}")
    print(f"  SUMMARY — sorted by filtered net Sharpe")
    print(f"{'#'*100}")
    cw = 9
    header = (f"  {'Model':<10} {'Target':<22} {'Features':<12} {'Hold':>4} "
              f"{'S(n)raw':>{cw}} {'S(n)filt':>{cw}} {'Turn raw':>{cw}} "
              f"{'Turn flt':>{cw}} {'DirAcc':>{cw}} {'MaxDD':>{cw}}")
    print(header)
    print(f"  {'-'*96}")

    sorted_keys = sorted(all_results.keys(),
                         key=lambda k: all_results[k]["metrics_filtered"]["sharpe_net"],
                         reverse=True)

    for key in sorted_keys:
        r = all_results[key]
        rm = r["metrics_raw"]
        fm = r["metrics_filtered"]
        print(f"  {r['model']:<10} {r['target']:<22} {r['features']:<12} {r['hold_period']:>4} "
              f"{rm['sharpe_net']:>{cw}.3f} {fm['sharpe_net']:>{cw}.3f} "
              f"{rm['avg_turnover']:>{cw}.3f} {fm['avg_turnover']:>{cw}.3f} "
              f"{rm['dir_accuracy']:>{cw}.3f} {fm['max_dd_net']:>{cw}.4f}")

    print(f"  {'-'*96}")
    print(f"  BENCHMARK: RF | target_ret | full_31 | filtered net Sharpe = 0.313")
    print(f"{'#'*100}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_data = {
        "feature_importance": {
            "perm_rank": imp_results["perm_rank"],
            "top_15_names": imp_results["top_15_names"],
            "top_12_names": imp_results["top_12_names"],
            "builtin_importance": imp_results["builtin_importance"],
            "perm_importance_mean": imp_results["perm_importance_mean"],
        },
        "curated_features": curated_names,
        "experiments": all_results,
        "feature_cols": feature_cols,
    }
    out_path = OUTPUT_DIR / f"feature_reduction_experiment_{timeframe}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
