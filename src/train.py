"""Base model training with walk-forward CV for QuantEnsemble.

Trains Ridge regression and XGBoost independently on daily SPY features,
evaluates out-of-sample performance, and saves results for ensemble use.
Includes turnover reduction filters and simple ensemble construction.
"""

import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data_loader import load_spy_daily
from features import build_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INITIAL_TRAIN = 1250  # ~5 years of daily bars
STEP_SIZE = 20        # ~1 month
COST_BPS = 5          # 5 bps one-way transaction cost
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# Correlated features to drop (per CLAUDE.md)
DROP_COLS = [
    "roc_10d",
    "cci_20",
    "adi",
    "close_sma20_ratio",
    "ret_std_20d",
    "ema_ratio_50",
]


# ---------------------------------------------------------------------------
# Walk-forward CV generator
# ---------------------------------------------------------------------------
def walk_forward_splits(n_samples: int, initial_train: int, step: int):
    """Yield (train_end, test_start, test_end) for expanding window CV."""
    t = initial_train
    while t < n_samples:
        test_end = min(t + step, n_samples)
        yield t, t, test_end
        t = test_end


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def make_ridge():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])


def make_random_forest():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=500,
            max_depth=5,
            max_features="sqrt",
            min_samples_leaf=50,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def make_xgboost():
    return XGBRegressor(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=500,  # high, rely on early stopping
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_walk_forward(X: np.ndarray, y: np.ndarray, model_name: str):
    """Run walk-forward CV for a single model type.

    Returns array of (index, prediction, actual) for all OOS periods.
    """
    results = []  # list of (idx, pred, actual)

    for train_end, test_start, test_end in walk_forward_splits(len(X), INITIAL_TRAIN, STEP_SIZE):
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        if model_name == "ridge":
            model = make_ridge()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        elif model_name == "rf":
            model = make_random_forest()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        elif model_name == "xgboost":
            model = make_xgboost()
            # Use last 20% of training data for early stopping
            val_size = max(1, int(len(X_train) * 0.2))
            X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
            y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            preds = model.predict(X_test)

        for i, (p, a) in enumerate(zip(preds, y_test)):
            results.append((test_start + i, p, a))

    return np.array(results)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def evaluate(preds: np.ndarray, actuals: np.ndarray, cost_bps: float = 0) -> dict:
    """Compute OOS metrics from predictions and actual returns."""
    # Strategy returns: go long/short based on prediction sign, earn actual return
    positions = np.sign(preds)
    gross_returns = positions * actuals

    # Transaction costs: pay cost on every position change
    turnover = np.abs(np.diff(positions, prepend=0))
    costs = turnover * (cost_bps / 10_000)
    net_returns = gross_returns - costs

    def _metrics(rets, label):
        m = {}
        mean_r = np.mean(rets)
        std_r = np.std(rets, ddof=1)
        m[f"sharpe_{label}"] = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
        m[f"annual_ret_{label}"] = mean_r * 252
        m[f"annual_vol_{label}"] = std_r * np.sqrt(252)

        # Directional accuracy
        correct = np.sum(np.sign(preds) == np.sign(actuals))
        m["dir_accuracy"] = correct / len(preds)

        # Profit factor
        profits = rets[rets > 0].sum()
        losses = abs(rets[rets < 0].sum())
        m[f"profit_factor_{label}"] = profits / losses if losses > 0 else np.inf

        # Max drawdown
        cum = np.cumsum(rets)
        running_max = np.maximum.accumulate(cum)
        drawdown = running_max - cum
        m[f"max_dd_{label}"] = np.max(drawdown) if len(drawdown) > 0 else 0.0

        return m

    metrics = {}
    metrics.update(_metrics(gross_returns, "gross"))
    metrics.update(_metrics(net_returns, "net"))
    metrics["n_predictions"] = len(preds)
    metrics["avg_turnover"] = np.mean(turnover)

    # Hit rate on high-conviction trades (|pred| > median)
    threshold = np.median(np.abs(preds))
    high_conv = np.abs(preds) > threshold
    if high_conv.sum() > 0:
        hc_correct = np.sum(np.sign(preds[high_conv]) == np.sign(actuals[high_conv]))
        metrics["hit_rate_high_conv"] = hc_correct / high_conv.sum()
    else:
        metrics["hit_rate_high_conv"] = np.nan

    return metrics


# ---------------------------------------------------------------------------
# Turnover reduction filters
# ---------------------------------------------------------------------------
def apply_filters(preds: np.ndarray, ema_alpha: float = None, deadzone_pct: float = None) -> np.ndarray:
    """Transform raw predictions into filtered positions.

    Filters are applied in order: EMA smoothing first, then dead-zone.
    No lookahead — only uses past predictions.

    Args:
        preds: Raw model predictions.
        ema_alpha: If set, smooth predictions with EMA before taking sign.
                   Higher alpha = less smoothing (more responsive).
        deadzone_pct: If set, percentile of |preds| below which we hold
                      previous position instead of flipping. E.g. 50 means
                      ignore signals weaker than the median.
    Returns:
        positions: Array of -1/0/+1 positions.
    """
    smoothed = preds.copy()

    # Filter 1: EMA smoothing (causal — each step uses only past)
    if ema_alpha is not None:
        for i in range(1, len(smoothed)):
            smoothed[i] = ema_alpha * smoothed[i] + (1 - ema_alpha) * smoothed[i - 1]

    # Compute dead-zone threshold from smoothed predictions
    if deadzone_pct is not None:
        threshold = np.percentile(np.abs(smoothed), deadzone_pct)
    else:
        threshold = 0.0

    # Build positions with dead-zone: hold previous position if signal is weak
    positions = np.zeros(len(smoothed))
    positions[0] = np.sign(smoothed[0])
    for i in range(1, len(smoothed)):
        if np.abs(smoothed[i]) > threshold:
            positions[i] = np.sign(smoothed[i])
        else:
            positions[i] = positions[i - 1]

    return positions


def evaluate_with_positions(positions: np.ndarray, actuals: np.ndarray,
                            preds: np.ndarray, cost_bps: float = 0) -> dict:
    """Evaluate strategy given explicit positions (not raw predictions).

    Similar to evaluate() but takes pre-computed positions from apply_filters().
    """
    gross_returns = positions * actuals
    turnover = np.abs(np.diff(positions, prepend=0))
    costs = turnover * (cost_bps / 10_000)
    net_returns = gross_returns - costs

    metrics = {}
    for rets, label in [(gross_returns, "gross"), (net_returns, "net")]:
        mean_r = np.mean(rets)
        std_r = np.std(rets, ddof=1)
        metrics[f"sharpe_{label}"] = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
        metrics[f"annual_ret_{label}"] = mean_r * 252
        metrics[f"annual_vol_{label}"] = std_r * np.sqrt(252)
        profits = rets[rets > 0].sum()
        losses = abs(rets[rets < 0].sum())
        metrics[f"profit_factor_{label}"] = profits / losses if losses > 0 else np.inf
        cum = np.cumsum(rets)
        running_max = np.maximum.accumulate(cum)
        drawdown = running_max - cum
        metrics[f"max_dd_{label}"] = np.max(drawdown) if len(drawdown) > 0 else 0.0

    # Directional accuracy (use original preds for this)
    correct = np.sum(np.sign(preds) == np.sign(actuals))
    metrics["dir_accuracy"] = correct / len(preds)
    metrics["n_predictions"] = len(preds)
    metrics["avg_turnover"] = np.mean(turnover)

    return metrics


def sweep_filters(preds: np.ndarray, actuals: np.ndarray, cost_bps: float,
                  model_label: str = "RF") -> list[dict]:
    """Sweep filter combinations and return results sorted by net Sharpe."""
    ema_options = [None, 0.3, 0.5]
    dz_options = [None, 25, 50]

    results = []
    for ema_alpha, dz_pct in product(ema_options, dz_options):
        if ema_alpha is None and dz_pct is None:
            label = "raw (no filter)"
        elif ema_alpha is None:
            label = f"dz={dz_pct}pct"
        elif dz_pct is None:
            label = f"ema={ema_alpha}"
        else:
            label = f"ema={ema_alpha}+dz={dz_pct}pct"

        positions = apply_filters(preds, ema_alpha=ema_alpha, deadzone_pct=dz_pct)
        metrics = evaluate_with_positions(positions, actuals, preds, cost_bps=cost_bps)
        results.append({
            "label": label,
            "ema_alpha": ema_alpha,
            "deadzone_pct": dz_pct,
            **metrics,
        })

    results.sort(key=lambda r: r["sharpe_net"], reverse=True)
    return results


def print_filter_table(filter_results: list[dict], model_label: str = "RF"):
    """Print a comparison table of filter sweep results."""
    print(f"\n{'='*85}")
    print(f"  Turnover Filter Sweep — {model_label}")
    print(f"{'='*85}")
    header = f"  {'Filter':<25} {'Sharpe(g)':>10} {'Sharpe(n)':>10} {'Turnover':>10} {'PF(net)':>10} {'MaxDD(n)':>10}"
    print(header)
    print(f"  {'-'*80}")
    for r in filter_results:
        print(f"  {r['label']:<25} {r['sharpe_gross']:>10.3f} {r['sharpe_net']:>10.3f} "
              f"{r['avg_turnover']:>10.3f} {r['profit_factor_net']:>10.2f} {r['max_dd_net']:>10.4f}")
    print(f"{'='*85}")


# ---------------------------------------------------------------------------
# Ensemble construction
# ---------------------------------------------------------------------------
def build_ensembles(all_results: dict, cost_bps: float,
                    best_ema: float = None, best_dz: float = None) -> dict:
    """Build ensemble predictions from aligned base model OOS predictions.

    Args:
        all_results: Dict of model_name -> {indices, predictions, actuals, ...}
        cost_bps: Transaction cost in basis points.
        best_ema: Best EMA alpha from filter sweep.
        best_dz: Best dead-zone percentile from filter sweep.

    Returns:
        Dict of ensemble_name -> {predictions, actuals, indices, metrics, metrics_filtered}
    """
    # Align predictions to common indices
    index_sets = [set(all_results[m]["indices"]) for m in all_results]
    common_idx = sorted(index_sets[0].intersection(*index_sets[1:]))
    common_idx = np.array(common_idx, dtype=int)

    aligned = {}
    for m in all_results:
        idx = all_results[m]["indices"]
        pred = all_results[m]["predictions"]
        act = all_results[m]["actuals"]
        # Build lookup: index -> (pred, actual)
        lookup = {int(idx[i]): (pred[i], act[i]) for i in range(len(idx))}
        aligned[m] = np.array([lookup[int(ci)] for ci in common_idx])

    # Actuals should be the same for all models (same target)
    actuals = aligned[list(all_results.keys())[0]][:, 1]

    model_names = list(all_results.keys())
    pred_matrix = np.column_stack([aligned[m][:, 0] for m in model_names])

    ensembles = {}

    # Equal weight ensemble
    eq_preds = pred_matrix.mean(axis=1)
    ensembles["equal_weight"] = eq_preds

    # RF-heavy ensemble (0.2 * ridge + 0.2 * xgboost + 0.6 * rf)
    if len(model_names) == 3:
        weights = {"ridge": 0.2, "xgboost": 0.2, "rf": 0.6}
        rf_heavy_preds = sum(
            weights[m] * aligned[m][:, 0] for m in model_names
        )
        ensembles["rf_heavy"] = rf_heavy_preds

    results = {}
    for ens_name, ens_preds in ensembles.items():
        # Raw evaluation (no filter)
        raw_metrics = evaluate(ens_preds, actuals, cost_bps=cost_bps)

        # Filtered evaluation
        positions_filt = apply_filters(ens_preds, ema_alpha=best_ema, deadzone_pct=best_dz)
        filt_metrics = evaluate_with_positions(positions_filt, actuals, ens_preds, cost_bps=cost_bps)

        results[ens_name] = {
            "predictions": ens_preds,
            "actuals": actuals,
            "indices": common_idx,
            "metrics_raw": raw_metrics,
            "metrics_filtered": filt_metrics,
        }

    return results


# ---------------------------------------------------------------------------
# Feature selection experiment: XGBoost with RF-selected features
# ---------------------------------------------------------------------------
def run_feature_selection_experiment(X: np.ndarray, y: np.ndarray,
                                     feature_cols: list[str],
                                     cost_bps: float) -> dict:
    """Train RF on initial window to get importances, then run XGBoost
    walk-forward with top-N features. No leakage — importances come only
    from the initial training window."""

    # Get feature importances from RF trained on initial window only
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=5, max_features="sqrt",
        min_samples_leaf=50, random_state=42, n_jobs=-1,
    )
    rf.fit(X[:INITIAL_TRAIN], y[:INITIAL_TRAIN])
    importances = rf.feature_importances_
    ranked = np.argsort(importances)[::-1]

    print("\n  RF Feature Importances (top 15):")
    for i in range(min(15, len(feature_cols))):
        idx = ranked[i]
        print(f"    {i+1:2d}. {feature_cols[idx]:<25} {importances[idx]:.4f}")

    results = {}
    for n_feat in [10, 15]:
        top_idx = ranked[:n_feat]
        X_sub = X[:, top_idx]
        sub_cols = [feature_cols[i] for i in top_idx]

        print(f"\n  XGBoost with top {n_feat} features: {sub_cols}")
        wf_results = train_walk_forward(X_sub, y, "xgboost")
        preds = wf_results[:, 1]
        actuals = wf_results[:, 2]
        metrics = evaluate(preds, actuals, cost_bps=cost_bps)

        # Also evaluate with best filter
        positions = apply_filters(preds, ema_alpha=0.5, deadzone_pct=50)
        filt_metrics = evaluate_with_positions(positions, actuals, preds, cost_bps=cost_bps)

        results[f"xgb_top{n_feat}"] = {
            "metrics_raw": metrics,
            "metrics_filtered": filt_metrics,
            "features_used": sub_cols,
        }

        print(f"    Sharpe (gross): {metrics['sharpe_gross']:.3f}")
        print(f"    Sharpe (net):   {metrics['sharpe_net']:.3f}")
        print(f"    Dir accuracy:   {metrics['dir_accuracy']:.3f}")
        print(f"    Turnover:       {metrics['avg_turnover']:.3f}")
        print(f"    Filtered net Sharpe: {filt_metrics['sharpe_net']:.3f} (turnover: {filt_metrics['avg_turnover']:.3f})")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_MODELS = ["ridge", "xgboost", "rf"]


def main():
    # CLI: python train.py [model1,model2,...] [daily|4h]
    models_arg = sys.argv[1] if len(sys.argv) > 1 else ",".join(ALL_MODELS)
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "daily"
    models_to_run = [m.strip() for m in models_arg.split(",")]

    print(f"Loading data and building features ({timeframe})...")
    if timeframe == "4h":
        from data_loader import load_spy_4h
        df = load_spy_4h()
    else:
        df = load_spy_daily()
    data = build_features(df, timeframe=timeframe)

    # Drop correlated features
    feature_cols = [c for c in data.columns if not c.startswith("target_") and c not in DROP_COLS]
    X = data[feature_cols].values
    y = data["target_ret"].values
    dates = data.index

    print(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Initial train window: {INITIAL_TRAIN}, step: {STEP_SIZE}")
    expected_folds = (len(X) - INITIAL_TRAIN) // STEP_SIZE
    print(f"Expected ~{expected_folds} folds, ~{len(X) - INITIAL_TRAIN} OOS predictions\n")

    all_results = {}

    for model_name in models_to_run:
        print(f"Training {model_name}...")
        results = train_walk_forward(X, y, model_name)
        indices = results[:, 0].astype(int)
        preds = results[:, 1]
        actuals = results[:, 2]

        metrics = evaluate(preds, actuals, cost_bps=COST_BPS)

        all_results[model_name] = {
            "indices": indices,
            "predictions": preds,
            "actuals": actuals,
            "dates": dates[indices],
            "metrics": metrics,
            "feature_cols": feature_cols,
        }

        print(f"  {model_name}: {len(preds)} OOS predictions")
        print(f"  Sharpe (gross): {metrics['sharpe_gross']:.3f}")
        print(f"  Sharpe (net):   {metrics['sharpe_net']:.3f}")
        print(f"  Dir accuracy:   {metrics['dir_accuracy']:.3f}")
        print(f"  Hit rate (HC):  {metrics['hit_rate_high_conv']:.3f}")
        print()

    # Print comparison table
    col_w = 15
    header = f"{'Metric':<30}" + "".join(f"{m:>{col_w}}" for m in models_to_run)
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    rows = [
        ("Sharpe (gross)", "sharpe_gross", ".3f"),
        ("Sharpe (net, 5bps)", "sharpe_net", ".3f"),
        ("Annual Return (gross)", "annual_ret_gross", ".4f"),
        ("Annual Return (net)", "annual_ret_net", ".4f"),
        ("Annual Vol", "annual_vol_gross", ".4f"),
        ("Dir Accuracy", "dir_accuracy", ".3f"),
        ("Hit Rate (high conv)", "hit_rate_high_conv", ".3f"),
        ("Profit Factor (gross)", "profit_factor_gross", ".2f"),
        ("Profit Factor (net)", "profit_factor_net", ".2f"),
        ("Max Drawdown (gross)", "max_dd_gross", ".4f"),
        ("Max Drawdown (net)", "max_dd_net", ".4f"),
        ("Avg Turnover", "avg_turnover", ".3f"),
        ("N Predictions", "n_predictions", "d"),
    ]

    for label, key, fmt in rows:
        vals = "".join(f"{all_results[m]['metrics'][key]:>{col_w}{fmt}}" for m in models_to_run)
        print(f"  {label:<28}{vals}")

    print("=" * len(header))

    # Save base model results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"base_model_results_{timeframe}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nBase results saved to {output_path}")

    # ------------------------------------------------------------------
    # Step 4.5: Turnover reduction filter sweep (on RF if available)
    # ------------------------------------------------------------------
    filter_model = "rf" if "rf" in all_results else models_to_run[0]
    print(f"\n--- Step 4.5: Turnover Filter Sweep ({filter_model} {timeframe}) ---")
    fm = all_results[filter_model]
    filter_results = sweep_filters(fm["predictions"], fm["actuals"], COST_BPS,
                                   model_label=f"{filter_model.upper()} {timeframe}")
    print_filter_table(filter_results, model_label=f"{filter_model.upper()} {timeframe}")

    # Pick best filter by net Sharpe
    best = filter_results[0]
    best_ema = best["ema_alpha"]
    best_dz = best["deadzone_pct"]
    print(f"\n  Best filter: {best['label']}")
    print(f"  Net Sharpe: {best['sharpe_net']:.3f}, Turnover: {best['avg_turnover']:.3f}")

    # ------------------------------------------------------------------
    # Step 5: Simple ensemble (only if all 3 models were run)
    # ------------------------------------------------------------------
    if len(all_results) >= 2:
        print(f"\n--- Step 5: Simple Ensemble ({timeframe}) ---")
        ensemble_results = build_ensembles(all_results, COST_BPS,
                                           best_ema=best_ema, best_dz=best_dz)

        # Print final comparison: all models + ensembles, raw and filtered
        all_names = list(models_to_run) + [f"ENS:{k}" for k in ensemble_results]

        # Apply best filter to individual models too for comparison
        filtered_individual = {}
        for m in models_to_run:
            pos = apply_filters(all_results[m]["predictions"],
                                ema_alpha=best_ema, deadzone_pct=best_dz)
            filtered_individual[m] = evaluate_with_positions(
                pos, all_results[m]["actuals"], all_results[m]["predictions"],
                cost_bps=COST_BPS)

        print(f"\n{'='*110}")
        print(f"  Final Comparison — Raw vs Filtered (best filter: {best['label']})")
        print(f"{'='*110}")
        cw = 12
        # Header
        h = f"  {'Model':<18}"
        for suffix in ["Raw", "Filtered"]:
            for col in ["Sharpe(n)", "Turnover", "PF(net)", "MaxDD(n)"]:
                h += f" {suffix+':'+col:>{cw}}"
        print(h)
        print(f"  {'-'*106}")

        for name in all_names:
            if name.startswith("ENS:"):
                ens_key = name.split(":")[1]
                raw_m = ensemble_results[ens_key]["metrics_raw"]
                filt_m = ensemble_results[ens_key]["metrics_filtered"]
            else:
                raw_m = all_results[name]["metrics"]
                filt_m = filtered_individual[name]
            row = f"  {name:<18}"
            for m in [raw_m, filt_m]:
                row += f" {m['sharpe_net']:>{cw}.3f}"
                row += f" {m['avg_turnover']:>{cw}.3f}"
                row += f" {m['profit_factor_net']:>{cw}.2f}"
                row += f" {m['max_dd_net']:>{cw}.4f}"
            print(row)

        print(f"{'='*110}")

        # Save ensemble results
        save_data = {
            "base_models": all_results,
            "ensembles": ensemble_results,
            "best_filter": {"ema_alpha": best_ema, "deadzone_pct": best_dz, "label": best["label"]},
            "filter_sweep": filter_results,
            "filtered_individual": filtered_individual,
        }
        ens_path = OUTPUT_DIR / f"ensemble_results_{timeframe}.pkl"
        with open(ens_path, "wb") as f:
            pickle.dump(save_data, f)
        print(f"\nEnsemble results saved to {ens_path}")
    else:
        print("\nSkipping ensemble — need at least 2 models. Run with 'ridge,xgboost,rf'.")

    # ------------------------------------------------------------------
    # Feature selection experiment: XGBoost with fewer features
    # ------------------------------------------------------------------
    if "xgboost" in models_to_run:
        print(f"\n--- Feature Selection Experiment: XGBoost with RF-selected features ---")
        fs_results = run_feature_selection_experiment(X, y, feature_cols, COST_BPS)

        # Compare all XGBoost variants
        print(f"\n  XGBoost Feature Selection Summary:")
        print(f"  {'Variant':<20} {'Sharpe(g)':>10} {'Sharpe(n)':>10} {'DirAcc':>8} {'Turnover':>10} {'Filt Sharpe(n)':>15}")
        print(f"  {'-'*73}")
        # Original XGBoost
        xgb_m = all_results["xgboost"]["metrics"]
        xgb_filt = filtered_individual.get("xgboost", {})
        filt_s = xgb_filt.get("sharpe_net", float("nan")) if xgb_filt else float("nan")
        print(f"  {'xgb (31 feat)':<20} {xgb_m['sharpe_gross']:>10.3f} {xgb_m['sharpe_net']:>10.3f} "
              f"{xgb_m['dir_accuracy']:>8.3f} {xgb_m['avg_turnover']:>10.3f} {filt_s:>15.3f}")
        for key, res in fs_results.items():
            rm = res["metrics_raw"]
            fm = res["metrics_filtered"]
            print(f"  {key:<20} {rm['sharpe_gross']:>10.3f} {rm['sharpe_net']:>10.3f} "
                  f"{rm['dir_accuracy']:>8.3f} {rm['avg_turnover']:>10.3f} {fm['sharpe_net']:>15.3f}")


if __name__ == "__main__":
    main()
