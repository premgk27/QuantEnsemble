"""Quick win experiments:
1. Vol-adjusted bucket target vs fixed bucket
2. RF + XGBoost ensemble on bucketed target (both now have positive edge)

Benchmark: RF + target_bucket + curated_14 = 0.607 net Sharpe
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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

CURATED_14 = [
    "log_ret_5d", "log_ret_20d", "mom_divergence",
    "ema_ratio_12", "ema_ratio_26", "macd_hist",
    "rsi_14",
    "bb_width", "gk_vol", "vol_ratio",
    "vol_ratio_20d",
    "close_pos_range", "gap",
    "dow_sin",
]


def walk_forward_splits(n_samples, initial_train, step):
    t = initial_train
    while t < n_samples:
        test_end = min(t + step, n_samples)
        yield t, t, test_end
        t = test_end


def preds_to_positions_bucketed(preds):
    """Class -1/0/+1 → positions. Neutral (0) holds previous."""
    positions = np.zeros(len(preds))
    positions[0] = preds[0]
    for i in range(1, len(preds)):
        if preds[i] != 0:
            positions[i] = preds[i]
        else:
            positions[i] = positions[i - 1]
    return positions


def evaluate_positions(positions, actuals_1bar, cost_bps=0):
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
    correct = np.sum(np.sign(positions) == np.sign(actuals_1bar))
    metrics["dir_accuracy"] = correct / len(positions)
    metrics["avg_turnover"] = np.mean(turnover)
    metrics["n_predictions"] = len(positions)
    metrics["pct_neutral"] = np.mean(positions == 0)  # how often flat after bucketing
    return metrics


def make_rf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=500, max_depth=5, max_features="sqrt",
            min_samples_leaf=50, random_state=42, n_jobs=-1,
        )),
    ])


def make_xgb():
    return XGBClassifier(
        max_depth=2, learning_rate=0.02, n_estimators=300,
        subsample=0.7, colsample_bytree=0.5, min_child_weight=50,
        reg_alpha=1.0, reg_lambda=5.0, gamma=0.1,
        random_state=42, verbosity=0, eval_metric="mlogloss",
    )


def train_walk_forward_clf(X, y, model_type):
    """Walk-forward for classifier. Remaps labels for XGBoost."""
    unique_labels = np.sort(np.unique(y[~np.isnan(y)]))
    remap = {orig: i for i, orig in enumerate(unique_labels)}
    remap_inv = {i: orig for orig, i in remap.items()}
    needs_remap = model_type == "xgb"

    if needs_remap:
        y_mapped = np.array([remap.get(v, v) for v in y])
    else:
        y_mapped = y

    results = []
    for train_end, test_start, test_end in walk_forward_splits(len(X), INITIAL_TRAIN, STEP_SIZE):
        X_train, y_train = X[:train_end], y_mapped[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        if model_type == "rf":
            model = make_rf()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        elif model_type == "xgb":
            model = make_xgb()
            val_size = max(1, int(len(X_train) * 0.2))
            X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
            y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_test)
            preds = np.array([remap_inv.get(p, p) for p in preds])

        for i, (p, a) in enumerate(zip(preds, y_test)):
            results.append((test_start + i, p, a))

    arr = np.array(results)
    return arr[:, 0].astype(int), arr[:, 1], arr[:, 2]


def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "daily"

    print(f"\n{'#'*70}")
    print(f"  QUICK WINS EXPERIMENT ({timeframe})")
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
    curated_idx = [feature_cols.index(f) for f in CURATED_14 if f in feature_cols]
    X = data[feature_cols].values[:, curated_idx]
    y_pnl = data["target_ret"].values

    # Check class balance for both targets
    for tgt in ["target_bucket", "target_bucket_vol"]:
        vals = data[tgt].value_counts()
        print(f"  {tgt} class balance: {dict(vals)}")

    # ------------------------------------------------------------------
    # Experiment 1: Fixed bucket vs vol-adjusted bucket
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Experiment 1: Fixed vs Vol-Adjusted Bucket")
    print(f"{'='*70}")

    all_results = {}

    for target_name in ["target_bucket", "target_bucket_vol"]:
        y_target = data[target_name].values

        for model_type in ["rf", "xgb"]:
            key = f"{model_type}|{target_name}"
            print(f"\n  Training {model_type} on {target_name}...")

            indices, preds, actuals_target = train_walk_forward_clf(X, y_target, model_type)
            actuals_pnl = y_pnl[indices]
            positions = preds_to_positions_bucketed(preds)
            metrics = evaluate_positions(positions, actuals_pnl, cost_bps=COST_BPS)

            all_results[key] = {
                "indices": indices,
                "predictions": preds,
                "positions": positions,
                "metrics": metrics,
            }

            print(f"    Net Sharpe: {metrics['sharpe_net']:.3f}  "
                  f"Turnover: {metrics['avg_turnover']:.3f}  "
                  f"DirAcc: {metrics['dir_accuracy']:.3f}  "
                  f"MaxDD: {metrics['max_dd_net']:.4f}  "
                  f"Neutral: {metrics['pct_neutral']:.1%}")

    # ------------------------------------------------------------------
    # Experiment 2: Ensemble RF + XGBoost on best bucket target
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Experiment 2: RF + XGBoost Ensemble (bucketed)")
    print(f"{'='*70}")

    # Find best bucket target from experiment 1
    best_bucket = max(
        [k for k in all_results if "rf|" in k],
        key=lambda k: all_results[k]["metrics"]["sharpe_net"]
    )
    best_target = best_bucket.split("|")[1]
    print(f"\n  Best bucket target: {best_target}")

    rf_key = f"rf|{best_target}"
    xgb_key = f"xgb|{best_target}"

    rf_preds = all_results[rf_key]["predictions"]
    xgb_preds = all_results[xgb_key]["predictions"]
    rf_idx = all_results[rf_key]["indices"]
    xgb_idx = all_results[xgb_key]["indices"]

    # Align (should already be aligned but be safe)
    assert np.array_equal(rf_idx, xgb_idx), "Indices mismatch"
    actuals_pnl = y_pnl[rf_idx]

    # Ensemble strategies for classification
    ensemble_configs = {
        "majority_vote": None,
        "rf_if_disagree": None,
        "agree_only": None,
    }

    # 1. Majority vote: if both agree, use that. If disagree, use RF (it's stronger).
    majority = np.zeros(len(rf_preds))
    for i in range(len(rf_preds)):
        if rf_preds[i] == xgb_preds[i]:
            majority[i] = rf_preds[i]
        else:
            majority[i] = rf_preds[i]  # RF is stronger, trust it on disagreement
    ensemble_configs["majority_vote"] = majority

    # 2. RF unless XGBoost strongly disagrees (opposite sign)
    rf_unless = rf_preds.copy()
    for i in range(len(rf_preds)):
        if rf_preds[i] * xgb_preds[i] < 0:  # opposite signs
            rf_unless[i] = 0  # go neutral when they disagree
    ensemble_configs["rf_unless_disagree"] = rf_unless
    del ensemble_configs["rf_if_disagree"]

    # 3. Only trade when both agree on direction (not neutral)
    agree_only = np.zeros(len(rf_preds))
    for i in range(len(rf_preds)):
        if rf_preds[i] == xgb_preds[i] and rf_preds[i] != 0:
            agree_only[i] = rf_preds[i]
        elif rf_preds[i] != 0 and xgb_preds[i] == 0:
            agree_only[i] = rf_preds[i]  # one active, one neutral → follow active
        elif xgb_preds[i] != 0 and rf_preds[i] == 0:
            agree_only[i] = xgb_preds[i]
        # both neutral or opposite → stay at 0
    ensemble_configs["agree_only"] = agree_only

    print(f"\n  {'Config':<25} {'Net Sharpe':>12} {'Turnover':>10} {'DirAcc':>10} {'MaxDD':>10} {'%Neutral':>10}")
    print(f"  {'-'*77}")

    # Print individual models first
    for key in [rf_key, xgb_key]:
        m = all_results[key]["metrics"]
        label = key.replace("|", " | ")
        print(f"  {label:<25} {m['sharpe_net']:>12.3f} {m['avg_turnover']:>10.3f} "
              f"{m['dir_accuracy']:>10.3f} {m['max_dd_net']:>10.4f} {m['pct_neutral']:>10.1%}")

    # Print ensembles
    for ens_name, ens_preds in ensemble_configs.items():
        positions = preds_to_positions_bucketed(ens_preds)
        m = evaluate_positions(positions, actuals_pnl, cost_bps=COST_BPS)
        all_results[f"ens|{ens_name}"] = {"metrics": m, "positions": positions}
        print(f"  ENS:{ens_name:<20} {m['sharpe_net']:>12.3f} {m['avg_turnover']:>10.3f} "
              f"{m['dir_accuracy']:>10.3f} {m['max_dd_net']:>10.4f} {m['pct_neutral']:>10.1%}")

    print(f"\n  BENCHMARK: RF + target_bucket + curated_14 = 0.607 net Sharpe")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"quick_wins_{timeframe}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
