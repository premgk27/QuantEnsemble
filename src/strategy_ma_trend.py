"""MA Trend + Pullback Entry Strategy.

Two-stage approach:
  Stage 1: RF predicts SMA_20 direction over next 5 bars (target_ma_dir)
  Stage 2: Combine MA direction with price position relative to MA (ema_ratio)
           to generate entry signals — trade pullbacks in direction of trend.

Signal logic:
  - MA rising + price below/near MA  → strong long  (+1.0)
  - MA falling + price above/near MA → strong short (-1.0)
  - MA rising + price above MA       → weak long    (+0.5)
  - MA falling + price below MA      → weak short   (-0.5)
  - Flat (no clear MA direction)     → hold previous position

Does NOT modify train.py or the base RF model.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_spy_daily
from features import build_features

# Reuse config from train.py
INITIAL_TRAIN = 1250
STEP_SIZE = 20
COST_BPS = 5
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

DROP_COLS = [
    "roc_10d", "cci_20", "adi", "close_sma20_ratio",
    "ret_std_20d", "ema_ratio_50",
]


# ---------------------------------------------------------------------------
# Walk-forward CV (same as train.py)
# ---------------------------------------------------------------------------
def walk_forward_splits(n_samples: int, initial_train: int, step: int):
    t = initial_train
    while t < n_samples:
        test_end = min(t + step, n_samples)
        yield t, t, test_end
        t = test_end


# ---------------------------------------------------------------------------
# Stage 1: RF predicts MA direction
# ---------------------------------------------------------------------------
def train_ma_direction_rf(X, y_ma_dir):
    """Walk-forward CV: RF classifier predicting MA direction."""
    results = []

    for train_end, test_start, test_end in walk_forward_splits(len(X), INITIAL_TRAIN, STEP_SIZE):
        X_train, y_train = X[:train_end], y_ma_dir[:train_end]
        X_test, y_test = X[test_start:test_end], y_ma_dir[test_start:test_end]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=500,
                max_depth=5,
                max_features="sqrt",
                min_samples_leaf=50,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        # Get probability of positive class for confidence
        proba = model.predict_proba(X_test)
        # Class order from model
        classes = model.named_steps["rf"].classes_
        pos_idx = list(classes).index(1.0) if 1.0 in classes else -1
        confidence = proba[:, pos_idx] if pos_idx >= 0 else np.full(len(preds), 0.5)

        for i in range(len(preds)):
            results.append((test_start + i, preds[i], y_test[i], confidence[i]))

    return np.array(results)


# ---------------------------------------------------------------------------
# Stage 2: Combine MA direction + price position → positions
# ---------------------------------------------------------------------------
def generate_positions(ma_pred, ema_ratio, confidence=None):
    """Two-stage signal: MA direction + pullback entry.

    Args:
        ma_pred: Predicted MA direction (+1 or -1) from RF.
        ema_ratio: close / EMA_26 at each bar (from features).
        confidence: RF probability of positive class (optional).

    Returns:
        positions: Array of position sizes [-1, -0.5, 0, +0.5, +1].
    """
    positions = np.zeros(len(ma_pred))

    for i in range(len(ma_pred)):
        direction = ma_pred[i]  # +1 = MA rising, -1 = MA falling
        ratio = ema_ratio[i]    # >1 = price above MA, <1 = price below

        if direction > 0:  # MA predicted to rise
            if ratio < 1.0:
                # Pullback in uptrend → strong long
                positions[i] = 1.0
            else:
                # Already above MA → weak long
                positions[i] = 0.5
        elif direction < 0:  # MA predicted to fall
            if ratio > 1.0:
                # Bounce in downtrend → strong short
                positions[i] = -1.0
            else:
                # Already below MA → weak short
                positions[i] = -0.5
        else:
            # No direction → hold previous
            positions[i] = positions[i - 1] if i > 0 else 0.0

    return positions


# ---------------------------------------------------------------------------
# Evaluation (supports fractional positions)
# ---------------------------------------------------------------------------
def evaluate_strategy(positions, actuals, cost_bps=0):
    """Evaluate with fractional positions (0.5 = half size)."""
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
        profits = rets[rets > 0].sum()
        losses = abs(rets[rets < 0].sum())
        metrics[f"profit_factor_{label}"] = profits / losses if losses > 0 else np.inf
        cum = np.cumsum(rets)
        running_max = np.maximum.accumulate(cum)
        drawdown = running_max - cum
        metrics[f"max_dd_{label}"] = np.max(drawdown) if len(drawdown) > 0 else 0.0

    # Direction accuracy (did MA direction prediction match actual?)
    metrics["avg_turnover"] = np.mean(turnover)
    metrics["n_predictions"] = len(positions)
    metrics["pct_full_size"] = np.mean(np.abs(positions) == 1.0)
    metrics["pct_half_size"] = np.mean(np.abs(positions) == 0.5)
    metrics["pct_flat"] = np.mean(positions == 0.0)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "daily"

    print(f"=== MA Trend + Pullback Strategy ({timeframe}) ===\n")
    print("Loading data...")
    if timeframe == "4h":
        from data_loader import load_spy_4h
        df = load_spy_4h()
    else:
        df = load_spy_daily()
    data = build_features(df, timeframe=timeframe)

    feature_cols = [c for c in data.columns
                    if not c.startswith("target_") and c not in DROP_COLS]
    X = data[feature_cols].values
    y_ma_dir = data["target_ma_dir"].values
    y_ret = data["target_ret"].values  # 1-bar return for PnL
    ema_ratio = data["ema_ratio_26"].values
    dates = data.index

    print(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"MA direction class balance: +1={np.mean(y_ma_dir==1):.1%}, -1={np.mean(y_ma_dir==-1):.1%}, 0={np.mean(y_ma_dir==0):.1%}")

    # ------------------------------------------------------------------
    # Stage 1: RF predicts MA direction
    # ------------------------------------------------------------------
    print("\nStage 1: Training RF classifier on target_ma_dir...")
    results = train_ma_direction_rf(X, y_ma_dir)
    indices = results[:, 0].astype(int)
    ma_preds = results[:, 1]
    ma_actuals = results[:, 2]
    confidence = results[:, 3]

    ma_accuracy = np.mean(ma_preds == ma_actuals)
    print(f"  MA direction accuracy: {ma_accuracy:.3f}")
    print(f"  Pred distribution: +1={np.mean(ma_preds==1):.1%}, -1={np.mean(ma_preds==-1):.1%}")

    # ------------------------------------------------------------------
    # Stage 2: Combine with price position → positions
    # ------------------------------------------------------------------
    print("\nStage 2: Generating positions (MA direction + pullback entry)...")
    ema_at_idx = ema_ratio[indices]
    actuals_at_idx = y_ret[indices]

    # Strategy A: Two-stage (MA direction + pullback sizing)
    positions_2stage = generate_positions(ma_preds, ema_at_idx, confidence)

    # Strategy B: Just MA direction (baseline, full size always)
    positions_ma_only = ma_preds.copy()  # +1 or -1

    # Strategy C: Two-stage with EMA filter on positions (from Step 4.5)
    # Smooth the positions to reduce flipping
    positions_smoothed = positions_2stage.copy()
    for i in range(1, len(positions_smoothed)):
        positions_smoothed[i] = 0.5 * positions_smoothed[i] + 0.5 * positions_smoothed[i - 1]
    # Re-discretize: round to nearest 0.5
    positions_smoothed = np.round(positions_smoothed * 2) / 2

    # ------------------------------------------------------------------
    # Evaluate all variants
    # ------------------------------------------------------------------
    strategies = {
        "MA direction only": positions_ma_only,
        "MA + pullback entry": positions_2stage,
        "MA + pullback + smooth": positions_smoothed,
    }

    print(f"\n{'='*90}")
    print(f"  Results Comparison — MA Trend Strategy ({timeframe})")
    print(f"{'='*90}")
    cw = 12
    header = f"  {'Strategy':<25} {'Sharpe(g)':>{cw}} {'Sharpe(n)':>{cw}} {'Turnover':>{cw}} {'PF(net)':>{cw}} {'MaxDD(n)':>{cw}} {'%Full':>{cw}} {'%Half':>{cw}}"
    print(header)
    print(f"  {'-'*85}")

    all_metrics = {}
    for name, pos in strategies.items():
        m = evaluate_strategy(pos, actuals_at_idx, cost_bps=COST_BPS)
        all_metrics[name] = m
        print(f"  {name:<25} {m['sharpe_gross']:>{cw}.3f} {m['sharpe_net']:>{cw}.3f} "
              f"{m['avg_turnover']:>{cw}.3f} {m['profit_factor_net']:>{cw}.2f} "
              f"{m['max_dd_net']:>{cw}.4f} {m['pct_full_size']:>{cw}.1%} {m['pct_half_size']:>{cw}.1%}")

    print(f"{'='*90}")

    # Also show the base RF (target_ret) benchmark for reference
    print(f"\n  Reference: Base RF (target_ret) had net Sharpe 0.242 raw, 0.313 filtered")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_data = {
        "indices": indices,
        "ma_preds": ma_preds,
        "ma_actuals": ma_actuals,
        "confidence": confidence,
        "ema_ratio": ema_at_idx,
        "actuals": actuals_at_idx,
        "strategies": {name: {"positions": pos, "metrics": all_metrics[name]}
                       for name, pos in strategies.items()},
        "ma_accuracy": ma_accuracy,
    }
    out_path = OUTPUT_DIR / f"ma_trend_results_{timeframe}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
