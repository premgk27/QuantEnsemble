"""Multi-Asset Trend Following Model (Phase A / Thesis 2).

Pools SPY + QQQ + IWM + DIA into one training set.
All features are vol-normalized / scale-invariant (ratios, returns, indicators).
Tests multiple targets: target_bucket (best single-asset), target_ret5_volnorm (trend),
target_trend_persist, target_ret_risknorm.

Walk-forward CV on pooled data with proper time alignment:
  - At each fold, train on ALL assets up to time T, predict ALL assets at T+1..T+step.
  - No asset leakage — same time boundary for all assets.

Evaluates:
  1. Pooled model performance (overall + per-asset breakdown)
  2. Comparison with single-asset models (does pooling help?)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from data_loader import load_ohlcv, DATA_DIR
from features import build_features

INITIAL_TRAIN = 1250  # per-asset equivalent ~5 years, but pooled this is ~1250 total rows
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

ASSETS = {
    "SPY": DATA_DIR / "BATS_SPY, 1D.csv",
    "QQQ": DATA_DIR / "BATS_QQQ, 1D.csv",
    "IWM": DATA_DIR / "BATS_IWM, 1D.csv",
    "DIA": DATA_DIR / "BATS_DIA, 1D.csv",
    "TLT": DATA_DIR / "BATS_TLT, 1D.csv",
    "GLD": DATA_DIR / "BATS_GLD, 1D.csv",
}


# ---------------------------------------------------------------------------
# Data loading & pooling
# ---------------------------------------------------------------------------
def load_all_assets(timeframe="daily"):
    """Load and compute features for all assets. Returns dict of DataFrames."""
    asset_data = {}
    for ticker, path in ASSETS.items():
        print(f"  Loading {ticker}...")
        df = load_ohlcv(path)
        data = build_features(df, timeframe=timeframe)
        data["asset"] = ticker
        asset_data[ticker] = data
        print(f"    {ticker}: {len(data)} rows, {data.index[0].date()} to {data.index[-1].date()}")
    return asset_data


def pool_assets(asset_data, feature_cols):
    """Stack all assets into one DataFrame, sorted by time.

    Each row keeps its asset label for per-asset evaluation.
    """
    frames = []
    for ticker, data in asset_data.items():
        # Only keep columns that exist in this asset's data
        available = [c for c in feature_cols if c in data.columns]
        target_cols = [c for c in data.columns if c.startswith("target_")]
        subset = data[available + target_cols + ["asset"]].copy()
        frames.append(subset)

    pooled = pd.concat(frames, axis=0).sort_index()

    # Drop rows with NaN in any feature
    pooled = pooled.dropna(subset=feature_cols)

    return pooled


# ---------------------------------------------------------------------------
# Walk-forward with time-based splits (no asset leakage)
# ---------------------------------------------------------------------------
def walk_forward_time_splits(dates, initial_train, step):
    """Yield train/test masks based on time, not row index.

    Critical: all assets share the same time boundary.
    Train on everything before date T, test on next `step` trading days.
    """
    unique_dates = np.sort(dates.unique())
    t = initial_train
    while t < len(unique_dates):
        test_end = min(t + step, len(unique_dates))
        train_cutoff = unique_dates[t - 1]  # last training date
        test_start_date = unique_dates[t]
        test_end_date = unique_dates[test_end - 1]

        train_mask = dates <= train_cutoff
        test_mask = (dates >= test_start_date) & (dates <= test_end_date)

        if test_mask.sum() > 0:
            yield train_mask, test_mask

        t = test_end


# ---------------------------------------------------------------------------
# Position generation
# ---------------------------------------------------------------------------
def preds_to_positions_bucketed(preds):
    """Bucketed classifier: -1/0/+1. Neutral holds previous."""
    positions = np.zeros(len(preds))
    positions[0] = preds[0]
    for i in range(1, len(preds)):
        if preds[i] != 0:
            positions[i] = preds[i]
        else:
            positions[i] = positions[i - 1]
    return positions


def preds_to_positions_regression(preds):
    return np.sign(preds)


def preds_to_positions_hold5(preds, hold_period=5):
    positions = np.zeros(len(preds))
    for i in range(len(preds)):
        if i % hold_period == 0:
            positions[i] = np.sign(preds[i])
        else:
            positions[i] = positions[i - 1]
    return positions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
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

    # Win rate: % of days where net return > 0 (only counting days with a position)
    active_mask = positions != 0
    if active_mask.sum() > 0:
        active_net = net_returns[active_mask]
        metrics["win_rate"] = np.sum(active_net > 0) / len(active_net)
    else:
        metrics["win_rate"] = 0.0

    # Cumulative and annualized returns
    cum_net = np.cumsum(net_returns)
    metrics["total_return_net"] = cum_net[-1] if len(cum_net) > 0 else 0.0
    n_years = len(net_returns) / 252
    if n_years > 0 and metrics["total_return_net"] > -1:
        # Convert log return sum to simple return, then annualize
        total_simple = np.exp(metrics["total_return_net"]) - 1
        metrics["annual_return_net"] = (1 + total_simple) ** (1 / n_years) - 1
    else:
        metrics["annual_return_net"] = 0.0

    return metrics


def monthly_yearly_breakdown(net_returns, dates, label=""):
    """Print monthly and yearly return breakdown for consistency analysis."""
    df = pd.DataFrame({"net_ret": net_returns}, index=dates)
    df["year"] = df.index.year
    df["month"] = df.index.month

    # Monthly aggregation
    monthly = df.groupby([df.index.year, df.index.month])["net_ret"].sum()
    monthly.index.names = ["year", "month"]
    monthly_df = monthly.reset_index()
    monthly_df.columns = ["year", "month", "ret"]

    # Monthly stats
    n_months = len(monthly_df)
    pct_positive = (monthly_df["ret"] > 0).mean()
    avg_monthly = monthly_df["ret"].mean()
    median_monthly = monthly_df["ret"].median()
    best_month = monthly_df["ret"].max()
    worst_month = monthly_df["ret"].min()

    print(f"\n    Monthly stats ({label}):")
    print(f"      Months: {n_months}  |  Positive: {pct_positive:.1%}  |  "
          f"Avg: {avg_monthly:.4f}  |  Median: {median_monthly:.4f}")
    print(f"      Best: {best_month:.4f}  |  Worst: {worst_month:.4f}  |  "
          f"Best/Worst ratio: {abs(best_month/worst_month):.2f}" if worst_month != 0 else "")

    # Yearly returns
    yearly = df.groupby("year")["net_ret"].agg(["sum", "count"])
    yearly.columns = ["return", "days"]
    yearly["sharpe"] = df.groupby("year")["net_ret"].mean() / df.groupby("year")["net_ret"].std() * np.sqrt(252)

    # Monthly win rate per year
    monthly_wins = monthly_df.copy()
    monthly_wins["win"] = monthly_wins["ret"] > 0
    yearly_monthly_wr = monthly_wins.groupby("year")["win"].mean()

    print(f"\n    {'Year':<6} {'Return':>8} {'Sharpe':>8} {'Mo WinR':>8} {'Days':>6}")
    print(f"    {'-'*40}")
    for year in sorted(yearly.index):
        yr = yearly.loc[year]
        mwr = yearly_monthly_wr.get(year, 0)
        print(f"    {year:<6} {yr['return']:>8.4f} {yr['sharpe']:>8.2f} "
              f"{mwr:>8.1%} {int(yr['days']):>6}")


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def make_rf_clf(balanced=False):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=500, max_depth=5, max_features="sqrt",
            min_samples_leaf=50, random_state=42, n_jobs=-1,
            class_weight="balanced" if balanced else None,
        )),
    ])


def make_rf_reg():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=500, max_depth=5, max_features="sqrt",
            min_samples_leaf=50, random_state=42, n_jobs=-1,
        )),
    ])


def make_xgb_clf():
    return XGBClassifier(
        max_depth=2, learning_rate=0.02, n_estimators=300,
        subsample=0.7, colsample_bytree=0.5, min_child_weight=50,
        reg_alpha=1.0, reg_lambda=5.0, gamma=0.1,
        random_state=42, verbosity=0, eval_metric="mlogloss",
    )


def make_xgb_reg():
    return XGBRegressor(
        max_depth=2, learning_rate=0.02, n_estimators=300,
        subsample=0.7, colsample_bytree=0.5, min_child_weight=50,
        reg_alpha=1.0, reg_lambda=5.0, gamma=0.1,
        random_state=42, verbosity=0, eval_metric="rmse",
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_pooled_model(pooled, feature_cols, target_name, model_type,
                       is_classifier=False, hold_period=1):
    """Walk-forward CV on pooled multi-asset data.

    Returns per-row results with asset labels for breakdown.
    """
    X = pooled[feature_cols].values
    y = pooled[target_name].values
    y_pnl = pooled["target_ret"].values
    assets = pooled["asset"].values
    dates = pooled.index

    # Remap labels for XGBoost classifier
    remap = None
    remap_inv = None
    if is_classifier and model_type == "xgb":
        unique_labels = np.sort(np.unique(y[~np.isnan(y)]))
        remap = {orig: i for i, orig in enumerate(unique_labels)}
        remap_inv = {i: orig for orig, i in remap.items()}
        y_mapped = np.array([remap.get(v, v) for v in y])
    else:
        y_mapped = y

    results = []  # (row_idx, pred, actual_target, actual_pnl, asset)
    n_folds = 0

    for train_mask, test_mask in walk_forward_time_splits(dates, INITIAL_TRAIN, STEP_SIZE):
        X_train, y_train = X[train_mask], y_mapped[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        y_pnl_test = y_pnl[test_mask]
        assets_test = assets[test_mask]
        test_indices = np.where(test_mask)[0]

        if model_type == "rf":
            model = make_rf_clf() if is_classifier else make_rf_reg()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        elif model_type == "xgb":
            model = make_xgb_clf() if is_classifier else make_xgb_reg()
            val_size = max(1, int(len(X_train) * 0.2))
            X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
            y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_test)
            if remap_inv is not None:
                preds = np.array([remap_inv.get(p, p) for p in preds])

        for i in range(len(preds)):
            results.append((test_indices[i], preds[i], y_test[i], y_pnl_test[i], assets_test[i]))

        n_folds += 1

    print(f"    {n_folds} folds, {len(results)} OOS predictions")

    arr = np.array(results, dtype=object)
    return {
        "indices": np.array(arr[:, 0], dtype=int),
        "predictions": np.array(arr[:, 1], dtype=float),
        "actuals_target": np.array(arr[:, 2], dtype=float),
        "actuals_pnl": np.array(arr[:, 3], dtype=float),
        "assets": np.array(arr[:, 4], dtype=str),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TARGET_CONFIGS = {
    "target_bucket": {
        "desc": "3-class bucketed (best single-asset)",
        "is_classifier": True,
        "hold_period": 1,
        "pos_fn": preds_to_positions_bucketed,
    },
    "target_ret5_volnorm": {
        "desc": "5d fwd return / vol (trend)",
        "is_classifier": False,
        "hold_period": 5,
        "pos_fn": lambda p: preds_to_positions_hold5(p, 5),
    },
    "target_ret_risknorm": {
        "desc": "1-bar risk-adjusted",
        "is_classifier": False,
        "hold_period": 1,
        "pos_fn": preds_to_positions_regression,
    },
    "target_trend_persist": {
        "desc": "trend persistence",
        "is_classifier": False,
        "hold_period": 5,
        "pos_fn": lambda p: preds_to_positions_hold5(p, 5),
    },
}


def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "daily"

    print(f"\n{'#'*70}")
    print(f"  MULTI-ASSET TREND MODEL ({timeframe})")
    print(f"{'#'*70}\n")

    # Load all assets
    print("Loading assets...")
    asset_data = load_all_assets(timeframe)

    # Build feature columns (curated 14)
    sample_data = list(asset_data.values())[0]
    all_feature_cols = [c for c in sample_data.columns
                        if not c.startswith("target_") and c != "asset" and c not in DROP_COLS]
    feature_cols = [f for f in CURATED_14 if f in all_feature_cols]
    print(f"\nUsing {len(feature_cols)} curated features: {feature_cols}")

    all_results = {}

    # ------------------------------------------------------------------
    # Single-asset RF on target_bucket (the winning recipe)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Single-Asset RF + target_bucket (curated_14)")
    print(f"{'='*70}")

    for ticker, data in asset_data.items():
        avail_cols = [f for f in feature_cols if f in data.columns]
        X_single = data[avail_cols].values
        y_single = data["target_bucket"].values
        y_pnl_single = data["target_ret"].values
        dates_single = data.index

        # Simple walk-forward (expanding window)
        results_single = []
        t = INITIAL_TRAIN
        while t < len(X_single):
            test_end = min(t + STEP_SIZE, len(X_single))
            X_tr, y_tr = X_single[:t], y_single[:t]
            X_te, y_te = X_single[t:test_end], y_single[t:test_end]

            model = make_rf_clf()
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            for i, (p, a) in enumerate(zip(preds, y_pnl_single[t:test_end])):
                results_single.append((t + i, p, a))
            t = test_end

        indices_s = np.array([r[0] for r in results_single], dtype=int)
        preds_s = np.array([r[1] for r in results_single])
        actuals_s = np.array([r[2] for r in results_single])
        pos_s = preds_to_positions_bucketed(preds_s)
        m = evaluate_positions(pos_s, actuals_s, cost_bps=COST_BPS)
        all_results[f"single|rf|{ticker}|bucket"] = {"metrics": m}
        print(f"\n  {ticker}: S(n)={m['sharpe_net']:.3f}  WinRate={m['win_rate']:.1%}  "
              f"AnnRet={m['annual_return_net']:.1%}  TotRet={m['total_return_net']:.2f}  "
              f"Turn={m['avg_turnover']:.3f}  MaxDD={m['max_dd_net']:.4f}")

        # Monthly/yearly breakdown
        oos_dates = dates_single[indices_s]
        turnover_s = np.abs(np.diff(pos_s, prepend=0))
        costs_s = turnover_s * (COST_BPS / 10_000)
        net_rets_s = pos_s * actuals_s - costs_s
        monthly_yearly_breakdown(net_rets_s, oos_dates, label=ticker)

    # ------------------------------------------------------------------
    # Probability threshold: only trade when RF is confident
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Probability Thresholds: trade only when P(class) > threshold")
    print(f"{'='*70}")

    # RF classes are [-1, 0, 1] — predict_proba gives [P(-1), P(0), P(+1)]
    THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]

    for ticker, data in asset_data.items():
        avail_cols = [f for f in feature_cols if f in data.columns]
        X_prob = data[avail_cols].values
        y_prob = data["target_bucket"].values
        y_pnl_prob = data["target_ret"].values

        # Walk-forward: collect probabilities
        all_probs = []  # (idx, prob_down, prob_neutral, prob_up, actual_pnl)
        t = INITIAL_TRAIN
        while t < len(X_prob):
            test_end = min(t + STEP_SIZE, len(X_prob))
            model = make_rf_clf()
            model.fit(X_prob[:t], y_prob[:t])
            probs = model.predict_proba(X_prob[t:test_end])
            # Ensure class order is [-1, 0, 1]
            classes = model.named_steps["rf"].classes_
            class_order = {c: i for i, c in enumerate(classes)}
            for i in range(len(probs)):
                p_down = probs[i][class_order[-1]] if -1 in class_order else 0
                p_neut = probs[i][class_order[0]] if 0 in class_order else 0
                p_up = probs[i][class_order[1]] if 1 in class_order else 0
                all_probs.append((t + i, p_down, p_neut, p_up, y_pnl_prob[t + i]))
            t = test_end

        probs_arr = np.array(all_probs)
        p_down = probs_arr[:, 1]
        p_neut = probs_arr[:, 2]
        p_up = probs_arr[:, 3]
        actuals = probs_arr[:, 4]

        cur_sharpe = all_results.get(f"single|rf|{ticker}|bucket", {}).get("metrics", {}).get("sharpe_net", 0)
        print(f"\n  {ticker} (default S(n)={cur_sharpe:.3f}):")
        print(f"    {'Thresh':>8} {'S(n)':>8} {'WinR':>8} {'AnnRet':>8} {'Turn':>8} {'MaxDD':>8} {'Trades%':>8}")

        for thresh in THRESHOLDS:
            # Position: +1 if P(up) > thresh, -1 if P(down) > thresh, else 0
            positions = np.zeros(len(p_up))
            for i in range(len(positions)):
                if p_up[i] > thresh:
                    positions[i] = 1.0
                elif p_down[i] > thresh:
                    positions[i] = -1.0
                else:
                    positions[i] = 0.0  # not confident enough

            # For neutral predictions, hold previous position
            for i in range(1, len(positions)):
                if positions[i] == 0:
                    positions[i] = positions[i - 1]

            m_prob = evaluate_positions(positions, actuals, cost_bps=COST_BPS)
            pct_active = np.mean(positions != 0) * 100
            key = f"prob|rf|{ticker}|t{int(thresh*100)}"
            all_results[key] = {"metrics": m_prob}

            marker = " ***" if m_prob["sharpe_net"] > cur_sharpe else ""
            print(f"    {thresh:>8.2f} {m_prob['sharpe_net']:>8.3f} "
                  f"{m_prob['win_rate']:>8.1%} {m_prob['annual_return_net']:>8.1%} "
                  f"{m_prob['avg_turnover']:>8.3f} {m_prob['max_dd_net']:>8.4f} "
                  f"{pct_active:>7.1f}%{marker}")

    # Best threshold per asset
    print(f"\n  {'='*70}")
    print(f"  Best probability threshold per asset:")
    print(f"  {'Asset':<6} {'Default':>10} {'Best Prob':>10} {'Thresh':>8} {'Delta':>8}")
    print(f"  {'-'*50}")
    for ticker in ASSETS:
        cur = all_results.get(f"single|rf|{ticker}|bucket", {}).get("metrics", {}).get("sharpe_net", 0)
        best_sharpe = cur
        best_thresh = "none"
        for thresh in THRESHOLDS:
            key = f"prob|rf|{ticker}|t{int(thresh*100)}"
            s = all_results.get(key, {}).get("metrics", {}).get("sharpe_net", 0)
            if s > best_sharpe:
                best_sharpe = s
                best_thresh = f"{thresh:.2f}"
        delta = best_sharpe - cur
        print(f"  {ticker:<6} {cur:>10.3f} {best_sharpe:>10.3f} {best_thresh:>8} {delta:>+8.3f}")

    # ------------------------------------------------------------------
    # Portfolio combination: equal-weight across single-asset models
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Portfolio: Equal-Weight Combination of Single-Asset Models")
    print(f"{'='*70}")

    # Collect per-asset daily net returns aligned by date
    asset_daily_rets = {}
    for ticker, data in asset_data.items():
        avail_cols = [f for f in feature_cols if f in data.columns]
        X_single = data[avail_cols].values
        y_single = data["target_bucket"].values
        y_pnl_single = data["target_ret"].values
        dates_single = data.index

        results_single = []
        t = INITIAL_TRAIN
        while t < len(X_single):
            test_end = min(t + STEP_SIZE, len(X_single))
            model = make_rf_clf()
            model.fit(X_single[:t], y_single[:t])
            preds = model.predict(X_single[t:test_end])
            for i, (p, a) in enumerate(zip(preds, y_pnl_single[t:test_end])):
                results_single.append((t + i, p, a))
            t = test_end

        indices_s = np.array([r[0] for r in results_single], dtype=int)
        preds_s = np.array([r[1] for r in results_single])
        actuals_s = np.array([r[2] for r in results_single])
        pos_s = preds_to_positions_bucketed(preds_s)
        turnover_s = np.abs(np.diff(pos_s, prepend=0))
        costs_s = turnover_s * (COST_BPS / 10_000)
        net_rets_s = pos_s * actuals_s - costs_s
        oos_dates = dates_single[indices_s]
        asset_daily_rets[ticker] = pd.Series(net_rets_s, index=oos_dates, name=ticker)

    # Align all assets on common dates and average
    rets_df = pd.DataFrame(asset_daily_rets)
    # Equal weight: average return across available assets each day
    portfolio_ret = rets_df.mean(axis=1).dropna()

    # Portfolio metrics
    port_mean = portfolio_ret.mean()
    port_std = portfolio_ret.std(ddof=1)
    port_sharpe = (port_mean / port_std * np.sqrt(252)) if port_std > 0 else 0
    port_cum = portfolio_ret.cumsum()
    port_total = port_cum.iloc[-1]
    port_dd = (port_cum.cummax() - port_cum).max()
    port_win = (portfolio_ret > 0).mean()
    n_years = len(portfolio_ret) / 252
    port_ann = (1 + (np.exp(port_total) - 1)) ** (1 / n_years) - 1 if n_years > 0 else 0

    print(f"\n  Equal-weight portfolio ({len(rets_df.columns)} assets, common dates only):")
    print(f"    Sharpe(net): {port_sharpe:.3f}  WinRate: {port_win:.1%}  "
          f"AnnRet: {port_ann:.1%}  TotRet: {port_total:.2f}  MaxDD: {port_dd:.4f}")
    print(f"    Dates: {portfolio_ret.index[0].date()} to {portfolio_ret.index[-1].date()}  "
          f"Days: {len(portfolio_ret)}")

    # Correlation of per-asset returns
    common = rets_df.dropna()
    if len(common) > 50:
        corr = common.corr()
        print(f"\n    Return correlation matrix:")
        print(f"    {'':>6}", end="")
        for c in corr.columns:
            print(f" {c:>6}", end="")
        print()
        for idx_name in corr.index:
            print(f"    {idx_name:>6}", end="")
            for c in corr.columns:
                print(f" {corr.loc[idx_name, c]:>6.3f}", end="")
            print()

    # Diversification ratio: portfolio Sharpe / avg individual Sharpe
    individual_sharpes = []
    for ticker in rets_df.columns:
        s = rets_df[ticker].dropna()
        if len(s) > 50:
            ind_sharpe = s.mean() / s.std(ddof=1) * np.sqrt(252)
            individual_sharpes.append(ind_sharpe)
    avg_ind_sharpe = np.mean(individual_sharpes) if individual_sharpes else 0
    div_ratio = port_sharpe / avg_ind_sharpe if avg_ind_sharpe > 0 else 0
    print(f"\n    Avg individual Sharpe: {avg_ind_sharpe:.3f}  |  Portfolio Sharpe: {port_sharpe:.3f}  |  "
          f"Diversification ratio: {div_ratio:.2f}x")

    monthly_yearly_breakdown(portfolio_ret.values, portfolio_ret.index, label="Portfolio")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n\n{'#'*90}")
    print(f"  FINAL SUMMARY — Single-Asset RF + target_bucket (curated_14)")
    print(f"{'#'*90}")
    cw = 9

    print(f"\n  {'Asset':<8} {'S(n)':>{cw}} {'WinRate':>{cw}} "
          f"{'AnnRet':>{cw}} {'TotRet':>{cw}} {'Turn':>{cw}} {'MaxDD':>{cw}}")
    print(f"  {'-'*70}")
    for key in sorted(all_results.keys()):
        if not key.startswith("single|"):
            continue
        ticker = key.split("|")[2]
        m = all_results[key]["metrics"]
        print(f"  {ticker:<8} "
              f"{m['sharpe_net']:>{cw}.3f} {m['win_rate']:>{cw}.1%} "
              f"{m['annual_return_net']:>{cw}.1%} {m['total_return_net']:>{cw}.2f} "
              f"{m['avg_turnover']:>{cw}.3f} {m['max_dd_net']:>{cw}.4f}")

    print(f"\n  BENCHMARK: SPY-only RF + target_bucket + curated_14 = 0.607 net Sharpe")
    print(f"{'#'*90}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"multi_asset_trend_{timeframe}.pkl"
    # Don't save wf_data to keep file size reasonable
    save_data = {}
    for key, val in all_results.items():
        if "wf_data" in val:
            save_data[key] = {k: v for k, v in val.items() if k != "wf_data"}
        else:
            save_data[key] = val
    with open(out_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
