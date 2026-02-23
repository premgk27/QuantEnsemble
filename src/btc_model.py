"""BTC Daily Model — RF + target_bucket_10 + forward-selected features.

Winning config from btc_feature_selection.py:
  - Target: target_bucket_10 (±1.0% threshold, 3-class)
  - Features: ema_ratio_12, atr_14, log_ret_20d, mfi_14 (forward-selected top-4)
  - Also tests: forward top-6, Gini top-8, equity curated, all features

Walk-forward CV (expanding window), 5bps transaction cost.
Data: Kraken BTCUSD daily (has volume).
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# Correlated pairs to drop
DROP_COLS = [
    "roc_10d", "cci_20", "adi", "close_sma20_ratio",
    "ret_std_20d", "ema_ratio_50",
]

# Feature sets to compare
FORWARD_4 = ["ema_ratio_12", "atr_14", "log_ret_20d", "mfi_14"]
FORWARD_5 = ["ema_ratio_12", "atr_14", "log_ret_20d", "mfi_14", "hash_rate_ratio"]  # +on-chain
FORWARD_6 = ["ema_ratio_12", "atr_14", "log_ret_20d", "mfi_14", "log_ret_1d", "vol_ratio"]
GINI_8 = ["bb_width", "intraday_range", "gk_vol", "log_ret_20d",
           "ema_ratio_12", "adx_14", "ema_ratio_26", "atr_14"]
EQUITY_CURATED = [
    "log_ret_5d", "log_ret_20d", "mom_divergence",
    "ema_ratio_12", "ema_ratio_26", "macd_hist",
    "rsi_14", "bb_width", "gk_vol", "vol_ratio",
    "vol_ratio_20d", "close_pos_range", "gap", "dow_sin",
]


# ---------------------------------------------------------------------------
# Model & evaluation
# ---------------------------------------------------------------------------
def make_rf_clf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=500, max_depth=5, max_features="sqrt",
            min_samples_leaf=50, random_state=42, n_jobs=-1,
        )),
    ])


def preds_to_positions(preds, flat_on_neutral=False):
    """Convert bucket predictions to positions.

    flat_on_neutral=False (default): neutral holds previous position.
    flat_on_neutral=True: neutral goes flat (position=0), reducing exposure.
    """
    if flat_on_neutral:
        return np.where(preds == 0, 0.0, preds.astype(float))

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

    # Profit factor
    profits = net_returns[net_returns > 0].sum()
    losses = abs(net_returns[net_returns < 0].sum())
    pf = profits / losses if losses > 0 else np.inf

    return {
        "sharpe_net": sharpe,
        "win_rate": win_rate,
        "annual_return_net": ann_ret,
        "total_return_net": total_ret,
        "avg_turnover": np.mean(turnover),
        "max_dd_net": max_dd,
        "profit_factor_net": pf,
    }


def monthly_yearly_breakdown(net_returns, dates, label=""):
    """Print monthly and yearly return breakdown."""
    df = pd.DataFrame({"net_ret": net_returns}, index=dates)
    df["year"] = df.index.year
    df["month"] = df.index.month

    monthly = df.groupby([df.index.year, df.index.month])["net_ret"].sum()
    monthly.index.names = ["year", "month"]
    monthly_df = monthly.reset_index()
    monthly_df.columns = ["year", "month", "ret"]

    n_months = len(monthly_df)
    pct_positive = (monthly_df["ret"] > 0).mean()
    avg_monthly = monthly_df["ret"].mean()
    median_monthly = monthly_df["ret"].median()
    best_month = monthly_df["ret"].max()
    worst_month = monthly_df["ret"].min()

    print(f"\n    Monthly stats ({label}):")
    print(f"      Months: {n_months}  |  Positive: {pct_positive:.1%}  |  "
          f"Avg: {avg_monthly:.4f}  |  Median: {median_monthly:.4f}")
    if worst_month != 0:
        print(f"      Best: {best_month:.4f}  |  Worst: {worst_month:.4f}  |  "
              f"Best/Worst ratio: {abs(best_month/worst_month):.2f}")

    yearly = df.groupby("year")["net_ret"].agg(["sum", "count"])
    yearly.columns = ["return", "days"]
    yearly["sharpe"] = df.groupby("year")["net_ret"].mean() / df.groupby("year")["net_ret"].std() * np.sqrt(252)

    monthly_wins = monthly_df.copy()
    monthly_wins["win"] = monthly_wins["ret"] > 0
    yearly_monthly_wr = monthly_wins.groupby("year")["win"].mean()

    # Cumulative return per year (simple, for dollar context)
    yearly["cum_simple"] = yearly["return"].apply(lambda x: (np.exp(x) - 1) * 100)

    print(f"\n    {'Year':<6} {'Return':>8} {'$(x100)':>8} {'Sharpe':>8} {'Mo WinR':>8} {'Days':>6}")
    print(f"    {'-'*48}")
    for year in sorted(yearly.index):
        yr = yearly.loc[year]
        mwr = yearly_monthly_wr.get(year, 0)
        print(f"    {year:<6} {yr['return']:>8.4f} {yr['cum_simple']:>7.1f}% "
              f"{yr['sharpe']:>8.2f} {mwr:>8.1%} {int(yr['days']):>6}")


def run_walk_forward(X, y, y_pnl, dates, flat_on_neutral=False):
    """Walk-forward CV, returns positions, actuals, indices, net_returns."""
    all_results = []
    t = INITIAL_TRAIN
    while t < len(X):
        test_end = min(t + STEP_SIZE, len(X))
        model = make_rf_clf()
        model.fit(X[:t], y[:t])
        preds = model.predict(X[t:test_end])
        for i, (p, a) in enumerate(zip(preds, y_pnl[t:test_end])):
            all_results.append((t + i, p, a))
        t = test_end

    indices = np.array([r[0] for r in all_results], dtype=int)
    preds = np.array([r[1] for r in all_results])
    actuals = np.array([r[2] for r in all_results])
    positions = preds_to_positions(preds, flat_on_neutral=flat_on_neutral)

    turnover = np.abs(np.diff(positions, prepend=0))
    costs = turnover * (COST_BPS / 10_000)
    net_rets = positions * actuals - costs

    return {
        "indices": indices,
        "preds": preds,
        "actuals": actuals,
        "positions": positions,
        "net_rets": net_rets,
        "dates": dates[indices],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "daily"
    include_onchain = "--onchain" in sys.argv

    print(f"\n{'#'*70}")
    print(f"  BTC MODEL — RF + target_bucket_10 ({timeframe}{'  +onchain' if include_onchain else ''})")
    print(f"{'#'*70}\n")

    # Load data
    path = BTC_4H if timeframe == "4h" else BTC_DAILY
    print(f"Loading BTC {timeframe} from {path}...")
    raw = load_ohlcv(path)
    print(f"  Raw: {len(raw)} rows, {raw.index[0]} to {raw.index[-1]}")

    # Load on-chain data if requested
    onchain_df = None
    if include_onchain:
        from onchain_loader import load_all_onchain
        onchain_df = load_all_onchain()

    # Build features
    print("Building features...")
    data = build_features(raw, timeframe=timeframe, onchain_df=onchain_df)

    # Add ±1.0% bucket target (winner from threshold sweep)
    data["target_bucket_10"] = pd.cut(
        data["target_ret"],
        bins=[-np.inf, -0.01, 0.01, np.inf],
        labels=[-1, 0, 1],
    ).astype(float)

    print(f"  {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"  {data.index[0]} to {data.index[-1]}")

    # All available features (after dropping correlated)
    all_feature_cols = [c for c in data.columns
                        if not c.startswith("target_") and c not in DROP_COLS]

    # Class distribution
    dist = data["target_bucket_10"].value_counts(normalize=True).sort_index()
    print(f"\n  Class distribution (±1.0%): Down={dist.get(-1,0):.1%}  "
          f"Neutral={dist.get(0,0):.1%}  Up={dist.get(1,0):.1%}")

    y_bucket = data["target_bucket_10"].values
    y_pnl = data["target_ret"].values
    dates = data.index

    # Feature sets to test
    feature_sets = {
        "forward_4": [f for f in FORWARD_4 if f in all_feature_cols],
        "forward_5_onchain": [f for f in FORWARD_5 if f in all_feature_cols],
        "forward_6": [f for f in FORWARD_6 if f in all_feature_cols],
        "gini_8": [f for f in GINI_8 if f in all_feature_cols],
        "equity_curated": [f for f in EQUITY_CURATED if f in all_feature_cols],
        "all_features": all_feature_cols,
    }

    all_results = {}

    # ======================================================================
    # Run each feature set (hold on neutral — original behaviour)
    # ======================================================================
    for set_name, feat_cols in feature_sets.items():
        print(f"\n{'='*70}")
        print(f"  {set_name} ({len(feat_cols)} features)")
        print(f"  {feat_cols}")
        print(f"{'='*70}")

        X = data[feat_cols].values
        wf = run_walk_forward(X, y_bucket, y_pnl, dates, flat_on_neutral=False)
        m = evaluate(wf["positions"], wf["actuals"])

        all_results[set_name] = {"metrics": m, "features": feat_cols}

        print(f"\n  S(n)={m['sharpe_net']:.3f}  WinRate={m['win_rate']:.1%}  "
              f"AnnRet={m['annual_return_net']:.1%}  TotRet={m['total_return_net']:.2f}  "
              f"Turn={m['avg_turnover']:.3f}  MaxDD={m['max_dd_net']:.4f}  "
              f"PF={m['profit_factor_net']:.2f}")

        monthly_yearly_breakdown(wf["net_rets"], wf["dates"], label=set_name)

    # ======================================================================
    # Neutral = FLAT comparison — forward_4 only
    # ======================================================================
    print(f"\n{'='*70}")
    print(f"  forward_4  [FLAT on neutral] — does going flat beat holding?")
    print(f"{'='*70}")
    X_f4 = data[[f for f in FORWARD_4 if f in all_feature_cols]].values
    wf_flat = run_walk_forward(X_f4, y_bucket, y_pnl, dates, flat_on_neutral=True)
    m_flat = evaluate(wf_flat["positions"], wf_flat["actuals"])
    all_results["forward_4_flat"] = {
        "metrics": m_flat,
        "features": [f for f in FORWARD_4 if f in all_feature_cols],
    }
    flat_time = np.mean(wf_flat["positions"] == 0)
    print(f"\n  S(n)={m_flat['sharpe_net']:.3f}  WinRate={m_flat['win_rate']:.1%}  "
          f"AnnRet={m_flat['annual_return_net']:.1%}  TotRet={m_flat['total_return_net']:.2f}  "
          f"Turn={m_flat['avg_turnover']:.3f}  MaxDD={m_flat['max_dd_net']:.4f}  "
          f"PF={m_flat['profit_factor_net']:.2f}  Flat={flat_time:.1%}")
    monthly_yearly_breakdown(wf_flat["net_rets"], wf_flat["dates"], label="forward_4_flat")

    # ======================================================================
    # Regime analysis on the best model
    # ======================================================================
    best_name = max(all_results, key=lambda k: all_results[k]["metrics"]["sharpe_net"])
    best_feats = all_results[best_name]["features"]
    print(f"\n{'='*70}")
    print(f"  REGIME ANALYSIS — {best_name} (best Sharpe)")
    print(f"{'='*70}")

    X_best = data[best_feats].values
    wf_best = run_walk_forward(X_best, y_bucket, y_pnl, dates)

    oos_df = pd.DataFrame({
        "position": wf_best["positions"],
        "actual": wf_best["actuals"],
        "net_ret": wf_best["net_rets"],
    }, index=wf_best["dates"])

    # Trailing 60-day return to classify regimes
    close_oos = data.loc[wf_best["dates"], "close"] if "close" in data.columns else None
    # Use cumulative actual returns as proxy
    oos_df["trail_60d"] = oos_df["actual"].rolling(60).sum()

    # Bull: trailing 60d > 10%, Bear: < -10%, Sideways: in between
    oos_df["regime"] = "sideways"
    oos_df.loc[oos_df["trail_60d"] > 0.10, "regime"] = "bull"
    oos_df.loc[oos_df["trail_60d"] < -0.10, "regime"] = "bear"

    for regime in ["bull", "bear", "sideways"]:
        subset = oos_df[oos_df["regime"] == regime]
        if len(subset) < 30:
            print(f"\n  {regime}: too few days ({len(subset)})")
            continue
        pos = subset["position"].values
        act = subset["actual"].values
        net = subset["net_ret"].values
        mean_r = np.mean(net)
        std_r = np.std(net, ddof=1)
        sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0
        wr = np.mean(net[pos != 0] > 0) if (pos != 0).sum() > 0 else 0
        total = np.sum(net)
        print(f"\n  {regime.upper()} ({len(subset)} days, {len(subset)/len(oos_df)*100:.1f}%):")
        print(f"    Sharpe={sharpe:.3f}  WinRate={wr:.1%}  TotalRet={total:.4f}  "
              f"AvgDailyRet={mean_r:.5f}")

    # Position distribution
    print(f"\n  Position distribution:")
    for pos_val, label in [(1.0, "Long"), (-1.0, "Short"), (0.0, "Flat")]:
        pct = np.mean(wf_best["positions"] == pos_val)
        print(f"    {label}: {pct:.1%}")

    # ======================================================================
    # Buy-and-hold comparison
    # ======================================================================
    print(f"\n{'='*70}")
    print(f"  BUY & HOLD COMPARISON")
    print(f"{'='*70}")

    bh_rets = wf_best["actuals"]  # next-bar returns over OOS period
    bh_cum = np.cumsum(bh_rets)
    bh_total = bh_cum[-1]
    bh_mean = np.mean(bh_rets)
    bh_std = np.std(bh_rets, ddof=1)
    bh_sharpe = (bh_mean / bh_std * np.sqrt(252)) if bh_std > 0 else 0
    bh_dd = np.max(np.maximum.accumulate(bh_cum) - bh_cum)
    n_years = len(bh_rets) / 252
    bh_ann = (1 + (np.exp(bh_total) - 1)) ** (1 / n_years) - 1 if n_years > 0 else 0

    m_best = all_results[best_name]["metrics"]
    print(f"\n  {'':20s} {'Model':>12} {'Buy&Hold':>12}")
    print(f"  {'-'*46}")
    print(f"  {'Sharpe':20s} {m_best['sharpe_net']:>12.3f} {bh_sharpe:>12.3f}")
    print(f"  {'Ann Return':20s} {m_best['annual_return_net']:>11.1%} {bh_ann:>11.1%}")
    print(f"  {'Total Return (log)':20s} {m_best['total_return_net']:>12.2f} {bh_total:>12.2f}")
    print(f"  {'Max Drawdown':20s} {m_best['max_dd_net']:>12.4f} {bh_dd:>12.4f}")
    print(f"  {'Win Rate':20s} {m_best['win_rate']:>11.1%} {'':>12}")

    # ======================================================================
    # Final summary
    # ======================================================================
    print(f"\n\n{'#'*80}")
    print(f"  FINAL SUMMARY — BTC RF Model Comparison")
    print(f"{'#'*80}")

    print(f"\n  {'FeatureSet':20s} {'N':>4} {'S(n)':>8} {'WinR':>8} {'AnnRet':>8} "
          f"{'TotRet':>8} {'Turn':>8} {'MaxDD':>8} {'PF':>6}")
    print(f"  {'-'*85}")
    for name in ["forward_4", "forward_4_flat", "forward_5_onchain", "forward_6", "gini_8", "equity_curated", "all_features"]:
        r = all_results[name]
        m = r["metrics"]
        print(f"  {name:20s} {len(r['features']):>4} {m['sharpe_net']:>8.3f} "
              f"{m['win_rate']:>8.1%} {m['annual_return_net']:>8.1%} "
              f"{m['total_return_net']:>8.2f} {m['avg_turnover']:>8.3f} "
              f"{m['max_dd_net']:>8.4f} {m['profit_factor_net']:>6.2f}")

    print(f"\n  Buy & Hold: Sharpe={bh_sharpe:.3f}  AnnRet={bh_ann:.1%}  MaxDD={bh_dd:.4f}")
    print(f"{'#'*80}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"btc_model_{timeframe}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
