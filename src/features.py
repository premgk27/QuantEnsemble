"""Feature engineering for QuantEnsemble.

Computes all 30 features from raw OHLCV data (groups A-H per CLAUDE.md).
For 4h bars, day-based periods are translated to bar counts (6.5 bars/day).
"""

import numpy as np
import pandas as pd


def _bars_per_day(timeframe: str) -> float:
    """Number of bars per trading day for period translation."""
    if timeframe == "4h":
        return 6.5  # 6.5 hours of trading / 4h bars ≈ ~1.6, but TV uses calendar 4h blocks
    return 1.0  # daily


def _period(days: int, timeframe: str) -> int:
    """Translate a day-based period to bar count for the given timeframe.

    For 4h bars: approximate 6.5 trading hours / 4h = ~1.6 bars per day,
    but TradingView 4h bars span the full session so we get ~2 bars per
    regular trading day plus extended. Empirically ~2 bars/day for US equities.
    Using the ratio from CLAUDE.md: 14-day RSI -> 91 bars -> ~6.5 bars/day.
    """
    if timeframe == "4h":
        return max(1, round(days * 6.5))
    return days


# ---------------------------------------------------------------------------
# Group A: Returns & Momentum (8 features)
# ---------------------------------------------------------------------------

def compute_returns_momentum(df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
    """Log returns at multiple scales, lagged returns, ROC, momentum divergence."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    # Log returns at 5 scales
    for d in [1, 2, 5, 10, 20]:
        p = _period(d, timeframe)
        out[f"log_ret_{d}d"] = np.log(close / close.shift(p))

    # Lagged returns (t-1, t-2, t-3 are the 1-bar return shifted back)
    ret_1 = np.log(close / close.shift(1))
    out["lag_ret_1"] = ret_1.shift(1)
    out["lag_ret_2"] = ret_1.shift(2)
    out["lag_ret_3"] = ret_1.shift(3)

    # ROC (10-day)
    p10 = _period(10, timeframe)
    out["roc_10d"] = close.pct_change(p10)

    # Momentum divergence
    out["mom_divergence"] = out["log_ret_5d"] - out["log_ret_20d"]

    return out


# ---------------------------------------------------------------------------
# Group B: Trend (5 features)
# ---------------------------------------------------------------------------

def compute_trend(df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
    """EMA ratios, MACD histogram, ADX."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA ratios
    for span in [12, 26, 50]:
        p = _period(span, timeframe)
        ema = close.ewm(span=p, adjust=False).mean()
        out[f"ema_ratio_{span}"] = close / ema

    # MACD histogram
    p12 = _period(12, timeframe)
    p26 = _period(26, timeframe)
    p9 = _period(9, timeframe)
    ema_fast = close.ewm(span=p12, adjust=False).mean()
    ema_slow = close.ewm(span=p26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=p9, adjust=False).mean()
    out["macd_hist"] = macd_line - signal_line

    # ADX-14 (manual computation)
    p14 = _period(14, timeframe)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_smooth = tr.ewm(alpha=1/p14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/p14, adjust=False).mean() / atr_smooth
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/p14, adjust=False).mean() / atr_smooth

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    out["adx_14"] = dx.ewm(alpha=1/p14, adjust=False).mean()

    return out


# ---------------------------------------------------------------------------
# Group C: Mean Reversion / Oscillators (5 features)
# ---------------------------------------------------------------------------

def compute_oscillators(df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
    """RSI, Stochastic %K, Bollinger Band %B and width, CCI."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    p14 = _period(14, timeframe)
    p20 = _period(20, timeframe)

    # RSI-14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/p14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/p14, adjust=False).mean()
    rs = avg_gain / avg_loss
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # Stochastic %K (14-period)
    lowest_low = low.rolling(p14).min()
    highest_high = high.rolling(p14).max()
    out["stoch_k_14"] = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # Bollinger Bands (20, 2std)
    sma20 = close.rolling(p20).mean()
    std20 = close.rolling(p20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["bb_pctb"] = (close - lower) / (upper - lower)
    out["bb_width"] = (upper - lower) / sma20

    # CCI-20
    typical_price = (high + low + close) / 3
    tp_sma = typical_price.rolling(p20).mean()
    tp_mad = typical_price.rolling(p20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    out["cci_20"] = (typical_price - tp_sma) / (0.015 * tp_mad)

    return out


# ---------------------------------------------------------------------------
# Group D: Volatility (5 features)
# ---------------------------------------------------------------------------

def compute_volatility(df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
    """ATR, Garman-Klass vol, vol ratio, rolling std, intraday range."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    op = df["open"]

    p14 = _period(14, timeframe)
    p5 = _period(5, timeframe)
    p20 = _period(20, timeframe)

    # ATR-14
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = tr.ewm(alpha=1/p14, adjust=False).mean()

    # Garman-Klass volatility (20-period rolling)
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / op) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    out["gk_vol"] = gk.rolling(p20).mean().apply(np.sqrt)

    # Vol ratio: 5d realized vol / 20d realized vol
    log_ret = np.log(close / close.shift(1))
    vol_5 = log_ret.rolling(p5).std()
    vol_20 = log_ret.rolling(p20).std()
    out["vol_ratio"] = vol_5 / vol_20

    # Rolling std of returns (20d)
    out["ret_std_20d"] = log_ret.rolling(p20).std()

    # Intraday range ratio
    out["intraday_range"] = (high - low) / close

    return out


# ---------------------------------------------------------------------------
# Group E: Volume (4 features)
# ---------------------------------------------------------------------------

def compute_volume(df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
    """OBV, volume ratio, MFI, ADI (Accumulation/Distribution)."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    p14 = _period(14, timeframe)
    p20 = _period(20, timeframe)

    # OBV
    direction = np.sign(close.diff())
    out["obv"] = (direction * volume).cumsum()

    # Volume ratio (current / 20d average)
    out["vol_ratio_20d"] = volume / volume.rolling(p20).mean()

    # MFI-14 (Money Flow Index)
    typical_price = (high + low + close) / 3
    raw_mf = typical_price * volume
    pos_mf = raw_mf.where(typical_price > typical_price.shift(1), 0.0)
    neg_mf = raw_mf.where(typical_price < typical_price.shift(1), 0.0)
    pos_sum = pos_mf.rolling(p14).sum()
    neg_sum = neg_mf.rolling(p14).sum()
    mfi_ratio = pos_sum / neg_sum
    out["mfi_14"] = 100 - (100 / (1 + mfi_ratio))

    # ADI (Accumulation/Distribution Index)
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)  # handle zero-range bars
    out["adi"] = (clv * volume).cumsum()

    return out


# ---------------------------------------------------------------------------
# Group F: Price Structure (3 features)
# ---------------------------------------------------------------------------

def compute_price_structure(df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
    """Close position in range, gap, rolling mean ratio."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    op = df["open"]

    p20 = _period(20, timeframe)

    # Close position in range (0 = low, 1 = high)
    out["close_pos_range"] = (close - low) / (high - low)

    # Gap
    out["gap"] = (op - close.shift(1)) / close.shift(1)

    # Rolling mean ratio
    sma20 = close.rolling(p20).mean()
    out["close_sma20_ratio"] = close / sma20

    return out


# ---------------------------------------------------------------------------
# Group G: Regime Detection (3 features)
# ---------------------------------------------------------------------------

def compute_regime(df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
    """Rolling autocorrelation, skewness, kurtosis of returns."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    p20 = _period(20, timeframe)

    log_ret = np.log(close / close.shift(1))

    # Rolling autocorrelation (lag-1)
    out["autocorr_20"] = log_ret.rolling(p20).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )

    # Rolling skewness
    out["skew_20"] = log_ret.rolling(p20).skew()

    # Rolling kurtosis
    out["kurt_20"] = log_ret.rolling(p20).kurt()

    return out


# ---------------------------------------------------------------------------
# Group H: External Context (3 features) — optional
# ---------------------------------------------------------------------------

def compute_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Day of week sin/cos encoded (2 features)."""
    out = pd.DataFrame(index=df.index)
    dow = df.index.dayofweek  # 0=Monday, 4=Friday
    out["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    return out


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------

def compute_target(df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
    """Next-period return and its sign."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    out["target_ret"] = np.log(close.shift(-1) / close)
    out["target_sign"] = np.sign(out["target_ret"])
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
    """Compute all features + target from raw OHLCV.

    Args:
        df: DataFrame with columns: open, high, low, close, volume.
        timeframe: "daily" or "4h" — controls period translation.

    Returns:
        DataFrame with all features and target, warmup NaNs trimmed.
    """
    parts = [
        compute_returns_momentum(df, timeframe),
        compute_trend(df, timeframe),
        compute_oscillators(df, timeframe),
        compute_volatility(df, timeframe),
        compute_volume(df, timeframe),
        compute_price_structure(df, timeframe),
        compute_regime(df, timeframe),
        compute_day_of_week(df),
        compute_target(df, timeframe),
    ]

    result = pd.concat(parts, axis=1)

    # Trim warmup NaNs (keep only rows where all features are valid)
    feature_cols = [c for c in result.columns if not c.startswith("target_")]
    result = result.dropna(subset=feature_cols)

    # Drop last row (target is NaN because we shifted -1)
    result = result.iloc[:-1]

    return result


if __name__ == "__main__":
    from data_loader import load_spy_4h, load_spy_daily

    for name, loader, tf in [("daily", load_spy_daily, "daily"), ("4h", load_spy_4h, "4h")]:
        df = loader()
        features = build_features(df, timeframe=tf)

        print(f"\n=== SPY {name} features ===")
        print(f"Shape: {features.shape}")
        print(f"Date range: {features.index[0]} to {features.index[-1]}")
        print(f"Columns ({len(features.columns)}):")
        for col in features.columns:
            nulls = features[col].isnull().sum()
            print(f"  {col:25s} nulls={nulls}  mean={features[col].mean():.6f}")

        # Correlation check
        feature_cols = [c for c in features.columns if not c.startswith("target_")]
        corr = features[feature_cols].corr().abs()
        corr_vals = corr.to_numpy().copy()
        np.fill_diagonal(corr_vals, 0)
        corr = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)
        high_corr = []
        for i in range(len(corr)):
            for j in range(i+1, len(corr)):
                if corr.iloc[i, j] > 0.9:
                    high_corr.append((corr.index[i], corr.columns[j], corr.iloc[i, j]))
        if high_corr:
            print(f"\nHighly correlated pairs (>0.9):")
            for a, b, c in sorted(high_corr, key=lambda x: -x[2]):
                print(f"  {a} <-> {b}: {c:.3f}")
        else:
            print("\nNo feature pairs with correlation > 0.9")
