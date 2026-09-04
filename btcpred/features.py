"""Feature engineering on closed 15m candles. All features use only past/present bars."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of features aligned with df's index."""
    f = pd.DataFrame(index=df.index)
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    logret = np.log(c / c.shift(1))

    # Returns and lags
    f["ret_1"] = logret
    for k in range(2, 9):
        f[f"ret_lag{k}"] = logret.shift(k - 1)
    for n in (4, 8, 16, 32, 96):
        f[f"ret_{n}"] = np.log(c / c.shift(n))

    # Volatility
    for n in (8, 16, 48, 96):
        f[f"vol_{n}"] = logret.rolling(n).std()
    f["vol_ratio_8_96"] = f["vol_8"] / f["vol_96"]
    atr14 = _atr(df, 14)
    f["atr14_pct"] = atr14 / c

    # Candle anatomy (normalised by ATR)
    rng = (h - l).replace(0, np.nan)
    f["body"] = (c - o) / atr14
    f["body_frac"] = (c - o) / rng
    f["upper_wick_frac"] = (h - np.maximum(c, o)) / rng
    f["lower_wick_frac"] = (np.minimum(c, o) - l) / rng
    f["range_atr"] = (h - l) / atr14
    f["close_pos"] = (c - l) / rng  # where close sits in the bar
    f["up_streak"] = _streak(np.sign(c - o))

    # Trend / mean-reversion
    for n in (8, 20, 50, 200):
        ema = c.ewm(span=n, adjust=False).mean()
        f[f"ema{n}_dist"] = (c - ema) / atr14
    f["ema8_20"] = (c.ewm(span=8, adjust=False).mean() - c.ewm(span=20, adjust=False).mean()) / atr14
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd"] = macd / atr14
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / atr14
    f["rsi7"] = _rsi(c, 7)
    f["rsi14"] = _rsi(c, 14)
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = (c - sma20) / (2 * sd20)
    f["bb_width"] = (4 * sd20) / sma20
    f["dist_high_96"] = (h.rolling(96).max() - c) / atr14
    f["dist_low_96"] = (c - l.rolling(96).min()) / atr14

    # Volume / order-flow
    vol_ma = v.rolling(96).mean()
    f["vol_z"] = (v - vol_ma) / v.rolling(96).std()
    f["vol_rel_4"] = v.rolling(4).mean() / vol_ma
    taker_ratio = df["taker_buy_base"] / v.replace(0, np.nan)
    f["taker_ratio"] = taker_ratio
    f["taker_ratio_8"] = taker_ratio.rolling(8).mean()
    f["taker_delta_z"] = ((2 * df["taker_buy_base"] - v) / vol_ma)
    f["trades_z"] = (df["trades"] - df["trades"].rolling(96).mean()) / df["trades"].rolling(96).std()
    f["vwap_dist"] = (c - (df["quote_volume"].rolling(16).sum() / v.rolling(16).sum())) / atr14

    # Time of day / week (UTC), cyclical
    t = df["open_time"].dt
    minute_of_day = t.hour * 60 + t.minute
    f["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    f["dow_sin"] = np.sin(2 * np.pi * t.dayofweek / 7)
    f["dow_cos"] = np.cos(2 * np.pi * t.dayofweek / 7)

    return f.replace([np.inf, -np.inf], np.nan)


def _streak(sign: pd.Series) -> pd.Series:
    """Consecutive run length of same-sign candles (positive for up, negative for down)."""
    s = sign.values
    out = np.zeros(len(s))
    for i in range(1, len(s)):
        if s[i] != 0 and s[i] == s[i - 1]:
            out[i] = out[i - 1] + s[i]
        else:
            out[i] = s[i]
    return pd.Series(out, index=sign.index)


def build_labels(df: pd.DataFrame) -> pd.Series:
    """1 if the NEXT candle closes above its open (bullish), else 0."""
    nxt_close, nxt_open = df["close"].shift(-1), df["open"].shift(-1)
    label = (nxt_close > nxt_open).astype(float)
    label[nxt_close.isna()] = np.nan
    return label


def build_dataset(df: pd.DataFrame):
    X = build_features(df)
    y = build_labels(df)
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask].astype(int), df.loc[mask, "open_time"]
