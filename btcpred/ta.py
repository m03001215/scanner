"""Rule-based technical-analysis predictor.

Each signal votes +1 (bullish), -1 (bearish) or 0 (neutral) for the NEXT candle.
Votes are summed into a score; sign gives the direction, |score|/n_signals the confidence.
Everything is vectorised so the same code backtests over history and scores the live bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import _atr, _rsi


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _stoch(df: pd.DataFrame, n: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    lo, hi = df["low"].rolling(n).min(), df["high"].rolling(n).max()
    k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(d).mean()


def signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of per-signal votes (+1 / -1 / 0), one column per signal."""
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    s = pd.DataFrame(index=df.index)
    sgn = lambda x: np.sign(x).fillna(0)

    # --- Trend ---
    ema8, ema20, ema50, ema200 = _ema(c, 8), _ema(c, 20), _ema(c, 50), _ema(c, 200)
    s["ema_8_20"] = sgn(ema8 - ema20)
    s["ema_50_200"] = sgn(ema50 - ema200)
    s["price_vs_ema20"] = sgn(c - ema20)

    # --- Momentum ---
    macd = _ema(c, 12) - _ema(c, 26)
    hist = macd - _ema(macd, 9)
    s["macd_hist"] = sgn(hist)
    s["macd_hist_rising"] = sgn(hist - hist.shift(1))
    s["roc_4"] = sgn(c - c.shift(4))

    # --- Oscillators (mean reversion) ---
    rsi = _rsi(c, 14)
    s["rsi_extreme"] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
    k, d = _stoch(df)
    s["stoch_cross"] = np.where((k > d) & (k.shift(1) <= d.shift(1)), 1,
                        np.where((k < d) & (k.shift(1) >= d.shift(1)), -1, 0))
    s["stoch_extreme"] = np.where(k < 20, 1, np.where(k > 80, -1, 0))

    # --- Volatility bands ---
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    upper, lower = sma20 + 2 * sd20, sma20 - 2 * sd20
    s["bollinger"] = np.where(c < lower, 1, np.where(c > upper, -1, 0))

    # --- Volume / order flow ---
    vwap16 = df["quote_volume"].rolling(16).sum() / v.rolling(16).sum()
    s["vwap"] = sgn(c - vwap16)
    vol_ma = v.rolling(48).mean()
    high_vol = v > 1.5 * vol_ma
    s["volume_confirm"] = np.where(high_vol, sgn(c - o), 0)  # big-volume bar → continuation
    taker = df["taker_buy_base"] / v.replace(0, np.nan)
    s["taker_flow"] = np.where(taker > 0.55, 1, np.where(taker < 0.45, -1, 0))

    # --- Candlestick patterns ---
    body = c - o
    prev_body = body.shift(1)
    rng = (h - l).replace(0, np.nan)
    atr = _atr(df, 14)
    bull_engulf = (body > 0) & (prev_body < 0) & (c > o.shift(1)) & (o < c.shift(1))
    bear_engulf = (body < 0) & (prev_body > 0) & (c < o.shift(1)) & (o > c.shift(1))
    s["engulfing"] = np.where(bull_engulf, 1, np.where(bear_engulf, -1, 0))
    lower_wick = (np.minimum(c, o) - l) / rng
    upper_wick = (h - np.maximum(c, o)) / rng
    small_body = body.abs() / rng < 0.3
    s["hammer_star"] = np.where(small_body & (lower_wick > 0.6), 1,
                        np.where(small_body & (upper_wick > 0.6), -1, 0))
    three = sgn(body) + sgn(body.shift(1)) + sgn(body.shift(2))
    s["three_soldiers_crows"] = np.where(three == 3, 1, np.where(three == -3, -1, 0))

    # --- Support / resistance ---
    hi96, lo96 = h.rolling(96).max().shift(1), l.rolling(96).min().shift(1)
    s["breakout_96"] = np.where(c > hi96, 1, np.where(c < lo96, -1, 0))
    near_sup = (c - lo96) < 0.5 * atr
    near_res = (hi96 - c) < 0.5 * atr
    s["near_sr"] = np.where(near_sup & ~s["breakout_96"].astype(bool), 1,
                    np.where(near_res & ~s["breakout_96"].astype(bool), -1, 0))

    return s.astype(float)


def score(sig: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Aggregate votes into score, direction and confidence."""
    w = pd.Series(1.0, index=sig.columns) if weights is None else pd.Series(weights).reindex(sig.columns).fillna(0)
    raw = (sig * w).sum(axis=1)
    out = pd.DataFrame(index=sig.index)
    out["score"] = raw
    out["n_bull"] = (sig > 0).sum(axis=1)
    out["n_bear"] = (sig < 0).sum(axis=1)
    out["direction"] = np.where(raw > 0, 1, np.where(raw < 0, 0, np.nan))  # tie → no call
    out["confidence"] = raw.abs() / w.abs().sum()
    return out


def calibrate(sig: pd.DataFrame, labels: pd.Series, lo: float = 0.48, hi: float = 0.52) -> dict[str, float]:
    """Weight each signal by its hit rate on a calibration window.

    +1 if the signal is right > hi of the time, -1 if it is right < lo (i.e. its inverse works),
    0 otherwise. Trend-following signals on 15m BTC typically come out negative: the market mean-reverts.
    """
    w = {}
    for col in sig.columns:
        s = sig[col]
        m = (s != 0) & labels.notna()
        if m.sum() < 200:
            w[col] = 0.0
            continue
        acc = float(((s[m] > 0).astype(int) == labels[m]).mean())
        w[col] = 1.0 if acc > hi else (-1.0 if acc < lo else 0.0)
    return w


def _composite_metrics(agg: pd.DataFrame, y: pd.Series) -> dict:
    d = agg["direction"]
    called = d.notna()
    comp = dict(n=int(len(y)), n_called=int(called.sum()), coverage=float(called.mean()),
                accuracy=float((d[called] == y[called]).mean()) if called.any() else None)
    for thr in (0.2, 0.3, 0.4):
        m = called & (agg["confidence"] >= thr)
        comp[f"acc_conf_{int(thr*100)}"] = float((d[m] == y[m]).mean()) if m.sum() > 20 else None
        comp[f"coverage_conf_{int(thr*100)}"] = float(m.mean())
    return comp


def backtest(df: pd.DataFrame, labels: pd.Series, calib_frac: float = 0.4) -> dict:
    """Calibrate weights on the first calib_frac of history, evaluate on the rest.

    Reports the textbook equal-weight vote and the calibrated vote, both out-of-sample,
    plus per-signal hit rates on the calibration window and the test window.
    """
    sig = signals(df)
    valid = labels.notna() & sig.notna().all(axis=1)
    sig, y = sig[valid], labels[valid].astype(int)
    split = int(len(sig) * calib_frac)
    sig_tr, y_tr = sig.iloc[:split], y.iloc[:split]
    sig_te, y_te = sig.iloc[split:], y.iloc[split:]

    weights = calibrate(sig_tr, y_tr)
    per_signal = {}
    for col in sig.columns:
        row = {}
        for name, s_, y_ in (("calib", sig_tr[col], y_tr), ("test", sig_te[col], y_te)):
            m = s_ != 0
            row[f"acc_{name}"] = float(((s_[m] > 0).astype(int) == y_[m]).mean()) if m.sum() else None
            row[f"coverage_{name}"] = float(m.mean())
        row["weight"] = weights[col]
        per_signal[col] = row

    return dict(
        calib_start=str(df.loc[sig_tr.index[0], "open_time"]), test_start=str(df.loc[sig_te.index[0], "open_time"]),
        test_end=str(df.loc[sig_te.index[-1], "open_time"]),
        weights=weights,
        textbook_oos=_composite_metrics(score(sig_te), y_te),
        calibrated_oos=_composite_metrics(score(sig_te, weights), y_te),
        per_signal=per_signal,
    )


def predict_last(df: pd.DataFrame, weights: dict[str, float] | None = None) -> dict:
    sig = signals(df)
    agg = score(sig, weights)
    last, row = sig.iloc[-1], agg.iloc[-1]
    bull = [k for k, v in last.items() if v > 0]
    bear = [k for k, v in last.items() if v < 0]
    if np.isnan(row["direction"]):
        direction = "NEUTRAL"
    else:
        direction = "BULLISH" if row["direction"] == 1 else "BEARISH"
    return dict(score=float(row["score"]), n_bull=int(row["n_bull"]), n_bear=int(row["n_bear"]),
                confidence=float(row["confidence"]), prediction=direction,
                bullish_signals=bull, bearish_signals=bear)
