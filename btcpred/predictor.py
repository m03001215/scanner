"""Shared prediction entry point used by the CLI and the web app."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, features, model, ta

ROOT = Path(__file__).resolve().parent.parent
INTERVALS = ("15m", "1h")
# Days of history to download when no cache exists (features need ~300 bars of warm-up
# plus the `recent` window). Override with BTCPRED_BOOT_DAYS.
_BOOT_DAYS = {"15m": 45, "1h": 150}

_CACHE: dict = {}
_MODELS: dict = {}


def cache_path(interval: str) -> Path:
    return ROOT / "data" / f"btcusdt_{interval}.parquet"


def model_dir(interval: str) -> Path:
    return ROOT / "models" / interval / "lgbm"


def ta_report_path(interval: str) -> Path:
    return ROOT / "models" / interval / "ta_report.json"


def boot_days(interval: str) -> int:
    env = os.environ.get("BTCPRED_BOOT_DAYS")
    return int(env) if env else _BOOT_DAYS.get(interval, 60)


def _check_interval(interval: str) -> str:
    if interval not in INTERVALS:
        raise ValueError(f"unsupported interval {interval!r}; choose one of {INTERVALS}")
    return interval


def _load_model(interval: str):
    if interval not in _MODELS:
        d = model_dir(interval)
        if not (d / "model.txt").exists():
            raise FileNotFoundError(f"no trained {interval} model; run `python main.py train --interval {interval}` first")
        _MODELS[interval] = model.load(d)
    return _MODELS[interval]


def _ta_report(interval: str) -> dict | None:
    p = ta_report_path(interval)
    return json.loads(p.read_text()) if p.exists() else None


def predict(interval: str = "15m", recent: int = 96, refresh: bool = True) -> dict:
    """Return ML + TA predictions for the next candle of `interval`, plus recent history.

    Result is cached per (interval, last-closed-candle, recent) so repeated calls within the
    same candle only hit Binance for a cheap 1-2 row update.
    """
    interval = _check_interval(interval)
    booster, meta = _load_model(interval)
    cache = cache_path(interval)
    df = (data.load_or_update(cache, interval=interval, days=boot_days(interval))
          if refresh else pd.read_parquet(cache))
    df = data.drop_open_candle(df)
    last_bar = df.iloc[-1]
    key = str(last_bar["open_time"])
    ck = (interval, key, recent)
    if ck in _CACHE:
        return _CACHE[ck]

    X = features.build_features(df)[meta["features"]]
    tail = X.iloc[-recent:]
    if tail.iloc[[-1]].isna().any(axis=1).iloc[0]:
        raise ValueError("not enough history to compute features")
    p_hist = booster.predict(tail.fillna(0))
    p = float(p_hist[-1])

    ta_rep = _ta_report(interval)
    weights = ta_rep["weights"] if ta_rep else None
    sig = ta.signals(df)
    agg_w = ta.score(sig, weights)
    agg_book = ta.score(sig)
    ta_res = ta.predict_last(df, weights)
    ta_book = ta.predict_last(df)

    ml_dir = "BULLISH" if p >= 0.5 else "BEARISH"
    step = pd.Timedelta(milliseconds=data.INTERVAL_MS[interval])
    next_open = (last_bar["close_time"] + pd.Timedelta(milliseconds=1)).floor(step)
    next_close = next_open + step

    # Recent history: candles + what each method said about the *following* candle + outcome
    rows = []
    idx = df.index[-recent:]
    for j, i in enumerate(idx):
        r = df.loc[i]
        nxt = df.loc[i + 1] if i + 1 in df.index else None
        rows.append(dict(
            time=int(r["open_time"].timestamp()),
            open=float(r["open"]), high=float(r["high"]), low=float(r["low"]), close=float(r["close"]),
            volume=float(r["volume"]),
            ml_p=round(float(p_hist[j]), 4),
            ta_score=float(agg_w.loc[i, "score"]),
            ta_dir=None if np.isnan(agg_w.loc[i, "direction"]) else int(agg_w.loc[i, "direction"]),
            actual=None if nxt is None else int(nxt["close"] > nxt["open"]),
        ))
    hits_ml = [(r["ml_p"] >= 0.5) == bool(r["actual"]) for r in rows if r["actual"] is not None]
    hits_ta = [(r["ta_dir"] == r["actual"]) for r in rows if r["actual"] is not None and r["ta_dir"] is not None]

    last_sig = sig.iloc[-1]
    signal_table = [
        dict(name=k, vote=int(last_sig[k]), weight=(weights or {}).get(k, 1.0),
             contribution=int(last_sig[k]) * (weights or {}).get(k, 1.0),
             acc_test=(ta_rep["per_signal"][k]["acc_test"] if ta_rep else None))
        for k in sig.columns
    ]

    out = dict(
        symbol="BTCUSDT", interval=interval, intervals=list(INTERVALS),
        last_closed_candle=key, last_close=float(last_bar["close"]),
        predicting_candle_open=str(next_open), predicting_candle_close=str(next_close),
        next_close_ts=int(next_close.timestamp()),
        ml=dict(p_bullish=round(p, 4), prediction=ml_dir, confidence=round(abs(p - 0.5) * 2, 4),
                oos_accuracy=round(meta["report"]["overall"]["accuracy"], 4),
                oos_auc=round(meta["report"]["overall"]["auc"], 4),
                recent_hit_rate=round(float(np.mean(hits_ml)), 4) if hits_ml else None),
        ta=dict(prediction=ta_res["prediction"], score=ta_res["score"],
                n_bull=ta_res["n_bull"], n_bear=ta_res["n_bear"], confidence=round(ta_res["confidence"], 4),
                bullish_signals=ta_book["bullish_signals"], bearish_signals=ta_book["bearish_signals"],
                oos_accuracy=round(ta_rep["calibrated_oos"]["accuracy"], 4) if ta_rep else None,
                textbook_oos_accuracy=round(ta_rep["textbook_oos"]["accuracy"], 4) if ta_rep else None,
                weights=weights,
                textbook_vote=dict(prediction=ta_book["prediction"], n_bull=ta_book["n_bull"], n_bear=ta_book["n_bear"]),
                signals=signal_table,
                recent_hit_rate=round(float(np.mean(hits_ta)), 4) if hits_ta else None,
                recent_calls=len(hits_ta)),
        agreement=(ml_dir == ta_res["prediction"]),
        recent=rows,
    )
    for k in [k for k in _CACHE if k[0] == interval and k[1] != key]:  # drop stale candles
        del _CACHE[k]
    _CACHE[ck] = out
    return out
