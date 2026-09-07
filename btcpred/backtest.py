"""Candle-by-candle replay of both predictors over the out-of-sample window."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import features, model, ta


def run(df: pd.DataFrame, n_folds: int = 5, calib_frac: float = 0.4) -> pd.DataFrame:
    """Return one row per out-of-sample candle: ML probability/call, TA call, actual outcome.

    ML calls come from the expanding walk-forward (model trained only on earlier candles).
    TA weights are calibrated on the oldest `calib_frac` of history and frozen, which is
    the same boundary the walk-forward uses for its first test fold.
    """
    X, y, t = features.build_dataset(df)
    rep = model.walk_forward(X, y, t, n_folds=n_folds, min_train_frac=calib_frac, return_predictions=True)
    ml = rep["predictions"]

    labels = features.build_labels(df)
    sig = ta.signals(df)
    valid = labels.notna() & sig.notna().all(axis=1)
    sig_v, y_v = sig[valid], labels[valid].astype(int)
    split = int(len(sig_v) * calib_frac)
    weights = ta.calibrate(sig_v.iloc[:split], y_v.iloc[:split])
    agg = ta.score(sig_v, weights)
    ta_df = pd.DataFrame({"open_time": df.loc[sig_v.index, "open_time"].values,
                          "ta_score": agg["score"].values, "ta_dir": agg["direction"].values,
                          "ta_conf": agg["confidence"].values})

    out = ml.merge(ta_df, on="open_time", how="inner")
    out["ml_dir"] = (out["ml_p"] >= 0.5).astype(int)
    out["ml_hit"] = (out["ml_dir"] == out["actual"]).astype(int)
    out["ta_hit"] = np.where(out["ta_dir"].isna(), np.nan, (out["ta_dir"] == out["actual"]).astype(float))
    out["agree"] = (out["ml_dir"] == out["ta_dir"])
    # the candle being predicted opens one bar after the feature bar
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    return out


def monthly(bt: pd.DataFrame) -> pd.DataFrame:
    g = bt.groupby(bt["open_time"].dt.tz_convert(None).dt.to_period("M"))
    rows = []
    for m, d in g:
        both = d[d["agree"]]
        rows.append(dict(
            month=str(m), candles=len(d), bull_rate=d["actual"].mean(),
            ml_acc=d["ml_hit"].mean(),
            ta_acc=d["ta_hit"].mean(), ta_calls=int(d["ta_hit"].notna().sum()),
            agree_acc=both["ml_hit"].mean() if len(both) else np.nan, agree_n=len(both),
        ))
    return pd.DataFrame(rows)


def ml_confidence(bt: pd.DataFrame) -> pd.Series:
    """ML confidence on the dashboard scale: 2 * |P(bullish) - 0.5|, so 0.10 means P >= 0.55 or <= 0.45."""
    return (bt["ml_p"] - 0.5).abs() * 2


def by_confidence(bt: pd.DataFrame) -> pd.DataFrame:
    conf = ml_confidence(bt)
    rows = []
    for lo, hi in ((0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)):
        d = bt[(conf >= lo) & (conf < hi)]
        label = f"ML conf {lo:.2f}-{hi:.2f}" if hi < 1 else f"ML conf >= {lo:.2f}"
        rows.append(dict(bucket=label, candles=len(d), share=len(d) / len(bt), ml_acc=d["ml_hit"].mean()))
    return pd.DataFrame(rows)


def agreement(bt: pd.DataFrame) -> pd.DataFrame:
    """Accuracy split by whether ML and TA agree, and by confidence within the agreeing set."""
    n = len(bt)
    ml_conf = ml_confidence(bt)
    ta_called = bt["ta_dir"].notna()
    agree = bt["agree"] & ta_called
    disagree = ~bt["agree"] & ta_called
    subsets = [
        ("All candles (ML call)", pd.Series(True, index=bt.index)),
        ("ML and TA agree", agree),
        ("ML and TA disagree", disagree),
        ("TA has no call (tie)", ~ta_called),
        ("Agree, ML conf >= 0.05", agree & (ml_conf >= 0.05)),
        ("Agree, ML conf >= 0.10", agree & (ml_conf >= 0.10)),
        ("Agree, ML conf >= 0.20", agree & (ml_conf >= 0.20)),
        ("Agree, TA conf >= 0.3", agree & (bt["ta_conf"] >= 0.3)),
        ("Agree, ML >= 0.10 and TA >= 0.3", agree & (ml_conf >= 0.10) & (bt["ta_conf"] >= 0.3)),
        ("Agree, ML >= 0.20 and TA >= 0.3", agree & (ml_conf >= 0.20) & (bt["ta_conf"] >= 0.3)),
    ]
    rows = []
    for name, m in subsets:
        d = bt[m]
        rows.append(dict(subset=name, candles=int(len(d)), share=len(d) / n if n else 0.0,
                         accuracy=float(d["ml_hit"].mean()) if len(d) else None,
                         bull_rate=float(d["actual"].mean()) if len(d) else None))
    return pd.DataFrame(rows)


def write_summary(bt: pd.DataFrame, interval: str, step: pd.Timedelta, path: Path) -> dict:
    """Compact JSON summary committed alongside the model, served by the web app."""
    def rec(df):
        return json.loads(df.to_json(orient="records"))
    both = bt[bt["agree"] & bt["ta_dir"].notna()]
    summary = dict(
        interval=interval,
        test_start=str((bt["open_time"].iloc[0] + step).date()), test_end=str((bt["open_time"].iloc[-1] + step).date()),
        candles=int(len(bt)), bull_rate=float(bt["actual"].mean()),
        ml_acc=float(bt["ml_hit"].mean()), ta_acc=float(bt["ta_hit"].mean()),
        ta_coverage=float(bt["ta_hit"].notna().mean()),
        agree_acc=float(both["ml_hit"].mean()), agree_share=float(len(both) / len(bt)),
        agreement=rec(agreement(bt)), monthly=rec(monthly(bt)), by_confidence=rec(by_confidence(bt)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1))
    return summary


HISTORY_COLUMNS = ["time", "ml_p", "ml_call", "ta_score", "ta_conf", "ta_call", "actual", "ml_hit", "ta_hit"]


def write_history(bt: pd.DataFrame, step: pd.Timedelta, path: Path) -> int:
    """Per-candle out-of-sample results, gzip CSV, small enough to commit and serve from the app.

    `time` is the open of the candle that was predicted (UTC, ISO-8601)."""
    lab = lambda v: np.where(pd.isna(v), "", np.where(v == 1, "BULL", "BEAR"))
    h = pd.DataFrame({
        "time": (bt["open_time"] + step).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ml_p": bt["ml_p"].round(4),
        "ml_call": lab(bt["ml_dir"]),
        "ta_score": bt["ta_score"].astype(int),
        "ta_conf": bt["ta_conf"].round(3),
        "ta_call": lab(bt["ta_dir"]),
        "actual": lab(bt["actual"]),
        "ml_hit": bt["ml_hit"].astype(int),
        "ta_hit": bt["ta_hit"].map(lambda v: "" if pd.isna(v) else int(v)),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(path, index=False, compression="gzip")
    return len(h)
