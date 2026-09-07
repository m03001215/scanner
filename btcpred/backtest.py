"""Candle-by-candle replay of both predictors over the out-of-sample window."""
from __future__ import annotations

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


def by_confidence(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi in ((0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.0)):
        d = bt[(abs(bt["ml_p"] - 0.5) >= lo) & (abs(bt["ml_p"] - 0.5) < hi)]
        rows.append(dict(bucket=f"|p-0.5| {lo:.2f}-{hi:.2f}", candles=len(d), share=len(d) / len(bt), ml_acc=d["ml_hit"].mean()))
    return pd.DataFrame(rows)
