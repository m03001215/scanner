"""Train / evaluate a LightGBM classifier with walk-forward validation."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

PARAMS = dict(
    objective="binary",
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=200,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    verbose=-1,
    seed=42,
)


def _fit(X_tr, y_tr, X_va=None, y_va=None, rounds: int = 2000) -> lgb.Booster:
    dtr = lgb.Dataset(X_tr, y_tr)
    if X_va is not None:
        dva = lgb.Dataset(X_va, y_va, reference=dtr)
        return lgb.train(PARAMS, dtr, num_boost_round=rounds, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(PARAMS, dtr, num_boost_round=rounds)


def walk_forward(X: pd.DataFrame, y: pd.Series, times: pd.Series,
                 n_folds: int = 5, min_train_frac: float = 0.4) -> dict:
    """Expanding-window walk-forward: train on [0, split), test on next block."""
    n = len(X)
    first = int(n * min_train_frac)
    edges = np.linspace(first, n, n_folds + 1, dtype=int)
    folds, all_p, all_y = [], [], []
    for i in range(n_folds):
        tr_end, te_end = edges[i], edges[i + 1]
        va_start = int(tr_end * 0.9)  # last 10% of train = early-stopping set
        booster = _fit(X.iloc[:va_start], y.iloc[:va_start],
                       X.iloc[va_start:tr_end], y.iloc[va_start:tr_end])
        p = booster.predict(X.iloc[tr_end:te_end], num_iteration=booster.best_iteration)
        yt = y.iloc[tr_end:te_end].values
        all_p.append(p); all_y.append(yt)
        folds.append(dict(
            test_start=str(times.iloc[tr_end]), test_end=str(times.iloc[te_end - 1]),
            n_test=int(te_end - tr_end), best_iter=int(booster.best_iteration),
            **_metrics(yt, p),
        ))
    p_all, y_all = np.concatenate(all_p), np.concatenate(all_y)
    return dict(folds=folds, overall=_metrics(y_all, p_all),
                baseline_majority=float(max(y_all.mean(), 1 - y_all.mean())))


def _metrics(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    out = dict(
        accuracy=float(accuracy_score(y, pred)),
        auc=float(roc_auc_score(y, p)),
        logloss=float(log_loss(y, p, labels=[0, 1])),
        bull_rate=float(np.mean(y)),
    )
    # Accuracy when the model is more confident (|p-0.5| >= 0.05 / 0.10)
    for thr in (0.05, 0.10):
        m = np.abs(p - 0.5) >= thr
        out[f"acc_conf_{int(thr*100):02d}"] = float(accuracy_score(y[m], pred[m])) if m.sum() > 20 else None
        out[f"coverage_conf_{int(thr*100):02d}"] = float(m.mean())
    return out


def train_final(X: pd.DataFrame, y: pd.Series, rounds: int) -> lgb.Booster:
    return _fit(X, y, rounds=rounds)


def save(booster: lgb.Booster, feature_names: list[str], report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.txt"))
    (out_dir / "meta.json").write_text(json.dumps(
        dict(features=feature_names, report=report), indent=2))


def load(out_dir: Path) -> tuple[lgb.Booster, dict]:
    booster = lgb.Booster(model_file=str(out_dir / "model.txt"))
    meta = json.loads((out_dir / "meta.json").read_text())
    return booster, meta
