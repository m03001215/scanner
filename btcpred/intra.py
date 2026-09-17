"""Intra-candle predictor (v2): predicts the NEXT candle's direction while the current candle is still forming.

At minute k (1..14) of the forming 15m candle t, the state is: the standard features of the last CLOSED candle
t-1, plus the forming candle's partial open/high/low/last/volume after k minutes (from 1-minute candles), plus k.
One model serves every k. The target is candle t+1: close > open. This is a separate version from the
at-close predictor in btcpred/predictor.py; nothing here is shared with it except the feature builder.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import data, features, model
from .features import _atr

ROOT = Path(__file__).resolve().parent.parent
MINUTES = {"15m": 15, "1h": 60}
CONF_ACT = 0.10          # act rule: first minute at which 2*|p-0.5| >= CONF_ACT
PARTIAL = ["p_ret", "p_ret_atr", "p_range_atr", "p_pos", "p_vol_ratio", "p_hi_atr", "p_lo_atr", "k_min", "k_frac"]


def intra_dir(interval: str) -> Path:
    return ROOT / "models" / interval / "intra"


def live_1m_path() -> Path:
    return ROOT / "data" / "btcusdt_1m_live.parquet"


# ----------------------------------------------------------------------------- partial-candle features
def minute_matrix(df: pd.DataFrame, m1: pd.DataFrame, n: int):
    """(len(df), n) arrays of the 1m high/low/close/volume inside each candle of df; NaN where 1m data is missing."""
    idx = pd.DatetimeIndex(df["open_time"]).repeat(n) + pd.to_timedelta(np.tile(np.arange(n), len(df)), unit="min")   # keeps the UTC tz
    m = m1.reindex(idx)
    out = {c: m[c].values.reshape(len(df), n) for c in ("high", "low", "close", "volume")}
    return out


def partial_features(df: pd.DataFrame, mm: dict, ks: list[int]) -> dict[int, pd.DataFrame]:
    """Partial-candle features for every k in ks, aligned with df rows. Uses ATR and volume norms from candle t-1."""
    n = mm["high"].shape[1]
    atr_prev = _atr(df, 14).shift(1).values
    vol_norm = df["volume"].rolling(96).mean().shift(1).values
    o = df["open"].values
    hi = np.fmax.accumulate(np.nan_to_num(mm["high"], nan=-np.inf), axis=1); lo = np.fmin.accumulate(np.nan_to_num(mm["low"], nan=np.inf), axis=1)
    vol = np.nancumsum(mm["volume"], axis=1); close = mm["close"]
    have = ~np.isnan(close)
    out = {}
    for k in ks:
        j = k - 1
        ok = have[:, :k].all(axis=1)
        h, l, c, v = hi[:, j], lo[:, j], close[:, j], vol[:, j]
        rng = np.where(h - l > 0, h - l, np.nan)
        f = pd.DataFrame({
            "p_ret": (c - o) / o, "p_ret_atr": (c - o) / atr_prev, "p_range_atr": (h - l) / atr_prev,
            "p_pos": (c - l) / rng, "p_vol_ratio": v / (vol_norm * k / n),
            "p_hi_atr": (h - o) / atr_prev, "p_lo_atr": (o - l) / atr_prev,
            "k_min": float(k), "k_frac": k / n,
        }, index=df.index)
        f[~ok] = np.nan
        out[k] = f
    return out


def build_training(df: pd.DataFrame, m1: pd.DataFrame, interval: str, feats: list[str]):
    """Stacked rows: for each candle t and each minute k, [features(t-1), partial_k(t), k] -> label(t+1)."""
    n = MINUTES[interval]
    Xc = features.build_features(df)[feats].shift(1)            # last closed candle at the time of prediction
    y = features.build_labels(df)                                 # candle t+1 direction
    mm = minute_matrix(df, m1, n)
    parts = partial_features(df, mm, list(range(1, n)))
    Xs, ys, ts, ks = [], [], [], []
    for k, P in parts.items():
        X = pd.concat([Xc, P], axis=1)
        ok = X.notna().all(axis=1) & y.notna()
        Xs.append(X[ok]); ys.append(y[ok].astype(int)); ts.append(df.loc[ok, "open_time"]); ks.append(np.full(ok.sum(), k))
    X = pd.concat(Xs); y = pd.concat(ys); t = pd.concat(ts); k = np.concatenate(ks)
    order = np.lexsort((k, t.values))
    return X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True), t.iloc[order].reset_index(drop=True), k[order]


# ----------------------------------------------------------------------------- walk-forward
def walk_forward(X, y, t, k, n_folds=5, min_train_frac=0.4):
    """Expanding walk-forward by candle time; rows of one candle never straddle a fold. Returns per-row OOS predictions."""
    cand = pd.factorize(t)[0]; ncand = cand.max() + 1
    first = int(ncand * min_train_frac); edges = np.linspace(first, ncand, n_folds + 1, dtype=int)
    pred = np.full(len(X), np.nan); folds = []
    for i in range(n_folds):
        tr = cand < edges[i]; va_cut = int(edges[i] * 0.9)
        fit_m, va_m = tr & (cand < va_cut), tr & (cand >= va_cut)
        te = (cand >= edges[i]) & (cand < edges[i + 1])
        dtr = lgb.Dataset(X[fit_m], y[fit_m]); dva = lgb.Dataset(X[va_m], y[va_m], reference=dtr)
        b = lgb.train(model.PARAMS, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        pred[te] = b.predict(X[te], num_iteration=b.best_iteration)
        folds.append(dict(test_start=str(t[te].iloc[0]), test_end=str(t[te].iloc[-1]), n_candles=int(te.sum() / max(1, len(np.unique(k)))), best_iter=int(b.best_iteration)))
    return pred, folds


def evaluate(pred, y, t, k, interval: str) -> dict:
    n = MINUTES[interval]
    d = pd.DataFrame({"t": t.values, "k": k, "p": pred, "y": y.values}).dropna(subset=["p"])
    d["hit"] = ((d.p >= 0.5).astype(int) == d.y).astype(int); d["conf"] = (d.p - 0.5).abs() * 2
    by_k = []
    for kk, g in d.groupby("k"):
        c = g[g.conf >= CONF_ACT]
        by_k.append(dict(k=int(kk), lead_min=int(n - kk), n=len(g), acc=float(g.hit.mean()), conf_share=float((g.conf >= CONF_ACT).mean()), conf_acc=float(c.hit.mean()) if len(c) else None))
    # act rule: first k where conf >= CONF_ACT
    first = d[d.conf >= CONF_ACT].sort_values(["t", "k"]).drop_duplicates("t")
    final = d[d.k == n - 1]
    nc = d.t.nunique()
    rule = dict(fired_share=len(first) / nc, acc=float(first.hit.mean()) if len(first) else None,
                lead_mean_min=float((n - first.k).mean()) if len(first) else None,
                lead_hist={int(n - kk): int(v) for kk, v in first.k.value_counts().sort_index().items()},
                acc_by_lead={int(n - kk): float(g.hit.mean()) for kk, g in first.groupby("k")})
    return dict(candles=int(nc), by_minute=by_k, act_rule=rule, conf_act=CONF_ACT,
                final_minute=dict(k=n - 1, acc=float(final.hit.mean()), conf_acc=float(final[final.conf >= CONF_ACT].hit.mean()) if (final.conf >= CONF_ACT).any() else None,
                                  conf_share=float((final.conf >= CONF_ACT).mean())))


def train(interval: str, days: int, log=print) -> dict:
    df = data.drop_open_candle(data.load_or_update(ROOT / "data" / f"btcusdt_{interval}.parquet", interval=interval, days=days))
    m1 = pd.read_parquet(ROOT / "data" / "btcusdt_1m.parquet").set_index("open_time")
    _, meta = model.load(ROOT / "models" / interval / "lgbm"); feats = meta["features"]
    X, y, t, k = build_training(df, m1, interval, feats)
    log(f"[{interval} intra] {len(X):,} rows = {t.nunique():,} candles x {len(np.unique(k))} minutes, {t.iloc[0]:%Y-%m-%d} .. {t.iloc[-1]:%Y-%m-%d}")
    pred, folds = walk_forward(X, y, t, k)
    rep = evaluate(pred, y, t, k, interval); rep["folds"] = folds
    for r in rep["by_minute"]:
        log(f"  minute {r['k']:2d} (lead {r['lead_min']:2d} min): acc {r['acc']*100:.2f}%  conf>={CONF_ACT} share {r['conf_share']*100:4.1f}% acc {(r['conf_acc'] or 0)*100:.2f}%")
    a = rep["act_rule"]; log(f"  act rule (first minute with conf >= {CONF_ACT}): fires on {a['fired_share']*100:.1f}% of candles, acc {a['acc']*100:.2f}%, mean lead {a['lead_mean_min']:.1f} min")
    rounds = int(np.median([f["best_iter"] for f in folds])) or 200
    booster = model.train_final(X, y, rounds)
    out = intra_dir(interval); out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.txt"))
    (out / "meta.json").write_text(json.dumps(dict(interval=interval, features=list(X.columns), base_features=feats, report=rep, rounds=rounds,
                                                    trained_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), days=days, label="next candle close > open (Binance)"), indent=1))
    return rep


# ----------------------------------------------------------------------------- live
_MODEL: dict = {}
_LOCK = threading.Lock()


def load(interval: str):
    d = intra_dir(interval)
    if not (d / "model.txt").exists():
        return None
    mt = (d / "model.txt").stat().st_mtime
    if interval not in _MODEL or _MODEL[interval][0] != mt:
        _MODEL[interval] = (mt, lgb.Booster(model_file=str(d / "model.txt")), json.loads((d / "meta.json").read_text()))
    return _MODEL[interval][1], _MODEL[interval][2]


def log_path(interval: str) -> Path:
    return ROOT / "data" / f"intra_log_{interval}.jsonl"


def _append(interval: str, rec: dict) -> None:
    p = log_path(interval); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f: f.write(json.dumps(rec) + "\n")


def load_log(interval: str) -> list[dict]:
    p = log_path(interval)
    if not p.exists(): return []
    out = []
    for line in p.read_text().splitlines():
        try: out.append(json.loads(line))
        except json.JSONDecodeError: pass
    return out


def snapshot(interval: str, record: bool = True) -> dict:
    """Prediction for the NEXT candle from the state of the forming candle right now. Called every minute by the
    scheduler (record=True appends to the log) and by the API."""
    loaded = load(interval)
    if loaded is None:
        raise FileNotFoundError(f"no intra model for {interval}; run `python main.py train-intra -i {interval}`")
    booster, meta = loaded
    n = MINUTES[interval]
    with _LOCK:
        df = data.drop_open_candle(data.load_or_update(ROOT / "data" / f"btcusdt_{interval}.parquet", interval=interval, days=45))
        m1 = data.drop_open_candle(data.load_or_update(live_1m_path(), interval="1m", days=3)).set_index("open_time")
    now = pd.Timestamp.now(tz="UTC")
    step = pd.Timedelta(minutes=n)
    forming_open = now.floor(f"{n}min")
    last_closed = df["open_time"].iloc[-1]
    if last_closed + step != forming_open:                       # the just-closed candle is not in the cache yet
        forming_open = last_closed + step
    inside = m1[(m1.index >= forming_open) & (m1.index < forming_open + step)]
    k = len(inside)
    target_open = forming_open + step
    base = dict(interval=interval, now=now.strftime("%Y-%m-%dT%H:%M:%SZ"), forming_candle_open=str(forming_open), target_candle_open=str(target_open),
                target_candle_close=str(target_open + step), target_close_ts=int((target_open + step).timestamp()), forming_close_ts=int(target_open.timestamp()),
                minute=k, minutes_total=n, lead_min=n - k, conf_act=CONF_ACT)
    if k == 0:
        return dict(base, state="waiting", message="First minute of the forming candle has not closed yet.")
    if k >= n:
        return dict(base, state="closed", message="Forming candle has closed; waiting for the next one.")
    row = df.iloc[[-1]]                                            # last closed candle t-1
    Xc = features.build_features(df)[meta["base_features"]].iloc[[-1]]
    tmp = pd.DataFrame({"open_time": [forming_open], "open": [inside["open"].iloc[0]], "volume": [np.nan]})
    # partial features need ATR/vol norms from the closed history: compute on df, take the last value
    atr_prev = float(_atr(df, 14).iloc[-1]); vol_norm = float(df["volume"].rolling(96).mean().iloc[-1]); o = float(inside["open"].iloc[0])
    h, l, c, v = float(inside["high"].max()), float(inside["low"].min()), float(inside["close"].iloc[-1]), float(inside["volume"].sum())
    rng = (h - l) if h > l else np.nan
    P = pd.DataFrame({"p_ret": [(c - o) / o], "p_ret_atr": [(c - o) / atr_prev], "p_range_atr": [(h - l) / atr_prev], "p_pos": [(c - l) / rng],
                      "p_vol_ratio": [v / (vol_norm * k / n)], "p_hi_atr": [(h - o) / atr_prev], "p_lo_atr": [(o - l) / atr_prev], "k_min": [float(k)], "k_frac": [k / n]}, index=Xc.index)
    X = pd.concat([Xc, P], axis=1)[meta["features"]]
    p = float(booster.predict(X.fillna(0))[0]); conf = abs(p - 0.5) * 2
    out = dict(base, state="forming", p_next=round(p, 4), prediction="BULLISH" if p >= 0.5 else "BEARISH", confidence=round(conf, 4), fires=conf >= CONF_ACT,
               forming=dict(open=o, high=h, low=l, last=c, volume=round(v, 3), ret_bp=round((c - o) / o * 1e4, 2)))
    if record:
        _append(interval, dict(forming_candle_open=str(forming_open), target_candle_open=str(target_open), minute=k, p=out["p_next"], at=base["now"]))
    return out


def history(interval: str, df: pd.DataFrame | None = None, candles: int = 96) -> dict:
    """Per-candle paths from the log, with outcomes and the act rule applied, for the page."""
    n = MINUTES[interval]
    recs = load_log(interval)
    if not recs:
        return dict(paths=[], rule=dict(calls=0, resolved=0, hit_rate=None), current_path=[])
    d = pd.DataFrame(recs).drop_duplicates(["target_candle_open", "minute"], keep="first")
    d["conf"] = (d.p - 0.5).abs() * 2
    if df is None:
        df = data.drop_open_candle(pd.read_parquet(ROOT / "data" / f"btcusdt_{interval}.parquet"))
    oc = {str(t): int(cl > op) for t, op, cl in zip(df.open_time, df.open, df.close)}
    paths, calls = [], []
    for tgt, g in d.groupby("target_candle_open", sort=True):
        g = g.sort_values("minute")
        first = g[g.conf >= CONF_ACT].head(1)
        actual = oc.get(tgt)
        call = None if first.empty else dict(minute=int(first.minute.iloc[0]), lead_min=int(n - first.minute.iloc[0]), p=float(first.p.iloc[0]), direction="BULLISH" if first.p.iloc[0] >= 0.5 else "BEARISH")
        final = g.iloc[-1]
        rec = dict(target=tgt, forming=str(g.forming_candle_open.iloc[0]), minutes=[int(x) for x in g.minute], p=[float(x) for x in g.p],
                   call=call, final_p=float(final.p), final_minute=int(final.minute), actual=actual,
                   call_hit=None if (call is None or actual is None) else int((call["direction"] == "BULLISH") == bool(actual)),
                   final_hit=None if actual is None else int((final.p >= 0.5) == bool(actual)))
        paths.append(rec)
        if call is not None and actual is not None: calls.append(rec["call_hit"])
    cur = paths[-1] if paths and paths[-1]["actual"] is None else None
    resolved = [p for p in paths if p["final_hit"] is not None]
    return dict(paths=paths[-candles:], current_path=cur,
                rule=dict(calls=sum(1 for p in paths if p["call"]), resolved=len(calls), hit_rate=round(float(np.mean(calls)), 4) if calls else None,
                          final_resolved=len(resolved), final_hit_rate=round(float(np.mean([p["final_hit"] for p in resolved])), 4) if resolved else None))
