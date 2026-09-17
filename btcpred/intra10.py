"""Intra-candle predictor v3: like v2 (predict the NEXT 15m candle while the current one forms) but the forming
candle is described by 10-second bars, and the prediction is recomputed every 10 seconds.

State at elapsed second e of forming candle t: the 50 standard features of the last CLOSED candle t-1, the forming
candle's partial state after e seconds (open/high/low/last/volume as in v2), fine-grained features from the 10 s bars
(returns over the last 30/60/90/180 s, realised volatility, taker-buy ratios, volume rate, up-bar share, VWAP
distance, drawdown/drawup), and e itself. One LightGBM model serves every e. Target: candle t+1 close > open.
Separate model, log, endpoint and page from v1 and v2.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from . import data, features, model
from .features import _atr

ROOT = Path(__file__).resolve().parent.parent
N_SEC = 900; BAR = 10; N_BARS = N_SEC // BAR
TRAIN_STEPS = list(range(30, N_SEC, 30))          # elapsed seconds used for training rows (29 per candle)
CONF_ACT = 0.10
FINE = ["p_ret", "p_ret_atr", "p_range_atr", "p_pos", "p_vol_ratio", "p_hi_atr", "p_lo_atr", "k_sec", "k_frac",
        "r30_atr", "r60_atr", "r90_atr", "r180_atr", "rv_atr", "taker_all", "taker_60", "vol_rate_60", "up_share", "vwap_dist_atr", "dd_atr", "du_atr", "n_bars"]
PARAMS = dict(model.PARAMS, learning_rate=0.05)   # 3M training rows: a faster learning rate keeps the walk-forward tractable


def v3_dir() -> Path: return ROOT / "models" / "15m" / "intra10"
def log_path() -> Path: return ROOT / "data" / "intra10_log_15m.jsonl"


# ----------------------------------------------------------------------------- bars
def load_bars() -> pd.DataFrame:
    files = sorted((ROOT / "data" / "bars10s").glob("*.parquet"))
    b = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("open_time").sort_values("open_time")
    return b.set_index("open_time")


def bar_matrix(df: pd.DataFrame, bars: pd.DataFrame) -> dict:
    """(len(df), 90) arrays of the 10 s bars inside each 15m candle; NaN where missing."""
    idx = pd.DatetimeIndex(df["open_time"]).repeat(N_BARS) + pd.to_timedelta(np.tile(np.arange(N_BARS) * BAR, len(df)), unit="s")
    m = bars.reindex(idx)
    return {c: m[c].values.reshape(len(df), N_BARS) for c in ("open", "high", "low", "close", "volume", "taker_buy")}


def _state_features(o, h, l, c, v, tb, atr, vol_norm, e):
    """Features from partial arrays: o scalar/array open; h,l,c,v,tb arrays (n, j) of the first j bars; e elapsed seconds."""
    j = c.shape[1]
    hi = np.nanmax(h, axis=1); lo = np.nanmin(l, axis=1); last = c[:, -1]; vol = np.nansum(v, axis=1); tbs = np.nansum(tb, axis=1)
    rng = np.where(hi - lo > 0, hi - lo, np.nan)
    def back(n):   # return over the last n bars, in ATR
        if j <= n: return np.full(len(last), np.nan)
        return (last - c[:, -n - 1]) / atr
    r10 = np.diff(np.log(c), axis=1) if j > 1 else np.zeros((len(last), 0))
    rv = np.sqrt(np.nansum(r10 ** 2, axis=1)) * last / atr if j > 1 else np.full(len(last), np.nan)
    n60 = min(j, 6)
    vwap = np.nansum(c * v, axis=1) / np.where(vol > 0, vol, np.nan)
    up = np.nanmean((c > np.concatenate([o[:, None], c[:, :-1]], axis=1)).astype(float), axis=1)
    return pd.DataFrame({
        "p_ret": (last - o) / o, "p_ret_atr": (last - o) / atr, "p_range_atr": (hi - lo) / atr, "p_pos": (last - lo) / rng,
        "p_vol_ratio": vol / (vol_norm * e / N_SEC), "p_hi_atr": (hi - o) / atr, "p_lo_atr": (o - lo) / atr, "k_sec": float(e), "k_frac": e / N_SEC,
        "r30_atr": back(3), "r60_atr": back(6), "r90_atr": back(9), "r180_atr": back(18), "rv_atr": rv,
        "taker_all": tbs / np.where(vol > 0, vol, np.nan), "taker_60": np.nansum(tb[:, -n60:], axis=1) / np.where(np.nansum(v[:, -n60:], axis=1) > 0, np.nansum(v[:, -n60:], axis=1), np.nan),
        "vol_rate_60": np.nansum(v[:, -n60:], axis=1) / (vol_norm / 15 * n60 / 6), "up_share": up,
        "vwap_dist_atr": (last - vwap) / atr, "dd_atr": (last - hi) / atr, "du_atr": (last - lo) / atr, "n_bars": float(j),
    })


def build_training(df: pd.DataFrame, bars: pd.DataFrame, feats: list[str]):
    Xc = features.build_features(df)[feats].shift(1); y = features.build_labels(df)
    mm = bar_matrix(df, bars)
    atr = _atr(df, 14).shift(1).values; vol_norm = df["volume"].rolling(96).mean().shift(1).values; o = df["open"].values
    Xs, ys, ts, es = [], [], [], []
    for e in TRAIN_STEPS:
        j = e // BAR
        sl = {k: a[:, :j] for k, a in mm.items()}
        ok = ~np.isnan(sl["close"]).any(axis=1)
        F = _state_features(o, sl["high"], sl["low"], sl["close"], sl["volume"], sl["taker_buy"], atr, vol_norm, e); F.index = df.index
        X = pd.concat([Xc, F], axis=1)
        good = ok & Xc.notna().all(axis=1).values & y.notna().values & np.isfinite(F["p_ret"].values)
        Xs.append(X[good]); ys.append(y[good].astype(int)); ts.append(df.loc[good, "open_time"]); es.append(np.full(good.sum(), e))
    X = pd.concat(Xs); y = pd.concat(ys); t = pd.concat(ts); e = np.concatenate(es)
    order = np.lexsort((e, t.values))
    return X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True), t.iloc[order].reset_index(drop=True), e[order]


def walk_forward(X, y, t, e, n_folds=5, min_train_frac=0.4, log=None):
    cand = pd.factorize(t)[0]; ncand = cand.max() + 1
    first = int(ncand * min_train_frac); edges = np.linspace(first, ncand, n_folds + 1, dtype=int)
    pred = np.full(len(X), np.nan); folds = []
    for i in range(n_folds):
        tr = cand < edges[i]; va_cut = int(edges[i] * 0.9)
        fit_m, va_m = tr & (cand < va_cut), tr & (cand >= va_cut); te = (cand >= edges[i]) & (cand < edges[i + 1])
        dtr = lgb.Dataset(X[fit_m], y[fit_m], free_raw_data=True); dva = lgb.Dataset(X[va_m], y[va_m], reference=dtr)
        b = lgb.train(PARAMS, dtr, num_boost_round=1500, valid_sets=[dva], callbacks=[lgb.early_stopping(80, verbose=False)])
        pred[te] = b.predict(X[te], num_iteration=b.best_iteration)
        folds.append(dict(test_start=str(t[te].iloc[0]), test_end=str(t[te].iloc[-1]), best_iter=int(b.best_iteration)))
        if log: log(f"  fold {i+1}: {folds[-1]['test_start'][:10]} .. {folds[-1]['test_end'][:10]}, {b.best_iteration} trees")
    return pred, folds


def evaluate(pred, y, t, e) -> dict:
    d = pd.DataFrame({"t": t.values, "e": e, "p": pred, "y": y.values}).dropna(subset=["p"])
    d["hit"] = ((d.p >= 0.5).astype(int) == d.y).astype(int); d["conf"] = (d.p - 0.5).abs() * 2
    by = [dict(e=int(k), lead_sec=int(N_SEC - k), n=len(g), acc=float(g.hit.mean()), conf_share=float((g.conf >= CONF_ACT).mean()),
               conf_acc=float(g[g.conf >= CONF_ACT].hit.mean()) if (g.conf >= CONF_ACT).any() else None) for k, g in d.groupby("e")]
    first = d[d.conf >= CONF_ACT].sort_values(["t", "e"]).drop_duplicates("t"); nc = d.t.nunique(); final = d[d.e == TRAIN_STEPS[-1]]
    rule = dict(fired_share=len(first) / nc, acc=float(first.hit.mean()) if len(first) else None, lead_mean_sec=float((N_SEC - first.e).mean()) if len(first) else None,
                acc_by_lead={int(N_SEC - k): float(g.hit.mean()) for k, g in first.groupby("e")}, n_by_lead={int(N_SEC - k): int(len(g)) for k, g in first.groupby("e")})
    return dict(candles=int(nc), by_step=by, act_rule=rule, conf_act=CONF_ACT,
                final_step=dict(e=TRAIN_STEPS[-1], acc=float(final.hit.mean()), conf_share=float((final.conf >= CONF_ACT).mean()),
                                conf_acc=float(final[final.conf >= CONF_ACT].hit.mean()) if (final.conf >= CONF_ACT).any() else None))


def train(days: int, log=print) -> dict:
    df = data.drop_open_candle(data.load_or_update(ROOT / "data" / "btcusdt_15m.parquet", interval="15m", days=days))
    bars = load_bars(); df = df[df["open_time"] >= bars.index[0].floor("15min") + pd.Timedelta(days=2)].reset_index(drop=True)
    _, meta = model.load(ROOT / "models" / "15m" / "lgbm"); feats = meta["features"]
    X, y, t, e = build_training(df, bars, feats)
    log(f"[15m intra10] {len(X):,} rows = {t.nunique():,} candles x {len(TRAIN_STEPS)} steps, {t.iloc[0]:%Y-%m-%d} .. {t.iloc[-1]:%Y-%m-%d}, {X.shape[1]} features")
    pred, folds = walk_forward(X, y, t, e, log=log)
    rep = evaluate(pred, y, t, e); rep["folds"] = folds
    for r in rep["by_step"]:
        if r["e"] % 60 == 0 or r["e"] == 30 or r["e"] == 870:
            log(f"  e={r['e']:3d}s (lead {r['lead_sec']:3d}s): acc {r['acc']*100:.2f}%  conf share {r['conf_share']*100:4.1f}%  conf acc {(r['conf_acc'] or 0)*100:.2f}%")
    a = rep["act_rule"]; log(f"  act rule: fires {a['fired_share']*100:.1f}%, acc {a['acc']*100:.2f}%, mean lead {a['lead_mean_sec']:.0f}s")
    rounds = int(np.median([f["best_iter"] for f in folds])) or 200
    booster = lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=rounds)
    out = v3_dir(); out.mkdir(parents=True, exist_ok=True); booster.save_model(str(out / "model.txt"))
    (out / "meta.json").write_text(json.dumps(dict(features=list(X.columns), base_features=feats, report=rep, rounds=rounds, params=PARAMS,
                                                    trained_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), days=days, label="next candle close > open (Binance)"), indent=1))
    return rep


# ----------------------------------------------------------------------------- live
_M: dict = {}; _LOCK = threading.Lock(); _K15: dict = {}


def load():
    d = v3_dir()
    if not (d / "model.txt").exists(): return None
    mt = (d / "model.txt").stat().st_mtime
    if "m" not in _M or _M["mt"] != mt:
        _M.update(mt=mt, m=lgb.Booster(model_file=str(d / "model.txt")), meta=json.loads((d / "meta.json").read_text()))
    return _M["m"], _M["meta"]


def _forming_bars(open_ts: pd.Timestamp) -> pd.DataFrame:
    """Complete 10 s bars of the forming candle from Binance 1 s klines (one request, weight 2)."""
    r = requests.get(data.BASE_URLS[0].replace("/api/v3/klines", "") + "/api/v3/klines", params=dict(symbol="BTCUSDT", interval="1s", startTime=int(open_ts.timestamp() * 1000), limit=1000), timeout=10)
    r.raise_for_status(); rows = r.json()
    if not rows: return pd.DataFrame()
    k = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 5, 9]]; k.columns = ["ts", "open", "high", "low", "close", "volume", "taker_buy"]
    k[["open", "high", "low", "close", "volume", "taker_buy"]] = k[["open", "high", "low", "close", "volume", "taker_buy"]].astype(float)
    now_ms = int(time.time() * 1000); k = k[k["ts"] + 999 <= now_ms]              # closed 1 s candles only
    k["b"] = (k["ts"] // 10_000) * 10_000; k = k[k["b"] + 10_000 <= now_ms]        # complete 10 s bins only
    g = k.groupby("b")
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(), "volume": g["volume"].sum(), "taker_buy": g["taker_buy"].sum()})


def snapshot(record: bool = True) -> dict:
    loaded = load()
    if loaded is None: raise FileNotFoundError("no v3 model; run `python main.py train-intra10`")
    booster, meta = loaded
    with _LOCK:
        df = data.drop_open_candle(data.load_or_update(ROOT / "data" / "btcusdt_15m.parquet", interval="15m", days=45))
    now = pd.Timestamp.now(tz="UTC"); step = pd.Timedelta(minutes=15)
    forming = now.floor("15min"); last_closed = df["open_time"].iloc[-1]
    if last_closed + step != forming: forming = last_closed + step
    target = forming + step
    base = dict(version="v3", interval="15m", now=now.strftime("%Y-%m-%dT%H:%M:%SZ"), forming_candle_open=str(forming), target_candle_open=str(target), target_candle_close=str(target + step),
                forming_close_ts=int(target.timestamp()), target_close_ts=int((target + step).timestamp()), seconds_total=N_SEC, conf_act=CONF_ACT)
    b = _forming_bars(forming)
    if b.empty: return dict(base, state="waiting", elapsed_sec=0, lead_sec=N_SEC, message="No complete 10-second bar of the forming candle yet.")
    j = len(b); e = j * BAR
    if e >= N_SEC: return dict(base, state="closed", elapsed_sec=e, lead_sec=0, message="Forming candle has closed; waiting for the next one.")
    Xc = features.build_features(df)[meta["base_features"]].iloc[[-1]]
    atr = np.array([float(_atr(df, 14).iloc[-1])]); vol_norm = np.array([float(df["volume"].rolling(96).mean().iloc[-1])])
    arr = lambda c: b[c].values[None, :]
    F = _state_features(np.array([b["open"].iloc[0]]), arr("high"), arr("low"), arr("close"), arr("volume"), arr("taker_buy"), atr, vol_norm, e); F.index = Xc.index
    X = pd.concat([Xc, F], axis=1)[meta["features"]]
    p = float(booster.predict(X)[0]); conf = abs(p - 0.5) * 2
    o = float(b["open"].iloc[0]); last = float(b["close"].iloc[-1])
    out = dict(base, state="forming", elapsed_sec=e, lead_sec=N_SEC - e, p_next=round(p, 4), prediction="BULLISH" if p >= 0.5 else "BEARISH", confidence=round(conf, 4), fires=conf >= CONF_ACT,
               forming=dict(open=o, high=float(b["high"].max()), low=float(b["low"].min()), last=last, volume=round(float(b["volume"].sum()), 3), ret_bp=round((last - o) / o * 1e4, 2),
                            taker_buy_share=round(float(F["taker_all"].iloc[0]), 3), taker_60=None if np.isnan(F["taker_60"].iloc[0]) else round(float(F["taker_60"].iloc[0]), 3),
                            r60_atr=None if np.isnan(F["r60_atr"].iloc[0]) else round(float(F["r60_atr"].iloc[0]), 3)))
    if record:
        p_ = log_path(); p_.parent.mkdir(parents=True, exist_ok=True)
        with p_.open("a") as f: f.write(json.dumps(dict(forming_candle_open=str(forming), target_candle_open=str(target), e=e, p=out["p_next"], at=base["now"])) + "\n")
    return out


def history(candles: int = 96) -> dict:
    p_ = log_path()
    if not p_.exists(): return dict(paths=[], current_path=None, rule=dict(calls=0, resolved=0, hit_rate=None, final_resolved=0, final_hit_rate=None))
    recs = [json.loads(l) for l in p_.read_text().splitlines() if l.strip()]
    d = pd.DataFrame(recs).drop_duplicates(["target_candle_open", "e"], keep="first"); d["conf"] = (d.p - 0.5).abs() * 2
    df = data.drop_open_candle(pd.read_parquet(ROOT / "data" / "btcusdt_15m.parquet"))
    oc = {str(t): int(cl > op) for t, op, cl in zip(df.open_time, df.open, df.close)}
    paths, calls = [], []
    for tgt, g in d.groupby("target_candle_open", sort=True):
        g = g.sort_values("e"); first = g[g.conf >= CONF_ACT].head(1); actual = oc.get(tgt); fin = g.iloc[-1]
        call = None if first.empty else dict(e=int(first.e.iloc[0]), lead_sec=int(N_SEC - first.e.iloc[0]), p=float(first.p.iloc[0]), direction="BULLISH" if first.p.iloc[0] >= 0.5 else "BEARISH")
        rec = dict(target=tgt, forming=str(g.forming_candle_open.iloc[0]), e=[int(x) for x in g.e], p=[float(x) for x in g.p], call=call, final_p=float(fin.p), final_e=int(fin.e), actual=actual,
                   call_hit=None if (call is None or actual is None) else int((call["direction"] == "BULLISH") == bool(actual)), final_hit=None if actual is None else int((fin.p >= 0.5) == bool(actual)))
        paths.append(rec)
        if call is not None and actual is not None: calls.append(rec["call_hit"])
    cur = paths[-1] if paths and paths[-1]["actual"] is None else None; res = [x for x in paths if x["final_hit"] is not None]
    return dict(paths=paths[-candles:], current_path=cur, rule=dict(calls=sum(1 for x in paths if x["call"]), resolved=len(calls), hit_rate=round(float(np.mean(calls)), 4) if calls else None,
                                                                      final_resolved=len(res), final_hit_rate=round(float(np.mean([x["final_hit"] for x in res])), 4) if res else None))
