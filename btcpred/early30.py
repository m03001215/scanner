"""v4: v1's model and TA vote, but called 30 s BEFORE the target candle opens.

At T-30 s the forming 15m candle t is replaced by a stand-in built from its first 870 s (open, running high/low,
last price as the close; volume, quote volume, trades and taker-buy scaled by 900/870) and treated as closed. v1's 50
features and the 18 TA signals are then computed exactly as in v1 and the model predicts candle t+1 (close > open).
Training uses the same stand-in for every historical candle (built from 10-second bars), so train and serve match.
Separate model, TA weights, log, endpoint and page; v1-v3 are untouched.
"""
from __future__ import annotations

import json
import threading
import time
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from . import data, features, model, ta

ROOT = Path(__file__).resolve().parent.parent
N_SEC, CUT_SEC, BAR = 900, 870, 10
N_CUT = CUT_SEC // BAR; SCALE = N_SEC / CUT_SEC
WINDOW = 800                      # closed candles of context per row; EMA200's residual weight at 800 bars is 0.03%
ML_GATE, TA_GATE = 0.10, 0.30


def v4_dir() -> Path: return ROOT / "models" / "15m" / "v4"
def log_path() -> Path: return ROOT / "data" / "v4_log_15m.jsonl"


# ----------------------------------------------------------------------------- stand-in candles
def standins_from_bars(df: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """For every candle in df, its stand-in from the first 870 s of 10 s bars. NaN rows where bars are missing."""
    n = len(df); nb = N_SEC // BAR
    idx = pd.DatetimeIndex(df["open_time"]).repeat(nb) + pd.to_timedelta(np.tile(np.arange(nb) * BAR, n), unit="s")
    m = bars.reindex(idx)
    g = {c: m[c].values.reshape(n, nb)[:, :N_CUT] for c in ("open", "high", "low", "close", "volume", "taker_buy", "trades")}
    ok = ~np.isnan(g["close"]).any(axis=1)
    out = pd.DataFrame({"open": g["open"][:, 0], "high": np.nanmax(np.where(np.isnan(g["high"]), -np.inf, g["high"]), axis=1),
                        "low": np.nanmin(np.where(np.isnan(g["low"]), np.inf, g["low"]), axis=1), "close": g["close"][:, -1],
                        "volume": np.nansum(g["volume"], axis=1) * SCALE, "quote_volume": np.nansum(g["close"] * g["volume"], axis=1) * SCALE,
                        "trades": np.nansum(g["trades"], axis=1) * SCALE, "taker_buy_base": np.nansum(g["taker_buy"], axis=1) * SCALE}, index=df.index)
    out[~ok] = np.nan
    return out


_G: dict = {}


def _one(i: int):
    df, st, feats = _G["df"], _G["st"], _G["feats"]
    w = df.iloc[i - WINDOW: i + 1].copy()
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"):
        w.iloc[-1, w.columns.get_loc(c)] = st[c].iloc[i]
    f = features.build_features(w)[feats].iloc[-1].values
    s = ta.signals(w).iloc[-1].values
    return i, f, s


def build_dataset(df: pd.DataFrame, bars: pd.DataFrame, feats: list[str], workers: int = 14, log=print):
    st = standins_from_bars(df, bars)
    rows = [i for i in range(WINDOW, len(df)) if not np.isnan(st["close"].iloc[i])]
    _G.update(df=df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "close_time"]], st=st, feats=feats)
    t0 = time.time()
    with Pool(workers) as p:
        res = p.map(_one, rows, chunksize=200)
    log(f"  built {len(res):,} stand-in rows in {time.time() - t0:.0f}s")
    idx = [r[0] for r in res]
    X = pd.DataFrame(np.vstack([r[1] for r in res]), index=idx, columns=feats)
    S = pd.DataFrame(np.vstack([r[2] for r in res]), index=idx, columns=ta.signals(df.iloc[:WINDOW]).columns)
    return X, S, st


# ----------------------------------------------------------------------------- training + paired comparison with v1
def _stats(p, y, tadir, taconf):
    d = (p >= 0.5).astype(int); c = np.abs(p - 0.5) * 2; hit = d == y
    gate = (tadir == d) & (c >= ML_GATE) & (taconf >= TA_GATE); agree = tadir == d
    from sklearn.metrics import roc_auc_score, log_loss
    return dict(acc=float(hit.mean()), auc=float(roc_auc_score(y, p)), logloss=float(log_loss(y, p)), conf10_acc=float(hit[c >= .1].mean()), conf10_share=float((c >= .1).mean()),
                conf20_acc=float(hit[c >= .2].mean()) if (c >= .2).any() else None, conf20_share=float((c >= .2).mean()), agree_acc=float(hit[agree].mean()), agree_share=float(agree.mean()),
                gate_acc=float(hit[gate].mean()) if gate.any() else None, gate_share=float(gate.mean()), gate_n=int(gate.sum())), hit, d, gate


def train(days: int = 1095, log=print) -> dict:
    from . import intra10
    df = data.drop_open_candle(data.load_or_update(ROOT / "data" / "btcusdt_15m.parquet", interval="15m", days=days + 12))
    bars = intra10.load_bars()
    if "trades" not in bars.columns: raise RuntimeError("bars10s lacks trades")
    df = df[df["open_time"] >= bars.index[0].floor("15min")].reset_index(drop=True)
    _, meta1 = model.load(ROOT / "models" / "15m" / "lgbm"); feats = meta1["features"]
    log(f"[15m v4] {len(df):,} candles {df.open_time.iloc[0]:%Y-%m-%d} .. {df.open_time.iloc[-1]:%Y-%m-%d}; stand-in = first {CUT_SEC}s, volumes x{SCALE:.4f}")
    X4, S4, st = build_dataset(df, bars, feats, log=log)
    y = features.build_labels(df); X1 = features.build_features(df)[feats]; S1 = ta.signals(df)
    ok = X4.notna().all(axis=1) & y.loc[X4.index].notna() & X1.loc[X4.index].notna().all(axis=1) & S4.notna().all(axis=1)
    U = X4.index[ok]; t = df.loc[U, "open_time"].reset_index(drop=True); yy = y.loc[U].astype(int).reset_index(drop=True)
    log(f"  rows with v4 + v1 features and labels: {len(U):,}; stand-in vs real close: median |diff| {((st.close.loc[U] / df.close.loc[U] - 1).abs() * 1e4).median():.2f} bp, direction of the candle matches on {((st.close.loc[U] > st.open.loc[U]) == (df.close.loc[U] > df.open.loc[U])).mean()*100:.1f}%")
    r4 = model.walk_forward(X4.loc[U].reset_index(drop=True), yy, t, n_folds=5, return_predictions=True)
    r1 = model.walk_forward(X1.loc[U].reset_index(drop=True), yy, t, n_folds=5, return_predictions=True)
    first = int(len(U) * 0.4); Uo = U[first:]; Y = yy.values[first:]
    # TA weights calibrated on the first 40% (as v1), each version on its own signals
    w4 = ta.calibrate(S4.loc[U[:first]], y.loc[U[:first]]); w1 = ta.calibrate(S1.loc[U[:first]], y.loc[U[:first]])
    a4 = ta.score(S4.loc[Uo], w4); a1 = ta.score(S1.loc[Uo], w1)
    s4, h4, d4, g4 = _stats(r4["predictions"]["ml_p"].values, Y, a4["direction"].values, a4["confidence"].values)
    s1, h1, d1, g1 = _stats(r1["predictions"]["ml_p"].values, Y, a1["direction"].values, a1["confidence"].values)
    b = int((h1 & ~h4).sum()); c = int((~h1 & h4).sum()); z = (c - b) / np.sqrt(b + c)
    ta4 = a4["direction"].notna().values; ta1 = a1["direction"].notna().values
    cmp_ = dict(n=int(len(Y)), v4=s4, v1=s1, mcnemar=dict(v1_only_right=b, v4_only_right=c, z=float(z)), same_direction=float((d1 == d4).mean()),
                gate_both=int((g1 & g4).sum()), gate_both_acc=float(h4[g1 & g4].mean()) if (g1 & g4).any() else None, gate_v1_only=int((g1 & ~g4).sum()), gate_v4_only=int((g4 & ~g1).sum()),
                ta_acc_v4=float((a4["direction"].values[ta4] == Y[ta4]).mean()), ta_acc_v1=float((a1["direction"].values[ta1] == Y[ta1]).mean()),
                ta_same=float((a4["direction"].values == a1["direction"].values)[ta4 & ta1].mean()), test_start=str(df.loc[Uo[0], "open_time"]), test_end=str(df.loc[Uo[-1], "open_time"]))
    mon = pd.DataFrame({"m": df.loc[Uo, "open_time"].dt.tz_convert(None).dt.to_period("M").astype(str).values, "v4": h4, "v1": h1, "g4": g4, "g1": g1})
    cmp_["monthly"] = [dict(month=m_, n=len(g), v4=float(g.v4.mean()), v1=float(g.v1.mean()), gate4=float(g.g4.mean()), gate1=float(g.g1.mean())) for m_, g in mon.groupby("m")]
    for name, s in (("v4 at -30 s", s4), ("v1 at   0 s", s1)):
        log(f"  {name}: acc {s['acc']*100:.2f}%  auc {s['auc']:.3f}  conf>=.10 {s['conf10_acc']*100:.2f}% ({s['conf10_share']*100:.1f}%)  gate {s['gate_acc']*100:.2f}% pass {s['gate_share']*100:.2f}%")
    log(f"  paired: v1-only right {b:,}, v4-only right {c:,}, z {z:+.2f}; same direction {cmp_['same_direction']*100:.1f}%; TA same call {cmp_['ta_same']*100:.1f}%")
    rounds = int(np.median([f["best_iter"] for f in r4["folds"]])) or 200
    booster = model.train_final(X4.loc[U].reset_index(drop=True), yy, rounds)
    w_final = ta.calibrate(S4.loc[U], y.loc[U])
    out = v4_dir(); out.mkdir(parents=True, exist_ok=True); booster.save_model(str(out / "model.txt"))
    (out / "meta.json").write_text(json.dumps(dict(features=feats, ta_weights=w_final, ta_weights_oos=w4, rounds=rounds, folds=r4["folds"], overall=r4["overall"], comparison=cmp_,
                                                    cut_sec=CUT_SEC, scale=SCALE, window=WINDOW, days=days, trained_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                    label="next candle close > open (Binance)"), indent=1, default=float))
    return cmp_


# ----------------------------------------------------------------------------- live
_M: dict = {}; _LOCK = threading.Lock(); _LAST: dict = {}


def load():
    d = v4_dir()
    if not (d / "model.txt").exists(): return None
    mt = (d / "model.txt").stat().st_mtime
    if _M.get("mt") != mt: _M.update(mt=mt, m=lgb.Booster(model_file=str(d / "model.txt")), meta=json.loads((d / "meta.json").read_text()))
    return _M["m"], _M["meta"]


def _standin_live(open_ts: pd.Timestamp) -> dict | None:
    """Stand-in for the forming candle from closed 1 s klines of its first 870 s; None until all 870 s are available."""
    base = data.BASE_URLS[0].replace("/api/v3/klines", "")
    r = requests.get(base + "/api/v3/klines", params=dict(symbol="BTCUSDT", interval="1s", startTime=int(open_ts.timestamp() * 1000), limit=1000), timeout=10)
    r.raise_for_status(); rows = r.json()
    if not rows: return None
    k = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 5, 8, 9]]; k.columns = ["ts", "open", "high", "low", "close", "volume", "trades", "taker_buy"]
    k = k.astype({c: float for c in ("open", "high", "low", "close", "volume", "trades", "taker_buy")})
    o_ms = int(open_ts.timestamp() * 1000); k = k[(k["ts"] >= o_ms) & (k["ts"] < o_ms + CUT_SEC * 1000)]
    if len(k) == 0 or k["ts"].max() < o_ms + (CUT_SEC - 1) * 1000 or int(time.time() * 1000) < o_ms + CUT_SEC * 1000: return None
    k["b"] = (k["ts"] // 10_000) * 10_000; g = k.groupby("b")
    b = pd.DataFrame({"close": g["close"].last(), "volume": g["volume"].sum()})
    return dict(open=float(k["open"].iloc[0]), high=float(k["high"].max()), low=float(k["low"].min()), close=float(k["close"].iloc[-1]), volume=float(k["volume"].sum() * SCALE),
                quote_volume=float((b["close"] * b["volume"]).sum() * SCALE), trades=float(k["trades"].sum() * SCALE), taker_buy_base=float(k["taker_buy"].sum() * SCALE), seconds=int(len(k)))


def timing() -> dict:
    now = pd.Timestamp.now(tz="UTC"); forming = now.floor("15min"); step = pd.Timedelta(minutes=15)
    return dict(now=now, forming=forming, target=forming + step, elapsed=(now - forming).total_seconds(), call_at=forming + pd.Timedelta(seconds=CUT_SEC))


def predict_now(record: bool = True, forming: pd.Timestamp | None = None) -> dict | None:
    """Make the T-30 s call for the candle that opens next. Returns None if the first 870 s are not complete yet.
    `forming` overrides the forming candle (a past one) so the live path can be tested without waiting."""
    loaded = load()
    if loaded is None: raise FileNotFoundError("no v4 model; run `python main.py train-v4`")
    booster, meta = loaded; tm = timing()
    if forming is not None: tm = dict(tm, forming=forming, target=forming + pd.Timedelta(minutes=15)); record = False
    key = str(tm["target"])
    with _LOCK:
        if forming is None and _LAST.get("target_candle_open") == key: return _LAST
        st = _standin_live(tm["forming"])
        if st is None: return None
        df = data.drop_open_candle(data.load_or_update(ROOT / "data" / "btcusdt_15m.parquet", interval="15m", days=45))
        df = df[df["open_time"] < tm["forming"]].iloc[-WINDOW:]
        if df["open_time"].iloc[-1] + pd.Timedelta(minutes=15) != tm["forming"]: raise RuntimeError("closed 15m history is not up to the forming candle")
        row = {c: st[c] for c in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base")}
        row.update(open_time=tm["forming"], close_time=tm["target"] - pd.Timedelta(milliseconds=1), taker_buy_quote=np.nan)
        w = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        for c in ("open_time", "close_time"): w[c] = pd.to_datetime(w[c], utc=True)      # concat of mixed-unit timestamps yields object dtype
        X = features.build_features(w)[meta["features"]].iloc[[-1]]
        p = float(booster.predict(X)[0]); conf = abs(p - 0.5) * 2; ml = "BULLISH" if p >= 0.5 else "BEARISH"
        sig = ta.signals(w); agg = ta.score(sig, meta["ta_weights"]).iloc[-1]
        ta_dir = "NEUTRAL" if np.isnan(agg["direction"]) else ("BULLISH" if agg["direction"] == 1 else "BEARISH")
        agree = ta_dir == ml; passed = bool(agree and conf >= ML_GATE and agg["confidence"] >= TA_GATE)
        out = dict(target_candle_open=key, target_candle_close=str(tm["target"] + pd.Timedelta(minutes=15)), forming_candle_open=str(tm["forming"]), made_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                   lead_sec=round((tm["target"] - pd.Timestamp.now(tz="UTC")).total_seconds(), 1), p_bullish=round(p, 4), prediction=ml, confidence=round(conf, 4),
                   ta=dict(prediction=ta_dir, score=float(agg["score"]), confidence=round(float(agg["confidence"]), 4)), gate=dict(agree=bool(agree), ml_ok=bool(conf >= ML_GATE), ta_ok=bool(agg["confidence"] >= TA_GATE), passed=passed),
                   standin=dict(st, ret_bp=round((st["close"] / st["open"] - 1) * 1e4, 2)))
        if forming is None: _LAST.clear(); _LAST.update(out)
        if record:
            p_ = log_path(); p_.parent.mkdir(parents=True, exist_ok=True)
            with p_.open("a") as f: f.write(json.dumps(out) + "\n")
        return out


def load_log() -> list[dict]:
    p_ = log_path()
    if not p_.exists(): return []
    seen, out = set(), []
    for line in p_.read_text().splitlines():
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        if r["target_candle_open"] in seen: continue
        seen.add(r["target_candle_open"]); out.append(r)
    return out


def history(n: int = 96) -> dict:
    recs = load_log()
    if not recs: return dict(rows=[], record=dict(calls=0, resolved=0, hit_rate=None, gate_calls=0, gate_resolved=0, gate_hit_rate=None, standin_dir_match=None))
    df = data.drop_open_candle(pd.read_parquet(ROOT / "data" / "btcusdt_15m.parquet"))
    oc = {str(t): (int(cl > op), float(op), float(cl)) for t, op, cl in zip(df.open_time, df.open, df.close)}
    rows, hits, ghits, sm = [], [], [], []
    for r in recs:
        a = oc.get(r["target_candle_open"]); f = oc.get(r["forming_candle_open"])
        hit = None if a is None else int((r["prediction"] == "BULLISH") == bool(a[0]))
        rows.append(dict(target=r["target_candle_open"], made_at=r["made_at"], lead_sec=r["lead_sec"], p=r["p_bullish"], prediction=r["prediction"], confidence=r["confidence"],
                         ta=r["ta"]["prediction"], ta_conf=r["ta"]["confidence"], gate=r["gate"]["passed"], actual=None if a is None else a[0], hit=hit,
                         standin_close=r["standin"]["close"], real_close=None if f is None else f[2]))
        if hit is not None:
            hits.append(hit)
            if r["gate"]["passed"]: ghits.append(hit)
        if f is not None: sm.append(int((r["standin"]["close"] > r["standin"]["open"]) == bool(f[0])))
    return dict(rows=rows[-n:], record=dict(calls=len(recs), resolved=len(hits), hit_rate=round(float(np.mean(hits)), 4) if hits else None, gate_calls=sum(1 for r in recs if r["gate"]["passed"]),
                                             gate_resolved=len(ghits), gate_hit_rate=round(float(np.mean(ghits)), 4) if ghits else None, standin_dir_match=round(float(np.mean(sm)), 4) if sm else None))
