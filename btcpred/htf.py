"""v7: v1 plus higher-timeframe context. Same 15m features, TA vote and gate as v1; adds features from CLOSED 1h and 4h
candles (each 15m row sees only higher-timeframe candles whose close_time <= its own close_time), so there is no look-ahead.
Separate model, log, endpoint (/api/v7) and page (/v7); v1 is untouched.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import data, features, model, predictor, ta
from .features import _atr, _rsi

ROOT = Path(__file__).resolve().parent.parent
TF = "15m"; HTFS = ("1h", "4h")
ML_GATE, TA_GATE = 0.10, 0.30


def v7_dir() -> Path: return ROOT / "models" / TF / "v7"
def log_path() -> Path: return ROOT / "data" / "v7_log_15m.jsonl"


def htf_frame(h: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Context features from a higher-timeframe candle series, in that timeframe's ATR units, keyed by close_time."""
    h = h.reset_index(drop=True); c, o, hi, lo, v = h["close"], h["open"], h["high"], h["low"], h["volume"]
    atr = _atr(h, 14); lr = np.log(c / c.shift(1))
    ema = {n: c.ewm(span=n, adjust=False).mean() for n in (8, 20, 50)}
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    sign = np.sign(c - o); streak = np.zeros(len(h)); s = sign.values
    for i in range(1, len(h)): streak[i] = streak[i - 1] + s[i] if s[i] != 0 and s[i] == s[i - 1] else s[i]
    f = pd.DataFrame({
        f"{tag}_ema8": (c - ema[8]) / atr, f"{tag}_ema20": (c - ema[20]) / atr, f"{tag}_ema50": (c - ema[50]) / atr,
        f"{tag}_rsi14": _rsi(c, 14), f"{tag}_ret4": (c - c.shift(4)) / atr, f"{tag}_ret12": (c - c.shift(12)) / atr,
        f"{tag}_vol_ratio": lr.rolling(6).std() / lr.rolling(48).std(), f"{tag}_atr_pct": atr / c,
        f"{tag}_range_pos": (c - lo.rolling(24).min()) / (hi.rolling(24).max() - lo.rolling(24).min()),
        f"{tag}_body": (c - o) / atr, f"{tag}_streak": streak, f"{tag}_bb": (c - sma20) / (2 * sd20),
        f"{tag}_vol_z": (v - v.rolling(48).mean()) / v.rolling(48).std(), f"{tag}_close": c, f"{tag}_atr": atr,
    })
    f["close_time"] = h["close_time"].values
    return f


def build_features_v7(df15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame, base_feats: list[str]) -> pd.DataFrame:
    """v1's features on df15 plus HTF context as of each 15m candle's close. Aligned with df15's index."""
    X = features.build_features(df15)[base_feats]
    _ts = lambda x: pd.to_datetime(x, utc=True).astype("datetime64[us, UTC]")     # one tz-aware unit on both sides of the join
    key = pd.DataFrame({"close_time": _ts(pd.Series(df15["close_time"].values, index=df15.index))}, index=df15.index).reset_index()
    for h, tag in ((h1, "h1"), (h4, "h4")):
        f = htf_frame(h, tag); f["close_time"] = _ts(f["close_time"])
        j = pd.merge_asof(key.sort_values("close_time"), f.sort_values("close_time"), on="close_time", direction="backward").set_index("index").sort_index()
        j[f"{tag}_dist"] = (df15["close"].values - j[f"{tag}_close"].values) / j[f"{tag}_atr"].values   # 15m close vs last closed HTF close
        for col in j.columns:
            if col.startswith(tag) and col not in (f"{tag}_close", f"{tag}_atr"): X[col] = j[col].values
    return X.replace([np.inf, -np.inf], np.nan)


def _load_htf(days: int):
    h1 = data.drop_open_candle(data.load_or_update(ROOT / "data" / "btcusdt_1h.parquet", interval="1h", days=days + 30))
    h4 = data.drop_open_candle(data.load_or_update(ROOT / "data" / "btcusdt_4h.parquet", interval="4h", days=days + 120))
    return h1, h4


def _stats(p, y, tadir, taconf):
    from sklearn.metrics import roc_auc_score, log_loss
    d = (p >= .5).astype(int); c = np.abs(p - .5) * 2; hit = d == y; gate = (tadir == d) & (c >= ML_GATE) & (taconf >= TA_GATE)
    return dict(acc=float(hit.mean()), auc=float(roc_auc_score(y, p)), logloss=float(log_loss(y, p)), conf10_acc=float(hit[c >= .1].mean()), conf10_share=float((c >= .1).mean()),
                conf20_acc=float(hit[c >= .2].mean()) if (c >= .2).any() else None, conf20_share=float((c >= .2).mean()), gate_acc=float(hit[gate].mean()), gate_share=float(gate.mean()), gate_n=int(gate.sum())), hit, d, gate


def train(days: int = 1095, log=print) -> dict:
    df = data.drop_open_candle(data.load_or_update(predictor.cache_path(TF), interval=TF, days=days))
    h1, h4 = _load_htf(days)
    _, meta1 = model.load(predictor.model_dir(TF)); base = meta1["features"]
    X7 = build_features_v7(df, h1, h4, base); X1 = X7[base]; y = features.build_labels(df)
    ok = X7.notna().all(axis=1) & y.notna()
    U = df.index[ok]; t = df.loc[U, "open_time"].reset_index(drop=True); yy = y.loc[U].astype(int).reset_index(drop=True)
    log(f"[15m v7] {len(U):,} candles {t.iloc[0]:%Y-%m-%d} .. {t.iloc[-1]:%Y-%m-%d}; {len(base)} v1 features + {X7.shape[1]-len(base)} higher-timeframe features (1h, 4h)")
    r7 = model.walk_forward(X7.loc[U].reset_index(drop=True), yy, t, n_folds=5, return_predictions=True)
    r1 = model.walk_forward(X1.loc[U].reset_index(drop=True), yy, t, n_folds=5, return_predictions=True)
    first = int(len(U) * 0.4); Uo = U[first:]; Y = yy.values[first:]
    rp = ta.score(ta.signals(df), json.load(open(predictor.ta_report_path(TF)))["weights"])
    tadir, taconf = rp["direction"].values[Uo], rp["confidence"].values[Uo]
    s7, h7, d7, g7 = _stats(r7["predictions"]["ml_p"].values, Y, tadir, taconf); s1, h1s, d1, g1 = _stats(r1["predictions"]["ml_p"].values, Y, tadir, taconf)
    b = int((h1s & ~h7).sum()); c = int((~h1s & h7).sum()); z = (c - b) / np.sqrt(b + c)
    mon = pd.DataFrame({"m": df.loc[Uo, "open_time"].dt.tz_convert(None).dt.to_period("M").astype(str).values, "v7": h7, "v1": h1s, "g7": g7, "g1": g1})
    fold_acc = []
    for i, f in enumerate(r7["folds"]): fold_acc.append(dict(test_start=f["test_start"][:10], test_end=f["test_end"][:10], v7=f["accuracy"], v1=r1["folds"][i]["accuracy"], trees_v7=f["best_iter"], trees_v1=r1["folds"][i]["best_iter"]))
    rounds = int(np.median([f["best_iter"] for f in r7["folds"]])) or 200
    booster = model.train_final(X7.loc[U].reset_index(drop=True), yy, rounds)
    imp = pd.Series(booster.feature_importance("gain"), index=X7.columns); imp = imp / imp.sum() * 100
    htf_cols = [k for k in X7.columns if k.startswith(("h1_", "h4_"))]
    cmp_ = dict(n=int(len(Y)), v7=s7, v1=s1, mcnemar=dict(v1_only_right=b, v7_only_right=c, z=float(z)), same_direction=float((d1 == d7).mean()),
                gate_both=int((g1 & g7).sum()), gate_both_acc=float(h7[g1 & g7].mean()) if (g1 & g7).any() else None, gate_v1_only=int((g1 & ~g7).sum()), gate_v7_only=int((g7 & ~g1).sum()),
                test_start=str(df.loc[Uo[0], "open_time"]), test_end=str(df.loc[Uo[-1], "open_time"]), folds=fold_acc,
                monthly=[dict(month=m_, n=len(g), v7=float(g.v7.mean()), v1=float(g.v1.mean()), gate7=float(g.g7.mean()), gate1=float(g.g1.mean())) for m_, g in mon.groupby("m")],
                htf_gain_share=float(imp[htf_cols].sum()), top_htf=[(k, round(float(v), 2)) for k, v in imp[htf_cols].sort_values(ascending=False).head(8).items()])
    for name, s in (("v7 (with 1h/4h)", s7), ("v1", s1)): log(f"  {name:16s}: acc {s['acc']*100:.2f}%  auc {s['auc']:.3f}  conf>=.10 {s['conf10_acc']*100:.2f}% ({s['conf10_share']*100:.1f}%)  gate {s['gate_acc']*100:.2f}% pass {s['gate_share']*100:.2f}%")
    log(f"  paired: v1-only right {b:,}, v7-only right {c:,}, z {z:+.2f}; same direction {cmp_['same_direction']*100:.1f}%; HTF features carry {cmp_['htf_gain_share']:.1f}% of gain; top: {cmp_['top_htf'][:5]}")
    out = v7_dir(); out.mkdir(parents=True, exist_ok=True); booster.save_model(str(out / "model.txt"))
    (out / "meta.json").write_text(json.dumps(dict(features=list(X7.columns), base_features=base, rounds=rounds, folds=r7["folds"], overall=r7["overall"], comparison=cmp_, days=days,
                                                    trained_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), label="next candle close > open (Binance)"), indent=1, default=float))
    return cmp_


# ----------------------------------------------------------------------------- live
_M: dict = {}; _LOCK = threading.Lock(); _LAST: dict = {}


def load():
    d = v7_dir()
    if not (d / "model.txt").exists(): return None
    mt = (d / "model.txt").stat().st_mtime
    if _M.get("mt") != mt: _M.update(mt=mt, m=lgb.Booster(model_file=str(d / "model.txt")), meta=json.loads((d / "meta.json").read_text()))
    return _M["m"], _M["meta"]


def predict_now(record: bool = True) -> dict:
    """v7's call for the candle forming now, alongside v1's (from the live v1 predictor) for comparison."""
    loaded = load()
    if loaded is None: raise FileNotFoundError("no v7 model; run `python main.py train-v7`")
    booster, meta = loaded
    with _LOCK:
        pr = predictor.predict(interval=TF, recent=16); key = pr["predicting_candle_open"]
        if _LAST.get("target_candle_open") == key: return _LAST
        df = data.drop_open_candle(data.load_or_update(predictor.cache_path(TF), interval=TF, days=45))
        h1, h4 = _load_htf(45)
        X = build_features_v7(df, h1, h4, meta["base_features"])[meta["features"]].iloc[[-1]]
        p = float(booster.predict(X.fillna(0))[0]); conf = abs(p - .5) * 2; ml = "BULLISH" if p >= .5 else "BEARISH"
        ta_ok = pr["ta"]["confidence"] >= TA_GATE; agree = pr["ta"]["prediction"] == ml
        out = dict(version="v7", target_candle_open=key, target_candle_close=pr["predicting_candle_close"], last_closed_candle=pr["last_closed_candle"], made_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                   p_bullish=round(p, 4), prediction=ml, confidence=round(conf, 4), gate=dict(agree=bool(agree), ml_ok=bool(conf >= ML_GATE), ta_ok=bool(ta_ok), passed=bool(agree and conf >= ML_GATE and ta_ok)),
                   ta=dict(prediction=pr["ta"]["prediction"], score=pr["ta"]["score"], confidence=pr["ta"]["confidence"]),
                   v1=dict(prediction=pr["ml"]["prediction"], p_bullish=pr["ml"]["p_bullish"], confidence=pr["ml"]["confidence"], gate_passed=bool(pr["ta"]["prediction"] == pr["ml"]["prediction"] and pr["ml"]["confidence"] >= ML_GATE and ta_ok)),
                   htf={k: (None if pd.isna(v) else round(float(v), 4)) for k, v in X.iloc[0].items() if k.startswith(("h1_", "h4_"))},
                   last_1h_close=str(h1["open_time"].iloc[-1]), last_4h_close=str(h4["open_time"].iloc[-1]))
        _LAST.clear(); _LAST.update(out)
        if record:
            p_ = log_path(); p_.parent.mkdir(parents=True, exist_ok=True)
            with p_.open("a") as f: f.write(json.dumps(out, default=float) + "\n")
        return out


def history(n: int = 96) -> dict:
    p_ = log_path()
    if not p_.exists(): return dict(rows=[], record={})
    seen, recs = set(), []
    for line in p_.read_text().splitlines():
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        if r["target_candle_open"] in seen: continue
        seen.add(r["target_candle_open"]); recs.append(r)
    c = data.drop_open_candle(pd.read_parquet(predictor.cache_path(TF))); oc = {str(t): int(cl > op) for t, op, cl in zip(c.open_time, c.open, c.close)}
    rows = []
    for r in recs:
        a = oc.get(r["target_candle_open"])
        rows.append(dict(target=r["target_candle_open"], p=r["p_bullish"], prediction=r["prediction"], conf=r["confidence"], gate=r["gate"]["passed"], ta=r["ta"]["prediction"],
                         v1_prediction=r["v1"]["prediction"], v1_p=r["v1"]["p_bullish"], v1_gate=r["v1"]["gate_passed"], actual=a,
                         hit=None if a is None else int((r["prediction"] == "BULLISH") == bool(a)), v1_hit=None if a is None else int((r["v1"]["prediction"] == "BULLISH") == bool(a))))
    res = [x for x in rows if x["hit"] is not None]
    rate = lambda key, sel: (lambda s: dict(n=len(s), hit_rate=round(float(np.mean(s)), 4) if s else None))([x[key] for x in res if sel(x)])
    return dict(rows=rows[-n:], record=dict(calls=len(rows), resolved=len(res), v7=rate("hit", lambda x: True), v1=rate("v1_hit", lambda x: True), v7_gate=rate("hit", lambda x: x["gate"]), v1_gate=rate("v1_hit", lambda x: x["v1_gate"]), same_direction=round(float(np.mean([x["prediction"] == x["v1_prediction"] for x in rows])), 4) if rows else None))
