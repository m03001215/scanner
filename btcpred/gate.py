"""v6: a learned gate on top of v1. v1 supplies the direction unchanged; v6 scores P(v1's call is right) from v1's own
outputs, the individual TA votes and pre-candle regime features, and passes the call when the score is in the top share.

Trained only on v1's out-of-sample walk-forward predictions (models/15m/history.csv.gz + live rows), itself walk-forward,
so the gate never learns from in-sample confidence. Separate model, log, endpoint (/api/v6) and page (/v6).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import data, predictor, ta
from .features import _atr

ROOT = Path(__file__).resolve().parent.parent
TF = "15m"; N_DAY = 96
PASS_SHARES = (0.30, 0.20, 0.10, 0.05)
# Deliberately tiny: with a 53% base rate and ~40k training rows, anything larger fits noise and scores below v1's own
# confidence rank (see reports/learned_gate_v6.md). Even this variant only matches it.
PARAMS = dict(objective="binary", learning_rate=0.01, num_leaves=4, min_child_samples=2000, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7)


def v6_dir() -> Path: return ROOT / "models" / TF / "v6"
def log_path() -> Path: return ROOT / "data" / "v6_log_15m.jsonl"


# ----------------------------------------------------------------------------- features
def regime_frame(c: pd.DataFrame) -> pd.DataFrame:
    """Pre-candle regime features indexed by the FEATURE candle's open_time (the closed candle before the target)."""
    c = c.reset_index(drop=True)
    lr = np.log(c.close / c.close.shift(1)); atr = _atr(c, 14)
    ret = c.close / c.open - 1; up = (ret > 0).astype(int)
    run = np.zeros(len(c)); s = up.values
    for i in range(1, len(c)): run[i] = run[i - 1] + 1 if s[i] == s[i - 1] else 1
    f = pd.DataFrame({
        "r_vol_day": lr.rolling(N_DAY).std() * np.sqrt(N_DAY), "r_spike": lr.rolling(16).std() / lr.rolling(N_DAY * 7).std(),
        "r_trend_4h": c.close / c.close.shift(16) - 1, "r_trend_day": c.close / c.close.shift(N_DAY) - 1,
        "r_last_ret_atr": (c.close - c.open) / atr, "r_last_size_atr": (c.high - c.low) / atr, "r_last_up": up,
        "r_run": run * np.where(up == 1, 1, -1),                       # +k: k consecutive up candles; -k: down
        "r_range_pos": (c.close - c.low.rolling(N_DAY).min()) / (c.high.rolling(N_DAY).max() - c.low.rolling(N_DAY).min()),
        "r_atr_pct": atr / c.close, "r_vol_ratio": c.volume / c.volume.rolling(N_DAY).mean(),
        "r_hour": c.open_time.dt.tz_convert("America/New_York").dt.hour, "r_dow": c.open_time.dt.dayofweek,
        "r_ret3_atr": (c.close - c.close.shift(3)) / atr, "r_ret8_atr": (c.close - c.close.shift(8)) / atr,
    })
    f.index = pd.DatetimeIndex(c.open_time)      # set after construction: passing index= would realign the integer-indexed columns to NaN
    return f


def gate_features(hist: pd.DataFrame, c: pd.DataFrame, weights: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """hist: v1 OOS rows (time = target open, ml_p, ml_call, ta_score, ta_conf, ta_call, ml_hit). Returns X, y, target times."""
    step = pd.Timedelta(minutes=15)
    reg = regime_frame(c); sig = ta.signals(c.reset_index(drop=True)); sig.index = c.reset_index(drop=True).open_time
    ft = pd.DatetimeIndex(hist["time"]) - step                         # feature candle
    R = reg.reindex(ft); S = sig.reindex(ft)
    S.columns = ["s_" + k for k in S.columns]
    dirn = np.where(hist["ml_call"].values == "BULL", 1, -1)
    mlc = np.abs(hist["ml_p"].values - 0.5) * 2
    tadir = np.where(hist["ta_call"].values == "BULL", 1, np.where(hist["ta_call"].values == "BEAR", -1, 0))
    # v1's own recent record, known at call time: misses in the last 4 / 16 resolved calls, current miss streak
    hit = hist["ml_hit"].values.astype(float)
    prev_hit = pd.Series(hit).shift(1)
    miss4 = (1 - prev_hit).rolling(4).sum().values; miss16 = (1 - prev_hit).rolling(16).sum().values
    streak = np.zeros(len(hit))
    for i in range(1, len(hit)): streak[i] = streak[i - 1] + 1 if hit[i - 1] == 0 else 0
    X = pd.DataFrame({
        "ml_conf": mlc, "ml_dir": dirn, "ta_score_signed": hist["ta_score"].values * dirn, "ta_conf": hist["ta_conf"].values,
        "agree": (tadir == dirn).astype(int), "ta_tie": (tadir == 0).astype(int),
        "call_vs_last": dirn * np.where(R["r_last_up"].values == 1, 1, -1),          # +1 call continues last candle, -1 contrarian
        "call_vs_run": dirn * np.sign(R["r_run"].values) * np.abs(R["r_run"].values), # signed run length relative to the call
        "call_vs_trend4h": dirn * R["r_trend_4h"].values, "call_vs_trend_day": dirn * R["r_trend_day"].values,
        "call_vs_ret3": dirn * R["r_ret3_atr"].values, "call_vs_ret8": dirn * R["r_ret8_atr"].values,
        "v1_miss4": miss4, "v1_miss16": miss16, "v1_streak": streak,
    }, index=hist.index)
    for k in ("r_vol_day", "r_spike", "r_last_size_atr", "r_range_pos", "r_atr_pct", "r_vol_ratio", "r_hour", "r_dow"): X[k] = R[k].values
    X["r_trend_4h_abs"] = np.abs(R["r_trend_4h"].values); X["r_trend_day_abs"] = np.abs(R["r_trend_day"].values); X["r_run_abs"] = np.abs(R["r_run"].values)
    for k in S.columns: X[k] = S[k].values * dirn                                     # each TA signal relative to the call
    return X, pd.Series(hist["ml_hit"].values, index=hist.index), pd.Series(pd.DatetimeIndex(hist["time"]), index=hist.index)


# ----------------------------------------------------------------------------- training
def _standard_gate(hist: pd.DataFrame) -> np.ndarray:
    mlc = np.abs(hist["ml_p"].values - 0.5) * 2
    return (hist["ta_call"].values == hist["ml_call"].values) & (mlc >= 0.10) & (hist["ta_conf"].values >= 0.3)


def train(log=print) -> dict:
    hist = predictor.full_history(TF).reset_index(drop=True)
    c = data.drop_open_candle(pd.read_parquet(predictor.cache_path(TF)))
    weights = json.load(open(predictor.ta_report_path(TF)))["weights"]
    X, y, t = gate_features(hist, c, weights)
    ok = X.notna().all(axis=1) & y.notna()
    X, y, t, hist = X[ok].reset_index(drop=True), y[ok].astype(int).reset_index(drop=True), t[ok].reset_index(drop=True), hist[ok].reset_index(drop=True)
    n = len(X); first = int(n * 0.4); edges = np.linspace(first, n, 6, dtype=int)
    log(f"[v6 gate] {n:,} v1 OOS calls {t.iloc[0]:%Y-%m-%d} .. {t.iloc[-1]:%Y-%m-%d}, {X.shape[1]} features; v1 accuracy {y.mean()*100:.2f}%")
    score = np.full(n, np.nan); folds = []
    for i in range(5):
        tr_end, te_end = edges[i], edges[i + 1]; va = int(tr_end * 0.9)
        b = lgb.train(PARAMS, lgb.Dataset(X.iloc[:va], y.iloc[:va]), 3000, valid_sets=[lgb.Dataset(X.iloc[va:tr_end], y.iloc[va:tr_end])], callbacks=[lgb.early_stopping(150, verbose=False)])
        score[tr_end:te_end] = b.predict(X.iloc[tr_end:te_end], num_iteration=b.best_iteration)
        folds.append(dict(test_start=str(t.iloc[tr_end]), test_end=str(t.iloc[te_end - 1]), best_iter=int(b.best_iteration)))
    oos = slice(first, n); S, Y, H = score[oos], y.values[oos], hist.iloc[oos]
    std = _standard_gate(H); mlc = np.abs(H["ml_p"].values - 0.5) * 2
    from sklearn.metrics import roc_auc_score
    rep = dict(n=int(len(Y)), v1_acc=float(Y.mean()), gate_auc=float(roc_auc_score(Y, S)), conf_auc=float(roc_auc_score(Y, mlc)),
               standard_gate=dict(pass_rate=float(std.mean()), acc=float(Y[std].mean())), by_share=[], thresholds={}, folds=folds,
               test_start=str(t.iloc[first]), test_end=str(t.iloc[-1]))
    for q in PASS_SHARES:
        thr = float(np.quantile(S, 1 - q)); m = S >= thr; cthr = float(np.quantile(mlc, 1 - q)); cm = mlc >= cthr
        rep["by_share"].append(dict(share=q, threshold=thr, pass_rate=float(m.mean()), acc=float(Y[m].mean()), conf_rank_acc=float(Y[cm].mean()), n=int(m.sum())))
        rep["thresholds"][str(q)] = thr
    # at the standard gate's own pass rate
    thr_std = float(np.quantile(S, 1 - std.mean())); m = S >= thr_std
    rep["at_standard_pass_rate"] = dict(pass_rate=float(m.mean()), acc=float(Y[m].mean()), overlap_with_standard=float((m & std).sum() / std.sum()))
    # monthly stability at 20%
    thr20 = rep["thresholds"]["0.2"]; g20 = S >= thr20
    mon = pd.DataFrame({"m": t.iloc[oos].dt.tz_convert(None).dt.to_period("M").astype(str).values, "y": Y, "g": g20, "s": std})
    rep["monthly"] = [dict(month=k, n=len(g), gate_acc=float(g.y[g.g].mean()) if g.g.any() else None, gate_pass=float(g.g.mean()), std_acc=float(g.y[g.s].mean()) if g.s.any() else None, std_pass=float(g.s.mean())) for k, g in mon.groupby("m")]
    # what the gate learned: importance
    rounds = int(np.median([f["best_iter"] for f in folds])) or 300
    final = lgb.train(PARAMS, lgb.Dataset(X, y), rounds)
    imp = pd.Series(final.feature_importance("gain"), index=X.columns); rep["importance"] = {k: round(float(v / imp.sum() * 100), 2) for k, v in imp.sort_values(ascending=False).head(15).items()}
    # streak-regime check: accuracy of gated calls when |run| >= 3 and the call is contrarian
    contra = (X["call_vs_last"].values[oos] == -1) & (X["r_run_abs"].values[oos] >= 3)
    rep["contrarian_in_run"] = dict(share=float(contra.mean()), v1_acc=float(Y[contra].mean()), standard_gate_pass=float(std[contra].mean()), learned_gate20_pass=float(g20[contra].mean()))
    for r in rep["by_share"]: log(f"  top {r['share']*100:.0f}% by gate score: acc {r['acc']*100:.2f}% (v1 confidence rank at same share: {r['conf_rank_acc']*100:.2f}%)")
    log(f"  standard gate: pass {rep['standard_gate']['pass_rate']*100:.1f}% acc {rep['standard_gate']['acc']*100:.2f}% | learned gate at the same pass rate: acc {rep['at_standard_pass_rate']['acc']*100:.2f}% | gate AUC {rep['gate_auc']:.3f} vs confidence AUC {rep['conf_auc']:.3f}")
    out = v6_dir(); out.mkdir(parents=True, exist_ok=True); final.save_model(str(out / "model.txt"))
    (out / "meta.json").write_text(json.dumps(dict(features=list(X.columns), rounds=rounds, report=rep, params=PARAMS, trained_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")), indent=1, default=float))
    return rep


# ----------------------------------------------------------------------------- live
_M: dict = {}; _LOCK = threading.Lock(); _LAST: dict = {}


def load():
    d = v6_dir()
    if not (d / "model.txt").exists(): return None
    mt = (d / "model.txt").stat().st_mtime
    if _M.get("mt") != mt: _M.update(mt=mt, m=lgb.Booster(model_file=str(d / "model.txt")), meta=json.loads((d / "meta.json").read_text()))
    return _M["m"], _M["meta"]


def score_now(record: bool = True) -> dict:
    """Score v1's live call for the candle forming now. v1 is called through predictor.predict, unchanged."""
    loaded = load()
    if loaded is None: raise FileNotFoundError("no v6 gate; run `python main.py train-v6`")
    booster, meta = loaded
    with _LOCK:
        pr = predictor.predict(interval=TF, recent=16)
        key = pr["predicting_candle_open"]
        if _LAST.get("target_candle_open") == key: return _LAST
        hist = predictor.full_history(TF).reset_index(drop=True)      # v1's resolved OOS record, for the streak features
        c = data.drop_open_candle(pd.read_parquet(predictor.cache_path(TF)))
        weights = json.load(open(predictor.ta_report_path(TF)))["weights"]
        ta_call = "BULL" if pr["ta"]["prediction"] == "BULLISH" else ("BEAR" if pr["ta"]["prediction"] == "BEARISH" else None)
        row = pd.DataFrame([dict(time=pd.Timestamp(key), ml_p=pr["ml"]["p_bullish"], ml_call="BULL" if pr["ml"]["prediction"] == "BULLISH" else "BEAR",
                                 ta_score=pr["ta"]["score"], ta_conf=pr["ta"]["confidence"], ta_call=ta_call, ml_hit=np.nan)])
        h2 = pd.concat([hist[["time", "ml_p", "ml_call", "ta_score", "ta_conf", "ta_call", "ml_hit"]], row], ignore_index=True)
        h2["time"] = pd.to_datetime(h2["time"], utc=True)
        X, _, _ = gate_features(h2, c, weights)
        x = X.iloc[[-1]][meta["features"]]
        s = float(booster.predict(x.fillna(0))[0]); th = meta["report"]["thresholds"]
        tier = next((q for q in ("0.05", "0.1", "0.2", "0.3") if s >= th[q]), None)
        out = dict(version="v6", target_candle_open=key, target_candle_close=pr["predicting_candle_close"], last_closed_candle=pr["last_closed_candle"],
                   made_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                   v1=dict(prediction=pr["ml"]["prediction"], p_bullish=pr["ml"]["p_bullish"], confidence=pr["ml"]["confidence"], ta=pr["ta"]["prediction"], ta_score=pr["ta"]["score"], ta_conf=pr["ta"]["confidence"],
                           standard_gate=bool(ta_call == row.ml_call.iloc[0] and pr["ml"]["confidence"] >= 0.10 and pr["ta"]["confidence"] >= 0.3)),
                   gate_score=round(s, 4), pass_top30=s >= th["0.3"], pass_top20=s >= th["0.2"], pass_top10=s >= th["0.1"], pass_top5=s >= th["0.05"], tier=tier,
                   decision=("ACT " + pr["ml"]["prediction"]) if s >= th["0.2"] else "SKIP",
                   regime={k: (None if pd.isna(v) else round(float(v), 4)) for k, v in x.iloc[0].items() if k.startswith(("r_", "call_vs", "v1_"))})
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
    c = data.drop_open_candle(pd.read_parquet(predictor.cache_path(TF)))
    oc = {str(t): int(cl > op) for t, op, cl in zip(c.open_time, c.open, c.close)}
    rows = []
    for r in recs:
        a = oc.get(r["target_candle_open"]); hit = None if a is None else int((r["v1"]["prediction"] == "BULLISH") == bool(a))
        rows.append(dict(target=r["target_candle_open"], prediction=r["v1"]["prediction"], p=r["v1"]["p_bullish"], conf=r["v1"]["confidence"], ta=r["v1"]["ta"], standard_gate=r["v1"]["standard_gate"],
                         score=r["gate_score"], tier=r["tier"], decision=r["decision"], actual=a, hit=hit))
    res = [x for x in rows if x["hit"] is not None]
    def rate(sel): s = [x["hit"] for x in res if sel(x)]; return dict(n=len(s), hit_rate=round(float(np.mean(s)), 4) if s else None)
    return dict(rows=rows[-n:], record=dict(calls=len(rows), resolved=len(res), all=rate(lambda x: True), standard_gate=rate(lambda x: x["standard_gate"]),
                                            top30=rate(lambda x: x["tier"] in ("0.05", "0.1", "0.2", "0.3")), top20=rate(lambda x: x["tier"] in ("0.05", "0.1", "0.2")), top10=rate(lambda x: x["tier"] in ("0.05", "0.1"))))
