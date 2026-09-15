"""All-timeframe snapshot for the dashboard overview: current calls, gate status, recent strip, gate statistics.

Each timeframe is built independently, so one failure (Binance hiccup, missing model) shows up as an error on
that timeframe only. Snapshots are cached briefly and built under a lock so many viewers share one computation.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import llm, predictor

ML_GATE, TA_GATE = 0.10, 0.30
_TTL_S = 15
_lock = threading.Lock()
_snap: dict = {"at": 0.0, "data": None}


def _gate_counts(g: pd.DataFrame) -> dict:
    agree = g["ta_call"].notna() & (g["ml_call"] == g["ta_call"])
    mlc = (g["ml_p"] - 0.5).abs() * 2
    gate = agree & (mlc >= ML_GATE) & (g["ta_conf"] >= TA_GATE)
    return dict(total=int(len(g)), hits=int(g["ml_hit"].sum()),
                agree=int(agree.sum()), agree_hits=int(g.loc[agree, "ml_hit"].sum()),
                gate=int(gate.sum()), gate_hits=int(g.loc[gate, "ml_hit"].sum()),
                first=g["time"].iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ") if len(g) else None,
                last=g["time"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ") if len(g) else None)


def _one(interval: str, llm_auto: list[str]) -> dict:
    t0 = time.time()
    pr = predictor.predict(interval=interval, recent=96)
    ml, ta = pr["ml"], pr["ta"]
    total_w = sum(abs(w) for w in (ta.get("weights") or {}).values()) or len(ta.get("signals") or []) or 16
    agree = ta["prediction"] != "NEUTRAL" and ml["prediction"] == ta["prediction"]
    ml_ok, ta_ok = ml["confidence"] >= ML_GATE, ta["confidence"] >= TA_GATE
    # recent strip: each row's call is for the following candle
    strip = []
    step = pr["interval_seconds"]
    for r in pr["recent"][-25:]:
        mlc = abs(r["ml_p"] - 0.5) * 2
        tac = abs(r["ta_score"]) / total_w
        ml_dir = int(r["ml_p"] >= 0.5)
        ag = r["ta_dir"] is not None and r["ta_dir"] == ml_dir
        strip.append(dict(t=r["time"] + step, ml=ml_dir, ta=r["ta_dir"], actual=r["actual"], agree=bool(ag),
                          gate=bool(ag and mlc >= ML_GATE and tac >= TA_GATE), llm=r.get("llm_dir")))
    # LLM: stored result only, never a paid call from the overview
    if not llm.available():
        llm_info = dict(state="off")
    else:
        hit = llm.lookup(interval, pr["predicting_candle_open"])
        if hit is not None:
            llm_info = dict(state="result", prediction=hit["prediction"], confidence=hit["confidence"], source=hit.get("source"))
        else:
            llm_info = dict(state="pending" if interval in llm_auto else "not_asked")
        llm_info["recent"] = pr.get("llm_recent")
    # gate statistics over several windows
    h = predictor.full_history(interval)
    now = pd.Timestamp.now(tz="UTC")
    stats = dict(backtest=_gate_counts(h[h["source"] == "backtest"]),
                 live=_gate_counts(h[h["source"] == "live"]),
                 d7=_gate_counts(h[h["time"] >= now - pd.Timedelta(days=7)]),
                 h24=_gate_counts(h[h["time"] >= now - pd.Timedelta(hours=24)]))
    return dict(interval=interval, ok=True, build_s=round(time.time() - t0, 2),
                last_closed_candle=pr["last_closed_candle"], last_close=pr["last_close"],
                candle_open=pr["predicting_candle_open"], candle_close=pr["predicting_candle_close"],
                next_close_ts=pr["next_close_ts"], interval_seconds=step,
                ml=dict(prediction=ml["prediction"], p=ml["p_bullish"], confidence=ml["confidence"]),
                ta=dict(prediction=ta["prediction"], score=ta["score"], confidence=ta["confidence"]),
                gate=dict(agree=agree, ml_ok=ml_ok, ta_ok=ta_ok, passed=bool(agree and ml_ok and ta_ok),
                          call=ml["prediction"] if agree else None, ml_min=ML_GATE, ta_min=TA_GATE),
                llm=llm_info, strip=strip, stats=stats)


def snapshot(llm_auto: list[str], force: bool = False) -> dict:
    with _lock:
        if not force and _snap["data"] is not None and time.time() - _snap["at"] < _TTL_S:
            return _snap["data"]
        def safe(iv):
            try:
                return _one(iv, llm_auto)
            except Exception as e:  # noqa: BLE001 - one timeframe failing must not break the others
                return dict(interval=iv, ok=False, error=f"{type(e).__name__}: {e}")
        with ThreadPoolExecutor(max_workers=len(predictor.INTERVALS)) as ex:
            items = list(ex.map(safe, predictor.INTERVALS))
        data = dict(generated_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                    gate=dict(ml_min=ML_GATE, ta_min=TA_GATE), intervals=items)
        _snap.update(at=time.time(), data=data)
        return data
