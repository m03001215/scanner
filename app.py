"""Web UI: `python app.py [--port 8765] [--host 0.0.0.0]`, then open http://127.0.0.1:8765"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import io
import time

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from btcpred import calibration, data, intra, llm, overview, predictor, rl

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

log = logging.getLogger("uvicorn.error")


# ---- automatic LLM calls: one per closed candle on these timeframes (BTCPRED_LLM_AUTO="" turns it off) ----
LLM_AUTO = [iv.strip() for iv in os.environ.get("BTCPRED_LLM_AUTO", "15m,1h").split(",") if iv.strip() in predictor.INTERVALS]
LLM_AUTO_DELAY_S = 20          # wait after the candle boundary so Binance has the final closed candle
LLM_STATUS: dict[str, dict] = {iv: {} for iv in LLM_AUTO}


def _llm_job(interval: str, boundary: pd.Timestamp) -> None:
    """Ask the LLM about the candle opening at `boundary`, retrying while that candle is still young."""
    step = pd.Timedelta(milliseconds=data.INTERVAL_MS[interval])
    st = LLM_STATUS[interval]
    now = lambda: pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    st.update(last_attempt_at=now(), candle=str(boundary))
    have = llm.lookup(interval, str(boundary))
    if have is not None:                           # someone already asked (a click, or before a restart)
        st.update(last_ok_candle=str(boundary), last_prediction=have["prediction"], last_confidence=have["confidence"],
                  last_cached=True, last_error=None, last_skip=None)
        return
    deadline = boundary + step / 2                # a call after half the candle is of little use
    if pd.Timestamp.now(tz="UTC") >= deadline:
        st.update(last_skip=f"candle {boundary} was more than half over when checked")
        log.info("auto llm %s: skipped candle %s, more than half over", interval, boundary)
        return
    attempt = 0
    while pd.Timestamp.now(tz="UTC") < deadline:
        attempt += 1
        try:
            df = data.drop_open_candle(data.load_or_update(predictor.cache_path(interval), interval=interval,
                                                           days=predictor.boot_days(interval)))
            if df["open_time"].iloc[-1] < boundary - step:     # Binance has not published the closed candle yet
                raise RuntimeError("closed candle not available yet")
            out = llm.predict(df, interval, source="auto")
            st.update(last_ok_candle=out["predicting_candle_open"], last_prediction=out["prediction"],
                      last_confidence=out["confidence"], last_cached=out.get("cached"), last_error=None, last_skip=None,
                      last_latency_s=out.get("latency_s"), last_ok_at=now())
            log.info("auto llm %s %s -> %s %.2f%s", interval, out["predicting_candle_open"], out["prediction"],
                     out["confidence"], " (already had a result)" if out.get("cached") else "")
            return
        except Exception as e:  # noqa: BLE001
            st.update(last_error=f"{type(e).__name__}: {e}", last_error_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
            log.warning("auto llm %s attempt %d failed: %s", interval, attempt, e)
            time.sleep(15 if "not available yet" in str(e) else 60)
    st.update(last_skip=f"gave up on candle {boundary} after {attempt} attempt(s)")
    log.warning("auto llm %s gave up on candle %s after %d attempt(s)", interval, boundary, attempt)


async def _llm_scheduler() -> None:
    loop = asyncio.get_running_loop()
    now = pd.Timestamp.now(tz="UTC")
    # cover the candle that is already forming when the server starts, if nobody asked yet
    jobs = [loop.run_in_executor(None, _llm_job, iv, now.floor(pd.Timedelta(milliseconds=data.INTERVAL_MS[iv]))) for iv in LLM_AUTO]
    await asyncio.gather(*jobs, return_exceptions=True)
    while True:
        now = pd.Timestamp.now(tz="UTC")
        nxt = {iv: now.floor(pd.Timedelta(milliseconds=data.INTERVAL_MS[iv])) + pd.Timedelta(milliseconds=data.INTERVAL_MS[iv]) for iv in LLM_AUTO}
        for iv, t in nxt.items():
            LLM_STATUS[iv]["next_run_at"] = (t + pd.Timedelta(seconds=LLM_AUTO_DELAY_S)).strftime("%Y-%m-%dT%H:%M:%SZ")
        boundary = min(nxt.values())
        await asyncio.sleep(max(0.0, (boundary - now).total_seconds() + LLM_AUTO_DELAY_S))
        due = [iv for iv, t in nxt.items() if t == boundary]
        await asyncio.gather(*(loop.run_in_executor(None, _llm_job, iv, boundary) for iv in due), return_exceptions=True)


INTRA_AUTO = [iv for iv in os.environ.get("BTCPRED_INTRA_AUTO", "15m").split(",") if iv.strip() in intra.MINUTES]
INTRA_STATUS: dict = {iv: {} for iv in INTRA_AUTO}


async def _intra_scheduler() -> None:
    """Recompute the intra-candle v2 prediction once per minute and log it, so every candle gets a full path."""
    loop = asyncio.get_running_loop()
    while True:
        now = pd.Timestamp.now(tz="UTC")
        nxt = now.floor("min") + pd.Timedelta(minutes=1) + pd.Timedelta(seconds=8)
        await asyncio.sleep(max(0.0, (nxt - now).total_seconds()))
        for iv in INTRA_AUTO:
            if intra.load(iv) is None:
                continue
            try:
                out = await loop.run_in_executor(None, intra.snapshot, iv, True)
                INTRA_STATUS[iv] = dict(last_run=out["now"], state=out["state"], minute=out.get("minute"), p=out.get("p_next"), error=None)
            except Exception as e:  # noqa: BLE001
                INTRA_STATUS[iv] = dict(last_run=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), error=f"{type(e).__name__}: {e}")
                log.warning("intra scheduler %s: %s", iv, e)


@asynccontextmanager
async def lifespan(_: FastAPI):
    for iv in predictor.INTERVALS:  # warm the data cache + model so the first page load is fast
        try:
            predictor.predict(interval=iv)
            log.info("warm-up prediction done for %s", iv)
        except Exception as e:  # noqa: BLE001
            log.warning("warm-up failed for %s (will retry on request): %s", iv, e)
    intra_task = asyncio.create_task(_intra_scheduler()) if INTRA_AUTO else None
    task = None
    if LLM_AUTO and llm.available():
        task = asyncio.create_task(_llm_scheduler())
        log.info("automatic LLM calls on: %s", ", ".join(LLM_AUTO))
    else:
        log.info("automatic LLM calls off (%s)", "no LLM key" if LLM_AUTO else "BTCPRED_LLM_AUTO is empty")
    yield
    if task:
        task.cancel()
    if intra_task:
        intra_task.cancel()


app = FastAPI(title="BTCUSDT 15m Predictor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/intra")
def intra_page():
    return FileResponse(STATIC / "intra.html")


@app.get("/api/intra")
def api_intra(interval: str = "15m", candles: int = 96):
    """Intra-candle v2: prediction for the NEXT candle from the forming candle's state right now, the minute-by-minute
    probability path of the current candle, recent candles' paths with outcomes, the act-rule track record, and the
    walk-forward report by minute. Separate model and log from the at-close predictors."""
    if interval not in intra.MINUTES:
        raise HTTPException(status_code=404, detail=f"intra model supports {list(intra.MINUTES)}")
    loaded = intra.load(interval)
    if loaded is None:
        raise HTTPException(status_code=404, detail=f"no intra model for {interval}; run `python main.py train-intra -i {interval}`")
    _, meta = loaded
    try:
        snap = intra.snapshot(interval, record=False)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"snapshot failed: {type(e).__name__}: {e}")
    hist = intra.history(interval, candles=min(max(candles, 8), 500))
    rep = meta["report"]
    return dict(snap, history=hist, scheduler=INTRA_STATUS.get(interval, {}), auto=interval in INTRA_AUTO,
                model=dict(trained_at=meta["trained_at"], days=meta["days"], label=meta["label"], candles=rep["candles"],
                           by_minute=rep["by_minute"], act_rule=rep["act_rule"], final_minute=rep["final_minute"], folds=rep["folds"]))


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/predict")
def api_predict(interval: str = "15m", recent: int = 96):
    try:
        return JSONResponse(predictor.predict(interval=interval, recent=min(max(recent, 16), 500)))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/backtest")
def api_backtest(interval: str = "15m"):
    path = predictor.backtest_summary_path(interval)
    if interval not in predictor.INTERVALS or not path.exists():
        raise HTTPException(status_code=404, detail=f"no backtest summary for {interval}")
    return FileResponse(path, media_type="application/json")


@app.get("/api/overview")
def api_overview():
    """All timeframes at once: current ML/TA/LLM calls, gate status, recent results strip, and gate statistics
    (backtest, live since training, last 7 days, last 24 hours). Never makes a paid LLM call."""
    return overview.snapshot(LLM_AUTO)


@app.get("/api/calibration/{interval}")
def api_calibration(interval: str):
    """Current p_bullish calibrator for the interval: version, label mode, the raw -> calibrated lookup table, and the
    held-out reliability tables (raw and calibrated, all rows and gated rows) it was validated on."""
    if interval not in predictor.INTERVALS:
        raise HTTPException(status_code=404, detail=f"unsupported interval {interval!r}")
    cal = calibration.current(interval)
    if cal is None:
        raise HTTPException(status_code=404, detail=f"no calibrator for {interval}; run `python main.py calibrate -i {interval}`")
    return dict(interval=interval, version=cal["version"], model_hash=cal["model_hash"], model_hash_current=calibration.model_hash(interval),
                stale=cal["model_hash"] != calibration.model_hash(interval), method=cal["method"], labels=cal["labels"], label_tag=cal["label_tag"],
                label_note=cal.get("label_note"), n=cal["n"], n_gated=cal["n_gated"], oos_period=cal["oos_period"], fitted_at=cal["fitted_at"],
                gate_score_threshold=cal["gate_score_threshold"], fold_stable=cal["fold_stable"], evaluation=cal["evaluation"],
                lookup=cal["lookup"], isotonic=cal["table"], platt=cal["platt"], heldout_reliability=cal["heldout_reliability"])


@app.get("/api/rl")
def api_rl(interval: str = "15m"):
    """Reinforcement-learning policy (offline fitted Q-iteration): LONG / FLAT / SHORT for the candle forming now,
    Q-values in basis points, the recent position path, and the strict time-split test results with baselines."""
    if interval not in predictor.INTERVALS:
        raise HTTPException(status_code=404, detail=f"unsupported interval {interval!r}")
    df = data.drop_open_candle(data.load_or_update(predictor.cache_path(interval), interval=interval,
                                                   days=predictor.boot_days(interval)))
    out = rl.predict_last(interval, df)
    if out is None:
        raise HTTPException(status_code=404, detail=f"no RL policy for {interval}; run `python main.py train-rl -i {interval}`")
    step = pd.Timedelta(milliseconds=data.INTERVAL_MS[interval])
    return dict(out, interval=interval, last_closed_candle=str(df["open_time"].iloc[-1]),
                predicting_candle_open=str((df["close_time"].iloc[-1] + pd.Timedelta(milliseconds=1)).floor(step)))


@app.get("/api/llm/status")
def api_llm_status():
    """Whether the LLM analyst is configured, which timeframes run automatically, and the scheduler's last runs."""
    return dict(configured=llm.available(), provider=llm.provider(), model=llm.model_name() if llm.available() else None,
                auto_intervals=LLM_AUTO if llm.available() else [], auto_delay_s=LLM_AUTO_DELAY_S,
                scheduler={iv: LLM_STATUS[iv] for iv in LLM_AUTO} if llm.available() else {})


@app.get("/api/llm")
def api_llm(interval: str = "15m", cached_only: bool = False):
    """The LLM's call for the candle forming now, with its reasoning.

    At most one paid call per candle. `cached_only=true` never calls the model: it returns the stored
    result for the current candle, or 404 if there is none yet."""
    if interval not in predictor.INTERVALS:
        raise HTTPException(status_code=404, detail=f"unsupported interval {interval!r}")
    if not llm.available():
        raise HTTPException(status_code=503, detail="LLM predictor not configured: set OPENAI_API_KEY (or ANTHROPIC_API_KEY) on the server")
    df = data.drop_open_candle(data.load_or_update(predictor.cache_path(interval), interval=interval,
                                                   days=predictor.boot_days(interval)))
    step = pd.Timedelta(milliseconds=data.INTERVAL_MS[interval])
    candle_open = str((df["close_time"].iloc[-1] + pd.Timedelta(milliseconds=1)).floor(step))
    if cached_only:
        out = llm.lookup(interval, candle_open)
        if out is None:
            raise HTTPException(status_code=404, detail="no LLM result for the current candle yet")
        out = dict(out, cached=True)
    else:
        try:
            out = llm.predict(df, interval)
        except Exception as e:  # noqa: BLE001 - surface SDK/auth/rate-limit errors to the UI as text
            log.warning("llm predict failed: %s", e)
            raise HTTPException(status_code=502, detail=f"LLM call failed: {type(e).__name__}: {e}")
    return dict(out, interval=interval, auto=interval in LLM_AUTO, track_record=llm.track_record(interval, df))


@app.get("/api/history")
def api_history(
    interval: str = "15m",
    start: str | None = Query(None, description="ISO date/time, inclusive (UTC unless an offset is given), e.g. 2026-09-10T17:33:00Z"),
    end: str | None = Query(None, description="ISO date/time, exclusive"),
    agree: bool = Query(False, description="only candles where ML and TA made the same call (TA ties excluded)"),
    limit: int = Query(1000, ge=1, le=20000),
    offset: int = Query(0, ge=0),
    order: str = Query("asc", pattern="^(asc|desc)$"),
    format: str = Query("json", pattern="^(json|csv)$"),
):
    """Per-candle out-of-sample results for any period: ML probability/call, TA score/call, real outcome.

    `time` is the open of the predicted candle (UTC). Rows with source="backtest" come from the stored
    walk-forward replay; rows with source="live" cover every closed candle after it, computed with the
    deployed models. Neither is a log of what the page displayed at the time.
    """
    if interval not in predictor.INTERVALS:
        raise HTTPException(status_code=404, detail=f"unsupported interval {interval!r}; choose one of {predictor.INTERVALS}")
    def ts(v, name):
        try:
            t = pd.Timestamp(v)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"bad {name}: {e}")
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    t_start = ts(start, "start") if start else None
    t_end = ts(end, "end") if end else None
    if t_start is not None and t_end is not None and t_end <= t_start:
        raise HTTPException(status_code=400, detail="end must be after start")
    try:
        df = predictor.full_history(interval)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001 - Binance or model failure while building live rows
        log.warning("live history failed: %s", e)
        raise HTTPException(status_code=502, detail=f"could not build live rows: {type(e).__name__}: {e}")
    if t_start is not None:
        df = df[df["time"] >= t_start]
    if t_end is not None:
        df = df[df["time"] < t_end]
    if agree:
        df = df[df["ta_call"].notna() & (df["ml_call"] == df["ta_call"])]
    if order == "desc":
        df = df.iloc[::-1]
    total = len(df)
    if format == "csv":  # whole filtered range, ignores paging
        buf = io.StringIO()
        out = df.copy(); out["time"] = out["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out.to_csv(buf, index=False)
        fname = f"btcusdt_{interval}_history{'_agree' if agree else ''}.csv"
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{fname}"'})
    page = df.iloc[offset: offset + limit].copy()
    page["time"] = page["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    page = page.astype(object).where(page.notna(), None)
    rows = page.to_dict(orient="records")
    bull, bear = df[df["ml_call"] == "BULL"], df[df["ml_call"] == "BEAR"]
    acc = lambda x: round(float(x.mean()), 4) if len(x) else None
    summary = dict(
        candles=total,
        ml_acc=acc(df["ml_hit"]),
        ta_acc=acc(df["ta_hit"].dropna()),
        ta_calls=int(df["ta_hit"].notna().sum()),
        bull_calls=len(bull), bull_acc=acc(bull["ml_hit"]),
        bear_calls=len(bear), bear_acc=acc(bear["ml_hit"]),
        actual_bull_rate=acc((df["actual"] == "BULL").astype(int)),
        backtest_rows=int((df["source"] == "backtest").sum()), live_rows=int((df["source"] == "live").sum()),
    )
    return dict(interval=interval, agree=agree,
                start=t_start.strftime("%Y-%m-%dT%H:%M:%SZ") if t_start is not None else None,
                end=t_end.strftime("%Y-%m-%dT%H:%M:%SZ") if t_end is not None else None,
                total=total, offset=offset, returned=len(rows),
                next_offset=(offset + limit) if offset + limit < total else None,
                range=dict(first=rows[0]["time"] if rows else None, last=rows[-1]["time"] if rows else None),
                summary=summary, rows=rows)


@app.get("/api/report")
def api_report(interval: str = "15m"):
    try:
        _, meta = predictor._load_model(interval)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return dict(interval=interval, ml=meta["report"], ta=predictor._ta_report(interval))


if __name__ == "__main__":
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    a = ap.parse_args()
    uvicorn.run("app:app", host=a.host, port=a.port, reload=False)
