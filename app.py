"""Web UI: `python app.py [--port 8765] [--host 0.0.0.0]`, then open http://127.0.0.1:8765"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import io

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from btcpred import data, llm, predictor

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    for iv in predictor.INTERVALS:  # warm the data cache + model so the first page load is fast
        try:
            predictor.predict(interval=iv)
            log.info("warm-up prediction done for %s", iv)
        except Exception as e:  # noqa: BLE001
            log.warning("warm-up failed for %s (will retry on request): %s", iv, e)
    yield


app = FastAPI(title="BTCUSDT 15m Predictor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


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


@app.get("/api/llm")
def api_llm(interval: str = "15m"):
    """The LLM's call for the candle forming now, with its reasoning. One API call per closed candle."""
    if interval not in predictor.INTERVALS:
        raise HTTPException(status_code=404, detail=f"unsupported interval {interval!r}")
    if not llm.available():
        raise HTTPException(status_code=503, detail="LLM predictor not configured: set OPENAI_API_KEY (or ANTHROPIC_API_KEY) on the server")
    df = data.drop_open_candle(data.load_or_update(predictor.cache_path(interval), interval=interval,
                                                   days=predictor.boot_days(interval)))
    try:
        out = llm.predict(df, interval)
    except Exception as e:  # noqa: BLE001 - surface SDK/auth/rate-limit errors to the UI as text
        log.warning("llm predict failed: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM call failed: {type(e).__name__}: {e}")
    out = dict(out, interval=interval, track_record=llm.track_record(interval, df))
    return out


@app.get("/api/history")
def api_history(
    interval: str = "15m",
    start: str | None = Query(None, description="ISO date/time, inclusive (UTC unless offset given), e.g. 2025-01-01"),
    end: str | None = Query(None, description="ISO date/time, exclusive"),
    limit: int = Query(1000, ge=1, le=20000),
    offset: int = Query(0, ge=0),
    order: str = Query("asc", pattern="^(asc|desc)$"),
    format: str = Query("json", pattern="^(json|csv)$"),
):
    """Per-candle out-of-sample backtest results: ML probability/call, TA score/call, real outcome.

    `time` is the open of the predicted candle (UTC). These are replayed backtest calls from
    models trained only on earlier data, not a log of what the live site showed at the time.
    """
    try:
        df = predictor.load_history(interval)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        if start:
            df = df[df["time"] >= pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)]
        if end:
            df = df[df["time"] < (pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end))]
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"bad start/end: {e}")
    if order == "desc":
        df = df.iloc[::-1]
    total = len(df)
    if format == "csv":  # whole filtered range, ignores paging
        buf = io.StringIO()
        out = df.copy(); out["time"] = out["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out.to_csv(buf, index=False)
        fname = f"btcusdt_{interval}_history.csv"
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{fname}"'})
    page = df.iloc[offset: offset + limit].copy()
    page["time"] = page["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    page = page.astype(object).where(page.notna(), None)
    rows = page.to_dict(orient="records")
    summary = dict(
        ml_acc=round(float(df["ml_hit"].mean()), 4) if total else None,
        ta_acc=round(float(df["ta_hit"].mean()), 4) if total and df["ta_hit"].notna().any() else None,
        ta_calls=int(df["ta_hit"].notna().sum()),
    )
    return dict(interval=interval, total=total, offset=offset, returned=len(rows),
                next_offset=(offset + limit) if offset + limit < total else None,
                range=dict(start=rows[0]["time"] if rows else None, end=rows[-1]["time"] if rows else None),
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
