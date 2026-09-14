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
