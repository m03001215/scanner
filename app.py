"""Web UI: `python app.py [--port 8765] [--host 0.0.0.0]`, then open http://127.0.0.1:8765"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from btcpred import predictor

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
