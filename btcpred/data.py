"""Fetch and cache Binance BTCUSDT klines."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

# Comma-separated list; the first host that answers is used. data-api.binance.vision is the
# market-data-only mirror, which is not geo-restricted the way api.binance.com is.
BASE_URLS = [u.strip().rstrip("/") + "/api/v3/klines" for u in os.environ.get(
    "BINANCE_BASE_URLS", "https://api.binance.com,https://data-api.binance.vision").split(",") if u.strip()]
_active_url: str | None = None
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
NUMERIC = ["open", "high", "low", "close", "volume", "quote_volume",
           "trades", "taker_buy_base", "taker_buy_quote"]

INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def _request(params: dict, retries: int = 5) -> list:
    global _active_url
    hosts = [_active_url] + [u for u in BASE_URLS if u != _active_url] if _active_url else list(BASE_URLS)
    last_err: Exception | None = None
    for url in hosts:
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, timeout=15)
                if r.status_code in (429, 418):
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code in (451, 403):  # geo-blocked: try the next host
                    last_err = requests.HTTPError(f"{r.status_code} from {url}")
                    break
                r.raise_for_status()
                _active_url = url
                return r.json()
            except requests.RequestException as e:
                last_err = e
                if attempt == retries - 1:
                    break
                time.sleep(2 ** attempt)
    raise RuntimeError(f"all Binance hosts failed: {last_err}")


def fetch_klines(symbol: str = "BTCUSDT", interval: str = "15m",
                 start_ms: int | None = None, end_ms: int | None = None,
                 limit: int = 1000) -> pd.DataFrame:
    """Fetch klines between start_ms and end_ms (inclusive), paginating as needed."""
    rows: list = []
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_ms is not None:
        params["endTime"] = end_ms
    cursor = start_ms
    while True:
        if cursor is not None:
            params["startTime"] = cursor
        batch = _request(params)
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if len(batch) < limit or cursor is None:
            break
        cursor = last_open + INTERVAL_MS[interval]
        if end_ms is not None and cursor > end_ms:
            break
        time.sleep(0.15)  # stay well under rate limits
    return _to_frame(rows)


def _to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        return df
    df = df.drop(columns=["ignore"])
    df[NUMERIC] = df[NUMERIC].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df


def load_or_update(cache: Path, symbol: str = "BTCUSDT", interval: str = "15m",
                   days: int = 730) -> pd.DataFrame:
    """Load cached klines and append anything newer from Binance."""
    now_ms = int(time.time() * 1000)
    if cache.exists():
        df = pd.read_parquet(cache)
        start = int(df["open_time"].iloc[-1].timestamp() * 1000) + INTERVAL_MS[interval]
    else:
        df = pd.DataFrame()
        start = now_ms - days * 86_400_000
    if start <= now_ms:
        new = fetch_klines(symbol, interval, start_ms=start, end_ms=now_ms)
        df = pd.concat([df, new], ignore_index=True) if not df.empty else new
        df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def drop_open_candle(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the still-forming last candle so features use only closed bars."""
    now = pd.Timestamp.now(tz="UTC")
    return df[df["close_time"] <= now].reset_index(drop=True)
