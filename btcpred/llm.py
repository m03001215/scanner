"""Third predictor: an LLM reads the recent candles and indicators and gives a call with a reason.

Providers: OpenAI (OPENAI_API_KEY) or Anthropic (ANTHROPIC_API_KEY); BTCPRED_LLM_PROVIDER overrides.
Independent of the ML and TA methods: the prompt contains only market data, never their calls.
One API call per closed candle per timeframe; results are cached in memory and appended to a
JSONL log so a running hit rate can be shown once outcomes are known.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from . import data
from .features import _atr, _rsi

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = {"openai": "gpt-5", "anthropic": "claude-opus-5"}
EFFORT = os.environ.get("BTCPRED_LLM_EFFORT", "medium")   # low | medium | high (OpenAI); + xhigh | max (Anthropic)
N_CANDLES = int(os.environ.get("BTCPRED_LLM_CANDLES", "48"))


def provider() -> str | None:
    forced = os.environ.get("BTCPRED_LLM_PROVIDER", "").strip().lower()
    if forced in ("openai", "anthropic"):
        return forced
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    return None


def model_name() -> str:
    return os.environ.get("BTCPRED_LLM_MODEL") or DEFAULT_MODEL.get(provider() or "openai", "gpt-5")

_CACHE: dict = {}


class CandleCall(BaseModel):
    direction: str = Field(description='"bullish" if the next candle is expected to close above its open, otherwise "bearish"')
    confidence: float = Field(ge=0, le=1, description="0 = coin flip, 1 = certain. Be calibrated: most calls on this timeframe deserve 0.05-0.3")
    reason: str = Field(description="2-4 sentences explaining the call, citing specific numbers from the data")
    key_factors: list[str] = Field(description="3-5 short bullet points, the strongest evidence for and against")


SYSTEM = """You are a disciplined short-term BTCUSDT market analyst. You are given the most recent closed
candles on one timeframe, a snapshot of standard indicators computed from them, and nothing else.
Predict whether the NEXT candle (the one opening right after the last closed candle) will close
above its open (bullish) or below (bearish).

Guidance from the historical record on this exact task:
- Short-horizon BTC candles are close to a coin flip. Directional accuracy above ~55% is excellent.
- On 15-minute and 1-hour candles, mean reversion has historically beaten momentum: after a strong
  move, a pullback is somewhat more likely than a continuation; oversold/overbought readings
  (RSI, stochastic, Bollinger touches, proximity to the recent range low/high) are the most
  reliable signals; textbook trend-following signals have been slightly worse than random.
- Order-flow (taker buy ratio, volume spikes) is informative mainly when it diverges from price.

Be honest about uncertainty. Keep confidence low unless several independent signals line up.
Cite concrete numbers from the data in your reason. Never mention that you are an AI or a language model."""


def available() -> bool:
    return provider() is not None


def log_path(interval: str) -> Path:
    return ROOT / "data" / f"llm_log_{interval}.jsonl"


def build_context(df: pd.DataFrame, interval: str, n: int = N_CANDLES) -> str:
    """Plain-text market brief: candle table + indicator snapshot, all from closed bars."""
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    atr = _atr(df, 14)
    rsi14, rsi7 = _rsi(c, 14), _rsi(c, 7)
    ema = {n_: c.ewm(span=n_, adjust=False).mean() for n_ in (8, 20, 50, 200)}
    macd = ema_12 = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    stoch = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)
    taker = df["taker_buy_base"] / v.replace(0, np.nan)
    vol_ma = v.rolling(96).mean()
    i = df.index[-1]
    last = df.loc[i]
    tail = df.iloc[-n:]

    rows = ["time_utc            open      high      low       close     vol      taker_buy%  body%"]
    for r in tail.itertuples():
        body = (r.close - r.open) / r.open * 100
        rows.append(f"{r.open_time:%Y-%m-%d %H:%M}  {r.open:9.1f} {r.high:9.1f} {r.low:9.1f} {r.close:9.1f} {r.volume:8.1f}  "
                    f"{(r.taker_buy_base / r.volume * 100 if r.volume else 0):5.1f}      {body:+.2f}")
    span_h = n * data.INTERVAL_MS[interval] / 3_600_000
    snap = f"""Indicator snapshot at the last close ({last['open_time']:%Y-%m-%d %H:%M} UTC, close {last['close']:.1f}):
- Return last 1 / 4 / 16 candles: {np.log(c.iloc[-1]/c.iloc[-2])*100:+.2f}% / {np.log(c.iloc[-1]/c.iloc[-5])*100:+.2f}% / {np.log(c.iloc[-1]/c.iloc[-17])*100:+.2f}%
- ATR(14): {atr.loc[i]:.1f} ({atr.loc[i]/c.iloc[-1]*100:.2f}% of price); last candle range = {(h.iloc[-1]-l.iloc[-1])/atr.loc[i]:.2f} ATR
- RSI(7) {rsi7.loc[i]:.1f}, RSI(14) {rsi14.loc[i]:.1f}; Stochastic(14) %K {stoch.loc[i]:.1f}
- Price vs EMA8 / EMA20 / EMA50 / EMA200: {(c.iloc[-1]-ema[8].loc[i])/atr.loc[i]:+.2f} / {(c.iloc[-1]-ema[20].loc[i])/atr.loc[i]:+.2f} / {(c.iloc[-1]-ema[50].loc[i])/atr.loc[i]:+.2f} / {(c.iloc[-1]-ema[200].loc[i])/atr.loc[i]:+.2f} ATR
- MACD(12,26,9) histogram: {hist.loc[i]:+.1f} (prev {hist.iloc[-2]:+.1f})
- Bollinger(20,2): position {((c.iloc[-1]-sma20.loc[i])/(2*sd20.loc[i])):+.2f} (-1 = lower band, +1 = upper band), width {4*sd20.loc[i]/sma20.loc[i]*100:.2f}%
- Last 96-candle high {h.iloc[-96:].max():.1f} ({(h.iloc[-96:].max()-c.iloc[-1])/atr.loc[i]:.2f} ATR above), low {l.iloc[-96:].min():.1f} ({(c.iloc[-1]-l.iloc[-96:].min())/atr.loc[i]:.2f} ATR below)
- Volume: last candle {v.iloc[-1]/vol_ma.loc[i]:.2f}x the 96-candle average; taker buy ratio last candle {taker.loc[i]*100:.1f}%, last 8 avg {taker.iloc[-8:].mean()*100:.1f}%
- Consecutive same-direction closes: {_streak(c - o)}"""
    return (f"Timeframe: {interval} candles. Last {n} closed candles ({span_h:.0f} hours):\n" + "\n".join(rows)
            + "\n\n" + snap + f"\n\nPredict the candle opening at {last['close_time'] + pd.Timedelta(milliseconds=1):%Y-%m-%d %H:%M} UTC.")


def _streak(body: pd.Series) -> str:
    s = np.sign(body.values); k = 1
    while k < len(s) and s[-k - 1] == s[-1] and s[-1] != 0:
        k += 1
    return f"{k} {'up' if s[-1] > 0 else 'down' if s[-1] < 0 else 'flat'}"


def predict(df: pd.DataFrame, interval: str) -> dict:
    """Return Claude's call for the candle after df's last (closed) row. Cached per candle."""
    prov = provider()
    if prov is None:
        raise RuntimeError("LLM predictor not configured: set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    last_bar = df.iloc[-1]
    key = (interval, str(last_bar["open_time"]))
    if key in _CACHE:
        return _CACHE[key]

    context = build_context(df, interval)
    model = model_name()
    t0 = time.time()
    call, usage = (_call_openai if prov == "openai" else _call_anthropic)(model, context)
    direction = "BULLISH" if call.direction.strip().lower().startswith("bull") else "BEARISH"
    step = pd.Timedelta(milliseconds=data.INTERVAL_MS[interval])
    out = dict(
        prediction=direction, confidence=round(float(call.confidence), 3),
        reason=call.reason.strip(), key_factors=[k.strip() for k in call.key_factors][:6],
        provider=prov, model=model, effort=EFFORT,
        last_closed_candle=str(last_bar["open_time"]),
        predicting_candle_open=str((last_bar["close_time"] + pd.Timedelta(milliseconds=1)).floor(step)),
        latency_s=round(time.time() - t0, 1),
        usage=usage,
    )
    for k in [k for k in _CACHE if k[0] == interval and k != key]:
        del _CACHE[k]
    _CACHE[key] = out
    _append_log(interval, out)
    return out


def _call_openai(model: str, context: str) -> tuple[CandleCall, dict]:
    """OpenAI Responses API with a Pydantic-enforced JSON schema."""
    from openai import OpenAI  # lazy import: the app must run without either SDK configured
    client = OpenAI()
    kwargs: dict = {}
    if model.startswith(("gpt-5", "o1", "o3", "o4")):   # reasoning models accept an effort knob
        kwargs["reasoning"] = {"effort": EFFORT if EFFORT in ("low", "medium", "high") else "medium"}
    response = client.responses.parse(
        model=model,
        input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": context}],
        text_format=CandleCall,
        **kwargs,
    )
    if response.output_parsed is None:
        raise RuntimeError(f"{model} returned no structured prediction (status={response.status})")
    u = response.usage
    cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
    return response.output_parsed, dict(input_tokens=u.input_tokens, output_tokens=u.output_tokens, cache_read_input_tokens=cached)


def _call_anthropic(model: str, context: str) -> tuple[CandleCall, dict]:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        max_tokens=4000,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": context}],
        output_format=CandleCall,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"{model} returned no prediction (stop_reason={response.stop_reason})")
    u = response.usage
    return response.parsed_output, dict(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                                        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0)


def _append_log(interval: str, out: dict) -> None:
    p = log_path(interval)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({k: out[k] for k in ("predicting_candle_open", "prediction", "confidence", "model")}) + "\n")


def track_record(interval: str, df: pd.DataFrame) -> dict:
    """Hit rate of logged Claude calls whose candle has since closed."""
    p = log_path(interval)
    if not p.exists():
        return dict(calls=0, resolved=0, hit_rate=None)
    seen, rows = set(), []
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["predicting_candle_open"] in seen:
            continue
        seen.add(r["predicting_candle_open"]); rows.append(r)
    outcome = {str(t): int(cl > op) for t, op, cl in zip(df["open_time"], df["open"], df["close"])}
    hits = [(r["prediction"] == "BULLISH") == bool(outcome[r["predicting_candle_open"]])
            for r in rows if r["predicting_candle_open"] in outcome]
    return dict(calls=len(rows), resolved=len(hits), hit_rate=round(float(np.mean(hits)), 4) if hits else None)


def replay(df: pd.DataFrame, interval: str, n: int, workers: int = 4) -> pd.DataFrame:
    """Point-in-time replay over the last n closed candles: for each target candle the model sees only
    the candles before it. Costs one API call per candle. Returns one row per candle with the outcome."""
    from concurrent.futures import ThreadPoolExecutor
    prov = provider()
    if prov is None:
        raise RuntimeError("LLM predictor not configured: set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    model = model_name()
    fn = _call_openai if prov == "openai" else _call_anthropic
    targets = list(range(len(df) - n, len(df)))          # index of each predicted candle

    def one(i):
        hist = df.iloc[:i]                                  # strictly before the target candle
        t0 = time.time()
        try:
            call, usage = fn(model, build_context(hist, interval))
            pred = "BULLISH" if call.direction.strip().lower().startswith("bull") else "BEARISH"
            return dict(candle=df["open_time"].iloc[i], prediction=pred, confidence=float(call.confidence),
                        reason=call.reason.strip(), actual="BULLISH" if df["close"].iloc[i] > df["open"].iloc[i] else "BEARISH",
                        latency_s=round(time.time() - t0, 1), error=None)
        except Exception as e:  # noqa: BLE001
            return dict(candle=df["open_time"].iloc[i], prediction=None, confidence=None, reason=None,
                        actual="BULLISH" if df["close"].iloc[i] > df["open"].iloc[i] else "BEARISH",
                        latency_s=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, targets))
    out = pd.DataFrame(rows)
    out["hit"] = (out["prediction"] == out["actual"]).where(out["prediction"].notna())
    out["model"] = model
    return out
