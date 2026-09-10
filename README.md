# BTCUSDT Next-Candle Predictor (5m / 15m / 1h / 4h)

Predicts whether the **next** Binance BTCUSDT candle, on the **5-minute, 15-minute,
1-hour or 4-hour** timeframe, will close bullish (close > open) or bearish, using
independent methods on multi-year public kline data:

1. **ML** - a LightGBM classifier over 50 engineered features.
2. **TA** - a vote of 18 classic technical-analysis signals (EMA, MACD, RSI,
   Stochastic, Bollinger, VWAP, candlestick patterns, support/resistance,
   volume and taker-flow), with per-signal weights calibrated on history.
3. **LLM analyst** (optional) - a language model (OpenAI GPT-5 by default, or Claude)
   reads the last 48 closed candles plus an indicator snapshot and returns a call, a
   confidence, and a written reason. Needs `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

`predict` prints both results and whether they agree. A web dashboard shows the
same plus a candlestick chart, the ML probability history, the TA signal
breakdown and a recent hit/miss table.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py fetch --days 1095   # download / update klines cache
.venv/bin/python main.py train --days 1095   # ML: walk-forward evaluate + train final model
.venv/bin/python main.py backtest-ta --days 1095  # TA: calibrate signal weights, evaluate out-of-sample
.venv/bin/python main.py predict             # both predictions for the candle forming now
.venv/bin/python main.py predict --json      # machine-readable output
```

`predict` refreshes the kline cache, drops the still-open candle, builds features
from the last closed bar, and prints `P(bullish)` for the next candle.

## How it works

| Module | Role |
|---|---|
| `btcpred/data.py` | Paginated Binance `/api/v3/klines` fetch with retry, parquet cache, incremental updates |
| `btcpred/features.py` | 50 features: lagged log returns, rolling vol, ATR-normalised candle anatomy, EMA/MACD/RSI/Bollinger, taker-buy order-flow, volume z-scores, VWAP distance, cyclical time-of-day / day-of-week |
| `btcpred/model.py` | LightGBM binary classifier, expanding-window walk-forward validation, early stopping |
| `btcpred/ta.py` | 18 rule-based TA signals voting +1/-1/0, weight calibration, backtest |
| `btcpred/predictor.py` | Shared prediction entry point (ML + TA + recent history), cached per closed candle |
| `main.py` | CLI |
| `app.py`, `static/index.html` | FastAPI backend + single-page dashboard (lightweight-charts) |

Label: `1` if candle *t+1* closes above its open, else `0`. Features for row *t*
use only bars `<= t`, so there is no look-ahead.

## Out-of-sample results

For each timeframe both methods are evaluated on the same window (the newest 60% of
history). The ML model uses 5 expanding walk-forward folds; the TA weights are
calibrated on the oldest 40% and frozen.

**Summary across timeframes** (ML confidence on the dashboard scale, 2·|p−0.5|)

| Timeframe | Test window | Candles | ML | ML conf ≥ 0.10 | ML conf ≥ 0.20 | TA calibrated | Agree, ML ≥ 0.10 & TA ≥ 0.3 |
|---|---|---|---|---|---|---|---|
| 5m | Jun 2025 – Sep 2026 | 126k | 51.6% | 55.1% (4% of candles) | 57.5% (0.05%) | 52.2% (53% coverage) | 54.9% (3%) |
| 15m | Nov 2024 – Sep 2026 | 63k | 52.9% | 56.1% (25%) | 58.4% (2%) | 52.4% (93%) | 56.7% (20%) |
| 1h | May 2023 – Sep 2026 | 29k | 53.7% | 56.6% (41%) | 60.7% (8%) | 52.8% (95%) | 57.3% (26%) |
| 4h | Apr 2021 – Sep 2026 | 12k | 54.3% | 57.0% (51%) | 59.7% (16%) | 54.0% (87%) | 57.0% (15%) |

The edge grows with the timeframe. 5m is almost pure noise; the model is confident
on fewer than 4% of candles. 4h has the strongest signal but far fewer candles, so its
monthly accuracy swings widely (44% to 64%).

**15m** - test window Nov 2024 - Sep 2026, ~63k candles

| Method | Accuracy | Coverage | Confident subset |
|---|---|---|---|
| ML (LightGBM) | **52.9%** | 100% | 56.1% on the 24% of candles with \|p-0.5\| >= 0.05 |
| TA, calibrated weights | **52.4%** | 93% (ties = no call) | 54.8% on the 36% of candles with confidence >= 0.4 |
| TA, textbook equal-weight vote | 47.6% | 94% | gets *worse* with confidence (45.2% at >= 0.4) |
| Majority-class baseline | 50.2% | 100% | |

ML AUC 0.542, log loss 0.6906 (coin flip = 0.6931). Every ML fold scored between 52.4% and 53.8%.

**1h** - test window May 2023 - Sep 2026, ~29k candles

| Method | Accuracy | Coverage | Confident subset |
|---|---|---|---|
| ML (LightGBM) | **53.7%** | 100% | 56.6% on the 41% of candles with \|p-0.5\| >= 0.05; 60.7% on the 8% with >= 0.10 |
| TA, calibrated weights | **52.8%** | 95% | 55.5% on the 27% of candles with confidence >= 0.4 |
| TA, textbook equal-weight vote | 47.0% | 93% | |
| Majority-class baseline | 50.7% | 100% | |

ML AUC 0.553, log loss 0.6888. ML folds ranged 52.3% - 54.7%. The 1h edge is slightly
larger than 15m, and the same mean-reversion pattern holds: every trend-following
signal is inverted by calibration on both timeframes.

**5m** - test window Jun 2025 - Sep 2026, ~126k candles

| Method | Accuracy | Coverage | Confident subset |
|---|---|---|---|
| ML (LightGBM) | **51.6%** | 100% | 54.9% on the 5% of candles with \|p-0.5\| >= 0.05 |
| TA, calibrated weights | **52.2%** | 54% (many ties) | 54.5% on the 10% with confidence >= 0.4 |
| TA, textbook vote | 48.5% | 94% | |
| Majority-class baseline | 50.3% | 100% | |

ML AUC 0.523. Five-minute candles are the noisiest timeframe: the edge is real but
tiny, and the model is rarely confident.

**4h** - test window Apr 2021 - Sep 2026, ~11.9k candles

| Method | Accuracy | Coverage | Confident subset |
|---|---|---|---|
| ML (LightGBM) | **54.4%** | 100% | 56.7% on the 51% with \|p-0.5\| >= 0.05; 59.6% on the 16% with >= 0.10 |
| TA, calibrated weights | **54.0%** | 87% | 55.3% on the 57% with confidence >= 0.2 |
| TA, textbook vote | 47.3% | 94% | |
| Majority-class baseline | 50.8% | 100% | |

ML AUC 0.561. The 4h timeframe has the largest edge, but also the fewest candles, so
its numbers carry wider error bars (about +/-1 point on the overall figure).

The edge grows with the timeframe: 51.6% (5m) -> 52.9% (15m) -> 53.7% (1h) -> 54.4% (4h).

### Why textbook TA loses on 15m BTC

Per-signal hit rates (test window) split cleanly into two camps:

| Camp | Signals | Hit rate |
|---|---|---|
| Mean reversion | Bollinger band touch, Stochastic < 20 / > 80, RSI < 30 / > 70, near 96-bar support/resistance | 54% - 56% |
| Trend / momentum following | price vs EMA20, MACD histogram, ROC(4), VWAP side, volume-confirmed bar, taker flow, three soldiers/crows, 96-bar breakout | 43% - 49% |

On this timeframe a bar that moves *with* the short-term trend is more likely to be
followed by a pullback than a continuation. The calibration step therefore gives the
mean-reversion signals weight +1, **inverts** the momentum signals (weight -1), and
drops the ones near 50% (EMA crossovers, Stochastic cross, engulfing). Weights are
learned only on the calibration window and held fixed on the test window, so the
52.4% is genuinely out-of-sample. `predict` shows the raw votes, the textbook call,
and which signals were inverted or ignored.

## Honest caveats

* A 15-minute BTC candle is close to a coin flip. ~53% directional accuracy is
  in line with what public OHLCV data can offer; anything claiming 70%+ on this
  task is almost certainly leaking future information.
* Accuracy is not profit. Fees (taker 0.1% round trip on Binance spot without
  discounts) and slippage exceed the typical 15m move, so this edge is **not**
  tradeable as-is. Treat the output as a weak signal, not a trading system.
* Use the `confidence` field: predictions near 0.5 carry almost no information.
* Retrain periodically (`train`) as market regime shifts.

### Exposing the UI from a VPS

`deploy/` contains a systemd unit, an nginx site (port 8090, HTTP basic auth) and
an installer. Run once:

```bash
sudo bash deploy/install.sh <username> <password>
```

Then open `http://<vps-ip>:8090`. Make sure TCP 8090 is also allowed in your
provider's cloud firewall. The app itself stays bound to 127.0.0.1; only nginx
is reachable from outside. For HTTPS, point a domain at the VPS, change
`server_name` in `/etc/nginx/sites-available/btcpred` and run
`sudo certbot --nginx -d yourdomain`.

Without any server changes, an SSH tunnel also works from your own machine:
`ssh -L 8765:127.0.0.1:8765 developer@<vps-ip>` then open `http://localhost:8765`.

## LLM analyst

```bash
export OPENAI_API_KEY=sk-proj-...                # or ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python main.py predict -i 1h --llm     # CLI
.venv/bin/python app.py                          # dashboard card 3 becomes active
```

| Env var | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | unset | Enables the analyst; OpenAI is used when both are set. On Render, set it in the service's Environment tab. |
| `BTCPRED_LLM_PROVIDER` | auto | Force `openai` or `anthropic` |
| `BTCPRED_LLM_MODEL` | `gpt-5` (OpenAI) / `claude-opus-5` (Anthropic) | Model id |
| `BTCPRED_LLM_EFFORT` | `medium` | Reasoning depth: `low`, `medium`, `high` (Anthropic also `xhigh`, `max`). GPT-5 at medium takes ~30 s per call; `low` is several times faster. |
| `BTCPRED_LLM_CANDLES` | `48` | Closed candles included in the prompt |

The prompt contains only market data (candle table + RSI, EMAs, MACD, Bollinger,
Stochastic, ATR, range, volume, taker flow), never the ML or TA calls, so the three
methods stay independent. Structured output (a JSON schema) guarantees a parseable
call every time. The prompt is ~1.6k tokens. Calls happen lazily, once per closed candle
per timeframe, only when the dashboard or API is hit. Each call is appended to
`data/llm_log_<tf>.jsonl` so the card can show a running hit rate. For a small
point-in-time replay, `main.py backtest-llm -i 1h --n 24` re-asks the model for each of
the last N closed candles using only the candles before it (one paid call each) and
writes `models/<tf>/llm_replay.csv`; a full multi-year backtest is not offered because
it would mean tens of thousands of paid calls.

## Deploying to a free platform (Render)

The repo ships a `Dockerfile` and a `render.yaml` Blueprint for a **free Render web
service in Frankfurt** (Binance's main API returns HTTP 451 to US IPs; the app also
falls back to the `data-api.binance.vision` mirror).

1. Push this repo to GitHub.
2. In Render: **New -> Blueprint**, pick the repo, click **Apply**. Render reads
   `render.yaml`, builds the image and gives you `https://btc-15m-predictor.onrender.com`.

Notes:
* The free instance sleeps after 15 min without traffic; the first request afterwards
  takes ~30-60 s while it wakes and downloads the last 45 days of candles (~5 s).
* The disk is ephemeral, so the kline cache is rebuilt on every start; the trained
  model and TA weights are committed in `models/` and need no retraining.
* Same image runs on Koyeb, Fly.io, Hugging Face Spaces or Cloud Run - just make sure
  the region is outside the US, or set `BINANCE_BASE_URLS=https://data-api.binance.vision`.
