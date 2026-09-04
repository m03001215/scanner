# BTCUSDT 15m Next-Candle Predictor

Predicts whether the **next** Binance BTCUSDT 15-minute candle will close bullish
(close > open) or bearish, using two independent methods on ~3 years of public
kline data:

1. **ML** - a LightGBM classifier over 50 engineered features.
2. **TA** - a vote of 18 classic technical-analysis signals (EMA, MACD, RSI,
   Stochastic, Bollinger, VWAP, candlestick patterns, support/resistance,
   volume and taker-flow), with per-signal weights calibrated on history.

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

Both methods are evaluated on the same window: **Nov 2024 - Sep 2026, ~63k candles**.
The ML model uses 5 expanding walk-forward folds; the TA weights are calibrated on
Sep 2023 - Nov 2024 and frozen.

| Method | Accuracy | Coverage | Confident subset |
|---|---|---|---|
| ML (LightGBM) | **52.9%** | 100% | 56.1% on the 24% of candles with \|p-0.5\| >= 0.05 |
| TA, calibrated weights | **52.4%** | 93% (ties = no call) | 54.8% on the 36% of candles with confidence >= 0.4 |
| TA, textbook equal-weight vote | 47.6% | 94% | gets *worse* with confidence (45.2% at >= 0.4) |
| Majority-class baseline | 50.2% | 100% | |

ML AUC 0.542, log loss 0.6906 (coin flip = 0.6931). Every ML fold scored between 52.4% and 53.8%.

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
