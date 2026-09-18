# BTCUSDT Next-Candle Predictor (5m / 15m / 1h / 4h)

Predicts whether the **next** Binance BTCUSDT candle, on the **5-minute, 15-minute,
1-hour or 4-hour** timeframe, will close bullish (close > open) or bearish, using
independent methods on multi-year public kline data:

1. **ML** - a LightGBM classifier over 50 engineered features.
2. **TA** - a vote of 18 classic technical-analysis signals (EMA, MACD, RSI,
   Stochastic, Bollinger, VWAP, candlestick patterns, support/resistance,
   volume and taker-flow), with per-signal weights calibrated on history.
3. **RL policy** - an offline reinforcement-learning agent (fitted Q-iteration with a LightGBM
   value function) that chooses long, flat or short for the next candle, trained on the
   fee-adjusted return so it learns when a call is worth trading. Evaluated on a strict time
   split against buy-and-hold and rule baselines, across several seeds.
4. **LLM analyst** (optional) - a language model (OpenAI GPT-5 by default, or Claude)
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

## API

All endpoints take `interval=5m|15m|1h|4h` (default `15m`). Times are UTC.

| Endpoint | Returns |
|---|---|
| `GET /api/overview` | All four timeframes at once: current ML/TA calls with confidence, gate status (agree, and agree + ML ≥ 0.10 + TA ≥ 0.3), the stored LLM result if any, a 25-candle results strip, and gate pass counts and accuracy for the backtest, live since training, last 7 days and last 24 hours. Cached for 15 s; never makes a paid LLM call; a failing timeframe is reported in its own entry |
| `GET /api/predict?interval=1h&recent=96` | Live ML + TA prediction for the candle forming now, plus the last `recent` candles (16-500) with each method's call and outcome |
| `GET /api/calibration/{interval}` | Current p_bullish calibrator: version (`<model hash>-<timestamp>`), label mode (`settlement` or `binance`), the raw → calibrated lookup table (201 points), the isotonic breakpoints and Platt parameters, the held-out reliability tables it was validated on, and `stale` (true when the model has been retrained since the calibrator was fit) |
| `GET /api/intra?interval=15m` | Intra-candle v2: prediction for the **next** candle from the forming candle's state at the current minute, the minute-by-minute probability path, recent candles' paths with outcomes, the act-rule track record, and the walk-forward report by minute. Separate model (`models/15m/intra/`) and log; page at `/intra` |
| `GET /api/intra10` | Intra-candle v3: same target as v2 (the next 15m candle) but the forming candle is described by 10-second bars and recomputed every 10 s. Separate model (`models/15m/intra10/`) and log; page at `/intra10`; includes v2's by-minute results for comparison |
| `GET /api/v4` | v4: v1's model and TA vote called 30 s before the target 15m candle opens, on a stand-in for the forming candle. Latest call (`p_bullish`, `prediction`, `confidence`, TA vote, gate), timing to the next call, logged calls with outcomes, and the paired walk-forward comparison with v1. A request in the last 30 s of a candle makes the call if the scheduler hasn't yet. Page at `/v4` |
| `GET /api/rl?interval=1h` | RL policy for the candle forming now: `action` (LONG/FLAT/SHORT), Q-values per action in bp, edge vs flat, position change and fee, the last 96 candles' position path, and the time-split `test` block (return after fees, buy-and-hold in the same window, excess over market drift per seed, seed agreement, Sharpe, drawdown, exposure) plus rule `baselines`. 404 until `train-rl` has run for that timeframe |
| `GET /api/llm?interval=1h` | LLM analyst call for the candle forming now, with reason and key factors. 503 when no LLM key is set; one paid call per candle, then cached |
| `GET /api/history?interval=1h&start=2026-09-10T17:33:00Z&end=2026-09-14&agree=true` | Per-candle out-of-sample results for any period, up to the last closed candle. See below |
| `GET /api/backtest?interval=1h` | Backtest summary: accuracy by month, by confidence, and when ML and TA agree |
| `GET /api/report?interval=1h` | Full ML walk-forward and TA calibration reports |
| `GET /healthz` | Liveness check |

### `/api/history`

| Parameter | Meaning |
|---|---|
| `start`, `end` | Period to return. A date or date-time, with or without an offset (`2026-09-10T13:33:00-04:00` works). Start inclusive, end exclusive. Both optional. |
| `agree` | `true` keeps only candles where ML and TA made the same call; TA ties are excluded |
| `limit`, `offset` | Paging; `limit` up to 20000, follow `next_offset` |
| `order` | `asc` or `desc` |
| `format` | `json`, or `csv` to download the whole filtered period |

Each row: `time` (open of the predicted candle), `ml_p`, `ml_call`, `ta_score`, `ta_conf`,
`ta_call`, `actual`, `ml_hit`, `ta_hit`, `source`. The `summary` block covers the whole
filtered period: candles, ML and TA accuracy, bull/bear call counts and accuracy, the actual
bull rate, and how many rows came from each source.

Rows with `source: "backtest"` come from the stored walk-forward replay
(`models/<tf>/history.csv.gz`, regenerated by `main.py backtest`). Every closed candle after
it is filled in with `source: "live"`, computed with the deployed ML model and TA weights,
which were trained on data ending before those candles. Neither is a log of what the page
displayed at the time, and live rows are recomputed if the models are retrained.

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

## Probability calibration of p_bullish

`ml.p_bullish` is a model score, not a probability. Every `/api/predict` response also carries
`ml.p_bullish_cal`, `ml.calibration_version`, `ml.calibration_labels` (`settlement` | `binance`)
and `ml.calibration_n`; `p_bullish` itself is never changed, the consumer chooses the field.

```bash
.venv/bin/python main.py calibrate -i 15m          # one interval: report + versioned calibrator
.venv/bin/python main.py calibrate --all           # all four + reports/calibration_summary.md
```

- **Data:** only stored walk-forward out-of-sample rows (`/api/history` source `backtest`); live rows and in-sample predictions are never used. One calibrator per (interval, model version).
- **Labels:** `data/settlement_labels_<tf>.csv` (`window_start_utc, settled_up`) when present → outputs tagged `settlement-calibrated`; otherwise Binance close ≥ open → `binance-calibrated`. Rows without a label are skipped and counted.
- **Method:** isotonic regression fitted to equal-count bin means with ≥ 1000 rows per step (plain per-row isotonic and Platt scaling are reported for comparison; plain isotonic overfits the fitting half badly on 5m). Fit on the first half of the OOS period, evaluated on the second; the deployed calibrator is refit on all OOS rows.
- **Refit policy:** the calibrator refits automatically at the end of `train` and `backtest` for that interval and the version bumps each time; `/api/calibration/{tf}.stale` flags a calibrator older than the model.
- **Outputs:** `reports/calibration_<tf>.md` with reliability diagrams (all rows, gated rows, by call direction), held-out Brier/ECE, fold stability; `reports/calibration_summary.md`; `calibrators/p_bullish_<tf>_<version>.json`; `models/<tf>/calibration.json` (served copy).

## Intra-candle predictor (v2, separate page at `/intra`)

The at-close predictors call the candle that has just opened. Version 2 predicts the candle
**after** the one forming now, and re-evaluates every minute as the forming candle develops, so a
call can be made 1-14 minutes before its target candle opens. State at minute k of the forming
15m candle: the 50 standard features of the last *closed* candle, the forming candle's partial
open/high/low/last/volume after k minutes (from 1-minute candles, `data/btcusdt_1m.parquet` for
training, a small live 1m cache at run time), and k itself. One LightGBM model serves every k;
label = next candle close > open on Binance. Trained with `main.py train-intra -i 15m`.

Walk-forward out of sample (63,500 test candles from a 105,833-candle history, Sep 2023 - Sep 2026):

| Lead before the target opens | Accuracy | Confident share (conf ≥ 0.10) | Confident accuracy |
|---|---|---|---|
| 14 min (minute 1) | 52.1% | 6.9% | 54.7% |
| 10 min (minute 5) | 52.3% | 7.4% | 54.7% |
| 5 min (minute 10) | 52.4% | 10.3% | 55.8% |
| 1 min (minute 14) | 52.4% | 10.5% | 56.0% |
| v1 at-close model, for reference | 52.8% | 25% | 55.9% |

Act rule: act at the first minute where confidence ≥ 0.10. In the backtest it fires on 18.7% of
candles, is right 54.4% of the time, and gives a mean lead of 10 minutes. The server recomputes
once per minute (`BTCPRED_INTRA_AUTO`, default `15m`; empty string turns it off) and logs each
minute to `data/intra_log_15m.jsonl`, so the page can show every candle's path and a live track
record. Nothing on the v1 dashboard or in its API uses this model.

## Intra-candle predictor v3: 10-second bars (separate page at `/intra10`)

Same target and protocol as v2, but the forming candle is described by 10-second bars built from
Binance 1-second klines (`data/bars10s/`, 9.9M bars from Aug 2023, no gaps), with 13 extra
features: returns over the last 30/60/90/180 s, realised volatility, taker-buy share since open
and over the last 60 s, volume rate, share of up bars, VWAP distance, drawdown and drawup, all in
ATR units, plus elapsed seconds. Training rows are taken every 30 s of each candle (29 per candle,
3.07M rows); the server recomputes every 10 s from live 1 s klines. Trained with
`main.py train-intra10`; `BTCPRED_INTRA10_AUTO=0` turns the scheduler off.

Walk-forward, 63,500 test candles, same folds as v2:

| Elapsed | Lead | v3 accuracy | v3 confident (share) | v2 accuracy | v2 confident (share) |
|---|---|---|---|---|---|
| 60 s | 840 s | 51.8% | 54.3% (11%) | 52.1% | 54.7% (7%) |
| 300 s | 600 s | 52.0% | 54.4% (13%) | 52.3% | 54.7% (7%) |
| 600 s | 300 s | 52.4% | 55.3% (15%) | 52.4% | 55.8% (10%) |
| 840 s | 60 s | 52.4% | 55.1% (15%) | 52.4% | 56.0% (11%) |

Act rule (first step with confidence ≥ 0.10): v3 fires on 30% of candles at 53.7% with a mean lead
of 662 s; v2 fires on 19% at 54.4% with a mean lead of 10 min. **The 10-second detail adds no
accuracy**: v3 matches v2 at every lead within sampling error and its confident tier is broader
but slightly less accurate. It exists to settle that question empirically and for the faster
refresh; v2 remains the better choice on the evidence.

## v4: v1 called 30 s before the candle opens (separate page at `/v4`)

Everything is the same as v1 (the 50 features, LightGBM, the 18-signal TA vote and the Agree + Confident
gate) except the moment of the call. At 870 s into the forming 15m candle, that candle is replaced by a
**stand-in**: its open, running high and low, last price as the close, and volume, quote volume, trades and
taker-buy scaled by 900/870. The stand-in is treated as closed, v1's features and TA signals are computed on
it, and the model predicts the candle after it. The model is trained on the same construction for every
historical candle (10-second bars, 105k rows), so train and serve match; TA weights are recalibrated on the
stand-in signals. `main.py train-v4`; `BTCPRED_V4_AUTO=0` turns the scheduler off.

The stand-in is a close proxy: its close is a median 1.25 bp from the real close, and it has the forming
candle's direction right 96.2% of the time.

Paired walk-forward against v1 on the identical 63,093 out-of-sample candles, same folds:

| | v4 at −30 s | v1 at 0 s |
|---|---|---|
| Accuracy | 52.42% | 52.80% |
| AUC | 0.538 | 0.541 |
| Accuracy at confidence ≥ 0.10 (share) | 55.69% (20.6%) | 56.45% (22.3%) |
| Gate pass rate | 16.99% | 18.90% |
| Gated accuracy | 56.32% | 56.75% |

Paired test z = -3.03: v1 is still ahead, by 0.39 points, but v4 keeps 90% of v1's
gate passes. For comparison, v2 at −60 s and v3 at −10 s keep about half (`reports/early_call_comparison_15m.md`).
Keeping v1's exact features, rather than adding partial-candle features beside the last closed candle, is
what preserves the model's confidence.

## Reinforcement-learning policy

```bash
.venv/bin/python main.py train-rl -i 1h --days 2000        # train + strict time-split evaluation, 3 seeds
.venv/bin/python main.py train-rl -i 4h --days 3300 --cost-bp 5 --gamma 0.9 --seeds 1 2 3 4 5
```

State = the 50 ML features plus the current position; actions = LONG / FLAT / SHORT for the next
candle; reward = position × next-candle return − 5 bp per side on every position change. The
Q-function is a heavily regularised LightGBM regressor refit for up to 8 fitted-Q iterations, with
the iteration chosen on a validation slice inside the training window. The oldest 60% of history
trains, the newest 40% tests; three seeds are trained so a policy that fits noise shows up as
seed disagreement. Baselines on the same test window: buy-and-hold, always-short, ML every
candle, the confidence gate for one candle, and gate-and-hold. The deployed policy is retrained
on all data. Artifacts: `models/<tf>/rl/q_model.txt` and `report.json`.

Results as of 2026-09-16 (test window is the newest 40% of each history, fees included):

| Timeframe | Test return | Buy-and-hold | Excess over drift (seeds) | Seed agreement | Verdict |
|---|---|---|---|---|---|
| 4h | +11,053 bp over 3.6 y | +15,506 bp | +4,747 to +7,128 bp | 65% | positive for every seed, Sharpe ≈ 0.7, but drawdowns near 100 bp-of-notional and only one market cycle |
| 1h | +3,358 bp over 2.2 y | +4,979 bp | −226 to +2,591 bp | 26% | profit mostly market drift; seeds disagree |
| 15m | −3,279 bp | −2,417 bp | negative | 80% | loses money on unseen data for every seed |
| 5m | −3,795 bp | −677 bp | negative | 0% | no skill: two seeds hold one position for the whole test, one loses money churning |

Direction accuracy when in the market is about 50–51% on every timeframe: whatever the 4h policy
earns comes from timing exposure, not from calling direction better than the ML model.

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
| `BTCPRED_LLM_AUTO` | `15m,1h` | Timeframes the server asks automatically after every candle closes. Set to an empty string to turn automatic calls off. Only active when an LLM key is set. |

The prompt contains only market data (candle table + RSI, EMAs, MACD, Bollinger,
Stochastic, ATR, range, volume, taker flow), never the ML or TA calls, so the three
methods stay independent. Structured output (a JSON schema) guarantees a parseable
call every time. The prompt is ~1.6k tokens.

**Automatic calls.** On the timeframes in `BTCPRED_LLM_AUTO` (15m and 1h by default) the
server asks the model about 20 s after each candle closes, retrying for up to half the
candle if Binance or the model call fails. On startup it also covers the current candle
if nobody has asked yet and less than half of it has passed. At GPT-5 medium effort that
is about 120 calls a day, roughly $3-4/day. `GET /api/llm/status` shows which timeframes
are on auto, the next run time, and the last result or error per timeframe.

**Manual calls.** On other timeframes card 3 has an "Ask the model" button. Either way
there is at most one paid call per candle per timeframe: a per-timeframe lock stops a
click and the scheduler from calling at the same time, and every answer (including its
reason) is appended to `data/llm_log_<tf>.jsonl`, so a restarted server shows it again
without a new call. `GET /api/llm?interval=15m&cached_only=true` returns the stored
result for the current candle and never calls the model. For a small
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
