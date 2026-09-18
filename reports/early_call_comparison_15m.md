# 15m early-call comparison: v1 at the open vs v2 at −60 s vs v3 at −10 s

Generated 2026-09-18 12:45 UTC from paired walk-forward backtests (`tools/compare_v2_minus60s_vs_v1.py`, `tools/compare_v3_minus10s_vs_v1.py`).

## Question

v1 predicts a 15m candle's direction at the moment it opens. v2 and v3 can predict the same candle before it opens: v2 at 60 s before the open (minute 14 of the candle before it, 1-minute bars), v3 at 10 s before (890 s elapsed, 10-second bars). What does the head start cost, and do the early versions pass the *Agree + Confident* gate as often as v1?

## Method

- Target: 15m candle close > open on Binance. The three versions are graded on the **same target candles**.
- Each pairing is a paired walk-forward: identical candles, identical fold boundaries (first 40% as the initial training block, five expanding folds, early stopping on the last 10% of each training block). Only the moment of prediction differs.
- *Agree + Confident* gate: the ML call agrees with the TA vote, ML confidence ≥ 0.10 and TA confidence ≥ 0.3. TA uses closed candles only, so it is the same for all three versions.
- v1 vs v2: 63,599 out-of-sample candles. v1 vs v3: 63,515. The two runs' candle sets differ by under 0.2%, so v1's figures differ slightly between them.

## Headline results

| | v1 at 0 s | v2 at −60 s | v3 at −10 s |
| --- | ---: | ---: | ---: |
| Accuracy | 52.86% | 52.30% | 52.36% |
| AUC | 0.540 | 0.533 | 0.533 |
| Log loss | 0.6908 | 0.6916 | 0.6916 |
| Accuracy at confidence ≥ 0.10 (share of candles) | 56.12% (24.4%) | 55.41% (13.8%) | 55.32% (15.3%) |
| Accuracy at confidence ≥ 0.20 (share) | 60.08% (2.4%) | 57.69% (0.3%) | 55.56% (0.5%) |
| **Gate pass rate** | **19.67%** | **10.02%** | **11.27%** |
| Gate passes per day | 18.9 | 9.6 | 10.8 |
| **Gated accuracy** | **56.63%** | **56.25%** | **56.23%** |
| Paired test vs v1 (McNemar z) | – | -3.07 | -2.52 |

v1 column from the v1-vs-v2 run. In the v1-vs-v3 run v1 scored 52.81% accuracy, a 20.89% gate pass rate and 56.44% gated accuracy. Sampling error on accuracy is ±0.20 points; |z| ≥ 2 is significant.

## Findings

1. **Calling early costs about half a point of accuracy, and the loss is real.** v2 is 0.56 points below v1 (z = -3.07); v3 is 0.45 points below (z = -2.52).
2. **The early versions pass the gate about half as often.** 10.0% for v2 and 11.3% for v3, against 19.7% for v1. The cause is ML confidence: the early models reach confidence 0.10 on 14%–15% of candles, v1 on 24%.
3. **The calls they do pass are as good as v1's.** Gated accuracy is 56.3% and 56.2% against 56.6%.
4. **v2 at −60 s and v3 at −10 s are equivalent.** Same accuracy, AUC and gated accuracy. The 50 seconds between them and the 10-second bar detail add nothing measurable; both pay the same price for not having the previous candle's close.
5. **The early gate is nearly a subset of v1's gate.** See the overlap tables: about four in five early passes also pass v1, and the few that don't are the weakest calls.
6. **A fixed confidence threshold can go quiet for months.** From Sep to Dec 2025 the pass rate fell to 7–8% for v1 and 0–2% for v2, because those folds' models were less confident.

## v1 vs v2 (−60 s)

- Same direction on 79.2% of candles; when they agree the call is right 53.26%. When they disagree (13,254 candles): v1 51.34%, v2 48.66%. Correlation of the probabilities: 0.770.
- v1 right / v2 wrong on 6,804 candles; v2 right / v1 wrong on 6,450.

| Gate outcome | Candles | Share | Accuracy |
| --- | ---: | ---: | --- |
| Both pass | 5,191 | 8.2% | 56.93% |
| v1 only | 7,322 | 11.5% | v1 56.42% |
| v2 only | 1,184 | 1.9% | v2 53.29% |
| Neither | 49,902 | 78.5% | |

81% of v2's passes also pass v1; 41% of v1's passes also pass v2.

Against real Polymarket settlement, 2,124 post-Aug-7 windows: accuracy v1 51.22%, v2 51.55%; gated v1 53.31% on 484, v2 54.01% on 237. Sampling error is about ±3.2 points on v2's gated calls, so settlement data does not separate them yet.

## v1 vs v3 (−10 s)

- Same direction on 79.9% of candles; when they agree the call is right 53.23%. When they disagree (12,787 candles): v1 51.11%, v3 48.89%. Correlation of the probabilities: 0.776.
- v1 right / v3 wrong on 6,536 candles; v3 right / v1 wrong on 6,251.

| Gate outcome | Candles | Share | Accuracy |
| --- | ---: | ---: | --- |
| Both pass | 5,902 | 9.3% | 56.78% |
| v1 only | 7,367 | 11.6% | v1 56.17% |
| v3 only | 1,258 | 2.0% | v3 53.66% |
| Neither | 48,988 | 77.1% | |

82% of v3's passes also pass v1; 44% of v1's passes also pass v3.

Against real Polymarket settlement, 2,124 post-Aug-7 windows: accuracy v1 51.18%, v3 50.38%; gated v1 54.08% on 466, v3 55.90% on 195. Sampling error is about ±3.6 points on v3's gated calls, so settlement data does not separate them yet.

## By month

| Month | v1 acc. | v2 acc. | v3 acc. | v1 gate pass | v2 gate pass | v3 gate pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-11 | 53.2% | 51.8% | 51.9% | 25% | 16% | 18% |
| 2024-12 | 53.3% | 52.6% | 52.5% | 24% | 17% | 16% |
| 2025-01 | 51.8% | 53.5% | 52.4% | 22% | 16% | 15% |
| 2025-02 | 52.1% | 52.7% | 52.5% | 25% | 17% | 16% |
| 2025-03 | 52.9% | 51.6% | 51.8% | 25% | 17% | 16% |
| 2025-04 | 53.8% | 53.4% | 54.4% | 23% | 11% | 16% |
| 2025-05 | 53.3% | 52.6% | 52.7% | 21% | 9% | 16% |
| 2025-06 | 51.1% | 51.6% | 51.7% | 23% | 10% | 17% |
| 2025-07 | 52.7% | 51.6% | 51.7% | 25% | 10% | 16% |
| 2025-08 | 52.3% | 51.7% | 50.9% | 18% | 6% | 10% |
| 2025-09 | 54.2% | 53.3% | 53.7% | 8% | 0% | 2% |
| 2025-10 | 52.4% | 50.7% | 51.1% | 8% | 0% | 4% |
| 2025-11 | 53.0% | 52.9% | 53.6% | 7% | 0% | 3% |
| 2025-12 | 54.4% | 52.6% | 52.5% | 8% | 2% | 4% |
| 2026-01 | 53.9% | 53.6% | 53.2% | 22% | 12% | 13% |
| 2026-02 | 51.9% | 51.4% | 51.4% | 23% | 13% | 14% |
| 2026-03 | 54.3% | 52.4% | 52.2% | 22% | 13% | 13% |
| 2026-04 | 53.6% | 52.3% | 52.8% | 22% | 11% | 12% |
| 2026-05 | 53.5% | 52.9% | 54.1% | 21% | 11% | 9% |
| 2026-06 | 51.7% | 51.5% | 51.8% | 23% | 12% | 10% |
| 2026-07 | 51.8% | 52.0% | 51.0% | 21% | 10% | 8% |
| 2026-08 | 52.6% | 51.9% | 51.9% | 21% | 10% | 8% |
| 2026-09 | 51.6% | 51.6% | 51.7% | 20% | 11% | 9% |

v1 was ahead of v2 in 18 of 23 months and ahead of v3 in 15 of 23.

## Recommendation

- **If accuracy and signal count matter most, call v1 at the open.** It is the most accurate and passes the gate twice as often.
- **If you need the lead time, use v2 at −60 s.** It gives the same result as v3 at −10 s with a simpler feed (one request a minute rather than one every 10 s), and 50 more seconds to act. Expect about 10 gate passes a day instead of 19, at the same ≈56% accuracy.
- **Best of both:** take v2's early pass as a provisional signal and confirm with v1 at the open. Candles that pass both gates were right 56.9% of the time, the highest of any subset here, and the two never pointed in opposite directions on those candles.
- **Replace the fixed 0.10 threshold with a rank-based one** (for example each model's top 20% most confident calls) so the gate does not go silent when a retrained model is less confident.

## Limits

- Labels are Binance candles; the Polymarket referee agrees with that label on 93.4% of windows.
- Accuracy is not profit: at these hit rates the gross edge is below trading fees.
- v1's figures here come from re-running its walk-forward on the comparison's candle set, so they differ by a few hundredths of a point from the dashboard's stored backtest.
