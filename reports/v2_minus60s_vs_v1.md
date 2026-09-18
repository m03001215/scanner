# 15m: v2 at −60 s vs v1 at 0 s, paired walk-forward

Run 2026-09-18 11:44 UTC. Target: the 15m candle's direction (close > open, Binance). **v1** predicts at the candle's open from the closed history; **v2** predicts the same candle 60 s earlier, from the closed history plus the first 14 minutes of the candle before it (1-minute bars). Identical 105,998 candles (2023-09-10 → 2026-09-18), identical fold boundaries, first 40% as the initial training block.

| Same target candles, out of sample (n = 63,599) | v1 at 0 s | v2 at −60 s | v3 at −10 s |
| --- | ---: | ---: | ---: |
| Accuracy | 52.86% | 52.30% | 52.37% |
| AUC | 0.540 | 0.533 | 0.533 |
| Log loss | 0.6908 | 0.6916 | 0.6916 |
| ECE (10 bins) | 0.75% | 0.77% | 0.85% |
| Accuracy, confidence ≥ 0.10 (share) | 56.12% (24.4%) | 55.41% (13.8%) | 55.34% (15.2%) |
| Accuracy, confidence ≥ 0.20 (share) | 60.08% (2.4%) | 57.69% (0.3%) | 55.41% (0.5%) |
| **Agree + Confident gate: pass rate** | **19.67%** (12,513) | **10.02%** (6,375) | **11.25%** (7,142) |
| **Agree + Confident gate: accuracy** | **56.63%** | **56.25%** | **56.26%** |
| Bullish calls | 55.8% | 56.6% | 56.9% |
| Trees per fold | [117, 109, 49, 308, 141] | [311, 120, 57, 423, 255] | see v3 report |

The v3 column is from the separate v3 run, matched on the 63,458 shared target candles; its fold boundaries differ by a few candles.

Sampling error on accuracy: ±0.20 points (1σ). Paired comparison (McNemar): v1 right / v2 wrong on 6,804 candles, v2 right / v1 wrong on 6,450; z = -3.07 (|z| ≥ 2 is significant).

## Do the two calls agree?

- Same direction on **79.2%** of candles. When they agree: 53.26% (n=50,345). When they disagree: v1 51.34%, v2 48.66% (n=13,254).
- Correlation of the two probabilities: 0.770.

## Agree + Confident gate: do the same candles pass?

| Outcome | Candles | Share | Accuracy |
| --- | ---: | ---: | --- |
| Both pass | 5,191 | 8.2% | 56.93% (opposite directions on 0) |
| v1 only | 7,322 | 11.5% | v1 56.42% |
| v2 only | 1,184 | 1.9% | v2 53.29% |
| Neither | 49,902 | 78.5% | |

81% of v2's passes also pass v1; 41% of v1's passes also pass v2. Passes per day: v1 18.9, v2 9.6.

## By month

| Month | Candles | v1 | v2 | v1 gate pass | v2 gate pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024-11 | 578 | 53.3% | 51.9% | 25% | 16% |
| 2024-12 | 2,976 | 53.3% | 52.6% | 24% | 17% |
| 2025-01 | 2,976 | 51.7% | 53.5% | 22% | 16% |
| 2025-02 | 2,688 | 52.2% | 52.7% | 25% | 17% |
| 2025-03 | 2,976 | 52.9% | 51.6% | 25% | 17% |
| 2025-04 | 2,880 | 53.8% | 53.4% | 23% | 11% |
| 2025-05 | 2,976 | 53.3% | 52.6% | 21% | 9% |
| 2025-06 | 2,880 | 51.1% | 51.6% | 23% | 10% |
| 2025-07 | 2,976 | 52.6% | 51.5% | 25% | 10% |
| 2025-08 | 2,976 | 52.3% | 51.7% | 18% | 6% |
| 2025-09 | 2,880 | 54.2% | 53.4% | 8% | 0% |
| 2025-10 | 2,976 | 52.4% | 50.7% | 8% | 0% |
| 2025-11 | 2,880 | 53.0% | 52.9% | 7% | 0% |
| 2025-12 | 2,976 | 54.4% | 52.6% | 8% | 2% |
| 2026-01 | 2,976 | 53.9% | 53.6% | 22% | 12% |
| 2026-02 | 2,688 | 52.0% | 51.4% | 23% | 13% |
| 2026-03 | 2,976 | 54.3% | 52.4% | 22% | 13% |
| 2026-04 | 2,880 | 53.6% | 52.3% | 22% | 11% |
| 2026-05 | 2,976 | 53.6% | 52.9% | 21% | 11% |
| 2026-06 | 2,880 | 51.7% | 51.5% | 23% | 12% |
| 2026-07 | 2,976 | 51.8% | 52.0% | 21% | 10% |
| 2026-08 | 2,976 | 52.6% | 51.9% | 21% | 10% |
| 2026-09 | 1,677 | 51.6% | 51.6% | 20% | 11% |

v1 ahead in 19 of 23 months.

## Against real Polymarket settlement, post-Aug-7 (n = 2,124)

| | v1 at 0 s | v2 at −60 s |
| --- | ---: | ---: |
| Accuracy | 51.22% | 51.55% |
| Gate pass rate | 22.8% (484) | 11.2% (237) |
| Gated accuracy | 53.31% | 54.01% |

Sampling error: ±1.1 points overall, about ±3.2 on v2's gated calls.
