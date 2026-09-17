# 15m: v3 at −10 s vs v1 at 0 s, paired walk-forward

Run 2026-09-17 18:41 UTC. Target: the 15m candle's direction (close > open, Binance). **v1** predicts at the candle's open from the closed history; **v3** predicts the same candle 10 s earlier, from the closed history plus the first 890 s of the candle before it (10-second bars). Both are trained and scored on the identical 105,857 candles (2023-09-10 → 2026-09-16) with identical fold boundaries; the first 40% is the initial training block.

| Same target candles, out of sample (n = 63,515) | v1 at 0 s | v3 at −10 s |
| --- | ---: | ---: |
| Accuracy | 52.81% | 52.36% |
| AUC | 0.540 | 0.533 |
| Log loss | 0.6909 | 0.6916 |
| ECE (10 bins) | 0.79% | 0.85% |
| Accuracy, confidence ≥ 0.10 (share) | 55.82% (26.1%) | 55.32% (15.3%) |
| Accuracy, confidence ≥ 0.20 (share) | 58.20% (3.0%) | 55.56% (0.5%) |
| Gated accuracy: agree with TA, ML ≥ 0.10, TA ≥ 0.3 (n) | 56.44% (13,269) | 56.23% (7,160) |
| Bullish calls | 55.5% | 56.9% |
| Trees per fold | [254, 93, 71, 176, 124] | [80, 133, 29, 133, 65] |

Sampling error on accuracy: ±0.20 points (1σ). Paired comparison (McNemar): v1 right / v3 wrong on 6,536 candles, v3 right / v1 wrong on 6,251; z = -2.52 (|z| ≥ 2 would be significant).

## Do the two calls agree?

- Same direction on **79.9%** of candles. When they agree: 53.23% accurate (n=50,728). When they disagree: v1 51.11%, v3 48.89% (n=12,787).
- Correlation of the two probabilities: 0.776. Mean |p1 − p3|: 0.0218.
- Both confident (≥ 0.10) and agreeing: 56.49% on 10.6% of candles.

## By month

| Month | Candles | v1 | v3 | Agree |
| --- | ---: | ---: | ---: | ---: |
| 2024-11 | 635 | 52.0% | 52.0% | 82% |
| 2024-12 | 2,976 | 52.7% | 52.5% | 79% |
| 2025-01 | 2,976 | 52.6% | 52.4% | 79% |
| 2025-02 | 2,688 | 52.6% | 52.5% | 81% |
| 2025-03 | 2,976 | 52.7% | 51.8% | 80% |
| 2025-04 | 2,880 | 54.1% | 54.4% | 80% |
| 2025-05 | 2,976 | 53.2% | 52.6% | 82% |
| 2025-06 | 2,880 | 51.2% | 51.8% | 80% |
| 2025-07 | 2,976 | 52.7% | 51.7% | 80% |
| 2025-08 | 2,976 | 52.0% | 50.9% | 79% |
| 2025-09 | 2,880 | 53.5% | 53.8% | 77% |
| 2025-10 | 2,976 | 52.0% | 51.1% | 79% |
| 2025-11 | 2,880 | 52.7% | 53.6% | 78% |
| 2025-12 | 2,976 | 54.9% | 52.6% | 76% |
| 2026-01 | 2,976 | 53.5% | 53.2% | 81% |
| 2026-02 | 2,688 | 51.9% | 51.4% | 82% |
| 2026-03 | 2,976 | 54.5% | 52.2% | 81% |
| 2026-04 | 2,880 | 53.5% | 52.8% | 79% |
| 2026-05 | 2,976 | 53.6% | 54.1% | 81% |
| 2026-06 | 2,880 | 51.5% | 51.8% | 83% |
| 2026-07 | 2,976 | 52.1% | 51.0% | 80% |
| 2026-08 | 2,976 | 52.3% | 51.9% | 80% |
| 2026-09 | 1,536 | 51.4% | 51.6% | 81% |

## Against real Polymarket settlement, post-Aug-7 (n = 2,124)

| | v1 at 0 s | v3 at −10 s |
| --- | ---: | ---: |
| Accuracy | 51.18% | 50.38% |
| Gated accuracy (n) | 54.08% (466) | 55.90% (195) |

Sampling error here: ±1.1 points.
