# 15m gate similarity: v5 (−10 s) vs v1 (0 s)

Generated 2026-09-21 05:08 UTC. Paired walk-forward, identical 63,093 out-of-sample candles (2024-11-28 → 2026-09-17) and folds. Gate = ML and TA agree, ML confidence ≥ 0.10, TA confidence ≥ 0.3. Each version uses its own ML model and its own TA weights.

## 1. Overlap of gate-passed windows

| | Candles | Share of all | Accuracy |
| --- | ---: | ---: | --- |
| v1 gate passes | 11,924 | 18.90% | 56.75% |
| v5 gate passes | 13,447 | 21.31% | 56.24% |
| **Both pass** | **10,549** | 16.72% | 56.98% |
| v1 only | 1,375 | 2.18% | v1's call 54.98%; v5's call on them 54.47% |
| v5 only | 2,898 | 4.59% | v5's call 53.52%; v1's call on them 53.76% |
| Neither | 48,271 | 76.51% | |

- **88.5% of v1's gate passes also pass v5's gate. 78.4% of v5's passes also pass v1's.**
- Jaccard similarity (both ÷ either): **0.712**. Correlation of the two pass/fail flags: 0.792. If the gates were independent only 2,541 candles would pass both; the actual 10,549 is 4.2× that.
- Direction on the 10,549 shared passes: identical on 100.00% (0 opposite).
- Probabilities: correlation 0.950 over all candles, 0.967 on candles passing either gate. Mean |p1 − p5| = 0.0099.

## 2. Why do the non-shared passes differ?

- **v1 only (v5 did not pass), 1,375 candles:** the other version's ML confidence was below 0.10 on 95.1% of them (median 0.088, so usually a near miss); it called the opposite direction on 1.1%; its TA confidence was short on 4.9%.
- **v5 only (v1 did not pass), 2,898 candles:** the other version's ML confidence was below 0.10 on 98.2% of them (median 0.084, so usually a near miss); it called the opposite direction on 0.6%; its TA confidence was short on 1.8%.
- Near misses: on 77.7% of v1-only candles v5's confidence was between 0.07 and 0.10; on 74.5% of v5-only candles v1's was.
- TA side: the two TA votes give the same direction on 99.38% of candles, and the TA-confident (≥ 0.3) sets have a Jaccard overlap of 0.949. The differences come from ML confidence, not TA.

## 3. At the same pass rate (rank-based gate)

v5's models are a little more confident, so a fixed 0.10 passes more candles. Matching the pass rates removes that effect.

| Top share by ML confidence, + agree + TA ≥ 0.3 | v1 passes | v5 passes | Both | Jaccard | v1's also in v5 | v1 acc. | v5 acc. | Both acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top 50% | 20,580 | 20,479 | 18,729 | 0.839 | 91.0% | 55.3% | 55.2% | 55.6% |
| top 30% | 14,919 | 14,934 | 12,869 | 0.758 | 86.3% | 56.2% | 55.9% | 56.3% |
| top 20% | 10,915 | 10,898 | 8,811 | 0.678 | 80.7% | 56.8% | 56.6% | 57.1% |
| top 10% | 5,857 | 5,843 | 4,081 | 0.536 | 69.7% | 56.9% | 56.6% | 56.2% |
| top 5% | 2,995 | 3,000 | 1,802 | 0.430 | 60.2% | 56.9% | 58.3% | 58.0% |

## 4. By walk-forward fold

| Fold | Period | v1 pass | v5 pass | Both | Jaccard | v1 passes also in v5 | Both acc. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2024-11-28 → 2025-04-09 | 25.1% | 24.5% | 2,802 | 0.810 | 88.5% | 55.2% |
| 2 | 2025-04-09 → 2025-08-18 | 11.2% | 24.3% | 1,397 | 0.452 | 98.4% | 58.7% |
| 3 | 2025-08-18 → 2025-12-28 | 17.7% | 17.7% | 1,960 | 0.779 | 87.6% | 57.0% |
| 4 | 2025-12-28 → 2026-05-08 | 19.1% | 20.7% | 2,141 | 0.743 | 88.7% | 58.1% |
| 5 | 2026-05-08 → 2026-09-17 | 21.3% | 19.3% | 2,249 | 0.782 | 83.7% | 57.1% |

Fold 2 stands out: v1's model for that block was unusually unconfident (11% pass rate) while v5's was not (24%), so v5 passed twice as many candles there. Nearly all of v1's passes in that fold were still inside v5's set.

## 5. Day by day, and value of each subset

- Passes per day: v1 mean 18.1, v5 mean 20.4; day-by-day correlation 0.540. Days where one fires and the other has none: v1 only 0, v5 only 0, of 659 days.
- Gross edge per trade before fees: shared passes +0.77 bp, v1 only +0.60 bp, v5 only +0.53 bp.

## Summary

The two gates select largely the same windows. 88% of v1's gated windows are also gated by v5, always in the same direction, and that shared set is the most accurate (57.0%). v5 adds 2,898 windows of its own that score 53.5%, which is why its overall gated accuracy is slightly below v1's. Almost all disagreement comes from ML confidence landing just either side of 0.10; the TA votes are practically identical. For a bot, a v5 gate pass at −10 s is a reliable preview of v1's gate at the open: when v5 passes, v1 passes too 78% of the time, and never in the opposite direction.
