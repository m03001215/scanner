# v6 learned gate: evaluation

Generated 2026-09-22 09:02 UTC. Gate trained walk-forward on v1's out-of-sample 15m calls; 38,609 evaluation calls, 2025-08-16 → 2026-09-22. v1 accuracy on them 52.91%.

## Deployed gate (tiny LightGBM, 44 features) vs v1's confidence rank

| Pass share | Learned gate | Confidence rank | Difference (pts) | Calls |
| --- | ---: | ---: | ---: | ---: |
| top 30% | 55.74% | 56.12% | -0.39 | 11,583 |
| top 20% | 56.55% | 57.31% | -0.76 | 7,722 |
| top 10% | 58.25% | 57.71% | +0.54 | 3,861 |
| top 5% | 58.42% | 57.34% | +1.07 | 1,931 |

Standard gate: pass 19.1%, accuracy 57.42%. Learned gate at that pass rate: 56.66% (69% of the standard gate's calls also pass it). AUC: gate 0.524, confidence 0.527. Sampling error about ±0.4 pts at 20%, ±0.9 at 5%.

## Variants tried on the same walk-forward

| Variant | AUC | top 30% | top 20% | top 10% | top 5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 confidence only | 0.527 | 56.12% | 57.31% | 57.71% | 57.34% |
| LightGBM, 44 features, 15 leaves | 0.522 | 55.64% | 56.42% | 56.95% | 56.65% |
| LightGBM, 44 features, 4 leaves, min 2,000 rows per leaf (deployed) | 0.524 | 55.74% | 56.55% | 58.25% | 58.42% |
| LightGBM, 13 regime features, tiny | 0.524 | 55.98% | 56.25% | 57.68% | 58.21% |
| Logistic regression, 13 features | 0.522 | 55.57% | 56.41% | 57.91% | 57.74% |
| Logistic, confidence + run features only | 0.521 | 55.52% | 56.77% | 57.83% | 57.85% |

## The run-skip rule from the miss-streak study

| Rule | Pass rate | Accuracy | Accuracy of the skipped calls |
| --- | ---: | ---: | ---: |
| Standard gate | 19.1% | 57.42% | |
| Standard gate, skip contrarian calls when run ≥ 2 | 4.8% | 56.89% | 57.59% (n=5,530) |
| … run ≥ 3 | 10.2% | 57.06% | 57.83% (n=3,448) |
| … run ≥ 4 | 14.2% | 56.87% | 58.99% (n=1,902) |
| Only contrarian calls in runs ≥ 3 | 8.9% | 57.83% | |

The skipped calls are *more* accurate than the ones kept. Contrarian gated calls inside runs are right about 58% of the time; the ≥ 4-miss streaks are the losing tail of a winning group, so removing the group loses accuracy.

## Per fold, top-20% share: learned gate vs confidence

| Fold | Learned gate | Confidence |
| --- | ---: | ---: |
| 1 | 56.8% | 56.7% |
| 2 | 56.4% | 57.1% |
| 3 | 57.3% | 58.3% |
| 4 | 55.9% | 57.3% |
| 5 | 56.0% | 57.6% |

## Conclusion

v1's confidence is already the best available predictor of whether its call is right. The regime features add no usable information at this sample size; the deployed gate's advantage at the top 10% and 5% is within sampling error and reverses at 20% and 30%. v6 is kept as a version so the live record can test this, but on the evidence the standard gate (or a rank-based confidence gate) should remain the decision rule.
