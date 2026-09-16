# Probability calibration · cross-interval summary

Generated 2026-09-16 14:06 UTC. Every interval: one isotonic calibrator per (interval, model version), fit on walk-forward OOS rows only, Brier/ECE measured on the held-out second half of the OOS period. No settlement label files were present, so all four intervals are **binance-calibrated** (close ≥ open); 4h's referee is not yet verified.

| Interval | Labels used | n OOS | n gated | Brier raw → cal | ECE raw → cal | Top-tier miscalibration | Direction asymmetry | Fold-stable | Gated bins within CI after cal |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| 5m | binance (binance-calibrated) | 126,096 | 3,482 | 0.24978 → 0.24984 | 1.13% → 1.36% | said 57.4% → got 53.0% | n | y | 11/11 |
| 15m | binance (binance-calibrated) | 63,014 | 12,626 | 0.24848 → 0.24855 | 1.32% → 1.45% | said 59.9% → got 56.8% | y | n | 10/15 ✗ |
| 1h | binance (binance-calibrated) | 28,730 | 7,534 | 0.24866 → 0.24873 | 1.87% → 2.05% | said 59.6% → got 60.7% | n | n | 9/11 ✗ |
| 4h | binance (binance-calibrated) | 11,849 | 1,714 | 0.24845 → 0.24810 | 3.14% → 2.66% | said 64.7% → got 61.0% | n | n | 6/6 |

**What the numbers say.** On 5m, 15m and 1h the raw p_bullish is already close to calibrated on the held-out half: the calibrator moves Brier by less than 0.0001, i.e. within noise. The top-tier drift that motivated this work is real in the first half of each OOS period but does not repeat consistently in the second half (15m: said 59.9% → got 56.8% held out; 5m: said 57.4% → got 53.0%; 4h: said 64.7% → got 61.0%; 1h ran the other way, said 59.6% → got 60.7%), which is why a calibrator fitted on the first half cannot fully correct it and why 15m and 1h fail the every-gated-bin-within-CI requirement. 4h is the one interval where calibration clearly helps (Brier −0.00035, ECE 3.1% → 2.7%). 15m is the only interval where bullish and bearish calls calibrate differently (bullish calls run ~1 pt too bold, bearish calls are on the diagonal; z = −3.0). The bot should treat `p_bullish_cal` as a mild shrink toward 0.5 on 5m/15m/1h and as a genuine correction on 4h.

Method: isotonic regression fitted to equal-count bin means (≥ 1000 rows per step) — plain per-row isotonic memorised the fitting half and made 5m materially worse (gated ECE 2.3% → 16.5%), so it is reported only as a comparison. Calibrators refit automatically on every `train` and `backtest` of an interval; the version string is `<model hash>-<timestamp>`. Per-interval detail: `reports/calibration_<tf>.md`.
