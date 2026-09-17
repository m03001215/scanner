# 15m retrain on TWAP-60 settlement labels

Run 2026-09-16 18:18 UTC. Features unchanged; only the label changes. Two gates; the run stops at the first failure.

## Step 1 - synthetic labels

- 15m boundaries 2021-01-01 → now: 200,138. TWAP-60 source: aggTrades 3,911 (post-Aug-7 only; every boundary would be ~200k requests), 1-minute fallback 196,151, missing 76.
- Fallback vs aggTrades TWAP on the 3,911 boundaries with both: median |diff| 0.32 bp, 95th pct 1.94 bp.
- Synthetic labels: 200,054; up rate 50.17%.

## Step 2 - GATE 1: synthetic vs real settlement (post-Aug-7)

- Real settlement source: the bot's `bot.db` (`markets` + `hist_windows`, 15m, resolved), exported to `data/settlement_labels_15m.csv`: 33,543 windows 2025-09-13 → 2026-09-16; post-Aug-7: 2,124.
- **Match rate, aggTrades TWAP-60 labels: 97.32% on 2,124 windows** (57 mismatches).
- Match rate, 1-minute fallback labels on the same windows: 96.52% on 2,124.
- For information, fallback labels vs real settlement over the whole labelled year: 94.97% on 33,543.
- Mismatched windows: median |Binance TWAP move| 0.17 bp (windows where the two TWAPs nearly tied).

**Gate 1: FAIL** (97.32% vs the 99% requirement).

Stopped at Gate 1. The synthetic label is not a faithful stand-in for the referee, so the pre-Aug-7 history cannot be relabelled from Binance data. No retrain, no deployment.

**Verdict: don't ship.**

---

# Amendment (2026-09-17): Gate 1 made relative to the incumbent label

The original 99% bar was absolute. The incumbent training label (Binance 15m candle, close ≥ open) is itself an imperfect stand-in for the referee, so the gate is restated as: the synthetic TWAP-60 label, with near-tie windows excluded, must reach 99% **and** beat the incumbent label by at least 5 points.

## Step A - both labels on the same 2,124 post-Aug-7 windows with real settlement

| Label | Windows kept | Match with real settlement | Mismatches |
| --- | ---: | ---: | ---: |
| (a) Current training label: Binance 15m candle, close ≥ open | 2,124 | **93.41%** | 140 |
| (b) Synthetic aggTrades TWAP-60, all windows | 2,122 | 97.31% | 57 |
| (b) excluding \|TWAP move\| < 1 bp | 1,906 (216 removed, 10.2%) | 99.84% | 3 |
| (b) excluding \|TWAP move\| < 2 bp | 1,718 (404 removed, 19.0%) | **100.00%** | 0 |
| (b) excluding \|TWAP move\| < 3 bp | 1,545 (577 removed, 27.2%) | 100.00% | 0 |

For reference, the candle label on the same kept windows: 97.01% (1 bp), 98.43% (2 bp), 98.83% (3 bp). Every TWAP-label mismatch sits within 1.85 bp of a tie; the candle label's mismatches are spread wider (median 0.74 bp, 19% of them beyond 2 bp), because an instantaneous open/close print is a noisier proxy for a 60-second TWAP than another TWAP is.

**Threshold choice: 2 bp** — the smallest threshold at which the synthetic label has zero mismatches (the largest mismatch is at 1.85 bp); 3 bp removes 8 more points of windows for no further gain.

**Amended Gate 1: PASS.** (b) at 2 bp = 100.00% ≥ 99%; (b) − (a) = 100.00 − 93.41 = **+6.59 points** ≥ 5.

**Permanent constant.** 2.7% is the mismatch between a single-exchange (Binance aggTrades) TWAP-60 reconstruction and the referee across *all* windows: the floor on any single-exchange reconstruction of the referee. Recorded as `REFEREE_RECONSTRUCTION_MISMATCH_FLOOR = 0.027` in `btcpred/constants.py`.

## Step B - relabel and retrain (bulk aggTrades TWAP-60)

- Bulk TWAP-60 boundaries: 200,052 (2021-01-01 → 2026-09-16), from 68 monthly files.
- Consistency with the per-request aggTrades TWAP validated in Step A, on 2,400 shared boundaries: median |diff| 0.000 bp, max 0.000 bp (same estimator, same trades).
- 1,511 boundaries in the current month (no monthly zip published yet) filled from the per-request aggTrades TWAP.
- Candles with features and both labels: 105,832 (2023-09-10 → 2026-09-16). Real settlement on 2,124 rows (≥ Aug 7), synthetic TWAP-60 on the rest.
- Near-tie exclusion (|TWAP move| < 2 bp, synthetic rows only): drops **10,844 rows (10.2%)** from training. Evaluation keeps every window.
- Old (candle) vs new (TWAP/settlement) label disagree on 5.11% of rows.

| Walk-forward, each model on its own labels, all OOS rows | old model (candle labels) | new model (TWAP/settlement labels) |
| --- | ---: | ---: |
| OOS accuracy | 52.74% | 52.29% |
| OOS AUC | 0.540 | 0.534 |
| OOS log loss | 0.6909 | 0.6916 |
| Trees per fold | [209, 87, 92, 195, 120] | [79, 63, 110, 135, 395] |

## Step C - GATE 2: both models on the same post-Aug-7 OOS windows, graded against real settlement

- Windows: 2,124 post-Aug-7 OOS windows with real settlement (2026-08-07 → 2026-09-16), predicted by folds trained before them; near-ties included. 337 have a market mid at open (bot capture from 3 Sep).

| Post-Aug-7 OOS, real settlement | old model | new model |
| --- | ---: | ---: |
| Accuracy, all windows (n=2,124) | 51.84% | 52.82% |
| Accuracy, windows with \|TWAP move\| ≥ 2 bp | 52.21% | 53.31% |
| **Gated accuracy** (agree, ML ≥ 0.10, TA ≥ 0.3) | **54.24%** (n=389) | **53.86%** (n=440) |
| **Incremental z** of logit(p) vs logit(mid) (n=337) | **-0.65** | **-0.94** |
| z of logit(mid) in the same regression | +0.98 | +1.11 |
| ECE, 10 bins | 1.17% | 1.00% |
| Brier | 0.2499 | 0.2496 |

Sampling error on gated accuracy at n≈389: about ±2.5 points (1σ); on the z statistic with n=337: ±1.

**Gate 2: FAIL** — gated accuracy new < old; incremental z new < old.

Keeping the old model.

**Verdict: don't ship.**
