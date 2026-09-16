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
