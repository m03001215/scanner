"""Empirical constants recorded from validation studies. Change only with a new study."""

# Share of 15m windows where a TWAP-60 reconstructed from a single exchange's trades (Binance aggTrades)
# disagrees with the Polymarket/Chainlink referee, measured on 2,122 post-2026-08-07 windows with real
# settlement (reports/twap_relabel_15m.md). The disagreements are near-ties (all within 1.85 bp), so this is
# the floor on any single-exchange reconstruction of the referee, not a defect of the reconstruction method.
REFEREE_RECONSTRUCTION_MISMATCH_FLOOR = 0.027
