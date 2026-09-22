# v1 15m: runs of four or more consecutive misses against the Binance candle

Generated 2026-09-22 05:11 UTC. Out-of-sample v1 history (walk-forward backtest rows plus live rows), 64,349 candles 2024-11-20 → 2026-09-22. Miss = v1's call did not match the candle's close-vs-open direction. Overall miss rate 47.23%.

## 1. How common are the streaks, and are they more than chance?

- Miss streaks of length ≥ 4: **1,592**, covering 7,679 candles (11.9% of all). Longest: 14 misses in a row (2025-11-13 15:30 UTC). Mean length of the ≥ 4 streaks: 4.82.
| Exact streak length | Observed | Expected if misses were independent | Ratio |
| --- | ---: | ---: | ---: |
| 4 | 882 | 892 | 0.99 |
| 5 | 375 | 421 | 0.89 |
| 6 | 201 | 199 | 1.01 |
| 7 | 66 | 94 | 0.70 |
| 8 | 38 | 44 | 0.86 |
| ≥ 9 | 30 | 40 | 0.76 |

At a 47% miss rate, four misses in a row happen by pure chance on about 5.0% of starts, so most streaks are expected. The ratios above measure how much more often than chance they occur.

## 2. Market conditions during the streaks vs the rest

| Condition (known before the candle unless stated) | During ≥ 4-miss streaks | All other candles | Ratio / diff |
| --- | ---: | ---: | ---: |
| Realised volatility over the prior day | 2.16% | 2.13% | 1.01× |
| Volatility spike: last 4 h vs the prior week | 0.90× | 0.89× | 1.01× |
| |Trend| over the prior day | 1.70% | 1.61% | 1.05× |
| |Trend| over the prior 4 h | 0.71% | 0.61% | 1.17× |
| Previous candle size (ATR) | 0.54 | 0.53 | 1.01× |
| **Size of the candle itself** (not known in advance, ATR) | 0.57 | 0.53 | 1.08× |
| Position in the prior 24 h range (0 = low, 1 = high) | 0.51 | 0.52 | 0.97× |
| Actual candle continues the previous candle's direction | 83.0% | 43.5% | +39.5 pts |
| v1 called the opposite of the previous candle | 83.0% | 67.7% | +15.3 pts |
| v1 ML confidence (mean) | 0.082 | 0.063 | 1.29× |
| ML and TA agreed | 83.7% | 75.6% | +8.0 pts |
| Standard gate passed | 32.1% | 18.4% | |
| Bullish candles | 42.0% | 50.9% | |

- Within a streak, the actual candles were **all in the same direction on 78.2%** of streaks (a one-way move), and strictly alternating on 0.8%. v1's calls were all the same direction on 78.2%.
- Signed trend over the prior day at streak candles: mean -0.01% (rest +0.02%); |trend| ≥ 2%: 32% vs 29%.

## 3. Where streaks start: rate of streak starts by regime


**Volatility spike (last 4 h vs prior week)**

| Quintile | Range | Candles | Streak starts | Starts per 1,000 candles | Miss rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Q1 | 0.11–0.52× | 12,870 | 280 | 21.8 | 46.4% |
| Q2 | 0.52–0.69× | 12,870 | 313 | 24.3 | 46.8% |
| Q3 | 0.69–0.89× | 12,869 | 344 | 26.7 | 48.1% |
| Q4 | 0.89–1.21× | 12,870 | 330 | 25.6 | 47.3% |
| Q5 | 1.21–5.20× | 12,870 | 325 | 25.3 | 47.6% |

**Realised volatility, prior day**

| Quintile | Range | Candles | Streak starts | Starts per 1,000 candles | Miss rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Q1 | 0.3–1.4% | 12,870 | 321 | 24.9 | 47.5% |
| Q2 | 1.4–1.8% | 12,870 | 311 | 24.2 | 47.2% |
| Q3 | 1.8–2.2% | 12,869 | 315 | 24.5 | 46.8% |
| Q4 | 2.2–2.8% | 12,870 | 301 | 23.4 | 47.2% |
| Q5 | 2.8–9.0% | 12,870 | 344 | 26.7 | 47.5% |

**Signed trend over the prior 4 h**

| Quintile | Range | Candles | Streak starts | Starts per 1,000 candles | Miss rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Q1 | -8.23 to -0.52% | 12,870 | 285 | 22.1 | 46.9% |
| Q2 | -0.52 to -0.13% | 12,870 | 319 | 24.8 | 46.9% |
| Q3 | -0.13 to +0.14% | 12,869 | 386 | 30.0 | 48.8% |
| Q4 | +0.14 to +0.54% | 12,870 | 313 | 24.3 | 47.0% |
| Q5 | +0.54 to +10.41% | 12,870 | 289 | 22.5 | 46.7% |

**Previous candle size (ATR)**

| Quintile | Range | Candles | Streak starts | Starts per 1,000 candles | Miss rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Q1 | 0.00–0.15 | 12,870 | 317 | 24.6 | 47.9% |
| Q2 | 0.15–0.31 | 12,870 | 335 | 26.0 | 48.0% |
| Q3 | 0.31–0.50 | 12,869 | 350 | 27.2 | 47.8% |
| Q4 | 0.50–0.80 | 12,870 | 314 | 24.4 | 47.1% |
| Q5 | 0.80–12.52 | 12,870 | 276 | 21.4 | 45.3% |

**Time of day (ET hour of the candle)**

| Hours ET | Candles | Starts per 1,000 | Miss rate |
| --- | ---: | ---: | ---: |
| 00–03 | 10,720 | 25.5 | 47.9% |
| 04–07 | 10,720 | 26.1 | 47.1% |
| 08–11 | 10,720 | 24.0 | 47.1% |
| 12–15 | 10,720 | 23.9 | 47.1% |
| 16–19 | 10,733 | 23.6 | 46.2% |
| 20–23 | 10,736 | 25.4 | 48.0% |

## 4. What happens after a streak ends

| Candles after the streak | v1 accuracy | Baseline |
| --- | ---: | ---: |
| next 1 | 100.0% | 52.8% |
| next 4 | 64.9% | 52.8% |
| next 8 | 58.7% | 52.8% |
| next 16 | 55.9% | 52.8% |

The first candle after a streak is a hit by construction (that is what ends the streak); the later rows show whether the model stays impaired. Streaks are also followed by new streaks more often than chance would give? Share of streak ends followed by another ≥ 4 streak within 16 candles: 35.7%.

## 5. The longest streaks

| Start (UTC) | Length | Actual sequence | Price move over the streak | Prior-day trend | Vol spike | Calls |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 2025-11-13 15:30 | 14 | ▼▼▼▼▼▼▼▼▼▼▼▼▼▼ | -3.54% | -1.21% | 1.63× | ▲▲▲▲▲▲▲▲▲▲▲▲▲▲ |
| 2025-11-19 02:30 | 13 | ▲▼▼▼▼▼▼▼▼▼▼▼▼ | -2.45% | +1.03% | 0.90× | ▼▲▲▲▲▲▲▲▲▲▲▲▲ |
| 2025-01-09 15:00 | 12 | ▲▲▲▲▲▼▲▲▼▼▼▼ | -0.18% | -2.54% | 1.53× | ▼▼▼▼▼▲▼▼▲▲▲▲ |
| 2025-04-06 16:00 | 12 | ▼▼▼▼▼▼▼▼▼▼▼▼ | -4.17% | -0.08% | 0.71× | ▲▲▲▲▲▲▲▲▲▲▲▲ |
| 2025-02-03 14:30 | 12 | ▲▲▲▲▲▲▼▲▲▼▼▲ | +3.96% | -4.68% | 1.46× | ▼▼▼▼▼▼▲▼▼▲▲▼ |
| 2025-10-14 00:30 | 12 | ▼▼▼▼▼▼▼▼▼▼▼▼ | -1.63% | -0.01% | 0.29× | ▲▲▲▲▲▲▲▲▲▲▲▲ |
| 2025-11-14 15:15 | 11 | ▲▲▲▼▲▲▼▼▼▼▼ | -0.27% | -6.55% | 1.97× | ▼▼▼▲▼▼▲▲▲▲▲ |
| 2025-07-30 22:30 | 11 | ▲▲▲▲▲▲▲▲▲▲▲ | +0.97% | -0.37% | 1.70× | ▼▼▼▼▼▼▼▼▼▼▼ |
| 2025-11-01 01:45 | 11 | ▼▼▲▼▲▲▲▲▲▲▲ | +0.43% | +0.25% | 0.36× | ▲▲▼▲▼▼▼▼▼▼▼ |
| 2026-09-21 07:15 | 10 | ▲▼▲▲▲▲▲▲▲▲ | +4.25% | +1.52% | 1.05× | ▼▲▼▼▼▼▼▼▼▼ |
| 2026-09-14 22:45 | 10 | ▼▼▼▼▼▼▼▼▼▼ | -1.05% | +2.50% | 1.07× | ▲▲▲▲▲▲▲▲▲▲ |
| 2026-07-17 15:45 | 10 | ▲▲▲▲▲▲▲▲▲▲ | +1.39% | -2.34% | 2.16× | ▼▼▼▼▼▼▼▼▼▼ |

## 6. Reading

- **Streaks are no more common than chance.** With a 47% miss rate, independent misses would produce about 1,690 runs of four or more; v1 produced 1,592 (0.94×), and every length from 4 to 9+ is at or below its chance count. Misses do not cluster in time. The 35.7% of streaks followed by another within 16 candles is also what independence predicts. So "four wrong in a row" is mostly the arithmetic of a 53% model, not a sign that the model has broken.
- **But the streaks have one clear character: a one-way move that v1 fades at every step.** 78% of streaks are candles all in the same direction; strictly alternating sequences are 0.8%. During streaks the candle continues the previous candle's direction 83% of the time (43% otherwise) while v1 calls the opposite of the previous candle 83% of the time (68% otherwise). The two figures are the same fact seen twice: a contrarian call that misses is, by definition, a candle that continued the move. The longest runs are the clearest cases: 14 straight misses in the −3.5% slide on 13 Nov 2025 and 12 in the −4.2% slide on 6 Apr 2025, every call bullish. Down moves dominate: 42% of streak candles were bullish against 51% elsewhere, so v1 fades sell-offs more than rallies.
- **Confidence makes it worse, not better.** v1's mean confidence during streaks is 0.082 against 0.063 elsewhere (1.29×), ML and TA agree more often (84% vs 76%), and the standard gate passes on 32% of streak candles against 18% elsewhere. In a trending burst every indicator reads "overextended", so the model and the vote become *more* sure of the reversal that does not come. The gate concentrates, rather than filters, these misses.
- **The streaks are hard to see coming.** Conditions known before the candle differ only mildly: prior-day volatility, the volatility spike ratio, trend size and time of day all give streak-start rates within 22–30 per 1,000 candles across their quintiles. The two mild signals: a flat prior 4 h (Q3, 30 per 1,000: streaks tend to begin as a new move starts from a quiet base, not mid-trend), and a small previous candle (a large previous candle, Q5, has the fewest starts and the lowest miss rate, 45.3%).
- **Recovery is immediate.** Accuracy over the 4, 8 and 16 candles after a streak, once the mechanical first hit is removed, is 53%, back at baseline. The model is not damaged by a streak; it was simply on the wrong side of a run while it lasted.
- **Practical implications.** (1) Do not treat a streak as a reason to distrust the next call; the next call is as good as any. (2) The gate's weakness is trend bursts: a rule such as "after three consecutive same-direction candles, skip the contrarian call or halve its weight" is the natural test, and it must act *during* the run. (3) Adding trend-persistence and higher-timeframe momentum features is the modelling route to the same end, since v1 currently has no feature that says "this move is continuing".
