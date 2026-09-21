"""Builds reports/gate_similarity_v5_v1.md from data/compare_v5_v1_preds.parquet (made by tools/gate_similarity_v5_v1.py)."""
import math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
d = pd.read_parquet(ROOT / "data" / "compare_v5_v1_preds.parquet"); n = len(d); y = d.y.values
def side(p, tdir, tconf):
    dr = (p >= .5).astype(int); c = np.abs(p - .5) * 2; return dr, c, (tdir == dr) & (c >= .10) & (tconf >= .3)
d1, c1, g1 = side(d.p1.values, d.ta1_dir.values, d.ta1_conf.values); d5, c5, g5 = side(d.p5.values, d.ta5_dir.values, d.ta5_conf.values)
h1, h5 = d1 == y, d5 == y; both, o1, o5 = g1 & g5, g1 & ~g5, g5 & ~g1; union = g1 | g5
pc = lambda x, k=1: f"{x*100:.{k}f}%"; acc = lambda m, h: h[m].mean() if m.any() else float("nan")
L = []; w = L.append
w("# 15m gate similarity: v5 (−10 s) vs v1 (0 s)\n")
w(f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC. Paired walk-forward, identical {n:,} out-of-sample candles ({str(d.target_open.min())[:10]} → {str(d.target_open.max())[:10]}) and folds. Gate = ML and TA agree, ML confidence ≥ 0.10, TA confidence ≥ 0.3. Each version uses its own ML model and its own TA weights.\n")
w("## 1. Overlap of gate-passed windows\n"); w("| | Candles | Share of all | Accuracy |"); w("| --- | ---: | ---: | --- |")
w(f"| v1 gate passes | {g1.sum():,} | {pc(g1.mean(),2)} | {pc(acc(g1,h1),2)} |"); w(f"| v5 gate passes | {g5.sum():,} | {pc(g5.mean(),2)} | {pc(acc(g5,h5),2)} |")
w(f"| **Both pass** | **{both.sum():,}** | {pc(both.mean(),2)} | {pc(acc(both,h1),2)} |"); w(f"| v1 only | {o1.sum():,} | {pc(o1.mean(),2)} | v1's call {pc(acc(o1,h1),2)}; v5's call on them {pc(acc(o1,h5),2)} |")
w(f"| v5 only | {o5.sum():,} | {pc(o5.mean(),2)} | v5's call {pc(acc(o5,h5),2)}; v1's call on them {pc(acc(o5,h1),2)} |"); w(f"| Neither | {(~union).sum():,} | {pc((~union).mean(),2)} | |")
exp = g1.mean() * g5.mean() * n
w(f"\n- **{pc(both.sum()/g1.sum())} of v1's gate passes also pass v5's gate. {pc(both.sum()/g5.sum())} of v5's passes also pass v1's.**")
w(f"- Jaccard similarity (both ÷ either): **{both.sum()/union.sum():.3f}**. Correlation of the two pass/fail flags: {np.corrcoef(g1, g5)[0,1]:.3f}. If the gates were independent only {exp:,.0f} candles would pass both; the actual {both.sum():,} is {both.sum()/exp:.1f}× that.")
w(f"- Direction on the {both.sum():,} shared passes: identical on {pc((d1[both]==d5[both]).mean(),2)} ({int((d1[both]!=d5[both]).sum())} opposite).")
w(f"- Probabilities: correlation {np.corrcoef(d.p1, d.p5)[0,1]:.3f} over all candles, {np.corrcoef(d.p1[union], d.p5[union])[0,1]:.3f} on candles passing either gate. Mean |p1 − p5| = {np.abs(d.p1-d.p5).mean():.4f}.\n")
w("## 2. Why do the non-shared passes differ?\n")
for name, m, od, ocf, otd, otc, md in (("v1 only (v5 did not pass)", o1, d5, c5, d.ta5_dir.values, d.ta5_conf.values, d1), ("v5 only (v1 did not pass)", o5, d1, c1, d.ta1_dir.values, d.ta1_conf.values, d5)):
    low = ocf[m] < .10; flip = od[m] != md[m]; tad = otd[m] != od[m]; tal = otc[m] < .3
    w(f"- **{name}, {m.sum():,} candles:** the other version's ML confidence was below 0.10 on {pc(low.mean())} of them (median {np.median(ocf[m]):.3f}, so usually a near miss); it called the opposite direction on {pc(flip.mean())}; its TA confidence was short on {pc((tal & ~low).mean())}.")
near = lambda cm, m: ((cm[m] >= .07) & (cm[m] < .10)).mean()
w(f"- Near misses: on {pc(near(c5,o1))} of v1-only candles v5's confidence was between 0.07 and 0.10; on {pc(near(c1,o5))} of v5-only candles v1's was.")
tt = d.ta1_dir.notna().values & d.ta5_dir.notna().values
w(f"- TA side: the two TA votes give the same direction on {pc((d.ta1_dir.values==d.ta5_dir.values)[tt].mean(),2)} of candles, and the TA-confident (≥ 0.3) sets have a Jaccard overlap of {((d.ta1_conf>=.3)&(d.ta5_conf>=.3)).sum()/((d.ta1_conf>=.3)|(d.ta5_conf>=.3)).sum():.3f}. The differences come from ML confidence, not TA.\n")
w("## 3. At the same pass rate (rank-based gate)\n"); w("v5's models are a little more confident, so a fixed 0.10 passes more candles. Matching the pass rates removes that effect.\n")
w("| Top share by ML confidence, + agree + TA ≥ 0.3 | v1 passes | v5 passes | Both | Jaccard | v1's also in v5 | v1 acc. | v5 acc. | Both acc. |"); w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for q in (.5, .3, .2, .1, .05):
    a = (d.ta1_dir.values == d1) & (c1 >= np.quantile(c1, 1 - q)) & (d.ta1_conf.values >= .3); b = (d.ta5_dir.values == d5) & (c5 >= np.quantile(c5, 1 - q)) & (d.ta5_conf.values >= .3); ab = a & b
    w(f"| top {q*100:.0f}% | {a.sum():,} | {b.sum():,} | {ab.sum():,} | {ab.sum()/(a|b).sum():.3f} | {pc(ab.sum()/a.sum())} | {pc(acc(a,h1))} | {pc(acc(b,h5))} | {pc(acc(ab,h1))} |")
w("\n## 4. By walk-forward fold\n"); w("| Fold | Period | v1 pass | v5 pass | Both | Jaccard | v1 passes also in v5 | Both acc. |"); w("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for f, g in d.assign(g1=g1, g5=g5, h=h1).groupby("fold"):
    ab = g.g1 & g.g5; w(f"| {f} | {str(g.target_open.min())[:10]} → {str(g.target_open.max())[:10]} | {pc(g.g1.mean())} | {pc(g.g5.mean())} | {ab.sum():,} | {ab.sum()/max(1,(g.g1|g.g5).sum()):.3f} | {pc(ab.sum()/max(1,g.g1.sum()))} | {pc(g.h[ab].mean()) if ab.any() else '–'} |")
w("\nFold 2 stands out: v1's model for that block was unusually unconfident (11% pass rate) while v5's was not (24%), so v5 passed twice as many candles there. Nearly all of v1's passes in that fold were still inside v5's set.\n")
w("## 5. Day by day, and value of each subset\n")
dd = d.assign(g1=g1, g5=g5, day=pd.to_datetime(d.target_open).dt.date).groupby("day")[["g1", "g5"]].sum()
w(f"- Passes per day: v1 mean {dd.g1.mean():.1f}, v5 mean {dd.g5.mean():.1f}; day-by-day correlation {dd.g1.corr(dd.g5):.3f}. Days where one fires and the other has none: v1 only {int(((dd.g1>0)&(dd.g5==0)).sum())}, v5 only {int(((dd.g5>0)&(dd.g1==0)).sum())}, of {len(dd)} days.")
bp = lambda m, dr: (np.where(dr[m] == 1, 1, -1) * d.ret_bp.values[m]).mean()
w(f"- Gross edge per trade before fees: shared passes {bp(both,d1):+.2f} bp, v1 only {bp(o1,d1):+.2f} bp, v5 only {bp(o5,d5):+.2f} bp.\n")
w("## Summary\n")
w(f"The two gates select largely the same windows. {pc(both.sum()/g1.sum(),0)} of v1's gated windows are also gated by v5, always in the same direction, and that shared set is the most accurate ({pc(acc(both,h1))}). v5 adds {o5.sum():,} windows of its own that score {pc(acc(o5,h5))}, which is why its overall gated accuracy is slightly below v1's. Almost all disagreement comes from ML confidence landing just either side of 0.10; the TA votes are practically identical. For a bot, a v5 gate pass at −10 s is a reliable preview of v1's gate at the open: when v5 passes, v1 passes too {pc(both.sum()/g5.sum(),0)} of the time, and never in the opposite direction.")
(ROOT / "reports" / "gate_similarity_v5_v1.md").write_text("\n".join(L) + "\n"); print("\n".join(L[-9:]))
