"""Paired walk-forward: v2 at -60 s (minute 14 of the forming candle) vs v1 at 0 s (candle open), same target candles.
Includes the Agree+Confident gate pass rates and overlap, and v3 at -10 s alongside. Writes reports/v2_minus60s_vs_v1.md."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btcpred import data, features, model, ta, predictor, intra
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "reports" / "v2_minus60s_vs_v1.md"; K = 14; DAYS = 1095
L = []; w = lambda s="": (print(s), L.append(s))
df = data.drop_open_candle(data.load_or_update(ROOT / "data" / "btcusdt_15m.parquet", interval="15m", days=DAYS))
m1 = pd.read_parquet(ROOT / "data" / "btcusdt_1m.parquet").set_index("open_time")
live = ROOT / "data" / "btcusdt_1m_live.parquet"
if live.exists(): m1 = pd.concat([m1, pd.read_parquet(live).set_index("open_time")]); m1 = m1[~m1.index.duplicated(keep="last")].sort_index()
_, meta = model.load(ROOT / "models" / "15m" / "lgbm"); feats = meta["features"]
X2, y2, t2, k2 = intra.build_training(df, m1, "15m", feats)
Xfull = features.build_features(df)[feats]; y = features.build_labels(df)
sel14 = k2 == K; t14 = pd.DatetimeIndex(t2[sel14]); X14 = X2[sel14].set_index(t14)
v1ok = Xfull.notna().all(axis=1) & y.notna(); d1 = df.loc[v1ok, "open_time"]
U = pd.DatetimeIndex(d1).intersection(t14); U = U.sort_values()
Xa = Xfull.set_index(pd.DatetimeIndex(df["open_time"])).loc[U]; ya = y.set_axis(pd.DatetimeIndex(df["open_time"])).loc[U].astype(int); X14 = X14.loc[U]
n = len(U); first = int(n * 0.4); edges = np.linspace(first, n, 6, dtype=int)
w("# 15m: v2 at −60 s vs v1 at 0 s, paired walk-forward\n")
w(f"Run {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC. Target: the 15m candle's direction (close > open, Binance). **v1** predicts at the candle's open from the closed history; **v2** predicts the same candle 60 s earlier, from the closed history plus the first 14 minutes of the candle before it (1-minute bars). Identical {n:,} candles ({U[0]:%Y-%m-%d} → {U[-1]:%Y-%m-%d}), identical fold boundaries, first 40% as the initial training block.\n")
p1 = np.full(n, np.nan); p2 = np.full(n, np.nan); fl = []; t2v = t2.values
for i in range(5):
    tr_end, te_end = edges[i], edges[i + 1]; va = int(tr_end * 0.9); t_end, t_va = U[tr_end], U[va]
    b1 = lgb.train(model.PARAMS, lgb.Dataset(Xa.iloc[:va], ya.iloc[:va]), 2000, valid_sets=[lgb.Dataset(Xa.iloc[va:tr_end], ya.iloc[va:tr_end])], callbacks=[lgb.early_stopping(100, verbose=False)])
    p1[tr_end:te_end] = b1.predict(Xa.iloc[tr_end:te_end], num_iteration=b1.best_iteration)
    fit = t2v < t_va.to_datetime64(); vam = (t2v >= t_va.to_datetime64()) & (t2v < t_end.to_datetime64())
    b2 = lgb.train(model.PARAMS, lgb.Dataset(X2[fit], y2[fit]), 2000, valid_sets=[lgb.Dataset(X2[vam], y2[vam])], callbacks=[lgb.early_stopping(100, verbose=False)])
    p2[tr_end:te_end] = b2.predict(X14.iloc[tr_end:te_end], num_iteration=b2.best_iteration)
    fl.append((b1.best_iteration, b2.best_iteration)); print("fold", i + 1, str(t_end)[:10], fl[-1], flush=True)
oos = slice(first, n); P1, P2, Y = p1[oos], p2[oos], ya.values[oos]; T = U[oos]
rp = ta.score(ta.signals(df), json.load(open(predictor.ta_report_path("15m")))["weights"]).set_index(pd.DatetimeIndex(df["open_time"]))
tadir = rp["direction"].loc[T].values; taconf = rp["confidence"].loc[T].values
pd.DataFrame({"target_open": T + pd.Timedelta(minutes=15), "p1": P1, "p2": P2, "y": Y, "ta_dir": tadir, "ta_conf": taconf}).to_parquet(ROOT / "data" / "compare_v2_v1_preds.parquet", index=False)
def ece(p, y, bins=10):
    q = np.clip((p * bins).astype(int), 0, bins - 1); return sum((q == b).mean() * abs(y[q == b].mean() - p[q == b].mean()) for b in range(bins) if (q == b).any())
def stats(p, Y=Y, tadir=tadir, taconf=taconf):
    d = (p >= 0.5).astype(int); c = np.abs(p - 0.5) * 2; hit = d == Y; gate = (tadir == d) & (c >= 0.10) & (taconf >= 0.3)
    return dict(acc=hit.mean(), auc=roc_auc_score(Y, p), ll=log_loss(Y, p), ece=ece(p, Y), c10=hit[c >= .1].mean(), s10=(c >= .1).mean(), c20=hit[c >= .2].mean() if (c >= .2).any() else float("nan"), s20=(c >= .2).mean(), gacc=hit[gate].mean(), gate=gate, d=d, hit=hit, bull=d.mean())
s1, s2 = stats(P1), stats(P2)
# v3 alongside, on the shared candles
v3 = None; f3 = ROOT / "data" / "compare_v3_v1_preds.parquet"
if f3.exists():
    q = pd.read_parquet(f3).set_index("target_open"); idx = (T + pd.Timedelta(minutes=15)); common = idx.isin(q.index)
    v3 = dict(n=int(common.sum()), p=q["p3"].reindex(idx[common]).values, mask=common)
w(f"| Same target candles, out of sample (n = {len(Y):,}) | v1 at 0 s | v2 at −60 s |" + (" v3 at −10 s |" if v3 else "")); w("| --- | ---: | ---: |" + (" ---: |" if v3 else ""))
s3 = stats(v3["p"], Y[v3["mask"]], tadir[v3["mask"]], taconf[v3["mask"]]) if v3 else None
row = lambda name, f: w(f"| {name} | {f(s1)} | {f(s2)} |" + (f" {f(s3)} |" if v3 else ""))
row("Accuracy", lambda s: f"{s['acc']*100:.2f}%"); row("AUC", lambda s: f"{s['auc']:.3f}"); row("Log loss", lambda s: f"{s['ll']:.4f}"); row("ECE (10 bins)", lambda s: f"{s['ece']*100:.2f}%")
row("Accuracy, confidence ≥ 0.10 (share)", lambda s: f"{s['c10']*100:.2f}% ({s['s10']*100:.1f}%)"); row("Accuracy, confidence ≥ 0.20 (share)", lambda s: f"{s['c20']*100:.2f}% ({s['s20']*100:.1f}%)")
row("**Agree + Confident gate: pass rate**", lambda s: f"**{s['gate'].mean()*100:.2f}%** ({int(s['gate'].sum()):,})"); row("**Agree + Confident gate: accuracy**", lambda s: f"**{s['gacc']*100:.2f}%**")
row("Bullish calls", lambda s: f"{s['bull']*100:.1f}%")
w(f"| Trees per fold | {[f[0] for f in fl]} | {[f[1] for f in fl]} |" + (" see v3 report |" if v3 else ""))
if v3: w(f"\nThe v3 column is from the separate v3 run, matched on the {v3['n']:,} shared target candles; its fold boundaries differ by a few candles.")
b = int((s1["hit"] & ~s2["hit"]).sum()); c_ = int((~s1["hit"] & s2["hit"]).sum()); z = (c_ - b) / math.sqrt(b + c_)
w(f"\nSampling error on accuracy: ±{math.sqrt(0.25/len(Y))*100:.2f} points (1σ). Paired comparison (McNemar): v1 right / v2 wrong on {b:,} candles, v2 right / v1 wrong on {c_:,}; z = {z:+.2f} (|z| ≥ 2 is significant).\n")
agree = s1["d"] == s2["d"]
w("## Do the two calls agree?\n")
w(f"- Same direction on **{agree.mean()*100:.1f}%** of candles. When they agree: {s1['hit'][agree].mean()*100:.2f}% (n={int(agree.sum()):,}). When they disagree: v1 {s1['hit'][~agree].mean()*100:.2f}%, v2 {s2['hit'][~agree].mean()*100:.2f}% (n={int((~agree).sum()):,}).")
w(f"- Correlation of the two probabilities: {np.corrcoef(P1, P2)[0,1]:.3f}.\n")
g1, g2 = s1["gate"], s2["gate"]; both, o1, o2 = g1 & g2, g1 & ~g2, g2 & ~g1
w("## Agree + Confident gate: do the same candles pass?\n"); w("| Outcome | Candles | Share | Accuracy |"); w("| --- | ---: | ---: | --- |")
w(f"| Both pass | {int(both.sum()):,} | {both.mean()*100:.1f}% | {s1['hit'][both].mean()*100:.2f}% (opposite directions on {int((both & ~agree).sum())}) |")
w(f"| v1 only | {int(o1.sum()):,} | {o1.mean()*100:.1f}% | v1 {s1['hit'][o1].mean()*100:.2f}% |"); w(f"| v2 only | {int(o2.sum()):,} | {o2.mean()*100:.1f}% | v2 {s2['hit'][o2].mean()*100:.2f}% |")
w(f"| Neither | {int((~g1 & ~g2).sum()):,} | {(~g1 & ~g2).mean()*100:.1f}% | |")
w(f"\n{both.sum()/g2.sum()*100:.0f}% of v2's passes also pass v1; {both.sum()/g1.sum()*100:.0f}% of v1's passes also pass v2. Passes per day: v1 {g1.sum()/(len(Y)/96):.1f}, v2 {g2.sum()/(len(Y)/96):.1f}.\n")
w("## By month\n"); w("| Month | Candles | v1 | v2 | v1 gate pass | v2 gate pass |"); w("| --- | ---: | ---: | ---: | ---: | ---: |")
mon = pd.DataFrame({"m": T.tz_convert(None).to_period("M").astype(str), "h1": s1["hit"], "h2": s2["hit"], "g1": g1, "g2": g2})
for m_, g in mon.groupby("m"): w(f"| {m_} | {len(g):,} | {g.h1.mean()*100:.1f}% | {g.h2.mean()*100:.1f}% | {g.g1.mean()*100:.0f}% | {g.g2.mean()*100:.0f}% |")
w(f"\nv1 ahead in {int((mon.groupby('m').h1.mean() > mon.groupby('m').h2.mean()).sum())} of {mon.m.nunique()} months.")
real = pd.read_csv(ROOT / "data" / "settlement_labels_15m.csv"); real["t"] = pd.to_datetime(real.window_start_utc, utc=True); real = real.set_index("t")["settled_up"]
tgt = T + pd.Timedelta(minutes=15); ys = real.reindex(tgt).values; hs = ~np.isnan(ys) & np.asarray(tgt >= pd.Timestamp("2026-08-07", tz="UTC"))
if hs.any():
    ys_ = ys[hs].astype(int); w(f"\n## Against real Polymarket settlement, post-Aug-7 (n = {int(hs.sum()):,})\n"); w("| | v1 at 0 s | v2 at −60 s |"); w("| --- | ---: | ---: |")
    w(f"| Accuracy | {(s1['d'][hs] == ys_).mean()*100:.2f}% | {(s2['d'][hs] == ys_).mean()*100:.2f}% |")
    w(f"| Gate pass rate | {g1[hs].mean()*100:.1f}% ({int(g1[hs].sum())}) | {g2[hs].mean()*100:.1f}% ({int(g2[hs].sum())}) |")
    w(f"| Gated accuracy | {(s1['d'][hs][g1[hs]] == ys_[g1[hs]]).mean()*100:.2f}% | {(s2['d'][hs][g2[hs]] == ys_[g2[hs]]).mean()*100:.2f}% |")
    w(f"\nSampling error: ±{math.sqrt(0.25/hs.sum())*100:.1f} points overall, about ±{math.sqrt(0.25/max(1,g2[hs].sum()))*100:.1f} on v2's gated calls.")
OUT.write_text("\n".join(L) + "\n"); print("wrote", OUT)
