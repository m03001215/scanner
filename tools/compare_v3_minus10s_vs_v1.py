"""Paired walk-forward: v3 at -10 s (elapsed 890 s of the forming candle) vs v1 at 0 s (candle open), same target candles.
Writes reports/v3_minus10s_vs_v1.md."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btcpred import data, features, model, ta, predictor, intra10
from btcpred.features import _atr
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "reports" / "v3_minus10s_vs_v1.md"
E_V3 = 890; DAYS = 1095
L = []; w = lambda s="": (print(s), L.append(s))

df = data.drop_open_candle(data.load_or_update(ROOT / "data" / "btcusdt_15m.parquet", interval="15m", days=DAYS))
bars = intra10.load_bars(); df = df[df["open_time"] >= bars.index[0].floor("15min") + pd.Timedelta(days=2)].reset_index(drop=True)
_, meta = model.load(ROOT / "models" / "15m" / "lgbm"); feats = meta["features"]
Xfull = features.build_features(df)[feats]                 # v1 state at the close of candle t -> predicts t+1
y = features.build_labels(df)
# v3 state at 890 s of candle t -> predicts t+1 (same target)
X3_train, y3, t3, e3 = intra10.build_training(df, bars, feats)
mm = intra10.bar_matrix(df, bars); j = E_V3 // intra10.BAR
sl = {k: a[:, :j] for k, a in mm.items()}
F = intra10._state_features(df["open"].values, sl["high"], sl["low"], sl["close"], sl["volume"], sl["taker_buy"], _atr(df, 14).shift(1).values, df["volume"].rolling(96).mean().shift(1).values, E_V3); F.index = df.index
X3_890 = pd.concat([Xfull.shift(1), F], axis=1)[X3_train.columns]
ok = Xfull.notna().all(axis=1) & X3_890.notna().all(axis=1) & y.notna() & ~np.isnan(sl["close"]).any(axis=1)
U = df.index[ok]; t = df.loc[U, "open_time"]; yy = y[U].astype(int)
n = len(U); first = int(n * 0.4); edges = np.linspace(first, n, 6, dtype=int)
w("# 15m: v3 at −10 s vs v1 at 0 s, paired walk-forward\n")
w(f"Run {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC. Target: the 15m candle's direction (close > open, Binance). **v1** predicts at the candle's open from the closed history; **v3** predicts the same candle 10 s earlier, from the closed history plus the first 890 s of the candle before it (10-second bars). Both are trained and scored on the identical {n:,} candles ({t.iloc[0]:%Y-%m-%d} → {t.iloc[-1]:%Y-%m-%d}) with identical fold boundaries; the first 40% is the initial training block.\n")
p1 = np.full(n, np.nan); p3 = np.full(n, np.nan); fl = []
cand3 = pd.factorize(t3)[0]; t3_arr = t3.values
for i in range(5):
    tr_end, te_end = edges[i], edges[i + 1]; va = int(tr_end * 0.9)
    t_end = t.iloc[tr_end]; t_va = t.iloc[va]
    # v1
    Xa = Xfull.loc[U]; ya = yy
    b1 = lgb.train(model.PARAMS, lgb.Dataset(Xa.iloc[:va], ya.iloc[:va]), 2000, valid_sets=[lgb.Dataset(Xa.iloc[va:tr_end], ya.iloc[va:tr_end])], callbacks=[lgb.early_stopping(100, verbose=False)])
    p1[tr_end:te_end] = b1.predict(Xa.iloc[tr_end:te_end], num_iteration=b1.best_iteration)
    # v3: train on all-step rows of candles before the boundary (same time cut), predict the 890 s rows of the test block
    fit = t3_arr < t_va.to_datetime64(); vam = (t3_arr >= t_va.to_datetime64()) & (t3_arr < t_end.to_datetime64())
    b3 = lgb.train(intra10.PARAMS, lgb.Dataset(X3_train[fit], y3[fit]), 1500, valid_sets=[lgb.Dataset(X3_train[vam], y3[vam])], callbacks=[lgb.early_stopping(80, verbose=False)])
    p3[tr_end:te_end] = b3.predict(X3_890.loc[U].iloc[tr_end:te_end], num_iteration=b3.best_iteration)
    fl.append((str(t_end)[:10], str(t.iloc[te_end - 1])[:10], b1.best_iteration, b3.best_iteration)); print("fold", i + 1, fl[-1], flush=True)
oos = slice(first, n); P1, P3, Y = p1[oos], p3[oos], yy.values[oos]; T = t.iloc[oos].reset_index(drop=True)
rp = ta.score(ta.signals(df), json.load(open(predictor.ta_report_path("15m")))["weights"])
tadir = rp["direction"].values[U][oos]; taconf = rp["confidence"].values[U][oos]
def ece(p, y, bins=10):
    q = np.clip((p * bins).astype(int), 0, bins - 1); return sum((q == b).mean() * abs(y[q == b].mean() - p[q == b].mean()) for b in range(bins) if (q == b).any())
def stats(p):
    d = (p >= 0.5).astype(int); c = np.abs(p - 0.5) * 2; hit = d == Y
    gate = (tadir == d) & (c >= 0.10) & (taconf >= 0.3)
    return dict(acc=hit.mean(), auc=roc_auc_score(Y, p), ll=log_loss(Y, p), ece=ece(p, Y), c10=hit[c >= .10].mean(), s10=(c >= .10).mean(), c20=hit[c >= .20].mean() if (c >= .20).any() else float("nan"), s20=(c >= .20).mean(), gated=hit[gate].mean(), ng=int(gate.sum()), bull=(d == 1).mean(), d=d, hit=hit, c=c)
s1, s3 = stats(P1), stats(P3)
w("| Same target candles, out of sample (n = {:,}) | v1 at 0 s | v3 at −10 s |".format(len(Y))); w("| --- | ---: | ---: |")
w(f"| Accuracy | {s1['acc']*100:.2f}% | {s3['acc']*100:.2f}% |"); w(f"| AUC | {s1['auc']:.3f} | {s3['auc']:.3f} |"); w(f"| Log loss | {s1['ll']:.4f} | {s3['ll']:.4f} |"); w(f"| ECE (10 bins) | {s1['ece']*100:.2f}% | {s3['ece']*100:.2f}% |")
w(f"| Accuracy, confidence ≥ 0.10 (share) | {s1['c10']*100:.2f}% ({s1['s10']*100:.1f}%) | {s3['c10']*100:.2f}% ({s3['s10']*100:.1f}%) |")
w(f"| Accuracy, confidence ≥ 0.20 (share) | {s1['c20']*100:.2f}% ({s1['s20']*100:.1f}%) | {s3['c20']*100:.2f}% ({s3['s20']*100:.1f}%) |")
w(f"| Gated accuracy: agree with TA, ML ≥ 0.10, TA ≥ 0.3 (n) | {s1['gated']*100:.2f}% ({s1['ng']:,}) | {s3['gated']*100:.2f}% ({s3['ng']:,}) |")
w(f"| Bullish calls | {s1['bull']*100:.1f}% | {s3['bull']*100:.1f}% |")
w(f"| Trees per fold | {[f[2] for f in fl]} | {[f[3] for f in fl]} |")
se = math.sqrt(0.25 / len(Y)) * 100
# paired test: McNemar on hits
b = int(((s1["hit"]) & (~s3["hit"])).sum()); c_ = int(((~s1["hit"]) & (s3["hit"])).sum())
z = (c_ - b) / math.sqrt(b + c_) if b + c_ else float("nan")
w(f"\nSampling error on accuracy: ±{se:.2f} points (1σ). Paired comparison (McNemar): v1 right / v3 wrong on {b:,} candles, v3 right / v1 wrong on {c_:,}; z = {z:+.2f} (|z| ≥ 2 would be significant).\n")
agree = s1["d"] == s3["d"]
w("## Do the two calls agree?\n")
w(f"- Same direction on **{agree.mean()*100:.1f}%** of candles. When they agree: {s1['hit'][agree].mean()*100:.2f}% accurate (n={int(agree.sum()):,}). When they disagree: v1 {s1['hit'][~agree].mean()*100:.2f}%, v3 {s3['hit'][~agree].mean()*100:.2f}% (n={int((~agree).sum()):,}).")
w(f"- Correlation of the two probabilities: {np.corrcoef(P1, P3)[0,1]:.3f}. Mean |p1 − p3|: {np.abs(P1-P3).mean():.4f}.")
both = agree & (s1["c"] >= .10) & (s3["c"] >= .10)
w(f"- Both confident (≥ 0.10) and agreeing: {s1['hit'][both].mean()*100:.2f}% on {both.mean()*100:.1f}% of candles.\n")
w("## By month\n"); w("| Month | Candles | v1 | v3 | Agree |"); w("| --- | ---: | ---: | ---: | ---: |")
mon = pd.DataFrame({"m": T.dt.tz_convert(None).dt.to_period("M").astype(str), "h1": s1["hit"], "h3": s3["hit"], "ag": agree})
for m_, g in mon.groupby("m"): w(f"| {m_} | {len(g):,} | {g.h1.mean()*100:.1f}% | {g.h3.mean()*100:.1f}% | {g.ag.mean()*100:.0f}% |")
# real settlement, post-Aug-7
real = pd.read_csv(ROOT / "data" / "settlement_labels_15m.csv"); real["t"] = pd.to_datetime(real.window_start_utc, utc=True); real = real.set_index("t")["settled_up"]
tgt = pd.DatetimeIndex(T) + pd.Timedelta(minutes=15); ys = real.reindex(tgt).values; hs = ~np.isnan(ys) & (tgt >= pd.Timestamp("2026-08-07", tz="UTC"))
if hs.any():
    ys_ = ys[hs].astype(int)
    w(f"\n## Against real Polymarket settlement, post-Aug-7 (n = {int(hs.sum()):,})\n"); w("| | v1 at 0 s | v3 at −10 s |"); w("| --- | ---: | ---: |")
    for name, s in (("v1", s1), ("v3", s3)): pass
    a1 = (s1["d"][hs] == ys_).mean(); a3 = (s3["d"][hs] == ys_).mean()
    g1 = (tadir[hs] == s1["d"][hs]) & (s1["c"][hs] >= .1) & (taconf[hs] >= .3); g3 = (tadir[hs] == s3["d"][hs]) & (s3["c"][hs] >= .1) & (taconf[hs] >= .3)
    w(f"| Accuracy | {a1*100:.2f}% | {a3*100:.2f}% |"); w(f"| Gated accuracy (n) | {(s1['d'][hs][g1] == ys_[g1]).mean()*100:.2f}% ({int(g1.sum())}) | {(s3['d'][hs][g3] == ys_[g3]).mean()*100:.2f}% ({int(g3.sum())}) |")
    w(f"\nSampling error here: ±{math.sqrt(0.25/hs.sum())*100:.1f} points.")
OUT.write_text("\n".join(L) + "\n"); print("wrote", OUT)
