"""Retrain the 15m model on TWAP-60 settlement labels, gated twice. See reports/twap_relabel_15m.md.

Step 1  synthetic labels: twap60(T) = time-weighted mean over [T-60s, T) from aggTrades where computed
        (data/twap60_aggtrades_15m.parquet), else the 1-minute candle ending at T, (O+H+L+C)/4.
        label(window starting T) = twap60(T+15m) >= twap60(T); ties -> Up.
Step 2  GATE 1: synthetic vs real Polymarket settlement on post-Aug-7 windows; need >= 99% match.
Step 3  relabel: synthetic before Aug 7, real settlement after; same walk-forward protocol.
Step 4  GATE 2: both models on the same post-Aug-7 OOS windows vs real settlement:
        gated accuracy, incremental z vs logit(mid), ECE. new >= old on accuracy AND z -> candidate saved.
Nothing is deployed by this script.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btcpred import data, features, model, ta, predictor

ROOT = Path(__file__).resolve().parent.parent
TF = "15m"; STEP = pd.Timedelta(minutes=15); CUT = pd.Timestamp("2026-08-07", tz="UTC"); DAYS = 1095
GATE1 = 0.99
OUT = ROOT / "reports" / "twap_relabel_15m.md"
L: list[str] = []
def w(s=""): print(s); L.append(s)

# ----------------------------------------------------------------------------- step 1
w("# 15m retrain on TWAP-60 settlement labels\n")
w(f"Run {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC. Features unchanged; only the label changes. Two gates; the run stops at the first failure.\n")
m1 = pd.read_parquet(ROOT / "data" / "btcusdt_1m.parquet")
m1["T"] = m1["open_time"] + pd.Timedelta(minutes=1)            # candle ending at T
m1["twap_fb"] = (m1["open"] + m1["high"] + m1["low"] + m1["close"]) / 4
fb = m1.set_index("T")["twap_fb"]
agg = pd.read_parquet(ROOT / "data" / "twap60_aggtrades_15m.parquet")
agg = agg.dropna(subset=["twap60_agg"]).set_index("boundary")["twap60_agg"]
bounds = pd.date_range(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp.now(tz="UTC").floor("15min"), freq="15min")
tw = pd.DataFrame(index=bounds)
tw["fallback"] = fb.reindex(bounds)
tw["agg"] = agg.reindex(bounds)
tw["twap"] = tw["agg"].fillna(tw["fallback"])
tw["src"] = np.where(tw["agg"].notna(), "aggTrades", np.where(tw["fallback"].notna(), "1m-fallback", "missing"))
nxt = tw["twap"].shift(-1)                                       # twap at T+15m
lab = pd.DataFrame(index=bounds)
lab["synthetic"] = (nxt >= tw["twap"]).astype(float).where(nxt.notna() & tw["twap"].notna())
lab["synthetic_fb"] = (tw["fallback"].shift(-1) >= tw["fallback"]).astype(float).where(tw["fallback"].shift(-1).notna() & tw["fallback"].notna())
lab["src"] = tw["src"]
cov = tw["src"].value_counts()
w("## Step 1 - synthetic labels\n")
w(f"- 15m boundaries 2021-01-01 → now: {len(bounds):,}. TWAP-60 source: aggTrades {cov.get('aggTrades', 0):,} (post-Aug-7 only; every boundary would be ~200k requests), 1-minute fallback {cov.get('1m-fallback', 0):,}, missing {cov.get('missing', 0):,}.")
both = tw.dropna(subset=["agg", "fallback"])
diff_bp = ((both["fallback"] / both["agg"] - 1) * 1e4)
w(f"- Fallback vs aggTrades TWAP on the {len(both):,} boundaries with both: median |diff| {diff_bp.abs().median():.2f} bp, 95th pct {diff_bp.abs().quantile(.95):.2f} bp.")
w(f"- Synthetic labels: {int(lab['synthetic'].notna().sum()):,}; up rate {lab['synthetic'].mean()*100:.2f}%.\n")

# ----------------------------------------------------------------------------- step 2: gate 1
real = pd.read_csv(ROOT / "data" / "settlement_labels_15m.csv")
real["t"] = pd.to_datetime(real["window_start_utc"], utc=True); real = real.set_index("t")
post = real[real.index >= CUT]
j = post.join(lab, how="inner")
def rate(col):
    s = j.dropna(subset=[col]); return (s[col] == s["settled_up"]).mean(), len(s)
r_primary, n_primary = rate("synthetic"); r_fb, n_fb = rate("synthetic_fb")
allj = real.join(lab, how="inner"); r_all, n_all = (allj.dropna(subset=["synthetic_fb"]).pipe(lambda s: ((s["synthetic_fb"] == s["settled_up"]).mean(), len(s))))
w("## Step 2 - GATE 1: synthetic vs real settlement (post-Aug-7)\n")
w(f"- Real settlement source: the bot's `bot.db` (`markets` + `hist_windows`, 15m, resolved), exported to `data/settlement_labels_15m.csv`: {len(real):,} windows {real.index.min():%Y-%m-%d} → {real.index.max():%Y-%m-%d}; post-Aug-7: {len(post):,}.")
w(f"- **Match rate, aggTrades TWAP-60 labels: {r_primary*100:.2f}% on {n_primary:,} windows** ({int(round((1-r_primary)*n_primary))} mismatches).")
w(f"- Match rate, 1-minute fallback labels on the same windows: {r_fb*100:.2f}% on {n_fb:,}.")
w(f"- For information, fallback labels vs real settlement over the whole labelled year: {r_all*100:.2f}% on {n_all:,}.")
mism = j[j["synthetic"].notna() & (j["synthetic"] != j["settled_up"])]
if len(mism):
    mv = ((tw["twap"].shift(-1) - tw["twap"]) / tw["twap"] * 1e4).reindex(mism.index)
    w(f"- Mismatched windows: median |Binance TWAP move| {mv.abs().median():.2f} bp (windows where the two TWAPs nearly tied).")
passed1 = r_primary >= GATE1
w(f"\n**Gate 1: {'PASS' if passed1 else 'FAIL'}** ({r_primary*100:.2f}% vs the 99% requirement).\n")
if not passed1:
    w("Stopped at Gate 1. The synthetic label is not a faithful stand-in for the referee, so the pre-Aug-7 history cannot be relabelled from Binance data. No retrain, no deployment.\n")
    w("**Verdict: don't ship.**")
    OUT.write_text("\n".join(L) + "\n"); print("wrote", OUT); sys.exit(0)

# ----------------------------------------------------------------------------- step 3: relabel + walk-forward
w("## Step 3 - retrain on relabelled history\n")
df = data.drop_open_candle(data.load_or_update(predictor.cache_path(TF), interval=TF, days=DAYS))
_, meta = model.load(predictor.model_dir(TF)); feats = meta["features"]
X = features.build_features(df)[feats]
y_old = features.build_labels(df)                                  # next candle close > open (Binance)
wstart = df["close_time"] + pd.Timedelta(milliseconds=1)           # the window each row predicts starts here
wstart = wstart.dt.floor("15min")
real_up = real["settled_up"].reindex(wstart.values).values
syn = lab["synthetic"].reindex(wstart.values).values
use_real = (wstart.values >= CUT.to_datetime64()) & ~np.isnan(real_up)
y_new = np.where(use_real, real_up, syn)
mask = X.notna().all(axis=1).values & ~np.isnan(y_new) & y_old.notna().values
Xm, t = X[mask].reset_index(drop=True), df.loc[mask, "open_time"].reset_index(drop=True)
yo = y_old[mask].astype(int).reset_index(drop=True); yn = pd.Series(y_new[mask].astype(int))
ws = pd.Series(wstart.values[mask])
w(f"- Training set: {len(Xm):,} candles {t.iloc[0]:%Y-%m-%d} → {t.iloc[-1]:%Y-%m-%d} (same {DAYS}-day window and features as the current model).")
w(f"- Labels: real settlement on {int(use_real[mask].sum()):,} rows (≥ Aug 7), synthetic on {int((~use_real[mask]).sum()):,}. Old vs new label disagree on {int((yo != yn).mean()*1e4)/100:.2f}% of rows.")
w("- Protocol: 5 expanding walk-forward folds, first 40% as the initial training set, early stopping on the last 10% of each training set. Both label sets run through the identical protocol on identical rows.\n")
rep_new = model.walk_forward(Xm, yn, t, n_folds=5, return_predictions=True)
rep_old = model.walk_forward(Xm, yo, t, n_folds=5, return_predictions=True)
pn, po = rep_new["predictions"], rep_old["predictions"]
w("| Walk-forward (own labels) | old model, Binance labels | new model, TWAP/settlement labels |")
w("| --- | ---: | ---: |")
w(f"| OOS accuracy | {rep_old['overall']['accuracy']*100:.2f}% | {rep_new['overall']['accuracy']*100:.2f}% |")
w(f"| OOS AUC | {rep_old['overall']['auc']:.3f} | {rep_new['overall']['auc']:.3f} |")
w(f"| OOS log loss | {rep_old['overall']['logloss']:.4f} | {rep_new['overall']['logloss']:.4f} |")
w("")

# ----------------------------------------------------------------------------- step 4: gate 2
w("## Step 4 - GATE 2: both models on the same post-Aug-7 OOS windows, graded against real settlement\n")
first = int(len(Xm) * 0.4)
oos = pd.DataFrame({"time": t.iloc[first:].values, "wstart": ws.iloc[first:].values, "p_new": pn["ml_p"].values, "p_old": po["ml_p"].values})
rp = ta.score(ta.signals(df), json.load(open(predictor.ta_report_path(TF)))["weights"])
tadir = rp["direction"].values[mask][first:]; taconf = rp["confidence"].values[mask][first:]
oos["ta_dir"], oos["ta_conf"] = tadir, taconf
oos["y"] = real["settled_up"].reindex(oos["wstart"]).values
oos["mid"] = real["up_mid_open"].reindex(oos["wstart"]).values
g = oos[(oos["wstart"] >= CUT.to_datetime64()) & oos["y"].notna()].copy()
w(f"- Windows: {len(g):,} post-Aug-7 OOS windows with a real settlement ({pd.Timestamp(g['wstart'].min()):%Y-%m-%d} → {pd.Timestamp(g['wstart'].max()):%Y-%m-%d}); both models' predictions for them come from walk-forward folds trained before those dates. {int(g['mid'].notna().sum())} of them have a market mid at open (bot capture from 3 Sep).")
def ece(p, y, bins=10):
    q = np.clip((p * bins).astype(int), 0, bins - 1); e = 0.0
    for b in range(bins):
        m_ = q == b
        if m_.any(): e += m_.mean() * abs(y[m_].mean() - p[m_].mean())
    return e
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
import statsmodels.api as sm
def grade(p, tag):
    d = (p >= 0.5).astype(int); acc = (d == g["y"].values).mean()
    conf = np.abs(p - 0.5) * 2
    gate = (g["ta_dir"].values == d) & (conf >= 0.10) & (g["ta_conf"].values >= 0.3)
    gacc = (d[gate] == g["y"].values[gate]).mean() if gate.any() else float("nan")
    hm = g["mid"].notna().values
    Xz = sm.add_constant(np.column_stack([logit(g["mid"].values[hm]), logit(p[hm])]))
    try:
        fit = sm.Logit(g["y"].values[hm].astype(float), Xz).fit(disp=0); z = fit.tvalues[2]; zmid = fit.tvalues[1]
    except Exception as e:
        z = zmid = float("nan")
    return dict(acc=acc, gated_acc=gacc, n_gated=int(gate.sum()), z=z, z_mid=zmid, n_z=int(hm.sum()), ece=ece(p, g["y"].values), brier=float(np.mean((p - g["y"].values) ** 2)))
ro, rn = grade(g["p_old"].values, "old"), grade(g["p_new"].values, "new")
w("")
w("| Post-Aug-7 OOS, real settlement | old model | new model |")
w("| --- | ---: | ---: |")
w(f"| Accuracy, all windows (n={len(g):,}) | {ro['acc']*100:.2f}% | {rn['acc']*100:.2f}% |")
w(f"| Gated accuracy (agree, ML ≥ 0.10, TA ≥ 0.3) | {ro['gated_acc']*100:.2f}% (n={ro['n_gated']}) | {rn['gated_acc']*100:.2f}% (n={rn['n_gated']}) |")
w(f"| Incremental z of logit(p) vs logit(mid) (n={ro['n_z']}) | {ro['z']:+.2f} | {rn['z']:+.2f} |")
w(f"| z of logit(mid) itself, same regression | {ro['z_mid']:+.2f} | {rn['z_mid']:+.2f} |")
w(f"| ECE, 10 bins | {ro['ece']*100:.2f}% | {rn['ece']*100:.2f}% |")
w(f"| Brier | {ro['brier']:.4f} | {rn['brier']:.4f} |")
se = math.sqrt(0.25 / max(1, ro["n_gated"])) * 100
w(f"\nSampling error on gated accuracy at this n: about ±{se:.1f} points (1σ).")
passed2 = (rn["gated_acc"] >= ro["gated_acc"]) and (rn["z"] >= ro["z"])
w(f"\n**Gate 2: {'PASS' if passed2 else 'FAIL'}** — new {'≥' if rn['gated_acc'] >= ro['gated_acc'] else '<'} old on gated accuracy, new {'≥' if rn['z'] >= ro['z'] else '<'} old on incremental z.\n")
if passed2:
    rounds = int(np.median([f["best_iter"] for f in rep_new["folds"]])) or 200
    booster = model.train_final(Xm, yn, rounds)
    cand = ROOT / "models" / TF / "lgbm_twap_candidate"
    model.save(booster, feats, dict(rep_new, predictions=None, labels="twap60-settlement", label_cut=str(CUT)), cand)
    w(f"Candidate saved to `{cand.relative_to(ROOT)}/` (not deployed; `models/{TF}/lgbm/` untouched). To ship: move it over `models/{TF}/lgbm/`, run `main.py backtest -i 15m` so the history and calibrator refit, and restart the server — **only after the bot's 150-fill verdict on the current model is in.**\n")
    w("**Verdict: ship** (pending the 150-fill verdict).")
else:
    w("Keeping the old model. The relabelled model did not beat it on the real-settlement windows on both required metrics.\n")
    w("**Verdict: don't ship.**")
OUT.write_text("\n".join(L) + "\n"); print("wrote", OUT)
