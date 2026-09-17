"""Amended run (2026-09-17): Steps B and C after the amended Gate 1 passed. Appends to reports/twap_relabel_15m.md.

Step B  TWAP-60 at every 15m boundary from data.binance.vision monthly aggTrades (data/twap60_bulk/*.parquet).
        Labels: synthetic TWAP-60 before 2026-08-07, real settlement after. Training rows whose |TWAP move| < THR bp
        are excluded; evaluation keeps every window. Same walk-forward protocol; features unchanged.
Step C  GATE 2: old (candle) vs new (TWAP) on the same post-Aug-7 OOS windows against real settlement:
        gated accuracy, incremental z vs logit(mid), ECE. Candidate saved only if new >= old on accuracy AND z.
Nothing is deployed by this script.
"""
from __future__ import annotations
import glob, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btcpred import data, features, model, ta, predictor

ROOT = Path(__file__).resolve().parent.parent
TF = "15m"; CUT = pd.Timestamp("2026-08-07", tz="UTC"); DAYS = 1095; THR_BP = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
OUT = ROOT / "reports" / "twap_relabel_15m.md"
L: list[str] = []
def w(s=""): print(s); L.append(s)

# ----------------------------------------------------------------------------- Step B: labels
w("\n## Step B - relabel and retrain (bulk aggTrades TWAP-60)\n")
bulk = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "data" / "twap60_bulk" / "*.parquet")))])
bulk = bulk.drop_duplicates("boundary").set_index("boundary").sort_index()
tw_bulk = bulk["twap60"]
per_req = pd.read_parquet(ROOT / "data" / "twap60_aggtrades_15m.parquet").dropna(subset=["twap60_agg"]).set_index("boundary")["twap60_agg"]
ov = pd.concat([tw_bulk.rename("bulk"), per_req.rename("api")], axis=1).dropna()
# the current month has no monthly zip yet: fill its boundaries from the per-request aggTrades TWAP (same estimator)
tw = tw_bulk.combine_first(per_req).sort_index()
filled = int(tw.index.difference(tw_bulk.index).size)
dev = ((ov.bulk / ov.api - 1) * 1e4).abs()
w(f"- Bulk TWAP-60 boundaries: {len(tw):,} ({tw.index.min():%Y-%m-%d} → {tw.index.max():%Y-%m-%d}), from {len(glob.glob(str(ROOT / 'data' / 'twap60_bulk' / '*.parquet')))} monthly files.")
w(f"- Consistency with the per-request aggTrades TWAP validated in Step A, on {len(ov):,} shared boundaries: median |diff| {dev.median():.3f} bp, max {dev.max():.3f} bp (same estimator, same trades).")
w(f"- {filled:,} boundaries in the current month (no monthly zip published yet) filled from the per-request aggTrades TWAP.")
nxt = tw.shift(-1); move_bp = (nxt / tw - 1) * 1e4
syn = (nxt >= tw).astype(float).where(nxt.notna())
real = pd.read_csv(ROOT / "data" / "settlement_labels_15m.csv"); real["t"] = pd.to_datetime(real.window_start_utc, utc=True); real = real.set_index("t")

df = data.drop_open_candle(data.load_or_update(predictor.cache_path(TF), interval=TF, days=DAYS))
_, meta = model.load(predictor.model_dir(TF)); feats = meta["features"]
X = features.build_features(df)[feats]
y_old = features.build_labels(df)
wstart = pd.DatetimeIndex((df["close_time"] + pd.Timedelta(milliseconds=1)).dt.floor("15min"))   # tz-aware, matches the label indexes
real_up = real["settled_up"].reindex(wstart).values
syn_v = syn.reindex(wstart).values; mv = move_bp.reindex(wstart).values
use_real = np.asarray(wstart >= CUT) & ~np.isnan(real_up)
y_new = np.where(use_real, real_up, syn_v)
base = X.notna().all(axis=1).values & ~np.isnan(y_new) & y_old.notna().values
near_tie = base & ~use_real & (np.abs(np.nan_to_num(mv, nan=np.inf)) < THR_BP)   # synthetic rows only; real-settled rows always kept
train_mask = base & ~near_tie
w(f"- Candles with features and both labels: {int(base.sum()):,} ({df.loc[base, 'open_time'].iloc[0]:%Y-%m-%d} → {df.loc[base, 'open_time'].iloc[-1]:%Y-%m-%d}). Real settlement on {int((use_real & base).sum()):,} rows (≥ Aug 7), synthetic TWAP-60 on the rest.")
w(f"- Near-tie exclusion (|TWAP move| < {THR_BP:g} bp, synthetic rows only): drops **{int(near_tie.sum()):,} rows ({near_tie.sum()/base.sum()*100:.1f}%)** from training. Evaluation keeps every window.")
w(f"- Old (candle) vs new (TWAP/settlement) label disagree on {(y_old.values[base] != y_new[base]).mean()*100:.2f}% of rows.")

# Walk-forward on the same protocol. Both models are trained on the same rows (near-ties removed) so the only
# difference is the label; both are then scored on ALL rows of the test blocks.
Xt = X[train_mask].reset_index(drop=True); tt = df.loc[train_mask, "open_time"].reset_index(drop=True)
yn_t = pd.Series(y_new[train_mask].astype(int)); yo_t = y_old[train_mask].astype(int).reset_index(drop=True)
Xa = X[base].reset_index(drop=True); ta_ = df.loc[base, "open_time"].reset_index(drop=True)
yn_a = pd.Series(y_new[base].astype(int)); yo_a = y_old[base].astype(int).reset_index(drop=True); ws_a = wstart[base]
import lightgbm as lgb
def walk(Xtr, ytr, ttr, Xall, tall, n_folds=5, frac=0.4):
    """Expanding walk-forward: fold boundaries by time; train on training rows before the boundary (with an
    early-stopping slice), predict every evaluation row in the block. Mirrors model.walk_forward."""
    n = len(Xall); first = int(n * frac); edges = np.linspace(first, n, n_folds + 1, dtype=int)
    preds = np.full(n, np.nan); folds = []
    for i in range(n_folds):
        t_end, t_next = tall.iloc[edges[i]], (tall.iloc[edges[i + 1] - 1])
        tr = ttr < t_end; ntr = int(tr.sum()); va = int(ntr * 0.9)
        dtr = lgb.Dataset(Xtr[tr].iloc[:va], ytr[tr].iloc[:va]); dva = lgb.Dataset(Xtr[tr].iloc[va:], ytr[tr].iloc[va:], reference=dtr)
        b = lgb.train(model.PARAMS, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        sl = slice(edges[i], edges[i + 1]); preds[sl] = b.predict(Xall.iloc[sl], num_iteration=b.best_iteration)
        folds.append(dict(test_start=str(t_end), test_end=str(t_next), best_iter=int(b.best_iteration), n_train=ntr))
    return preds, folds, first
p_new, f_new, first = walk(Xt, yn_t, tt, Xa, ta_)
p_old, f_old, _ = walk(Xt, yo_t, tt, Xa, ta_)
oos = slice(first, len(Xa))
from sklearn.metrics import roc_auc_score, log_loss
def own(p, y): d = (p >= 0.5).astype(int); return (d == y).mean(), roc_auc_score(y, p), log_loss(y, p)
o_new = own(p_new[oos], yn_a.values[oos]); o_old = own(p_old[oos], yo_a.values[oos])
w("\n| Walk-forward, each model on its own labels, all OOS rows | old model (candle labels) | new model (TWAP/settlement labels) |")
w("| --- | ---: | ---: |")
w(f"| OOS accuracy | {o_old[0]*100:.2f}% | {o_new[0]*100:.2f}% |"); w(f"| OOS AUC | {o_old[1]:.3f} | {o_new[1]:.3f} |"); w(f"| OOS log loss | {o_old[2]:.4f} | {o_new[2]:.4f} |")
w(f"| Trees per fold | {[f['best_iter'] for f in f_old]} | {[f['best_iter'] for f in f_new]} |")

# ----------------------------------------------------------------------------- Step C: Gate 2
w("\n## Step C - GATE 2: both models on the same post-Aug-7 OOS windows, graded against real settlement\n")
rp = ta.score(ta.signals(df), json.load(open(predictor.ta_report_path(TF)))["weights"])
g = pd.DataFrame({"wstart": ws_a, "p_new": p_new, "p_old": p_old, "ta_dir": rp["direction"].values[base], "ta_conf": rp["confidence"].values[base]}).iloc[oos]
gi = pd.DatetimeIndex(g["wstart"])
g["y"] = real["settled_up"].reindex(gi).values; g["mid"] = real["up_mid_open"].reindex(gi).values
g["move"] = np.abs(move_bp.reindex(gi).values)
g = g[np.asarray(gi >= CUT) & g["y"].notna().values].copy()
w(f"- Windows: {len(g):,} post-Aug-7 OOS windows with real settlement ({pd.Timestamp(g.wstart.min()):%Y-%m-%d} → {pd.Timestamp(g.wstart.max()):%Y-%m-%d}), predicted by folds trained before them; near-ties included. {int(g['mid'].notna().sum())} have a market mid at open (bot capture from 3 Sep).")
def ece(p, y, bins=10):
    q = np.clip((p * bins).astype(int), 0, bins - 1); return sum(((q == b).mean() * abs(y[q == b].mean() - p[q == b].mean())) for b in range(bins) if (q == b).any())
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def grade(p):
    y = g["y"].values; d = (p >= 0.5).astype(int); conf = np.abs(p - 0.5) * 2
    gate = (g["ta_dir"].values == d) & (conf >= 0.10) & (g["ta_conf"].values >= 0.3)
    hm = g["mid"].notna().values
    try:
        fit = sm.Logit(y[hm].astype(float), sm.add_constant(np.column_stack([logit(g["mid"].values[hm]), logit(p[hm])]))).fit(disp=0); z, zm = fit.tvalues[2], fit.tvalues[1]
    except Exception: z = zm = float("nan")
    nt = g["move"].values >= THR_BP
    return dict(acc=(d == y).mean(), acc_nt=(d[nt] == y[nt]).mean(), gated=(d[gate] == y[gate]).mean() if gate.any() else float("nan"), n_gated=int(gate.sum()),
                z=z, z_mid=zm, n_z=int(hm.sum()), ece=ece(p, y), brier=float(np.mean((p - y) ** 2)))
ro, rn = grade(g["p_old"].values), grade(g["p_new"].values)
w("\n| Post-Aug-7 OOS, real settlement | old model | new model |")
w("| --- | ---: | ---: |")
w(f"| Accuracy, all windows (n={len(g):,}) | {ro['acc']*100:.2f}% | {rn['acc']*100:.2f}% |")
w(f"| Accuracy, windows with \\|TWAP move\\| ≥ {THR_BP:g} bp | {ro['acc_nt']*100:.2f}% | {rn['acc_nt']*100:.2f}% |")
w(f"| **Gated accuracy** (agree, ML ≥ 0.10, TA ≥ 0.3) | **{ro['gated']*100:.2f}%** (n={ro['n_gated']}) | **{rn['gated']*100:.2f}%** (n={rn['n_gated']}) |")
w(f"| **Incremental z** of logit(p) vs logit(mid) (n={ro['n_z']}) | **{ro['z']:+.2f}** | **{rn['z']:+.2f}** |")
w(f"| z of logit(mid) in the same regression | {ro['z_mid']:+.2f} | {rn['z_mid']:+.2f} |")
w(f"| ECE, 10 bins | {ro['ece']*100:.2f}% | {rn['ece']*100:.2f}% |")
w(f"| Brier | {ro['brier']:.4f} | {rn['brier']:.4f} |")
w(f"\nSampling error on gated accuracy at n≈{ro['n_gated']}: about ±{math.sqrt(0.25 / max(1, ro['n_gated']))*100:.1f} points (1σ); on the z statistic with n={ro['n_z']}: ±1.")
passed = (rn["gated"] >= ro["gated"]) and (rn["z"] >= ro["z"])
w(f"\n**Gate 2: {'PASS' if passed else 'FAIL'}** — gated accuracy new {'≥' if rn['gated'] >= ro['gated'] else '<'} old; incremental z new {'≥' if rn['z'] >= ro['z'] else '<'} old.\n")
if passed:
    rounds = int(np.median([f["best_iter"] for f in f_new])) or 200
    booster = model.train_final(Xt, yn_t, rounds)
    cand = ROOT / "models" / TF / "lgbm_twap_candidate"
    model.save(booster, feats, dict(folds=f_new, overall=dict(accuracy=o_new[0], auc=o_new[1], logloss=o_new[2]), labels="twap60-bulk+settlement", label_cut=str(CUT), near_tie_bp=THR_BP, gate2=dict(old=ro, new=rn)), cand)
    w(f"Candidate saved to `{cand.relative_to(ROOT)}/`. Not deployed: `models/{TF}/lgbm/` is untouched. To ship after the bot's 150-fill verdict: move the candidate over `models/{TF}/lgbm/`, run `main.py backtest -i 15m` (history + calibrator refit, now settlement-calibrated), restart the server.\n")
    w("**Verdict: ship** (as a new version, pending the 150-fill verdict on the current model).")
else:
    w("Keeping the old model.\n"); w("**Verdict: don't ship.**")
with OUT.open("a") as f: f.write("\n".join(L) + "\n")
print("appended to", OUT)
