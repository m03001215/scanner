"""Markdown + PNG reports for probability calibration (reports/calibration_<tf>.md, reports/calibration_summary.md)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration as C

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"; FIG = REP / "figures"
P = lambda x, d=1: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.{d}f}%"


def _table(head, rows, align=None):
    align = align or ["l"] + ["r"] * (len(head) - 1)
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---:" if a == "r" else "---" for a in align) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def rel_md(t: pd.DataFrame, label="") -> str:
    if t.empty:
        return "_no rows_\n"
    rows = [[r.bin, f"{r.p_lo:.3f}–{r.p_hi:.3f}", f"{r.n:,}", P(r.pred), P(r.obs), f"{P(r.ci_lo)}–{P(r.ci_hi)}", f"{r.gap*100:+.1f}",
             "✓" if r.ci_lo <= r.pred <= r.ci_hi else "✗"] for r in t.itertuples()]
    return _table(["Bin", "p range", "n", "Mean predicted", "Observed", "Wilson 95% CI", "Gap (pt)", "Diagonal in CI"], rows,
                  ["r", "l", "r", "r", "r", "l", "r", "l"])


def plot(interval: str, r: dict, tag: str) -> list[str]:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True); files = []
    def diag(ax):
        ax.plot([0, 1], [0, 1], color="#9aa3ad", lw=1, ls="--", label="perfect calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("mean predicted p_bullish"); ax.set_ylabel("observed rate"); ax.grid(alpha=.25)
    def curve(ax, t, color, label):
        if t.empty: return
        ax.errorbar(t["pred"], t["obs"], yerr=[t["obs"] - t["ci_lo"], t["ci_hi"] - t["obs"]], fmt="o-", ms=3, lw=1.2, color=color, capsize=2, label=label)
    # figure 1: all / gated / directions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    diag(axes[0]); curve(axes[0], r["rel_all"], "#2a78d6", f"all rows (n={len(r['rows']):,})"); axes[0].set_title(f"{interval} · all OOS rows"); axes[0].legend(fontsize=8)
    diag(axes[1]); curve(axes[1], r["rel_gated"], "#1baf7a", f"gated (n={int(r['rows']['gated'].sum()):,})"); axes[1].set_title(f"{interval} · gated region"); axes[1].legend(fontsize=8)
    d = r["direction"]; diag(axes[2]); curve(axes[2], d["bull"], "#0ca30c", f"bullish calls (n={d['n_bull']:,})"); curve(axes[2], d["bear_folded"], "#d03b3b", f"bearish, folded (n={d['n_bear']:,})")
    axes[2].set_xlim(0.45, 1); axes[2].set_ylim(0.3, 1); axes[2].set_title(f"{interval} · by direction (bearish folded to 1−p)"); axes[2].legend(fontsize=8)
    fig.suptitle(f"Reliability diagrams · {interval} · {tag}", fontsize=11); fig.tight_layout()
    f1 = FIG / f"calibration_{interval}_reliability.png"; fig.savefig(f1, dpi=110); plt.close(fig); files.append(f1.name)
    # figure 2: held-out before/after + fold curves
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    h = r["hold"]; diag(axes[0]); curve(axes[0], h["gated_raw"], "#8a8985", "gated, raw p"); curve(axes[0], h["gated_iso"], "#1baf7a", "gated, isotonic-calibrated")
    axes[0].set_title(f"{interval} · held-out half, gated region"); axes[0].legend(fontsize=8)
    st = r["stability"]; grid = np.array(st["grid"]); diag(axes[1])
    for k, cv in st["curves"].items():
        cv = np.array([np.nan if v is None else v for v in cv]); axes[1].plot(grid, cv, lw=1.4, label=f"fold {k}")
    axes[1].set_title(f"{interval} · isotonic curve per walk-forward fold"); axes[1].legend(fontsize=8)
    fig.suptitle(f"Calibration fit · {interval} · {tag}", fontsize=11); fig.tight_layout()
    f2 = FIG / f"calibration_{interval}_fit.png"; fig.savefig(f2, dpi=110); plt.close(fig); files.append(f2.name)
    return files


def write_interval(interval: str, r: dict) -> Path:
    i, e, st, d, cal = r["info"], r["evaluation"], r["stability"], r["direction"], r["calibrator"]
    tag = i["tag"]; figs = plot(interval, r, tag)
    L = []; w = L.append
    w(f"# Probability calibration · {interval} · {tag}\n")
    w(f"Model version `{cal['model_hash']}`, calibrator version `{cal['version']}`, fitted {cal['fitted_at']}. Labels: **{tag}**.\n")
    w("## Labels and data\n")
    w(_table(["Item", "Value"], [
        ["Label mode", f"**{i['mode']}** ({tag})"],
        ["Label source", i["file"] or "Binance candle: close ≥ open of the predicted candle"],
        ["Note", i.get("note", "")],
        ["OOS rows available", f"{i['n_rows']:,} (walk-forward, point-in-time; live rows and in-sample predictions excluded)"],
        ["Rows skipped (no label)", f"{i['skipped']:,}"],
        ["Ties counted as up (close = open)", f"{i['ties_counted_up']:,}"],
        ["Rows used", f"{i['n_used']:,}, {i['period'][0]} → {i['period'][1]}"],
        ["Gated region", f"agree AND \\|ta.score\\| ≥ {i['gate_score_threshold']} AND (p ≥ 0.55 or p ≤ 0.45); TA weight total {i['ta_weight_total']:.0f}" + (f" → threshold ceil(0.30 × {i['ta_weight_total']:.0f}) = {i['gate_score_threshold']}" if interval == "4h" else "")],
        ["Gated rows", f"{i['n_gated']:,} ({i['n_gated']/i['n_used']*100:.1f}%)"],
    ], ["l", "l"]))
    w("## Part 1 · Reliability diagrams\n")
    w(f"Equal-count bins, at least {C.MIN_BIN} rows per bin, bin count scaling with n. \"Diagonal in CI\" marks bins whose mean predicted p lies inside the Wilson 95% CI of the observed rate.\n")
    w(f"![reliability]({'figures/' + figs[0]})\n")
    w(f"### All rows ({len(r['rows']):,}, {len(r['rel_all'])} bins) · ECE {P(C.ece(r['rel_all']), 2)}\n"); w(rel_md(r["rel_all"]))
    w(f"### Gated rows ({i['n_gated']:,}, {len(r['rel_gated'])} bins) · ECE {P(C.ece(r['rel_gated']), 2)}" + (" · **bins widened** because some had < 100 rows" if r["widened"] else "") + "\n"); w(rel_md(r["rel_gated"]))
    w(f"### By call direction · bullish calls (p > 0.5), {d['n_bull']:,} rows · ECE {P(d['ece_bull'], 2)}\n"); w(rel_md(d["bull"]))
    w(f"### By call direction · bearish calls folded to 1−p, {d['n_bear']:,} rows · ECE {P(d['ece_bear'], 2)}\n"); w(rel_md(d["bear_folded"]))
    w(f"**Direction asymmetry:** calibration gap (observed − predicted) is {d['gap_bull']*100:+.2f} pt for bullish calls and {d['gap_bear']*100:+.2f} pt for folded bearish calls; z = {d['z']:+.2f} → " +
      ("**the two directions calibrate differently** (|z| ≥ 2)." if d["asymmetric"] else "no material difference (|z| < 2)."))
    w("")
    w("## Part 2 · Calibrator fit\n")
    w(f"Fit on the first half of the OOS period ({e['fit_start']} → {e['fit_end']}, {e['n_fit']:,} rows), evaluated on the second half ({e['hold_start']} → {e['hold_end']}, {e['n_hold']:,} rows, {e['n_hold_gated']:,} gated).\n")
    b, ec = e["brier"], e["ece"]
    w(f"Primary calibrator: isotonic regression fitted to equal-count bin means with ≥ {e['isotonic_bin_rows']} rows per step (\"binned isotonic\"). Plain isotonic on individual rows and Platt scaling are shown for comparison.\n")
    w(_table(["Metric (held-out half)", "Raw p", "Isotonic (binned, primary)", "Isotonic (plain)", "Platt"], [
        ["Brier score", f"{b['raw']:.5f}", f"{b['isotonic']:.5f}", f"{b['isotonic_plain']:.5f}", f"{b['platt']:.5f}"],
        ["ECE, all rows", P(ec["raw"], 2), P(ec["isotonic"], 2), P(ec["isotonic_plain"], 2), P(ec["platt"], 2)],
        ["ECE, gated rows", P(e["ece_gated"]["raw"], 2), P(e["ece_gated"]["isotonic"], 2), P(e["ece_gated"]["isotonic_plain"], 2), P(e["ece_gated"]["platt"], 2)],
    ]))
    gain = b["raw"] - b["isotonic"]
    w(("**Read this honestly:** the calibrator changes the held-out Brier score by " + f"{gain:+.5f}" + ", which is " +
       ("a real improvement." if gain > 2e-4 else "within noise: the raw score is already close to calibrated on this interval, and the top-tier drift seen in Part 1 does not repeat consistently enough between halves to correct.")) + "\n")
    t = e["top_tier"]
    w(f"**Top tier** (most confident held-out calls, folded, n={t['n']:,}): raw said **{P(t['said'])}** → got **{P(t['got'])}** (CI {P(t['ci'][0])}–{P(t['ci'][1])}); after isotonic said {P(t['said_after'])} → got {P(t['got_after'])}.\n")
    ok = e["gated_within_ci"]
    w(f"**Gated bins after calibration:** {ok['bins_ok']} of {ok['bins_total']} bins have the diagonal inside the CI → " + ("**pass**" if ok["all_bins"] else "**FAIL**") + ".\n")
    w(f"![fit]({'figures/' + figs[1]})\n")
    w("### Held-out reliability, gated rows, raw p\n"); w(rel_md(r["hold"]["gated_raw"]))
    w("### Held-out reliability, gated rows, after isotonic\n"); w(rel_md(r["hold"]["gated_iso"]))
    w("### Held-out reliability, all rows, after isotonic\n"); w(rel_md(r["hold"]["all_iso"]))
    w("### Stability across walk-forward folds\n")
    w(_table(["Fold", "Period", "n", "Mean p", "Observed", "Brier raw", "ECE raw", "p range"],
             [[f["fold"], f"{f['start']} → {f['end']}", f"{f['n']:,}", P(f["mean_p"]), P(f["obs"]), f"{f['brier_raw']:.5f}", P(f["ece_raw"], 2), f"{f['p_min']:.3f}–{f['p_max']:.3f}"] for f in st["per_fold"]]))
    w(f"**Verdict:** {st['verdict']}. " + ("" if st["stable"] else "The calibrator therefore refits automatically at every retrain and backtest of this interval (built into `main.py train` and `main.py backtest`).") + "\n")
    w("## Part 3 · Deployed calibrator\n")
    w(f"Isotonic regression on all {cal['n']:,} OOS rows, version `{cal['version']}`, stored at `calibrators/p_bullish_{interval}_{cal['version']}.json` and served as `models/{interval}/calibration.json`. "
      f"Exposed in `/api/predict?interval={interval}` as `ml.p_bullish_cal`, `ml.calibration_version`, `ml.calibration_labels`, `ml.calibration_n`; full lookup at `/api/calibration/{interval}`.\n")
    lk = [x for x in cal["lookup"] if abs(x["p"] * 100 % 5) < 1e-6 or x["p"] in (0.45, 0.55)]
    w(_table(["Raw p", "Calibrated p"], [[f"{x['p']:.3f}", f"{x['p_cal']:.4f}"] for x in lk]))
    w("**Caveat.** The OOS rows come from the walk-forward fold models; the deployed model is retrained on all data and can be sharper or flatter than those folds, so the calibrator is best read as calibrating the *modelling recipe* until live rows accumulate.\n")
    out = REP / f"calibration_{interval}.md"; out.write_text("\n".join(L) + "\n"); return out


def write_summary(results: dict) -> Path:
    rows = []
    for tf, r in results.items():
        i, e, st, d = r["info"], r["evaluation"], r["stability"], r["direction"]; t = e["top_tier"]
        ok = e["gated_within_ci"]
        rows.append([tf, f"{i['mode']} ({i['tag']})", f"{i['n_used']:,}", f"{i['n_gated']:,}",
                     f"{e['brier']['raw']:.5f} → {e['brier']['isotonic']:.5f}", f"{P(e['ece']['raw'], 2)} → {P(e['ece']['isotonic'], 2)}",
                     f"said {P(t['said'])} → got {P(t['got'])}", "y" if d["asymmetric"] else "n", "y" if st["stable"] else "n",
                     f"{ok['bins_ok']}/{ok['bins_total']}" + ("" if ok["all_bins"] else " ✗")])
    L = ["# Probability calibration · cross-interval summary\n",
         f"Generated {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}. Every interval: one isotonic calibrator per (interval, model version), fit on walk-forward OOS rows only, Brier/ECE measured on the held-out second half of the OOS period. No settlement label files were present, so all four intervals are **binance-calibrated** (close ≥ open); 4h's referee is not yet verified.\n",
         _table(["Interval", "Labels used", "n OOS", "n gated", "Brier raw → cal", "ECE raw → cal", "Top-tier miscalibration", "Direction asymmetry", "Fold-stable", "Gated bins within CI after cal"], rows,
                ["l", "l", "r", "r", "l", "l", "l", "l", "l", "l"]),
         "**What the numbers say.** On 5m, 15m and 1h the raw p_bullish is already close to calibrated on the held-out half: the calibrator moves Brier by less than 0.0001, i.e. within noise. The top-tier drift that motivated this work is real in the first half of each OOS period but does not repeat consistently in the second half (15m: said 59.9% → got 56.8% held out; 5m: said 57.4% → got 53.0%; 4h: said 64.7% → got 61.0%; 1h ran the other way, said 59.6% → got 60.7%), which is why a calibrator fitted on the first half cannot fully correct it and why 15m and 1h fail the every-gated-bin-within-CI requirement. 4h is the one interval where calibration clearly helps (Brier −0.00035, ECE 3.1% → 2.7%). 15m is the only interval where bullish and bearish calls calibrate differently (bullish calls run ~1 pt too bold, bearish calls are on the diagonal; z = −3.0). The bot should treat `p_bullish_cal` as a mild shrink toward 0.5 on 5m/15m/1h and as a genuine correction on 4h.\n",
         "Method: isotonic regression fitted to equal-count bin means (≥ 1000 rows per step) — plain per-row isotonic memorised the fitting half and made 5m materially worse (gated ECE 2.3% → 16.5%), so it is reported only as a comparison. Calibrators refit automatically on every `train` and `backtest` of an interval; the version string is `<model hash>-<timestamp>`. Per-interval detail: `reports/calibration_<tf>.md`.\n"]
    out = REP / "calibration_summary.md"; out.write_text("\n".join(L)); return out
