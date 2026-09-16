"""Probability calibration for p_bullish, one calibrator per (interval, model version).

Data: only stored walk-forward OUT-OF-SAMPLE rows (models/<tf>/history.csv.gz, source="backtest").
Live rows and in-sample predictions are never used.
Labels: data/settlement_labels_<tf>.csv (window_start_utc, settled_up) when present -> "settlement";
        otherwise Binance close >= open of the predicted candle -> "binance".
Fit: isotonic regression (primary) and Platt scaling (comparison) on the first half of the OOS period,
     evaluated on the second half; the deployed calibrator is the isotonic fit on all OOS rows.
No trading logic lives here. TA scores, confidence definitions and gate logic are read, never changed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, predictor

ROOT = Path(__file__).resolve().parent.parent
CAL_DIR = ROOT / "calibrators"
MIN_BIN = 100
# Gated region in raw units (spec): agree AND |ta.score| >= threshold AND (p >= 0.55 or p <= 0.45).
GATE_SCORE = {"5m": 2, "15m": 5, "1h": 5}   # 4h is derived from its TA weight total at run time
P_HI, P_LO = 0.55, 0.45


# ----------------------------------------------------------------------------- helpers
def model_hash(interval: str) -> str:
    return hashlib.md5((predictor.model_dir(interval) / "model.txt").read_bytes()).hexdigest()[:8]


def ta_weight_total(interval: str) -> float:
    w = json.load(open(predictor.ta_report_path(interval)))["weights"]
    return float(sum(abs(v) for v in w.values()))


def gate_threshold(interval: str) -> int:
    if interval in GATE_SCORE:
        return GATE_SCORE[interval]
    return math.ceil(0.30 * ta_weight_total(interval))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def n_bins(n: int) -> int:
    """Equal-count bins that scale with n while keeping >= MIN_BIN rows per bin."""
    if n < MIN_BIN:
        return 1
    return int(max(1, min(n // MIN_BIN, 2 * math.sqrt(n / MIN_BIN), 40)))


def reliability(p: np.ndarray, y: np.ndarray, bins: int | None = None) -> pd.DataFrame:
    """Equal-count bins over p: n, mean predicted, observed rate, Wilson 95% CI, gap."""
    n = len(p)
    if n == 0:
        return pd.DataFrame(columns=["bin", "p_lo", "p_hi", "n", "pred", "obs", "ci_lo", "ci_hi", "gap"])
    b = bins or n_bins(n)
    order = np.argsort(p, kind="stable"); idx = np.array_split(order, b)
    rows = []
    for i, ix in enumerate(idx):
        if len(ix) == 0:
            continue
        pp, yy = p[ix], y[ix]; k = int(yy.sum()); lo, hi = wilson(k, len(ix))
        rows.append(dict(bin=i + 1, p_lo=float(pp.min()), p_hi=float(pp.max()), n=int(len(ix)), pred=float(pp.mean()), obs=float(yy.mean()),
                         ci_lo=lo, ci_hi=hi, gap=float(yy.mean() - pp.mean())))
    return pd.DataFrame(rows)


def ece(table: pd.DataFrame) -> float:
    if table.empty:
        return float("nan")
    return float((table["n"] * (table["obs"] - table["pred"]).abs()).sum() / table["n"].sum())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def fold_labels(interval: str, times: pd.Series) -> np.ndarray:
    folds = json.load(open(predictor.model_dir(interval) / "meta.json"))["report"]["folds"]
    step = pd.Timedelta(milliseconds=data.INTERVAL_MS[interval])
    starts = np.array([(pd.Timestamp(f["test_start"]) + step).to_datetime64() for f in folds], dtype="datetime64[ns]")
    return np.searchsorted(starts, times.dt.tz_convert(None).values.astype("datetime64[ns]"), side="right")


# ----------------------------------------------------------------------------- data + labels
def load_rows(interval: str) -> tuple[pd.DataFrame, dict]:
    """Walk-forward OOS rows with the label used for calibration. Returns (rows, label_info)."""
    h = predictor.load_history(interval)
    h = h[h["source"] == "backtest"].copy() if "source" in h.columns else h.copy()
    h["time"] = pd.to_datetime(h["time"], utc=True)
    n_total = len(h)
    info = dict(mode="binance", tag="binance-calibrated", file=None, skipped=0, ties_counted_up=0, n_rows=n_total)
    settle = ROOT / "data" / f"settlement_labels_{interval}.csv"
    if settle.exists():
        s = pd.read_csv(settle)
        s["window_start_utc"] = pd.to_datetime(s["window_start_utc"], utc=True)
        s = s.dropna(subset=["settled_up"]).drop_duplicates("window_start_utc")
        m = h.merge(s[["window_start_utc", "settled_up"]], left_on="time", right_on="window_start_utc", how="left")
        info.update(mode="settlement", tag="settlement-calibrated", file=str(settle.relative_to(ROOT)),
                    skipped=int(m["settled_up"].isna().sum()))
        m = m[m["settled_up"].notna()].copy(); m["y"] = m["settled_up"].astype(int)
        h = m.drop(columns=["window_start_utc", "settled_up"])
    else:
        c = pd.read_parquet(predictor.cache_path(interval))[["open_time", "open", "close"]]
        m = h.merge(c, left_on="time", right_on="open_time", how="left")
        info["skipped"] = int(m["close"].isna().sum())
        m = m[m["close"].notna()].copy()
        m["y"] = (m["close"] >= m["open"]).astype(int)          # spec: Binance close >= open
        info["ties_counted_up"] = int((m["close"] == m["open"]).sum())
        h = m.drop(columns=["open_time", "open", "close"])
    if interval == "1h":
        info["note"] = "Binance 1H candle is the Polymarket referee for this interval; settlement file optional."
    elif interval == "4h":
        info["note"] = "Referee NOT YET VERIFIED for 4h; Binance fallback used. Treat as provisional."
    elif info["mode"] == "binance":
        info["note"] = f"No settlement file at data/settlement_labels_{interval}.csv; Binance fallback. The Chainlink TWAP-60 referee differs (drag ~{'1.3' if interval == '5m' else '0.3'} pt)."
    h["p"] = h["ml_p"].astype(float)
    h["agree"] = h["ta_call"].notna() & (h["ml_call"] == h["ta_call"])
    thr = gate_threshold(interval)
    h["gated"] = h["agree"] & (h["ta_score"].abs() >= thr) & ((h["p"] >= P_HI) | (h["p"] <= P_LO))
    h["fold"] = fold_labels(interval, h["time"])
    h = h.sort_values("time").reset_index(drop=True)
    info.update(gate_score_threshold=thr, ta_weight_total=ta_weight_total(interval), n_used=len(h), n_gated=int(h["gated"].sum()),
                period=(h["time"].iloc[0].strftime("%Y-%m-%d"), h["time"].iloc[-1].strftime("%Y-%m-%d")))
    return h, info


# ----------------------------------------------------------------------------- calibrators
class Isotonic:
    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.m = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    def fit(self, p, y): self.m.fit(p, y); return self
    def predict(self, p): return np.clip(self.m.predict(np.asarray(p, dtype=float)), 0.0, 1.0)
    def table(self):
        return dict(kind="isotonic", x=[float(v) for v in self.m.X_thresholds_], y=[float(v) for v in self.m.y_thresholds_])


class BinnedIsotonic(Isotonic):
    """Isotonic regression fitted to equal-count bin means (n-weighted). With noisy 0/1 labels the plain
    isotonic step function memorises the fitting half; pooling >= min_bin rows per step keeps it honest."""
    def __init__(self, min_bin: int = 1000):
        super().__init__(); self.min_bin = min_bin
    def fit(self, p, y):
        p = np.asarray(p, dtype=float); y = np.asarray(y, dtype=float)
        mb = self.min_bin
        while len(p) // mb < 4 and mb > MIN_BIN:      # small samples: shrink the bin until >= 4 steps exist
            mb //= 2
        t = reliability(p, y, bins=max(1, len(p) // mb))
        self.m.fit(t["pred"].values, t["obs"].values, sample_weight=t["n"].values); self.bin_rows = int(mb); return self
    def table(self):
        d = super().table(); d["kind"] = "isotonic"; d["binned_rows_per_step"] = self.bin_rows; return d


class Platt:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p)).reshape(-1, 1)
    def fit(self, p, y): self.m.fit(self._logit(p), y); return self
    def predict(self, p): return self.m.predict_proba(self._logit(p))[:, 1]
    def table(self): return dict(kind="platt", a=float(self.m.coef_[0][0]), b=float(self.m.intercept_[0]))


def apply_table(tab: dict, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if tab["kind"] == "isotonic":
        return np.clip(np.interp(p, tab["x"], tab["y"]), 0.0, 1.0)
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(tab["a"] * z + tab["b"])))


# ----------------------------------------------------------------------------- analysis
def direction_cut(rows: pd.DataFrame) -> dict:
    """Bullish (p > .5) vs bearish (p < .5, folded to 1-p / 1-y) reliability, plus an asymmetry test."""
    bull = rows[rows["p"] > 0.5]; bear = rows[rows["p"] < 0.5]
    pb, yb = bull["p"].values, bull["y"].values
    pr, yr = 1 - bear["p"].values, 1 - bear["y"].values
    tb, tr = reliability(pb, yb), reliability(pr, yr)
    gb = float(yb.mean() - pb.mean()) if len(pb) else float("nan"); gr = float(yr.mean() - pr.mean()) if len(pr) else float("nan")
    # z-test on the difference of calibration gaps (obs - pred), treating pred as fixed
    se = math.sqrt(yb.var() / max(len(yb), 1) + yr.var() / max(len(yr), 1)) if len(yb) and len(yr) else float("nan")
    z = (gb - gr) / se if se and se > 0 else float("nan")
    return dict(bull=tb, bear_folded=tr, gap_bull=gb, gap_bear=gr, z=z, n_bull=int(len(pb)), n_bear=int(len(pr)),
                ece_bull=ece(tb), ece_bear=ece(tr), asymmetric=bool(abs(z) >= 2.0) if not math.isnan(z) else False)


def within_ci(table_after: pd.DataFrame) -> tuple[bool, int, int]:
    ok = ((table_after["pred"] >= table_after["ci_lo"]) & (table_after["pred"] <= table_after["ci_hi"]))
    return bool(ok.all()) if len(ok) else True, int(ok.sum()), int(len(ok))


def fold_stability(rows: pd.DataFrame) -> dict:
    """Fit (binned) isotonic per walk-forward fold; compare curves on a p grid where both folds have >= 200 rows of support."""
    grid = np.round(np.concatenate([np.arange(0.20, 0.451, 0.01), np.arange(0.55, 0.801, 0.01)]), 3)
    curves, per_fold = {}, []
    for k, g in rows.groupby("fold"):
        if len(g) < 2 * MIN_BIN:
            continue
        iso = BinnedIsotonic().fit(g["p"].values, g["y"].values)
        lo, hi = g["p"].min(), g["p"].max()
        # only trust the curve where this fold has real support: >= 50 rows within +-0.02 of the grid point
        gp = g["p"].values
        support = np.array([((gp >= v - 0.02) & (gp <= v + 0.02)).sum() >= 200 for v in grid])
        cv = np.where((grid >= lo) & (grid <= hi) & support, iso.predict(grid), np.nan)
        curves[int(k)] = cv
        per_fold.append(dict(fold=int(k), n=int(len(g)), start=g["time"].iloc[0].strftime("%Y-%m-%d"), end=g["time"].iloc[-1].strftime("%Y-%m-%d"),
                             mean_p=float(g["p"].mean()), obs=float(g["y"].mean()), brier_raw=brier(g["p"].values, g["y"].values),
                             ece_raw=ece(reliability(g["p"].values, g["y"].values)), p_min=float(lo), p_max=float(hi)))
    keys = sorted(curves); max_diff, where = 0.0, None
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = np.abs(curves[keys[i]] - curves[keys[j]]); m = np.nanmax(d) if np.isfinite(d).any() else 0.0
            if m > max_diff:
                max_diff, where = float(m), (keys[i], keys[j], float(grid[int(np.nanargmax(d))]))
    stable = max_diff < 0.03
    return dict(grid=[float(v) for v in grid], curves={k: [None if np.isnan(v) else round(float(v), 4) for v in c] for k, c in curves.items()},
                per_fold=per_fold, max_diff=max_diff, where=where, stable=stable,
                verdict=("stable: fold curves within 3 pt of each other wherever two folds both have support on the gated p range" if stable else
                         f"NOT stable: fold curves differ by up to {max_diff*100:.1f} pt where both folds have support (folds {where[0]} vs {where[1]} at p={where[2]:.2f})"))


def build(interval: str, write: bool = True) -> dict:
    """Full calibration run for one interval: diagrams, half-split fit/evaluation, fold stability, deployed
    calibrator on all OOS rows. Writes calibrators/p_bullish_<tf>_<version>.json and models/<tf>/calibration.json."""
    rows, info = load_rows(interval)
    p, y = rows["p"].values, rows["y"].values
    g = rows[rows["gated"]]
    # ---- Part 1: reliability
    rel_all = reliability(p, y); rel_gated = reliability(g["p"].values, g["y"].values)
    widened = False
    if interval == "4h" and len(g) and (rel_gated["n"] < MIN_BIN).any():
        widened = True; rel_gated = reliability(g["p"].values, g["y"].values, bins=max(1, len(g) // (2 * MIN_BIN)))
    direction = direction_cut(rows)
    # ---- Part 2: half split by time
    half = len(rows) // 2
    fit, hold = rows.iloc[:half], rows.iloc[half:]
    iso = BinnedIsotonic().fit(fit["p"].values, fit["y"].values); plain = Isotonic().fit(fit["p"].values, fit["y"].values)
    pl = Platt().fit(fit["p"].values, fit["y"].values)
    ph, yh = hold["p"].values, hold["y"].values
    p_iso, p_plain, p_pl = iso.predict(ph), plain.predict(ph), pl.predict(ph)
    hold_all_raw, hold_all_iso, hold_all_pl = reliability(ph, yh), reliability(p_iso, yh), reliability(p_pl, yh)
    hold_all_plain = reliability(p_plain, yh)
    hg = hold[hold["gated"]]; pg = hg["p"].values; yg = hg["y"].values
    hold_g_raw, hold_g_iso = reliability(pg, yg), reliability(iso.predict(pg), yg)
    if interval == "4h" and len(hg) and (hold_g_iso["n"] < MIN_BIN).any():
        widened = True; b = max(1, len(hg) // (2 * MIN_BIN)); hold_g_raw, hold_g_iso = reliability(pg, yg, b), reliability(iso.predict(pg), yg, b)
    ok_all, ok_n, ok_tot = within_ci(hold_g_iso)
    # top tier = most confident held-out calls (folded); "said X% -> got Y%"
    fold_p = np.where(ph >= 0.5, ph, 1 - ph); fold_y = np.where(ph >= 0.5, yh, 1 - yh)
    top = reliability(fold_p, fold_y).iloc[-1]
    top_cal = reliability(np.where(ph >= 0.5, p_iso, 1 - p_iso), fold_y).iloc[-1]
    evaluation = dict(fit_start=fit["time"].iloc[0].strftime("%Y-%m-%d"), fit_end=fit["time"].iloc[-1].strftime("%Y-%m-%d"),
                      hold_start=hold["time"].iloc[0].strftime("%Y-%m-%d"), hold_end=hold["time"].iloc[-1].strftime("%Y-%m-%d"),
                      n_fit=int(half), n_hold=int(len(hold)), n_hold_gated=int(len(hg)),
                      brier=dict(raw=brier(ph, yh), isotonic=brier(p_iso, yh), isotonic_plain=brier(p_plain, yh), platt=brier(p_pl, yh)),
                      ece=dict(raw=ece(hold_all_raw), isotonic=ece(hold_all_iso), isotonic_plain=ece(hold_all_plain), platt=ece(hold_all_pl)),
                      ece_gated=dict(raw=ece(hold_g_raw), isotonic=ece(hold_g_iso), isotonic_plain=ece(reliability(plain.predict(pg), yg)), platt=ece(reliability(pl.predict(pg), yg))),
                      isotonic_bin_rows=iso.bin_rows,
                      gated_within_ci=dict(all_bins=ok_all, bins_ok=ok_n, bins_total=ok_tot),
                      top_tier=dict(said=float(top["pred"]), got=float(top["obs"]), n=int(top["n"]), ci=(float(top["ci_lo"]), float(top["ci_hi"])),
                                    said_after=float(top_cal["pred"]), got_after=float(top_cal["obs"])),
                      platt=pl.table())
    stability = fold_stability(rows)
    # ---- deployed calibrator: isotonic on ALL OOS rows
    final = BinnedIsotonic().fit(p, y)
    mh = model_hash(interval)
    version = f"{mh}-{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}"
    grid = np.round(np.arange(0.0, 1.0001, 0.005), 3)
    lookup = [dict(p=float(v), p_cal=round(float(c), 4)) for v, c in zip(grid, final.predict(grid))]
    cal = dict(interval=interval, version=version, model_hash=mh, method=f"isotonic (binned, >= {final.bin_rows} rows per step)", labels=info["mode"], label_tag=info["tag"],
               label_note=info.get("note"), n=int(len(rows)), n_gated=int(len(g)), oos_period=info["period"],
               fitted_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), table=final.table(), lookup=lookup,
               heldout_reliability=dict(raw=hold_all_raw.round(4).to_dict("records"), isotonic=hold_all_iso.round(4).to_dict("records"),
                                        gated_raw=hold_g_raw.round(4).to_dict("records"), gated_isotonic=hold_g_iso.round(4).to_dict("records")),
               evaluation={k: v for k, v in evaluation.items() if k != "platt"}, platt=evaluation["platt"],
               fold_stable=stability["stable"], fold_max_diff=stability["max_diff"], gate_score_threshold=info["gate_score_threshold"],
               refit_policy="refit automatically on every retrain/backtest of this interval")
    if write:
        CAL_DIR.mkdir(exist_ok=True)
        (CAL_DIR / f"p_bullish_{interval}_{version}.json").write_text(json.dumps(cal, indent=1))
        (predictor.model_dir(interval).parent / "calibration.json").write_text(json.dumps(cal, indent=1))
    return dict(rows=rows, info=info, rel_all=rel_all, rel_gated=rel_gated, widened=widened, direction=direction, evaluation=evaluation,
                hold=dict(all_raw=hold_all_raw, all_iso=hold_all_iso, all_platt=hold_all_pl, gated_raw=hold_g_raw, gated_iso=hold_g_iso),
                stability=stability, calibrator=cal)


# ----------------------------------------------------------------------------- runtime
_CUR: dict = {}


def current(interval: str) -> dict | None:
    path = predictor.model_dir(interval).parent / "calibration.json"
    if not path.exists():
        return None
    mt = path.stat().st_mtime; hit = _CUR.get(interval)
    if hit is None or hit[0] != mt:
        _CUR[interval] = (mt, json.loads(path.read_text()))
    return _CUR[interval][1]


def calibrate_p(interval: str, p: float) -> dict:
    """Fields to attach to /api/predict's ml block; p_bullish itself is left untouched by the caller."""
    cal = current(interval)
    if cal is None:
        return dict(p_bullish_cal=None, calibration_version=None, calibration_labels=None, calibration_n=None)
    stale = cal["model_hash"] != model_hash(interval)
    return dict(p_bullish_cal=round(float(apply_table(cal["table"], [p])[0]), 4), calibration_version=cal["version"],
                calibration_labels=cal["labels"], calibration_n=cal["n"], calibration_stale=stale)
