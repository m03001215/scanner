"""Reinforcement-learning predictor: offline fitted Q-iteration (FQI) over the candle history.

State  : the 50 engineered features at candle t plus the position held during candle t (-1, 0, +1).
Action : target position for candle t+1 -> SHORT (-1), FLAT (0), LONG (+1).
Reward : a * (close_{t+1} / open_{t+1} - 1) - cost * |a - p|, i.e. the fee-adjusted return of holding
         `a` through the next candle, paying `cost` (per side, in return units) for every unit of position change.
Q(s,p,a) is a LightGBM regressor with p and a as extra input columns. Each FQI iteration refits it on
         y = r + gamma * max_a' Q_prev(s', p'=a, a'). Because reward and transition are known for every
         (p, a), the training set is the history expanded over all 3 x 3 (p, a) combinations.
Policy : argmax_a Q(s, p, a). "Confidence" is the Q-gap between the chosen action and staying flat,
         expressed in basis points of expected discounted return.

Evaluation is a strict time split (train on the oldest `train_frac`, test on the rest), repeated over
several seeds so a policy that merely fits noise shows up as seed disagreement. The deployed policy is
trained on all data with the first seed, like the ML model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import data, features

ACTIONS = (-1, 0, 1)
ACTION_NAME = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
DEFAULTS = dict(cost_bp=5.0, gamma=0.9, iterations=8, rounds=60, train_frac=0.6, seeds=(1, 2, 3), val_frac=0.2)
# Heavily regularised: the reward is almost pure noise, so a flexible Q-function memorises it. Each leaf must
# cover >= 3000 expanded rows (~330 distinct candles); shallow trees; few boosting rounds; strong L2.
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=7, min_child_samples=3000,
              feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0, verbose=-1, num_threads=0)


def rl_dir(interval: str) -> Path:
    return Path(__file__).resolve().parent.parent / "models" / interval / "rl"


# ----------------------------------------------------------------------------- data
def build_frames(df: pd.DataFrame, feature_names: list[str] | None = None):
    """Features X_t and next-candle return r_{t+1}, aligned so row t decides the position for candle t+1."""
    X = features.build_features(df)
    if feature_names is not None:
        X = X[feature_names]
    nxt_ret = (df["close"].shift(-1) / df["open"].shift(-1) - 1)
    mask = X.notna().all(axis=1) & nxt_ret.notna()
    return X[mask].reset_index(drop=True), nxt_ret[mask].reset_index(drop=True).values, df.loc[mask, "open_time"].reset_index(drop=True)


def _expand(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every row x every (p, a): returns stacked design matrix plus the p and a columns."""
    n = len(X)
    P, A = np.meshgrid(ACTIONS, ACTIONS, indexing="ij")  # 3x3
    p = np.repeat(P.ravel(), n); a = np.repeat(A.ravel(), n)
    Xr = np.tile(X, (9, 1))
    return np.column_stack([Xr, p, a]), p, a


def _q_all(model: lgb.Booster, X: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Q(s, p, a) for all three actions -> shape (n, 3) in ACTIONS order."""
    n = len(X)
    Xa = np.vstack([np.column_stack([X, p, np.full(n, a)]) for a in ACTIONS])
    return model.predict(Xa).reshape(3, n).T


# ----------------------------------------------------------------------------- training
def fit_fqi(X: np.ndarray, ret: np.ndarray, cost: float, gamma: float, iterations: int, rounds: int, seed: int,
            val_frac: float = 0.0, log=None) -> tuple[lgb.Booster, dict]:
    """Fitted Q-iteration. With val_frac > 0 the last slice of X is held out and the iteration whose greedy
    policy earns the most there is kept (offline model selection); the returned model is then that Q-function.
    Returns (model, info) where info records per-iteration fit/validation PnL."""
    n_all = len(X); n = int(n_all * (1 - val_frac)) if val_frac > 0 else n_all
    Xf, rf = X[:n], ret[:n]
    Xpa, p, a = _expand(Xf)
    r = a * np.tile(rf, 9) - cost * np.abs(a - p)
    idx = np.tile(np.arange(n), 9)
    nxt_ok = idx < n - 1
    Xn = Xf[np.minimum(idx + 1, n - 1)]
    model, best, hist = None, None, []
    for k in range(iterations):
        y = r if model is None else r + gamma * np.where(nxt_ok, _q_all(model, Xn, a).max(axis=1), 0.0)
        params = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        model = lgb.train(params, lgb.Dataset(Xpa, y), num_boost_round=rounds)
        fit_pnl = simulate(act(model, Xf)[0], rf, cost)["total_bp"]
        rec = dict(iteration=k + 1, fit_bp=fit_pnl)
        if val_frac > 0:
            va = act(model, X[n:], pos0=0)[0]
            rec["val_bp"] = simulate(va, ret[n:], cost)["total_bp"]
            if best is None or rec["val_bp"] > best[1]:
                best = (model, rec["val_bp"], k + 1)
        hist.append(rec)
        if log:
            log(f"    iter {k + 1}/{iterations}: fit {fit_pnl:+.0f} bp" + (f", validation {rec['val_bp']:+.0f} bp" if val_frac > 0 else ""))
    if best is not None:
        model = best[0]
        if log: log(f"    kept iteration {best[2]} (best validation PnL {best[1]:+.0f} bp)")
    return model, dict(iterations=hist, kept=best[2] if best else iterations)


# ----------------------------------------------------------------------------- policy & simulation
def act(model: lgb.Booster, X: np.ndarray, pos0: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Run the greedy policy sequentially through X starting from position pos0. Returns actions and Q rows."""
    n = len(X); actions = np.zeros(n, dtype=int); qs = np.zeros((n, 3)); p = pos0
    # Q for all p in one pass, then walk the position path
    Q = {pp: _q_all(model, X, np.full(n, pp)) for pp in ACTIONS}
    for i in range(n):
        q = Q[p][i]; a = ACTIONS[int(np.argmax(q))]
        actions[i] = a; qs[i] = q; p = a
    return actions, qs


def simulate(actions: np.ndarray, ret: np.ndarray, cost: float, pos0: int = 0) -> dict:
    prev = np.concatenate([[pos0], actions[:-1]])
    pnl = actions * ret - cost * np.abs(actions - prev)
    cum = np.cumsum(pnl); dd = (np.maximum.accumulate(cum) - cum).max() if len(cum) else 0.0
    active = actions != 0
    dir_hits = ((np.sign(ret[active]) == actions[active]).mean() if active.any() else float("nan"))
    return dict(total_bp=float(pnl.sum() * 1e4), mean_bp=float(pnl.mean() * 1e4), std_bp=float(pnl.std() * 1e4),
                max_dd_bp=float(dd * 1e4), time_in_market=float(active.mean()),
                position_changes=int((actions != prev).sum()), long_share=float((actions == 1).mean()),
                short_share=float((actions == -1).mean()), dir_accuracy=None if math.isnan(dir_hits) else float(dir_hits),
                n=int(len(actions)))


def sharpe(mean_bp: float, std_bp: float, per_day: float) -> float | None:
    return None if std_bp == 0 else float(mean_bp / std_bp * math.sqrt(per_day * 365))


# ----------------------------------------------------------------------------- baselines
def baselines(interval: str, times: pd.Series, ret: np.ndarray, cost: float, history: pd.DataFrame | None) -> dict:
    out = {"buy_and_hold": simulate(np.ones(len(ret), dtype=int), ret, cost),
           "always_short": simulate(-np.ones(len(ret), dtype=int), ret, cost)}
    if history is None or history.empty:
        return out
    h = history.copy(); h["time"] = pd.to_datetime(h["time"], utc=True).dt.tz_convert(None)
    h = h.set_index("time")
    key = (pd.to_datetime(times, utc=True) + pd.Timedelta(milliseconds=data.INTERVAL_MS[interval])).dt.tz_convert(None)  # predicted candle
    j = h.reindex(key.values)
    have = j["ml_call"].notna().values
    if have.mean() < 0.5:
        return out
    ml_dir = np.where(j["ml_call"].values == "BULL", 1, -1)
    mlc = (j["ml_p"].values - 0.5) * 2
    agree = (j["ta_call"].values == j["ml_call"].values)
    gate = agree & (np.abs(mlc) >= 0.10) & (j["ta_conf"].values >= 0.3)
    sub = have
    r = ret[sub]
    out["ml_every_candle"] = simulate(ml_dir[sub], r, cost)
    a = np.where(gate[sub], ml_dir[sub], 0); out["gate_one_candle"] = simulate(a, r, cost)
    held = np.zeros(sub.sum(), dtype=int); cur = 0
    for i, (g, d) in enumerate(zip(gate[sub], ml_dir[sub])):
        if g: cur = d
        held[i] = cur
    out["gate_and_hold"] = simulate(held, r, cost)
    out["baseline_coverage"] = float(have.mean())
    return out


# ----------------------------------------------------------------------------- end to end
def train_and_evaluate(interval: str, df: pd.DataFrame, feature_names: list[str], history: pd.DataFrame | None,
                       cost_bp: float = DEFAULTS["cost_bp"], gamma: float = DEFAULTS["gamma"],
                       iterations: int = DEFAULTS["iterations"], rounds: int = DEFAULTS["rounds"],
                       train_frac: float = DEFAULTS["train_frac"], seeds=DEFAULTS["seeds"], val_frac: float = DEFAULTS["val_frac"], log=print) -> dict:
    X, ret, times = build_frames(df, feature_names)
    Xv = X.values.astype(np.float32)
    n = len(Xv); split = int(n * train_frac); cost = cost_bp / 1e4
    per_day = 86_400_000 / data.INTERVAL_MS[interval]
    days_test = (n - split) / per_day
    log(f"[{interval}] {n} candles, train {split} ({times.iloc[0]:%Y-%m-%d} .. {times.iloc[split-1]:%Y-%m-%d}), "
        f"test {n - split} ({times.iloc[split]:%Y-%m-%d} .. {times.iloc[-1]:%Y-%m-%d}), cost {cost_bp} bp/side, gamma {gamma}")
    seed_results, policies = [], []
    for s in seeds:
        log(f"  seed {s}:")
        m, info = fit_fqi(Xv[:split], ret[:split], cost, gamma, iterations, rounds, s, val_frac=val_frac, log=log)
        a_tr, _ = act(m, Xv[:split]); a_te, _ = act(m, Xv[split:])
        tr, te = simulate(a_tr, ret[:split], cost), simulate(a_te, ret[split:], cost)
        tr["sharpe"] = sharpe(tr["mean_bp"], tr["std_bp"], per_day); te["sharpe"] = sharpe(te["mean_bp"], te["std_bp"], per_day)
        te["per_day_bp"] = te["total_bp"] / days_test
        seed_results.append(dict(seed=s, train=tr, test=te, kept_iteration=info["kept"], iterations=info["iterations"])); policies.append(a_te)
        log(f"    train {tr['total_bp']:+.0f} bp, test {te['total_bp']:+.0f} bp ({te['per_day_bp']:+.2f}/day), in market {te['time_in_market']*100:.0f}%, "
            f"changes {te['position_changes']}, dir acc {(te['dir_accuracy'] or 0)*100:.1f}%, sharpe {te['sharpe'] if te['sharpe'] is None else round(te['sharpe'], 2)}")
    # seed agreement on the test window
    P = np.array(policies)
    agreement = float((P == P[0]).all(axis=0).mean()) if len(P) > 1 else None
    tests = [r["test"]["total_bp"] for r in seed_results]
    base = baselines(interval, times.iloc[split:], ret[split:], cost, history)
    for k, v in base.items():
        if isinstance(v, dict): v["sharpe"] = sharpe(v["mean_bp"], v["std_bp"], per_day); v["per_day_bp"] = v["total_bp"] / days_test
    log(f"  seeds test total: mean {np.mean(tests):+.0f} bp, std {np.std(tests):.0f} bp, agreement {agreement}")
    log("  baselines (test): " + ", ".join(f"{k} {v['total_bp']:+.0f}bp" for k, v in base.items() if isinstance(v, dict)))
    log(f"  training deployed policy on all {n} candles (seed {seeds[0]})...")
    final, finfo = fit_fqi(Xv, ret, cost, gamma, iterations, rounds, seeds[0], val_frac=val_frac)
    report = dict(interval=interval, config=dict(cost_bp=cost_bp, gamma=gamma, iterations=iterations, rounds=rounds, train_frac=train_frac, seeds=list(seeds), val_frac=val_frac, params=PARAMS),
                  deployed_kept_iteration=finfo["kept"],
                  n=n, train_start=str(times.iloc[0]), train_end=str(times.iloc[split - 1]), test_start=str(times.iloc[split]), test_end=str(times.iloc[-1]),
                  test_days=days_test, seeds=seed_results, seed_agreement=agreement,
                  test_mean_bp=float(np.mean(tests)), test_std_bp=float(np.std(tests)), baselines=base, features=feature_names,
                  trained_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    return dict(model=final, report=report)


def save(model: lgb.Booster, report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "q_model.txt"))
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))


_MODELS: dict = {}


def load(interval: str):
    d = rl_dir(interval)
    if not (d / "q_model.txt").exists():
        return None
    mt = (d / "q_model.txt").stat().st_mtime
    hit = _MODELS.get(interval)
    if hit is None or hit[0] != mt:
        _MODELS[interval] = (mt, lgb.Booster(model_file=str(d / "q_model.txt")), json.loads((d / "report.json").read_text()))
    return _MODELS[interval][1], _MODELS[interval][2]


def predict_last(interval: str, df: pd.DataFrame, lookback: int = 96) -> dict | None:
    """Greedy action for the candle forming now, having walked the policy through the last `lookback` closed candles
    from a flat start so the position state is realistic. Returns action, Q-values and recent policy path."""
    loaded = load(interval)
    if loaded is None:
        return None
    model, rep = loaded
    X = features.build_features(df)[rep["features"]]
    X = X.iloc[-lookback:]
    if X.isna().any(axis=None):
        X = X.dropna()
    Xv = X.values.astype(np.float32)
    actions, qs = act(model, Xv, pos0=0)
    a, q = int(actions[-1]), qs[-1]
    prev = int(actions[-2]) if len(actions) > 1 else 0
    gap_bp = float((q[ACTIONS.index(a)] - q[ACTIONS.index(0)]) * 1e4)
    te = rep["seeds"][0]["test"]
    bh = rep["baselines"].get("buy_and_hold", {}).get("total_bp", 0.0)
    excess = [r["test"]["total_bp"] - (r["test"]["long_share"] - r["test"]["short_share"]) * bh for r in rep["seeds"]]
    path = []
    ret = (df["close"] / df["open"] - 1).values
    for i, t in enumerate(X.index[:-1]):
        # action at row t applies to candle t+1 (both closed here)
        nxt = t + 1
        if nxt in df.index:
            path.append(dict(time=int(df.loc[nxt, "open_time"].timestamp()), action=int(actions[i]),
                             ret_bp=round(float(ret[df.index.get_loc(nxt)] * 1e4), 2)))
    return dict(action=ACTION_NAME[a], position=a, previous_position=prev,
                changed=a != prev, q_bp={ACTION_NAME[k]: round(float(v * 1e4), 2) for k, v in zip(ACTIONS, q)},
                edge_vs_flat_bp=round(gap_bp, 2), path=path[-lookback:],
                test=dict(total_bp=round(te["total_bp"], 1), per_day_bp=round(te["per_day_bp"], 2), sharpe=None if te["sharpe"] is None else round(te["sharpe"], 2),
                          time_in_market=round(te["time_in_market"], 3), dir_accuracy=None if te["dir_accuracy"] is None else round(te["dir_accuracy"], 4),
                          test_start=rep["test_start"][:10], test_end=rep["test_end"][:10], seed_agreement=rep["seed_agreement"],
                          seed_mean_bp=round(rep["test_mean_bp"], 1), seed_std_bp=round(rep["test_std_bp"], 1),
                          buy_and_hold_bp=round(bh, 1), excess_over_drift_bp=[round(e, 1) for e in excess],
                          long_share=round(te["long_share"], 3), short_share=round(te["short_share"], 3),
                          max_dd_bp=round(te["max_dd_bp"], 1), position_changes=te["position_changes"], test_days=round(rep["test_days"], 1),
                          seeds=len(rep["seeds"])),
                baselines={k: round(v["total_bp"], 1) for k, v in rep["baselines"].items() if isinstance(v, dict)},
                cost_bp=rep["config"]["cost_bp"], trained_at=rep["trained_at"])
