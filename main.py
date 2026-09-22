#!/usr/bin/env python
"""CLI: fetch data, train/evaluate the model, and predict the next BTCUSDT 15m candle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from btcpred import backtest, calibration, calibration_report, data, early10, early30, features, gate, intra, intra10, llm, model, predictor, rl, ta
from btcpred.predictor import INTERVALS, backtest_summary_path, cache_path, history_path, model_dir, ta_report_path


def cmd_fetch(args):
    df = data.load_or_update(cache_path(args.interval), interval=args.interval, days=args.days, rebuild=args.rebuild)
    print(f"{len(df)} candles cached: {df['open_time'].iloc[0]} -> {df['open_time'].iloc[-1]}")
    if args.verify:
        import time as _t
        tail = df.tail(args.verify)
        live = data.fetch_klines("BTCUSDT", args.interval, start_ms=int(tail["open_time"].iloc[0].timestamp() * 1000),
                                 end_ms=int(_t.time() * 1000))
        m = tail.merge(live, on="open_time", suffixes=("_c", "_l"))
        bad = m[(m["close_c"] != m["close_l"]) | (m["high_c"] != m["high_l"]) | (m["low_c"] != m["low_l"]) | (m["volume_c"] != m["volume_l"])]
        print(f"verified last {len(m)} candles against Binance: {len(bad)} mismatch(es)")
        for r in bad.itertuples():
            print(f"  {r.open_time}: close {r.close_c} vs {r.close_l}, vol {r.volume_c} vs {r.volume_l}")


def cmd_train(args):
    df = data.drop_open_candle(data.load_or_update(cache_path(args.interval), interval=args.interval, days=args.days))
    X, y, t = features.build_dataset(df)
    print(f"[{args.interval}] dataset: {len(X)} rows, {X.shape[1]} features, {t.iloc[0]} -> {t.iloc[-1]}")
    print(f"bullish rate: {y.mean():.4f}")

    report = model.walk_forward(X, y, t, n_folds=args.folds)
    print("\nWalk-forward folds:")
    for f in report["folds"]:
        print(f"  {f['test_start'][:10]} .. {f['test_end'][:10]}  n={f['n_test']:5d}  "
              f"acc={f['accuracy']:.4f}  auc={f['auc']:.4f}  logloss={f['logloss']:.4f}  iters={f['best_iter']}")
    o = report["overall"]
    print(f"\nOverall out-of-sample:  acc={o['accuracy']:.4f}  auc={o['auc']:.4f}  "
          f"logloss={o['logloss']:.4f}  (majority baseline acc={report['baseline_majority']:.4f})")
    print(f"Confident-only (|p-0.5|>=0.05): acc={o['acc_conf_05']}  coverage={o['coverage_conf_05']:.3f}")
    print(f"Confident-only (|p-0.5|>=0.10): acc={o['acc_conf_10']}  coverage={o['coverage_conf_10']:.3f}")

    rounds = int(np.median([f["best_iter"] for f in report["folds"]])) or 200
    booster = model.train_final(X, y, rounds)
    model.save(booster, list(X.columns), report, model_dir(args.interval))
    _refit_calibrator(args.interval)
    imp = pd.Series(booster.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print("\nTop features (gain):")
    print(imp.head(12).round(1).to_string())
    print(f"\nmodel saved to {model_dir(args.interval)}")


def cmd_backtest_ta(args):
    df = data.drop_open_candle(data.load_or_update(cache_path(args.interval), interval=args.interval, days=args.days))
    labels = features.build_labels(df)
    rep = ta.backtest(df, labels)
    print(f"[{args.interval}] calibration window: {rep['calib_start'][:10]} .. {rep['test_start'][:10]}")
    print(f"test window       : {rep['test_start'][:10]} .. {rep['test_end'][:10]}  (n={rep['calibrated_oos']['n']})\n")
    for title, c in (("Textbook equal-weight vote (out-of-sample)", rep["textbook_oos"]),
                     ("Calibrated vote (out-of-sample)", rep["calibrated_oos"])):
        print(title)
        print(f"  coverage={c['coverage']:.3f}  acc={c['accuracy']:.4f}")
        for thr in (20, 30, 40):
            a = c[f"acc_conf_{thr}"]
            a_txt = f"{a:.4f}" if a is not None else "  n/a "
            print(f"  confidence>={thr/100:.1f}: acc={a_txt}  coverage={c[f'coverage_conf_{thr}']:.3f}")
        print()
    print("Per-signal hit rate (calibration | test) and assigned weight:")
    rows = sorted(rep["per_signal"].items(), key=lambda kv: -(kv[1]["acc_test"] or 0))
    for name, r in rows:
        print(f"  {name:22s} {r['acc_calib']:.4f} | {r['acc_test']:.4f}   cov={r['coverage_test']:.3f}   w={r['weight']:+.0f}")
    out = ta_report_path(args.interval)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nweights + report saved to {out}")


def cmd_backtest(args):
    df = data.drop_open_candle(data.load_or_update(cache_path(args.interval), interval=args.interval, days=args.days))
    bt = backtest.run(df, n_folds=args.folds)
    step = pd.Timedelta(milliseconds=data.INTERVAL_MS[args.interval])
    out = ta_report_path(args.interval).parent / "backtest.csv"
    save = bt.copy(); save["predicted_candle"] = save["open_time"] + step
    save.to_csv(out, index=False)
    lab = lambda v: "-" if pd.isna(v) else ("BULL" if int(v) == 1 else "BEAR")
    pct = lambda v: "   -  " if pd.isna(v) else f"{v*100:5.1f}%"

    print(f"[{args.interval}] out-of-sample replay: {len(bt)} candles, "
          f"{(bt['open_time'].iloc[0] + step):%Y-%m-%d} .. {(bt['open_time'].iloc[-1] + step):%Y-%m-%d}")
    print(f"  ML acc {bt['ml_hit'].mean()*100:.2f}%   TA acc {bt['ta_hit'].mean()*100:.2f}% "
          f"(calls on {bt['ta_hit'].notna().mean()*100:.0f}%)   when both agree: "
          f"{bt.loc[bt['agree'], 'ml_hit'].mean()*100:.2f}% on {bt['agree'].mean()*100:.0f}% of candles   "
          f"bull rate {bt['actual'].mean()*100:.1f}%")
    print("\nBy month:")
    print(f"  {'month':8s} {'n':>6s} {'bull%':>6s} {'ML':>7s} {'TA':>7s} {'agree':>7s} {'agree n':>8s}")
    for r in backtest.monthly(bt).itertuples():
        print(f"  {r.month:8s} {r.candles:6d} {r.bull_rate*100:5.1f}% {pct(r.ml_acc):>7s} {pct(r.ta_acc):>7s} {pct(r.agree_acc):>7s} {r.agree_n:8d}")
    print("\nAccuracy when ML and TA agree (ML conf on the dashboard scale = 2*|p-0.5|; TA conf = |score|/16):")
    print(f"  {'subset':34s} {'n':>6s} {'share':>6s} {'acc':>7s}")
    for r in backtest.agreement(bt).itertuples():
        print(f"  {r.subset:34s} {r.candles:6d} {r.share*100:5.1f}% {pct(r.accuracy):>7s}")
    print("\nML accuracy by confidence (dashboard scale):")
    for r in backtest.by_confidence(bt).itertuples():
        print(f"  {r.bucket:20s} n={r.candles:6d} ({r.share*100:4.1f}%)  acc={pct(r.ml_acc)}")
    n = args.rows
    print(f"\nLast {n} candles (predicted candle open, UTC):")
    print(f"  {'candle':16s} {'ML p':>6s} {'ML':>5s} {'TA':>5s} {'real':>5s}  ML  TA")
    for r in bt.tail(n).itertuples():
        mh = "✓" if r.ml_hit else "✗"; th = "-" if pd.isna(r.ta_hit) else ("✓" if r.ta_hit else "✗")
        print(f"  {(r.open_time + step):%Y-%m-%d %H:%M} {r.ml_p:6.3f} {lab(r.ml_dir):>5s} {lab(r.ta_dir):>5s} {lab(r.actual):>5s}   {mh}   {th}")
    backtest.write_summary(bt, args.interval, step, backtest_summary_path(args.interval))
    _refit_calibrator(args.interval, after_history=True)
    n_hist = backtest.write_history(bt, step, history_path(args.interval))
    print(f"\nfull per-candle table saved to {out}; summary -> {backtest_summary_path(args.interval)}; "
          f"history ({n_hist} rows) -> {history_path(args.interval)}")


def cmd_backtest_llm(args):
    """Replay the LLM analyst over the last N closed candles (one paid API call each)."""
    if not llm.available():
        sys.exit("not configured: set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    df = data.drop_open_candle(data.load_or_update(cache_path(args.interval), interval=args.interval, days=730))
    print(f"[{args.interval}] replaying {llm.provider()} {llm.model_name()} over the last {args.n} closed candles "
          f"({args.workers} parallel calls)...")
    rep = llm.replay(df, args.interval, args.n, workers=args.workers)
    out = ta_report_path(args.interval).parent / "llm_replay.csv"
    rep.to_csv(out, index=False)
    ok = rep[rep["hit"].notna()]
    print(f"done: {len(ok)} calls ok, {int(rep['error'].notna().sum())} failed; accuracy {ok['hit'].mean()*100:.1f}% ({int(ok['hit'].sum())}/{len(ok)})")
    for r in rep.itertuples():
        if r.error:
            print(f"  {r.candle:%m-%d %H:%M} UTC  ERROR {r.error[:80]}"); continue
        print(f"  {r.candle:%m-%d %H:%M} UTC  {r.prediction:7s} conf {r.confidence:.2f}  real {r.actual:7s}  {'hit ' if r.hit else 'MISS'}  {r.latency_s:5.1f}s")
    print(f"saved to {out}")


def cmd_train_rl(args):
    """Offline fitted Q-iteration: learn long/flat/short with fees in the reward; strict time-split evaluation."""
    df = data.drop_open_candle(data.load_or_update(cache_path(args.interval), interval=args.interval, days=args.days))
    _, meta = model.load(model_dir(args.interval))
    hist_path = predictor.history_path(args.interval)
    hist = predictor.load_history(args.interval) if hist_path.exists() else None
    res = rl.train_and_evaluate(args.interval, df, meta["features"], hist, cost_bp=args.cost_bp, gamma=args.gamma,
                                iterations=args.iterations, rounds=args.rounds, train_frac=args.train_frac,
                                seeds=tuple(args.seeds))
    rl.save(res["model"], res["report"], rl.rl_dir(args.interval))
    print(f"saved RL policy + report to {rl.rl_dir(args.interval)}")


def _refit_calibrator(interval, after_history=False):
    """Probability calibrator is tied to the model version: refit whenever the model or its OOS history changes."""
    if not predictor.history_path(interval).exists():
        print("calibration skipped: no OOS history yet (run `backtest`)"); return
    try:
        r = calibration.build(interval)
        print(f"calibrator refit: version {r['calibrator']['version']}, {r['calibrator']['n']} OOS rows, labels {r['calibrator']['label_tag']}"
              + ("" if after_history else " (note: history predates this retrain until `backtest` runs)"))
    except Exception as e:  # noqa: BLE001
        print(f"calibration refit failed: {type(e).__name__}: {e}")


def cmd_calibrate(args):
    """Fit and report the p_bullish probability calibrator for one interval, or all with --all."""
    results = {}
    for iv in (INTERVALS if args.all else [args.interval]):
        r = calibration.build(iv)
        out = calibration_report.write_interval(iv, r); results[iv] = r
        e = r["evaluation"]
        print(f"[{iv}] {r['info']['tag']}: n {r['info']['n_used']} gated {r['info']['n_gated']} | held-out Brier {e['brier']['raw']:.5f}->{e['brier']['isotonic']:.5f} "
              f"ECE {e['ece']['raw']*100:.2f}%->{e['ece']['isotonic']*100:.2f}% | top tier said {e['top_tier']['said']*100:.1f}% got {e['top_tier']['got']*100:.1f}% | "
              f"asym {'y' if r['direction']['asymmetric'] else 'n'} | {r['stability']['verdict']} | version {r['calibrator']['version']} | {out}")
    if args.all or args.summary:
        allr = results
        if not args.all:  # summary needs every interval: rebuild the others without rewriting their reports
            allr = {iv: (results[iv] if iv in results else calibration.build(iv, write=False)) for iv in INTERVALS}
        print("summary:", calibration_report.write_summary(allr))


def cmd_train_intra(args):
    """Intra-candle v2: train the minute-by-minute next-candle model and evaluate it walk-forward."""
    if not (predictor.ROOT / "data" / "btcusdt_1m.parquet").exists():
        sys.exit("needs data/btcusdt_1m.parquet (1-minute history)")
    intra.train(args.interval, args.days)
    print(f"saved to {intra.intra_dir(args.interval)}")


def cmd_train_intra10(args):
    """Intra-candle v3 (10-second bars): train + walk-forward evaluate the next-candle model."""
    intra10.train(args.days)
    print(f"saved to {intra10.v3_dir()}")


def cmd_train_v4(args):
    """v4: v1's features and TA on a stand-in candle cut 30 s before the close; paired walk-forward against v1."""
    early30.train(args.days)
    print(f"saved to {early30.v4_dir()}")


def cmd_train_v5(args):
    """v5: as v4 but the stand-in candle is cut 10 s before the close."""
    early10.train(args.days)
    print(f"saved to {early10.model_dir()}")


def cmd_train_v6(args):
    """v6: learned gate over v1 (v1 unchanged). Trains on v1's OOS walk-forward calls, walk-forward again."""
    gate.train()
    print(f"saved to {gate.v6_dir()}")


def cmd_predict(args):
    try:
        out = predictor.predict(interval=args.interval)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    if args.json:
        out = {k: v for k, v in out.items() if k != "recent"}
        print(json.dumps(out, indent=2)); return
    ml, t = out["ml"], out["ta"]
    print(f"Timeframe          : {out['interval']}")
    print(f"Last closed candle : {out['last_closed_candle']}  close={out['last_close']:.2f}")
    print(f"Next candle opens  : {out['predicting_candle_open']}\n")
    print("[1] ML model (LightGBM)")
    print(f"    P(bullish)  : {ml['p_bullish']:.4f}")
    print(f"    Prediction  : {ml['prediction']}  (confidence {ml['confidence']:.1%}, walk-forward OOS acc {ml['oos_accuracy']:.1%})\n")
    print("[2] Technical analysis (indicator vote, weights calibrated on history)")
    acc_txt = f", OOS acc {t['oos_accuracy']:.1%}" if t["oos_accuracy"] else ""
    print(f"    Prediction  : {t['prediction']}  (score {t['score']:+.0f}, confidence {t['confidence']:.1%}{acc_txt})")
    tb = t["textbook_vote"]
    print(f"    Raw votes   : {tb['n_bull']} bullish / {tb['n_bear']} bearish  -> textbook vote {tb['prediction']}")
    print(f"    Bullish     : {', '.join(t['bullish_signals']) or '-'}")
    print(f"    Bearish     : {', '.join(t['bearish_signals']) or '-'}")
    if t["weights"]:
        print(f"    Inverted    : {', '.join(k for k, w in t['weights'].items() if w < 0) or '-'}")
        print(f"    Ignored     : {', '.join(k for k, w in t['weights'].items() if w == 0) or '-'}")
    print()
    if t["prediction"] == "NEUTRAL":
        print("Agreement          : TA has no call (tie); only the ML prediction applies")
    else:
        print("Agreement          : " + ("YES, both methods agree" if out["agreement"] else "NO, methods disagree"))
    if args.llm:
        print("\n[3] LLM")
        if not llm.available():
            print("    not configured: set OPENAI_API_KEY or ANTHROPIC_API_KEY"); return
        df = data.drop_open_candle(pd.read_parquet(cache_path(args.interval)))
        try:
            r = llm.predict(df, args.interval)
        except Exception as e:  # noqa: BLE001
            print(f"    call failed: {e}"); return
        print(f"    Prediction  : {r['prediction']}  (confidence {r['confidence']:.0%}, {r['model']}, effort {r['effort']}, {r['latency_s']}s)")
        print(f"    Reason      : {r['reason']}")
        for k in r["key_factors"]:
            print(f"      - {k}")
        tr = llm.track_record(args.interval, df)
        if tr["resolved"]:
            print(f"    Track record: {tr['hit_rate']:.1%} on {tr['resolved']} resolved calls")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def add(name, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--interval", "-i", choices=INTERVALS, default="15m", help="candle timeframe (default 15m)")
        return sp
    s = add("fetch", "download / update cached klines"); s.add_argument("--days", type=int, default=730)
    s.add_argument("--rebuild", action="store_true", help="discard the cache and download everything again")
    s.add_argument("--verify", type=int, metavar="N", default=0, help="compare the last N cached candles with Binance")
    s = add("train", "walk-forward evaluate and train final ML model")
    s.add_argument("--days", type=int, default=730); s.add_argument("--folds", type=int, default=5)
    s = add("backtest-ta", "calibrate and evaluate the rule-based TA predictor"); s.add_argument("--days", type=int, default=730)
    s = add("backtest", "replay ML + TA candle by candle over the out-of-sample window")
    s.add_argument("--days", type=int, default=730); s.add_argument("--folds", type=int, default=5)
    s.add_argument("--rows", type=int, default=20, help="how many recent candles to print")
    s = add("backtest-llm", "replay the LLM analyst over the last N closed candles (paid API calls)")
    s.add_argument("--n", type=int, default=24); s.add_argument("--workers", type=int, default=4)
    s = add("train-rl", "train the reinforcement-learning policy (fitted Q-iteration) and evaluate it on a time split")
    s.add_argument("--days", type=int, default=730); s.add_argument("--cost-bp", type=float, default=5.0, help="fee per side in bp (default 5 = 10 bp round trip)")
    s.add_argument("--gamma", type=float, default=0.9); s.add_argument("--iterations", type=int, default=8); s.add_argument("--rounds", type=int, default=200)
    s.add_argument("--train-frac", type=float, default=0.6); s.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    s = add("calibrate", "fit + report the p_bullish probability calibrator (isotonic, walk-forward OOS rows only)")
    s.add_argument("--all", action="store_true", help="all intervals plus the cross-interval summary"); s.add_argument("--summary", action="store_true", help="also write the cross-interval summary")
    s = add("train-intra", "intra-candle v2: train + walk-forward evaluate the minute-by-minute next-candle model")
    s.add_argument("--days", type=int, default=1095)
    s = add("train-intra10", "intra-candle v3: 10-second bars, recomputed every 10 s (15m only)"); s.add_argument("--days", type=int, default=1095)
    s = add("train-v4", "v4: v1 called 30 s before the candle opens, using a stand-in for the forming candle (15m only)"); s.add_argument("--days", type=int, default=1095)
    s = add("train-v5", "v5: as v4 but called 10 s before the candle opens (15m only)"); s.add_argument("--days", type=int, default=1095)
    s = add("train-v6", "v6: learned gate with regime features over v1's calls (15m)")
    s = add("predict", "predict direction of the next candle (ML + TA, optionally Claude)")
    s.add_argument("--json", action="store_true"); s.add_argument("--llm", action="store_true", help="also ask the LLM (needs OPENAI_API_KEY or ANTHROPIC_API_KEY)")
    args = ap.parse_args()
    {"fetch": cmd_fetch, "train": cmd_train, "backtest-ta": cmd_backtest_ta, "backtest": cmd_backtest,
     "backtest-llm": cmd_backtest_llm, "train-rl": cmd_train_rl, "calibrate": cmd_calibrate, "train-intra": cmd_train_intra, "train-intra10": cmd_train_intra10, "train-v4": cmd_train_v4, "train-v5": cmd_train_v5, "train-v6": cmd_train_v6, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
