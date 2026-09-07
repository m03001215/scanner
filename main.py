#!/usr/bin/env python
"""CLI: fetch data, train/evaluate the model, and predict the next BTCUSDT 15m candle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from btcpred import backtest, data, features, model, predictor, ta
from btcpred.predictor import INTERVALS, backtest_summary_path, cache_path, history_path, model_dir, ta_report_path


def cmd_fetch(args):
    df = data.load_or_update(cache_path(args.interval), interval=args.interval, days=args.days)
    print(f"{len(df)} candles cached: {df['open_time'].iloc[0]} -> {df['open_time'].iloc[-1]}")


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
    n_hist = backtest.write_history(bt, step, history_path(args.interval))
    print(f"\nfull per-candle table saved to {out}; summary -> {backtest_summary_path(args.interval)}; "
          f"history ({n_hist} rows) -> {history_path(args.interval)}")


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def add(name, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--interval", "-i", choices=INTERVALS, default="15m", help="candle timeframe (default 15m)")
        return sp
    s = add("fetch", "download / update cached klines"); s.add_argument("--days", type=int, default=730)
    s = add("train", "walk-forward evaluate and train final ML model")
    s.add_argument("--days", type=int, default=730); s.add_argument("--folds", type=int, default=5)
    s = add("backtest-ta", "calibrate and evaluate the rule-based TA predictor"); s.add_argument("--days", type=int, default=730)
    s = add("backtest", "replay ML + TA candle by candle over the out-of-sample window")
    s.add_argument("--days", type=int, default=730); s.add_argument("--folds", type=int, default=5)
    s.add_argument("--rows", type=int, default=20, help="how many recent candles to print")
    s = add("predict", "predict direction of the next candle (ML + TA)"); s.add_argument("--json", action="store_true")
    args = ap.parse_args()
    {"fetch": cmd_fetch, "train": cmd_train, "backtest-ta": cmd_backtest_ta, "backtest": cmd_backtest, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
