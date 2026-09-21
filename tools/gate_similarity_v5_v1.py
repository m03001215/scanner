"""Per-candle paired walk-forward of v5 (T-10 s) and v1 (0 s) for gate-overlap analysis. Analysis only: saves
predictions to data/compare_v5_v1_preds.parquet and does not touch the deployed models."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btcpred import data, features, model, ta, intra10, early10
m = early10._m; ROOT = Path(__file__).resolve().parent.parent
df = data.drop_open_candle(data.load_or_update(ROOT / "data" / "btcusdt_15m.parquet", interval="15m", days=1095 + 12))
bars = intra10.load_bars(); df = df[df["open_time"] >= bars.index[0].floor("15min")].reset_index(drop=True)
_, meta1 = model.load(ROOT / "models" / "15m" / "lgbm"); feats = meta1["features"]
X5, S5, st = m.build_dataset(df, bars, feats)
y = features.build_labels(df); X1 = features.build_features(df)[feats]; S1 = ta.signals(df)
ok = X5.notna().all(axis=1) & y.loc[X5.index].notna() & X1.loc[X5.index].notna().all(axis=1) & S5.notna().all(axis=1)
U = X5.index[ok]; t = df.loc[U, "open_time"].reset_index(drop=True); yy = y.loc[U].astype(int).reset_index(drop=True)
r5 = model.walk_forward(X5.loc[U].reset_index(drop=True), yy, t, n_folds=5, return_predictions=True)
r1 = model.walk_forward(X1.loc[U].reset_index(drop=True), yy, t, n_folds=5, return_predictions=True)
first = int(len(U) * 0.4); Uo = U[first:]
w5 = ta.calibrate(S5.loc[U[:first]], y.loc[U[:first]]); w1 = ta.calibrate(S1.loc[U[:first]], y.loc[U[:first]])
a5 = ta.score(S5.loc[Uo], w5); a1 = ta.score(S1.loc[Uo], w1)
folds = np.searchsorted(np.linspace(first, len(U), 6, dtype=int)[1:], np.arange(first, len(U)), side="right") + 1
out = pd.DataFrame({"target_open": (df.loc[Uo, "open_time"] + pd.Timedelta(minutes=15)).values, "fold": folds, "y": yy.values[first:],
                    "p1": r1["predictions"]["ml_p"].values, "p5": r5["predictions"]["ml_p"].values,
                    "ta1_dir": a1["direction"].values, "ta1_conf": a1["confidence"].values, "ta5_dir": a5["direction"].values, "ta5_conf": a5["confidence"].values,
                    "ret_bp": ((df["close"].shift(-1) / df["open"].shift(-1) - 1) * 1e4).loc[Uo].values})
out.to_parquet(ROOT / "data" / "compare_v5_v1_preds.parquet", index=False); print("saved", len(out))
