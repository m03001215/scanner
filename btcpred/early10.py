"""v5: identical to v4 (btcpred/early30.py) but the stand-in candle is cut at 890 s, so the call is made 10 s before
the target candle opens. This loads a second, independent instance of the early30 module with its own settings, model
directory (models/15m/v5), TA weights, log and caches; v4 is not affected."""
import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("btcpred._early10_impl", Path(__file__).with_name("early30.py"))
_m = importlib.util.module_from_spec(_spec)
sys.modules["btcpred._early10_impl"] = _m          # registered so multiprocessing workers can resolve its functions
_spec.loader.exec_module(_m)
_m.NAME = "v5"; _m.CUT_SEC = 890; _m.N_CUT = _m.CUT_SEC // _m.BAR; _m.SCALE = _m.N_SEC / _m.CUT_SEC

NAME, CUT_SEC, ML_GATE, TA_GATE = _m.NAME, _m.CUT_SEC, _m.ML_GATE, _m.TA_GATE
train, load, timing, predict_now, history, load_log, model_dir, log_path = _m.train, _m.load, _m.timing, _m.predict_now, _m.history, _m.load_log, _m.v4_dir, _m.log_path
