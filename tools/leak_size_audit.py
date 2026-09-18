"""
How large was the in-sample residual leak? (reviewer point R1-1, round 2)

The revision replaces in-sample first-stage residuals with cross-fitted ones. This
script quantifies the difference the reviewer was worried about, on the fitting block
itself, without training any second stage:

  * RMS of the IN-SAMPLE residual   y - S1(X)          with S1 fitted on the same rows;
  * RMS of the CROSS-FITTED residual y - S1_cf(X)      with S1 fitted on the other four
    of five contiguous chronological folds;
  * RMS of the residual on the VALIDATION block, i.e. what an honest out-of-block
    residual looks like for the same first stage.

A first stage that memorises its own rows has an in-sample residual that is too small;
the gap between the first two numbers is the size of the leak, and the third says how
much of that gap is optimism rather than genuine fit. The numbers reported in the
response letter for the retail V3 training block (0.155 / 0.163 / 0.202 on the log
scale, structural first stage) come from this script.

Usage (repository root):
    python tools/leak_size_audit.py                 # V3, all three first stages
    python tools/leak_size_audit.py --variants V1_customers_dropped V3_sales_lagged
    python tools/leak_size_audit.py --stages structural

Output: results/crossfit_leak_size.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from src.experiments import arlrx as A          # noqa: E402
from src.experiments import protocol as P       # noqa: E402

OUT = os.path.join(ROOT, "results", "crossfit_leak_size.csv")


def rms(x) -> float:
    x = np.asarray(x, dtype=float).ravel()
    return float(np.sqrt(np.mean(x ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["V3_sales_lagged"])
    ap.add_argument("--stages", nargs="+", default=list(A.STAGE1_KINDS))
    ap.add_argument("--folds", type=int, default=A.N_CROSS_FIT_FOLDS)
    args = ap.parse_args()

    P.set_global_seed()
    print("Environment:", P.environment_stamp(), flush=True)
    frame = P.build_rossmann_frame(os.path.join(ROOT, "data/raw/rossmann/train.csv"),
                                   os.path.join(ROOT, "data/raw/rossmann/store.csv"))
    rows = []
    for variant in args.variants:
        d = P.build_rossmann_dataset(frame, variant, target="log1p")
        X_tr, y_tr = d.X_train, d.y_train
        X_va, y_va = d.X_val, d.y_val
        for kind in args.stages:
            stage1 = A.make_stage1(kind, d.feature_names)
            # in-sample: fit on the whole training block and predict the same rows,
            # plus the honest prediction of the untouched validation block
            pred_tr, pred_va = stage1(X_tr, y_tr, X_va)
            in_sample = y_tr - pred_tr
            val = y_va - pred_va
            # cross-fitted: out-of-fold predictions on the same training block
            cf = A._cross_fitted_stage1(stage1, X_tr, y_tr, args.folds)
            rows.append({"variant": variant, "stage1": kind, "n_folds": args.folds,
                         "n_train": len(y_tr), "n_val": len(y_va),
                         "rms_resid_in_sample": rms(in_sample),
                         "rms_resid_cross_fitted": rms(y_tr - cf),
                         "rms_resid_validation": rms(val),
                         "leak_pct": (rms(y_tr - cf) - rms(in_sample)) / rms(in_sample) * 100})
            r = rows[-1]
            print(f"{variant} {kind:14s} in-sample {r['rms_resid_in_sample']:.4f}  "
                  f"cross-fitted {r['rms_resid_cross_fitted']:.4f}  "
                  f"validation {r['rms_resid_validation']:.4f}  "
                  f"({r['leak_pct']:+.1f}%)", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("written:", OUT)


if __name__ == "__main__":
    main()
