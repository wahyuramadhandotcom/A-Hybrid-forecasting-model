"""
Before/after audit for reviewer point R1-1 (IJIES Paper ID 20265893, round 2)
on the Rossmann panel.

Re-runs the residual-based models of exp05b/exp05d twice -- in-sample residuals
(`cross_fit_folds=None`, the procedure behind the submitted manuscript) and
cross-fitted residuals (the revised default) -- in the SAME environment, and writes
a side-by-side table. Both sides must come from one environment: XGBoost 3.2.0 with
subsample/colsample < 1 does not give bit-identical results on Windows and Linux
(verified on exp05c: plain XGBoost with identical parameters differs in the 4th
significant digit), so pairing a Linux "after" with an archived Windows "before"
would mix a platform effect into the R1-1 effect. The archived value is still
reported, as a reproduction check: on the archiving machine (Windows) the
legacy column must equal it. Models whose residual procedure is unaffected
(seasonal naive, first stages alone, plain XGBoost) are not re-run.

This is a preview/audit tool, not the source of the manuscript tables: the
canonical rerun is notebooks exp05b -> exp05d -> exp06b (table/DM cells), which
save the prediction files every later analysis reads.

Usage (repository root):
    python tools/crossfit_audit_rossmann.py                       # V3, all residual models
    python tools/crossfit_audit_rossmann.py --variants V1_customers_dropped V2_customers_lagged V3_sales_lagged
    python tools/crossfit_audit_rossmann.py --models "AR-LRX-Aug [structural]"
    python tools/crossfit_audit_rossmann.py --stores 80   # smoke test on 80 stores;
        # writes results/rossmann_crossfit_audit_smoke.csv and never touches the real file

Output: results/rossmann_crossfit_audit.csv (appended row by row, so an
interrupted run keeps what finished; rows already present are skipped).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from src.experiments import arlrx as A          # noqa: E402
from src.experiments import protocol as P       # noqa: E402

RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "rossmann_crossfit_audit.csv")


def model_specs(kind_list=A.STAGE1_KINDS):
    """(label, archived source, callable(d, cff)) - arguments copied from the notebooks."""
    g = A.GRID_XGB_ARLRX
    inv = np.expm1
    specs = []
    for kind in kind_list:                                  # exp05b cell 4
        specs.append((f"AR-LRX [{kind}]", "exp05b_rossmann_arlrx_audit",
                      lambda d, c, k=kind: A.run_arlrx(f"AR-LRX [{k}]", d, k, g,
                                                       inverse_transform=inv,
                                                       cross_fit_folds=c)))
    specs.append(("LR-XGB (residual, tanpa gerbang)", "exp05b_rossmann_arlrx_audit",
                  lambda d, c: P.run_model(
                      "LR-XGB (residual, tanpa gerbang)",
                      lambda Xf, yf, Xe, p: P.fp_lr_xgb_residual(Xf, yf, Xe, p, cross_fit_folds=c),
                      d, g, inverse_transform=inv)))
    for kind in ("structural", "struct_linear"):            # exp05d cell 5
        specs.append((f"AR-LRX-Aug [{kind}]", "exp05d_rossmann_arlrx_dev",
                      lambda d, c, k=kind: A.run_arlrx_segmented(
                          f"AR-LRX-Aug [{k}]", d, k, g, schemes=("global",),
                          augment_stage1=True, inverse_transform=inv, cross_fit_folds=c)))
        specs.append((f"AR-LRX-Seg [{kind}]", "exp05d_rossmann_arlrx_dev",
                      lambda d, c, k=kind: A.run_arlrx_segmented(
                          f"AR-LRX-Seg [{k}]", d, k, g, inverse_transform=inv,
                          cross_fit_folds=c)))
        specs.append((f"AR-LRX-Aug-Seg [{kind}]", "exp05d_rossmann_arlrx_dev",
                      lambda d, c, k=kind: A.run_arlrx_segmented(
                          f"AR-LRX-Aug-Seg [{k}]", d, k, g, augment_stage1=True,
                          inverse_transform=inv, cross_fit_folds=c)))
    return specs


def archived(source, variant, model):
    df = pd.read_csv(os.path.join(RESULTS, f"{source}.csv"))
    sub = df[(df.feature_set == variant) & (df.model == model)]
    return None if sub.empty else sub.iloc[0]


COLS = ["orig_RMSE", "orig_MAE", "orig_RMSPE", "orig_R2", "test_RMSE",
        "gate_w", "resid_val_r2", "gate_val_gain_pct", "stage1_only_test_RMSE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["V3_sales_lagged"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--stores", type=int, default=None,
                    help="smoke test: keep only the first N stores (archived values then do not apply)")
    args = ap.parse_args()
    out = OUT if args.stores is None else OUT.replace(".csv", "_smoke.csv")

    P.set_global_seed()
    print("Environment:", P.environment_stamp(), flush=True)
    frame = P.build_rossmann_frame(os.path.join(ROOT, "data/raw/rossmann/train.csv"),
                                   os.path.join(ROOT, "data/raw/rossmann/store.csv"))
    if args.stores:
        keep = np.sort(frame["Store"].unique())[: args.stores]
        frame = frame[frame["Store"].isin(keep)].reset_index(drop=True)
    specs = [s for s in model_specs() if args.models is None or s[0] in args.models]
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=["variant", "model"])

    for variant in args.variants:
        d = P.build_rossmann_dataset(frame, variant, target="log1p")
        for label, src, fn in specs:
            if ((done.variant == variant) & (done.model == label)).any():
                print(f"skip (already in audit file): {variant} {label}")
                continue
            ref = archived(src, variant, label)
            t0 = time.time()
            old = fn(d, None)
            new = fn(d, A.N_CROSS_FIT_FOLDS)
            env = P.environment_stamp()
            line = {"variant": variant, "model": label, "source_archived": src,
                    "platform": env["platform"], "python": env["python"],
                    "cross_fit_folds": A.N_CROSS_FIT_FOLDS,
                    "runtime_min": round((time.time() - t0) / 60, 2)}
            val = lambda r, c: float(r[c]) if c in r and r[c] is not None else np.nan
            for c in COLS:
                line[f"{c}_archived"] = np.nan if ref is None or c not in ref else float(ref[c])
                line[f"{c}_before"] = val(old, c)
                line[f"{c}_after"] = val(new, c)
            b, a = line["orig_RMSE_before"], line["orig_RMSE_after"]
            line["orig_RMSE_change_pct"] = (a - b) / b * 100
            line["legacy_reproduces_archived"] = bool(
                abs(b - line["orig_RMSE_archived"]) < 1e-6)
            pd.DataFrame([line]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            done = pd.read_csv(out)
            print(f"{variant} {label:32s} archived {line['orig_RMSE_archived']:9.2f} "
                  f"(reproduced here: {line['legacy_reproduces_archived']})  "
                  f"in-sample {b:9.2f} -> cross-fitted {a:9.2f} "
                  f"({line['orig_RMSE_change_pct']:+.2f}%)  w* {line['gate_w_before']:.2f} -> "
                  f"{line['gate_w_after']:.2f}  resid R2 {line['resid_val_r2_before']:.3f} -> "
                  f"{line['resid_val_r2_after']:.3f}  [{line['runtime_min']:.1f} min]", flush=True)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
