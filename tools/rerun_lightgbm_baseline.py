"""
Re-train the LightGBM baseline of exp06b with row bagging actually enabled
(IJIES Paper ID 20265893, round 2 - baseline fidelity fix).

`baselines.fp_lightgbm` used to pass subsample=0.8 without subsample_freq, and
LightGBM ignores row subsampling while the frequency is 0, so the submitted
baseline ran without row bagging. This script

  1. re-runs the ORIGINAL setting (subsample_freq=0) and checks that it
     reproduces the archived exp06b row (reproduction check);
  2. runs the FIXED setting (subsample_freq=1, the new default);
  3. writes the before/after comparison to results/lightgbm_fix_audit.csv;
  4. replaces ONLY the LightGBM row of results/exp06b_rossmann_baselines_strong.csv
     and ONLY its prediction array in ..._predictions.npz (every other baseline
     is left bit-identical, which the script verifies).

Afterwards re-run, in this order:
    notebooks/exp06b_rossmann_baselines_strong.ipynb   (reuse path; skip the smoke-test cell)
    python tools/groupB_inference.py
    python tools/exp07_rolling_origin.py --part rossmann-origins --redo-model "LightGBM (Zeng)"
    python tools/exp07_rolling_origin.py --part rossmann-seeds   --redo-model "LightGBM (Zeng)"
    python tools/exp07_rolling_origin.py --part summary

Usage (repository root):  python tools/rerun_lightgbm_baseline.py        (~3-4 min)
"""
from __future__ import annotations

import functools
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from src.experiments import baselines as B      # noqa: E402
from src.experiments import protocol as P       # noqa: E402

RES = os.path.join(ROOT, "results")
EXP = "exp06b_rossmann_baselines_strong"
NAME = "LightGBM (Zeng)"
VARIANT = "V3_sales_lagged"


def main():
    P.set_global_seed()
    print("Environment:", P.environment_stamp(), flush=True)
    csv = os.path.join(RES, f"{EXP}.csv")
    npz = os.path.join(RES, f"{EXP}_predictions.npz")
    saved = pd.read_csv(csv)
    z = dict(np.load(npz))
    assert NAME in set(saved.model), f"{NAME} not in {csv}"
    old_row = saved[saved.model == NAME].iloc[0]

    frame = P.build_rossmann_frame(os.path.join(ROOT, "data/raw/rossmann/train.csv"),
                                   os.path.join(ROOT, "data/raw/rossmann/store.csv"))
    d = P.build_rossmann_dataset(frame, VARIANT, target="log1p")
    assert np.allclose(z["y_test"], d.y_test, atol=1e-5), "saved y_test does not match the split"

    runs = {}
    for freq in (0, 1):
        t0 = time.time()
        fp = functools.partial(B.fp_lightgbm, subsample_freq=freq)
        runs[freq] = P.run_model(NAME, fp, d, B.GRID_LIGHTGBM, inverse_transform=np.expm1,
                                 extra={"lgbm_subsample_freq": freq})
        print(f"  subsample_freq={freq}: orig RMSE {runs[freq]['orig_RMSE']:.4f}  "
              f"params {runs[freq]['params']}  [{(time.time()-t0)/60:.1f} min]", flush=True)

    dev = abs(runs[0]["orig_RMSE"] - float(old_row.orig_RMSE))
    reproduced = dev < 1e-6
    print(f"\nOriginal setting reproduces archived exp06b row: {reproduced} "
          f"(|diff| = {dev:.2e})")
    if not reproduced:
        sys.exit("Reproduction failed - nothing was written. Check the environment first.")

    cols = ["orig_RMSE", "orig_MAE", "orig_RMSPE", "orig_R2", "test_RMSE", "val_RMSE", "val_R2"]
    audit = pd.DataFrame([{
        "model": NAME, "platform": P.environment_stamp()["platform"],
        **{f"{c}_archived": float(old_row[c]) for c in cols},
        **{f"{c}_freq0": float(runs[0][c]) for c in cols},
        **{f"{c}_freq1": float(runs[1][c]) for c in cols},
        "params_freq0": runs[0]["params"], "params_freq1": runs[1]["params"],
        "orig_RMSE_change_pct": 100 * (runs[1]["orig_RMSE"] / runs[0]["orig_RMSE"] - 1),
    }])
    audit.to_csv(os.path.join(RES, "lightgbm_fix_audit.csv"), index=False)

    # replace only the LightGBM row and prediction
    for f in (csv, npz):
        bak = f + ".before_lgbm_fix"
        if not os.path.exists(bak):
            shutil.copy2(f, bak)
    new = {k: v for k, v in runs[1].items() if not k.startswith("_")}
    rows = []
    for rec in saved.to_dict("records"):
        rows.append({**{k: np.nan for k in new}, **new} if rec["model"] == NAME else rec)
    out = pd.DataFrame(rows)
    out.to_csv(csv, index=False)
    z_new = dict(z)
    z_new[f"{NAME}|test_pred"] = np.asarray(runs[1]["_test_pred"], dtype=np.float32)
    np.savez_compressed(npz, **z_new)

    # verify every other baseline is untouched
    chk = pd.read_csv(csv)
    others = saved.model != NAME
    for c in ["orig_RMSE", "val_R2", "params"]:
        assert (chk.loc[others.values, c].astype(str).values
                == saved.loc[others, c].astype(str).values).all(), f"other rows changed ({c})"
    zc = np.load(npz)
    for k in z:
        if k != f"{NAME}|test_pred":
            assert np.array_equal(zc[k], z[k]), f"prediction {k} changed"
    print(f"\nUpdated {NAME} in {os.path.basename(csv)} and {os.path.basename(npz)}: "
          f"orig RMSE {runs[0]['orig_RMSE']:.2f} -> {runs[1]['orig_RMSE']:.2f} "
          f"({audit.orig_RMSE_change_pct.iloc[0]:+.2f}%). Other baselines verified unchanged.")
    print("Backups: *.before_lgbm_fix (not tracked by git). Audit: results/lightgbm_fix_audit.csv")


if __name__ == "__main__":
    main()
