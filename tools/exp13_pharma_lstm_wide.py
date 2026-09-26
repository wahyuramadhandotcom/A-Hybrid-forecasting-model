"""
exp13 -- LSTM on PharmaSales with a wider width grid.

Why
---
In exp12 the validation-selected LSTM width fell on the edge of the grid
{64, 128, 256} in 23 of 32 configurations (64 in 12, 256 in 11), so a wider grid
might still improve it. The dissertation lists this as a limitation of the
comparison. This script removes it by adding one width below and one above the
exp12 grid. Pre-registered in PREREG_exp10_exp13_exp14.md (prediction P4) before
any run of this script.

Protocol (contract C1-C8 unchanged; everything identical to exp12 except the grid)
--------------------------------------------------------------------------------
  LSTM (wide)   the strengthened Keras builder of exp06b/exp12 (target standardised
                on the active block, early stopping on its chronological tail),
                batch size 32, width grid {32, 64, 128, 256, 512}, learning rate 0.001,
                selected on validation RMSE (C3), refit on train+validation (C6).
  S1 [linear], AR-LRX [linear]   recomputed for paired Diebold-Mariano tests, using
                the exp12 analysis code unchanged.

The exp12 LSTM rows are read from results/exp12_pharma_baselines.csv and reported
next to the new rows (change in test RMSE, selected width), so the effect of the
wider grid is visible per configuration.

Run on the Windows reference machine (TensorFlow needed):
    python tools/exp13_pharma_lstm_wide.py
    python tools/exp13_pharma_lstm_wide.py --quick   # 2 categories, 2 widths, smoke test
Outputs: results/exp13_pharma_lstm_wide.csv (+ .meta.json), _dm.csv, _summary.csv,
         _vs_exp12.csv
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from src.experiments import protocol as P   # noqa: E402
from src.experiments import arlrx as A      # noqa: E402
import exp12_pharma_baselines_completion as E12   # noqa: E402

warnings.filterwarnings("ignore")
EXPERIMENT = "exp13_pharma_lstm_wide"
GRID_LSTM_WIDE = {"units": [32, 64, 128, 256, 512], "learning_rate": [0.001]}
MODEL = "LSTM (wide grid)"


def run(quick: bool):
    from src.experiments import baselines as B
    P.set_global_seed()
    lstm_fp = B.make_keras_fp_v2("lstm", batch_size=E12.LSTM_BATCH)
    grid = {"units": [32, 512], "learning_rate": [0.001]} if quick else GRID_LSTM_WIDE
    cats = E12.CATEGORIES[:2] if quick else E12.CATEGORIES
    rows, data = [], {}
    t0 = time.time()
    for gran, path, sp in E12.GRANS:
        raw = pd.read_csv(path)
        for cat in cats:
            for fs in P.FEATURE_SETS:
                d = P.build_pharma_dataset(raw, cat, fs, seasonal_period=sp, lag_rule="pacf_train")
                data[(gran, cat, fs)] = d
                add = lambda r: rows.append({**r, "granularity": gran})
                add(A.run_stage1_only("S1 [linear]", d, "linear"))
                add(A.run_arlrx("AR-LRX [linear]", d, "linear", P.GRID_XGB_PHARMA))
                add(P.run_model(MODEL, lstm_fp, d, grid))
                print(f"  {gran:6s} {cat:6s} {fs:7s} ({(time.time()-t0)/60:.1f} min)", flush=True)
    return rows, data


def compare_exp12(rows) -> pd.DataFrame:
    f = P.RESULTS_DIR / "exp12_pharma_baselines.csv"
    new = pd.DataFrame([r for r in rows if r["model"] == MODEL])
    new["units_new"] = new.params.map(lambda s: eval(s)["units"] if isinstance(s, str) else np.nan)
    key = ["granularity", "category", "feature_set"]
    if not f.exists():
        return new[key + ["test_RMSE", "units_new"]]
    old = pd.read_csv(f)
    old = old[old.model == "LSTM"][key + ["test_RMSE", "params"]].rename(
        columns={"test_RMSE": "test_RMSE_exp12", "params": "params_exp12"})
    old["units_exp12"] = old.params_exp12.map(lambda s: eval(s)["units"])
    out = new[key + ["test_RMSE", "units_new", "n_lags"]].merge(old, on=key, how="left")
    out["change_pct"] = 100 * (out.test_RMSE / out.test_RMSE_exp12 - 1)
    out["new_width_selected"] = out.units_new.isin([32, 512])
    return out.drop(columns="params_exp12")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    rows, data = run(args.quick)
    dm, summ = E12.analyse(rows, data)
    name = EXPERIMENT + ("_quick" if args.quick else "")
    cmp_ = compare_exp12(rows)
    P.save_results(rows, name)
    dm.to_csv(P.RESULTS_DIR / f"{name}_dm.csv", index=False)
    summ.to_csv(P.RESULTS_DIR / f"{name}_summary.csv", index=False)
    cmp_.to_csv(P.RESULTS_DIR / f"{name}_vs_exp12.csv", index=False)
    pd.set_option("display.width", 200)
    show = summ[summ.scope == "distinct"].copy(); show["pooled_pct"] = show.pooled_pct.round(2)
    print("\nAR-LRX [linear] against each reference, distinct configurations "
          "(negative pooled_pct = AR-LRX better):")
    print(show.drop(columns="scope").to_string(index=False))
    if "change_pct" in cmp_:
        print("\nSelected width (new grid):", cmp_.units_new.value_counts().sort_index().to_dict())
        print(f"Configurations selecting a new width (32 or 512): {int(cmp_.new_width_selected.sum())}"
              f" of {len(cmp_)}")
        r = cmp_.test_RMSE / cmp_.test_RMSE_exp12
        print(f"Pooled change in LSTM test RMSE vs exp12: {100*(np.exp(np.log(r).mean())-1):+.2f}%")
    print(f"\nTotal runtime {(time.time()-t0)/60:.1f} min -> results/{name}*.csv")


if __name__ == "__main__":
    main()
