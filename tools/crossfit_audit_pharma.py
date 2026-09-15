"""
Before/after audit for reviewer point R1-1 (IJIES Paper ID 20265893, round 2)
on the pharmaceutical experiment (exp05c).

Runs the exact model set of `notebooks/exp05c_pharma_arlrx.ipynb` (cell 5) twice:

  legacy   residuals formed in-sample (`cross_fit_folds=None`) -- the procedure
           behind the submitted manuscript. Its rows must reproduce
           `results/exp05c_pharma_arlrx.csv`; the script checks this and reports
           the largest deviation, which doubles as a platform-reproducibility check.
  crossfit residuals formed from cross-fitted first-stage predictions (the
           revised default, K = N_CROSS_FIT_FOLDS chronological folds).

and writes, per mode, the same summaries the manuscript reports (Tables 7-9,
Section 4.4/4.5, Figure 3 inputs), plus a side-by-side comparison.

Nothing here selects anything on the test block; the script only re-runs the
existing protocol under two residual-generation procedures.

Usage (from the repository root):
    python tools/crossfit_audit_pharma.py              # both modes
    python tools/crossfit_audit_pharma.py --mode crossfit
    python tools/crossfit_audit_pharma.py --quick      # 2 categories, smoke test
    python tools/crossfit_audit_pharma.py --mode crossfit --folds 3   # fold-count sensitivity

Outputs (results/):
    exp05c_crossfit_<mode>.csv              one row per (config, model)
    exp05c_crossfit_<mode>_predictions.npz  test predictions, same keys as exp05c
    exp05c_crossfit_<mode>_dm.csv           per-configuration DM tests (as exp05c)
    exp05c_crossfit_audit_summary.csv       legacy vs crossfit, one line per statistic
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from src.experiments import arlrx as A          # noqa: E402
from src.experiments import protocol as P       # noqa: E402

CATEGORIES = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
GRANS = [("daily", "data/raw/pharma-sales/salesdaily.csv", 7),
         ("weekly", "data/raw/pharma-sales/salesweekly.csv", 52)]
REFS = ["Naive", "S1 [linear]", "XGBoost", "LR-XGB (residual, tanpa gerbang)"]
RESULTS = os.path.join(ROOT, "results")


def build_datasets(quick: bool):
    cats = CATEGORIES[:2] if quick else CATEGORIES
    out = {}
    for gran, path, sp in GRANS:
        data = pd.read_csv(os.path.join(ROOT, path))
        for cat in cats:
            for fs in P.FEATURE_SETS:
                out[(gran, cat, fs)] = P.build_pharma_dataset(
                    data, cat, fs, seasonal_period=sp, lag_rule="pacf_train")
    return out


def run_mode(datasets, cff):
    """Model loop of exp05c cell 5, with the residual procedure made explicit.

    cff: None (in-sample residuals) or the number of cross-fitting folds."""
    rows = []
    lrx_resid = functools.partial(P.fp_lr_xgb_residual, cross_fit_folds=cff)
    for (gran, cat, fs), d in datasets.items():
        n_val = int(d.describe()["n_val"])
        n_folds = int(min(5, max(2, n_val // 30)))
        extra = {"granularity": gran, "n_folds": n_folds}
        for r in P.naive_rows(d):
            r.update(extra); rows.append(r)
        rows.append({**A.run_stage1_only("S1 [linear]", d, "linear"), **extra})
        rows.append({**A.run_arlrx("AR-LRX [linear]", d, "linear", P.GRID_XGB_PHARMA,
                                   cross_fit_folds=cff), **extra})
        rows.append({**A.run_arlrx_segmented("AR-LRX-g [linear]", d, "linear",
                                             P.GRID_XGB_PHARMA, schemes=("global",),
                                             n_folds=n_folds, augment_stage1=False,
                                             cross_fit_folds=cff), **extra})
        rows.append({**A.run_arlrx_segmented("AR-LRX-Aug [linear]", d, "linear",
                                             P.GRID_XGB_PHARMA, schemes=("global",),
                                             n_folds=n_folds, augment_stage1=True,
                                             cross_fit_folds=cff), **extra})
        rows.append({**P.run_model("XGBoost", P.fp_xgboost, d, P.GRID_XGB_PHARMA), **extra})
        rows.append({**P.run_model("LR-XGB (residual, tanpa gerbang)", lrx_resid,
                                   d, P.GRID_XGB_PHARMA), **extra})
        print(f"    {gran:7s} {cat:6s} {fs:8s} done", flush=True)
    return rows


def summarise(rows, datasets):
    """The statistics the manuscript quotes, computed as in exp05c cells 7-15."""
    by = {(r["granularity"], r["category"], r["feature_set"], r["model"]): r for r in rows}
    keys = list(datasets)
    s = {}

    # Table 7 / Section 4.1: gate closures and mean selected weight
    w = np.array([float(by[k + ("AR-LRX [linear]",)]["gate_w"]) for k in keys]).round(3)
    s["T7 gate closed (w*=0), of N"] = int((w == 0).sum())
    s["T7 gate closed, daily"] = int(sum(1 for k, x in zip(keys, w) if k[0] == "daily" and x == 0))
    s["T7 gate closed, weekly"] = int(sum(1 for k, x in zip(keys, w) if k[0] == "weekly" and x == 0))
    s["T7 mean selected w*"] = float(w.mean())
    for v, c in zip(*np.unique(w, return_counts=True)):
        s[f"T7 count w*={v:g}"] = int(c)

    # Section 1 / 4.1 / Fig. 3: residual R2 on validation
    r2 = np.array([float(by[k + ("AR-LRX [linear]",)]["resid_val_r2"]) for k in keys])
    s["resid_val_r2 negative, of N"] = int((r2 < 0).sum())
    s["resid_val_r2 median"] = float(np.median(r2))
    s["resid_val_r2 min"] = float(r2.min())
    s["resid_val_r2 max"] = float(r2.max())

    # Table 8: degradation relative to the linear first stage
    def degr(model):
        v = []
        for k in keys:
            s1 = by[k + ("S1 [linear]",)]["test_RMSE"]
            v.append((by[k + (model,)]["test_RMSE"] - s1) / s1 * 100)
        return np.array(v)
    for label, model in [("ungated", "LR-XGB (residual, tanpa gerbang)"),
                         ("gated", "AR-LRX [linear]"),
                         ("gated+aug", "AR-LRX-Aug [linear]")]:
        v = degr(model)
        s[f"T8 {label}: worse than S1, of N"] = int((v > 1e-9).sum())
        s[f"T8 {label}: mean %"] = float(v.mean())
        s[f"T8 {label}: worst %"] = float(v.max())

    # Table 9: wins / ties / significant wins of AR-LRX against each reference
    dm_rows = []
    for k, d in datasets.items():
        prop = by[k + ("AR-LRX [linear]",)]
        for ref in REFS:
            rr = by.get(k + (ref,))
            if rr is None:
                continue
            t = P.diebold_mariano(d.y_test, prop["_test_pred"], rr["_test_pred"])
            same = bool(np.array_equal(prop["_test_pred"], rr["_test_pred"]))
            dm_rows.append({"granularity": k[0], "category": k[1], "feature_set": k[2],
                            "reference": ref, "DM": t["DM"], "p": t["p_value"],
                            "identical_predictions": same,
                            "win": bool(t["DM"] < 0),
                            "sig_win": bool(t["DM"] < 0 and t["p_value"] < 0.05),
                            "sig_loss": bool(t["DM"] > 0 and t["p_value"] < 0.05)})
    dm = pd.DataFrame(dm_rows)
    for ref in REFS:
        g = dm[dm.reference == ref]
        s[f"T9 vs {ref}: wins"] = int(g.win.sum())
        s[f"T9 vs {ref}: ties (identical)"] = int(g.identical_predictions.sum())
        s[f"T9 vs {ref}: significant wins"] = int(g.sig_win.sum())
        s[f"T9 vs {ref}: significant losses"] = int(g.sig_loss.sum())
    s["DM tests defined (non-NaN p)"] = int(dm.p.notna().sum())

    # Section 4.4: augmentation effect (Aug vs its paired gated control)
    e = np.array([(by[k + ("AR-LRX-Aug [linear]",)]["test_RMSE"]
                   - by[k + ("AR-LRX-g [linear]",)]["test_RMSE"])
                  / by[k + ("AR-LRX-g [linear]",)]["test_RMSE"] * 100 for k in keys])
    s["S4.4 augmentation effect: mean %"] = float(e.mean())
    s["S4.4 augmentation improves, of N"] = int((e < 0).sum())

    # Section 4.5: validation-gain indicator vs realised test improvement
    gain = np.array([float(by[k + ("AR-LRX [linear]",)]["gate_val_gain_pct"]) for k in keys])
    real = np.array([(by[k + ("S1 [linear]",)]["test_RMSE"] - by[k + ("AR-LRX [linear]",)]["test_RMSE"])
                     / by[k + ("S1 [linear]",)]["test_RMSE"] * 100 for k in keys])
    s["S4.5 Pearson r (val gain vs test gain)"] = float(np.corrcoef(gain, real)[0, 1])
    s["S4.5 Spearman rho"] = float(stats.spearmanr(gain, real).statistic)
    open_ = w > 0
    s["S4.5 open gates: mean val gain %"] = float(gain[open_].mean()) if open_.any() else np.nan
    s["S4.5 open gates: mean test gain %"] = float(real[open_].mean()) if open_.any() else np.nan
    s["S4.5 open gates: improved on test"] = int((real[open_] > 1e-9).sum())
    s["N configurations"] = len(keys)
    return s, dm


def save_mode(rows, datasets, mode):
    res = P.save_results(rows, f"exp05c_crossfit_{mode}")
    np.savez_compressed(
        os.path.join(RESULTS, f"exp05c_crossfit_{mode}_predictions.npz"),
        **{f"{r['granularity']}|{r['category']}|{r['feature_set']}|{r['model']}|test_pred":
           np.asarray(r["_test_pred"], dtype=np.float32) for r in rows},
        **{f"{g}|{c}|{f}|y_test": np.asarray(d.y_test, dtype=np.float32)
           for (g, c, f), d in datasets.items()})
    return res


def check_against_submitted(res):
    """Legacy mode must reproduce the archived exp05c result file."""
    ref_path = os.path.join(RESULTS, "exp05c_pharma_arlrx.csv")
    if not os.path.exists(ref_path):
        print("  (no archived exp05c file - reproduction check skipped)")
        return None
    ref = pd.read_csv(ref_path)
    key = ["granularity", "category", "feature_set", "model"]
    cols = ["test_RMSE", "val_RMSE", "gate_w", "resid_val_r2"]
    m = res[key + cols].merge(ref[key + cols], on=key, suffixes=("_now", "_ref"))
    worst = 0.0
    for c in cols:
        a, b = m[f"{c}_now"].astype(float), m[f"{c}_ref"].astype(float)
        both_nan = a.isna() & b.isna()
        dev = (a - b).abs()[~both_nan]
        worst = max(worst, float(dev.max()))
        print(f"  reproduce {c:14s} max |now - archived| = {dev.max():.3e}  over {len(dev)} rows")
    print(f"  LEGACY REPRODUCES ARCHIVED exp05c: {worst < 1e-6}  (worst {worst:.3e})")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["both", "legacy", "crossfit"], default="both")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--folds", type=int, default=A.N_CROSS_FIT_FOLDS,
                    help="cross-fitting folds for the crossfit mode (default: the protocol value)")
    args = ap.parse_args()
    k = args.folds
    tag = "crossfit" if k == A.N_CROSS_FIT_FOLDS else f"crossfit_k{k}"

    P.set_global_seed()
    print("Environment:", P.environment_stamp())
    datasets = build_datasets(args.quick)
    print(f"{len(datasets)} configurations; cross-fit folds = {k}")
    modes = ["legacy", tag] if args.mode == "both" else ([tag] if args.mode == "crossfit" else ["legacy"])
    summaries = {}
    for mode in modes:
        cff = None if mode == "legacy" else k
        t0 = time.time()
        print(f"\n=== mode {mode} (cross_fit_folds={cff}) ===", flush=True)
        rows = run_mode(datasets, cff)
        res = save_mode(rows, datasets, mode)
        if mode == "legacy" and not args.quick:
            check_against_submitted(res)
        summ, dm = summarise(rows, datasets)
        dm.to_csv(os.path.join(RESULTS, f"exp05c_crossfit_{mode}_dm.csv"), index=False)
        summaries[mode] = summ
        print(f"  finished in {(time.time()-t0)/60:.1f} min")

    out = os.path.join(RESULTS, "exp05c_crossfit_audit_summary.csv")
    table = pd.DataFrame(summaries)
    if os.path.exists(out) and not args.quick:
        # keep columns written by earlier runs (e.g. legacy + other fold counts)
        prev = pd.read_csv(out, index_col=0)
        for col in prev.columns:
            if col not in table.columns and col != "change":
                table[col] = prev[col]
    if {"legacy", "crossfit"} <= set(table.columns):
        table["change"] = table["crossfit"] - table["legacy"]
        table = table[[c for c in table.columns if c != "change"] + ["change"]]
    if not args.quick:
        table.to_csv(out)
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print("\n", table.round(4).to_string())
    print(f"\nwritten: {out if not args.quick else '(quick run - summary not written)'}")


if __name__ == "__main__":
    main()
