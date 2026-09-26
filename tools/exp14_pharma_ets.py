"""
exp14 -- Exponential smoothing (ETS / Holt-Winters) as an additional PharmaSales reference.

Why
---
Exponential smoothing is the most common family in pharmaceutical demand forecasting
and was not yet among the retrained references. Pre-registered in
PREREG_exp10_exp13_exp14.md (prediction P5) before any run of this script.

Protocol (contract C1-C8 unchanged, the 32 PharmaSales configurations of exp05c)
------------------------------------------------------------------------------
ETS with additive errors (the series contain zeros, so multiplicative components are
not admissible). Candidate specifications, selected on validation RMSE only (C3):

    (trend, damped, seasonal) in
        (None,  False, None)   simple exponential smoothing
        ("add", False, None)   Holt linear trend
        ("add", True,  None)   damped trend
        (None,  False, "add")  seasonal (period 7 daily / 52 weekly)
        ("add", True,  "add")  damped trend + seasonal (Holt-Winters)

Rolling one-step-ahead evaluation (C7), exactly like the rolling ARIMA of exp12:
parameters (smoothing weights and initial states) are estimated on the fitting block
(train for tuning, train+validation for the final refit, C6); the model is then
FILTERED over the observed series with those fixed parameters, so each prediction
uses observations up to t-1 only. A specification whose fit fails is skipped and
recorded in `n_failed`; fits that stop before the optimiser's convergence
criterion are kept when their predictions are finite and are counted in
`n_nonconverged_val` / `final_fit_converged` (reported, not hidden).

S1 [linear], AR-LRX [linear] and ARIMA(5,1,0) rolling are recomputed for paired
Diebold-Mariano tests, using the exp12 analysis code unchanged.

Run on the Windows reference machine:
    python tools/exp14_pharma_ets.py
    python tools/exp14_pharma_ets.py --quick     # 2 categories, smoke test only
Outputs: results/exp14_pharma_ets.csv (+ .meta.json), _dm.csv, _summary.csv
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
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.simplefilter("ignore", ConvergenceWarning)
except Exception:
    pass
EXPERIMENT = "exp14_pharma_ets"
SPECS = [(None, False, None), ("add", False, None), ("add", True, None),
         (None, False, "add"), ("add", True, "add")]


def _ets(y, spec, sp):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    trend, damped, seasonal = spec
    return ETSModel(np.asarray(y, dtype=float), error="add", trend=trend,
                    damped_trend=damped if trend else False, seasonal=seasonal,
                    seasonal_periods=sp if seasonal else None,
                    initialization_method="estimated")


def ets_one_step(y, n_fit, n_end, spec, sp):
    """Fit on y[:n_fit]; one-step-ahead predictions for positions n_fit..n_end-1.
    Returns (predictions, converged flag of the maximum-likelihood fit)."""
    res = _ets(y[:n_fit], spec, sp).fit(disp=False, maxiter=2000)
    conv = bool(getattr(res, "mle_retvals", {}).get("converged", True))
    full = _ets(y[:n_end], spec, sp).smooth(res.params)
    pred = np.asarray(full.fittedvalues, dtype=float)[n_fit:n_end]
    return np.maximum(pred, 0.0), conv


def ets_row(d: P.Dataset, sp: int) -> dict:
    t0 = time.time()
    y = d.y
    best, best_rmse, best_val, n_failed, n_nonconv = None, np.inf, None, 0, 0
    for spec in SPECS:
        try:
            vp, conv = ets_one_step(y, d.i_train_end, d.i_val_end, spec, sp)
            n_nonconv += int(not conv)
            if not np.all(np.isfinite(vp)):
                raise ValueError("non-finite")
        except Exception:
            n_failed += 1
            continue
        r = float(np.sqrt(np.mean((d.y_val - vp) ** 2)))
        if r < best_rmse:                      # tie-break: first specification wins
            best, best_rmse, best_val = spec, r, vp
    test_pred, test_conv = ets_one_step(y, d.i_val_end, len(y), best, sp)
    row = {"model": "ETS rolling 1-step", **d.describe(),
           "params": str({"trend": best[0], "damped": best[1], "seasonal": best[2],
                          "seasonal_periods": sp if best[2] else None}),
           "n_grid": len(SPECS), "n_failed": n_failed, "n_nonconverged_val": n_nonconv,
           "final_fit_converged": test_conv, "scaler": "none", "seed": P.SEED,
           **P.compute_metrics(d.y_val, best_val, prefix="val_"),
           **P.compute_metrics(d.y_test, test_pred, prefix="test_"),
           "runtime_s": round(time.time() - t0, 3)}
    row["_val_pred"], row["_test_pred"] = best_val, test_pred
    return row


def run(quick: bool):
    P.set_global_seed()
    rows, data = [], {}
    t0 = time.time()
    cats = E12.CATEGORIES[:2] if quick else E12.CATEGORIES
    for gran, path, sp in E12.GRANS:
        raw = pd.read_csv(path)
        for cat in cats:
            for fs in P.FEATURE_SETS:
                d = P.build_pharma_dataset(raw, cat, fs, seasonal_period=sp, lag_rule="pacf_train")
                data[(gran, cat, fs)] = d
                add = lambda r: rows.append({**r, "granularity": gran})
                add(A.run_stage1_only("S1 [linear]", d, "linear"))
                add(A.run_arlrx("AR-LRX [linear]", d, "linear", P.GRID_XGB_PHARMA))
                add(E12.arima_rolling_row(d))
                add(ets_row(d, sp))
                print(f"  {gran:6s} {cat:6s} {fs:7s} ({(time.time()-t0)/60:.1f} min)", flush=True)
    return rows, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    rows, data = run(args.quick)
    dm, summ = E12.analyse(rows, data)
    name = EXPERIMENT + ("_quick" if args.quick else "")
    P.save_results(rows, name)
    dm.to_csv(P.RESULTS_DIR / f"{name}_dm.csv", index=False)
    summ.to_csv(P.RESULTS_DIR / f"{name}_summary.csv", index=False)
    pd.set_option("display.width", 200)
    show = summ[summ.scope == "distinct"].copy(); show["pooled_pct"] = show.pooled_pct.round(2)
    print("\nAR-LRX [linear] against each reference, distinct configurations "
          "(negative pooled_pct = AR-LRX better):")
    print(show.drop(columns="scope").to_string(index=False))
    ets = pd.DataFrame([r for r in rows if r["model"] == "ETS rolling 1-step"])
    print("\nSelected ETS specification:")
    print(ets.groupby(["granularity", "params"]).size().to_string())
    print(f"\nTotal runtime {(time.time()-t0)/60:.1f} min -> results/{name}*.csv")


if __name__ == "__main__":
    main()
