"""
exp12 -- Completing the PharmaSales baseline set named in the dissertation RQ1.

Why
---
The approved proposal lists ARIMA, LSTM, Random Forest, GRNN and XGBoost as
single-model references. On PharmaSales the protocol (exp01/exp02, exp05c)
already covers LR, GRNN, PNN, RBFNN and XGBoost, but:
  * Random Forest was only used as a RESIDUAL learner (exp08), never as a forecaster;
  * LSTM was only evaluated on Rossmann;
  * ARIMA(5,1,0) in exp01/exp02 forecast the whole test block MULTI-STEP from the
    end of train+val, whereas every other model forecasts ROLLING ONE-STEP-AHEAD
    (contract C7). That asymmetry handicaps ARIMA and must be removed before ARIMA
    is used as a reference in the dissertation.

What this script runs (contract C1-C8 unchanged, 32 configurations as exp05c)
-----------------------------------------------------------------------------
  ARIMA(5,1,0) multi-step      the exp01/exp02 procedure, kept for comparison
  ARIMA(5,1,0) rolling 1-step  parameters estimated on the fitting block, then the
                               model is FILTERED over the observed series so that each
                               prediction uses observations up to t-1 only (C7)
  Random Forest                grid of 8 settings, selected on validation (C3)
  LSTM                         the strengthened Keras builder of exp06b (target
                               standardised on the active block, early stopping on
                               its chronological tail, width grid {64,128,256});
                               batch size 32 because the pharmaceutical series are short.
                               Input (n, p, 1): for B_rich the p positions ARE lags,
                               so here the LSTM sees a genuine lag window.
  S1 [linear], AR-LRX [linear], XGBoost   recomputed for paired Diebold-Mariano tests

Run on the Windows reference machine (TensorFlow needed for LSTM):
    python tools/exp12_pharma_baselines_completion.py
    python tools/exp12_pharma_baselines_completion.py --skip-lstm   # no TensorFlow
Outputs: results/exp12_pharma_baselines.csv (+ .meta.json), _dm.csv, _summary.csv
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
from src.experiments import protocol as P   # noqa: E402
from src.experiments import arlrx as A      # noqa: E402

warnings.filterwarnings("ignore")
EXPERIMENT = "exp12_pharma_baselines"
CATEGORIES = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
GRANS = [("daily", ROOT / "data/raw/pharma-sales/salesdaily.csv", 7),
         ("weekly", ROOT / "data/raw/pharma-sales/salesweekly.csv", 52)]
ORDER = (5, 1, 0)
GRID_RF = {"n_estimators": [300], "max_depth": [None, 10], "min_samples_leaf": [1, 5],
           "max_features": [1.0, "sqrt"]}
LSTM_BATCH = 32


def fp_rf(X_fit, y_fit, X_eval, params):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(random_state=P.SEED, n_jobs=-1, **params)
    return m.fit(X_fit, y_fit).predict(X_eval)


def arima_rolling_row(d: P.Dataset) -> dict:
    """One-step-ahead ARIMA: parameters from the fitting block, then filtering over
    the observed series (no refitting inside the evaluation block)."""
    from statsmodels.tsa.arima.model import ARIMA
    t0 = time.time()
    y = d.y

    def one_step(n_fit, n_end):
        res = ARIMA(y[:n_fit], order=ORDER).fit()
        full = ARIMA(y[:n_end], order=ORDER).filter(res.params)
        return np.maximum(np.asarray(full.get_prediction(start=n_fit).predicted_mean), 0.0)

    val_pred = one_step(d.i_train_end, d.i_val_end)
    test_pred = one_step(d.i_val_end, len(y))
    row = {"model": "ARIMA(5,1,0) rolling 1-step", **d.describe(),
           "params": str({"order": ORDER}), "n_grid": 1, "scaler": "none", "seed": P.SEED,
           **P.compute_metrics(d.y_val, val_pred, prefix="val_"),
           **P.compute_metrics(d.y_test, test_pred, prefix="test_"),
           "runtime_s": round(time.time() - t0, 3)}
    row["_val_pred"], row["_test_pred"] = val_pred, test_pred
    return row


def run(skip_lstm: bool):
    P.set_global_seed()
    grid_x = P.GRID_XGB_PHARMA
    lstm_fp = None
    if not skip_lstm:
        from src.experiments import baselines as B
        lstm_fp = B.make_keras_fp_v2("lstm", batch_size=LSTM_BATCH)
        grid_lstm = B.GRID_NEURAL_V2
    rows, data = [], {}
    t0 = time.time()
    for gran, path, sp in GRANS:
        raw = pd.read_csv(path)
        for cat in CATEGORIES:
            for fs in P.FEATURE_SETS:
                d = P.build_pharma_dataset(raw, cat, fs, seasonal_period=sp, lag_rule="pacf_train")
                data[(gran, cat, fs)] = d
                ex = {"granularity": gran}
                add = lambda r: rows.append({**r, **ex})
                for r in P.naive_rows(d):
                    if r["model"] == "Naive":
                        add(r)
                add(A.run_stage1_only("S1 [linear]", d, "linear"))
                add(A.run_arlrx("AR-LRX [linear]", d, "linear", grid_x))
                add(P.run_model("XGBoost", P.fp_xgboost, d, grid_x))
                add(P.run_model("ARIMA(5,1,0) multi-step", P.fp_arima, d, {"order": [ORDER]}))
                add(arima_rolling_row(d))
                add(P.run_model("Random Forest", fp_rf, d, GRID_RF))
                if lstm_fp is not None:
                    add(P.run_model("LSTM", lstm_fp, d, grid_lstm))
                print(f"  {gran:6s} {cat:6s} {fs:7s} ({(time.time()-t0)/60:.1f} min)", flush=True)
    return rows, data


def analyse(rows, data):
    by = {(r["granularity"], r["category"], r["feature_set"], r["model"]): r for r in rows}
    PROP = "AR-LRX [linear]"
    dm = []
    for (g, c, f), d in data.items():
        for m in sorted({k[3] for k in by if k[:3] == (g, c, f)} - {PROP}):
            a, b = by[(g, c, f, PROP)], by[(g, c, f, m)]
            t = P.diebold_mariano(d.y_test, a["_test_pred"], b["_test_pred"])
            dm.append({"granularity": g, "category": c, "feature_set": f, "reference": m,
                       "rmse_arlrx": a["test_RMSE"], "rmse_ref": b["test_RMSE"],
                       "delta_pct": 100 * (a["test_RMSE"] / b["test_RMSE"] - 1),
                       "DM": t["DM"], "p": t["p_value"],
                       "distinct": not (f == P.FEATURE_SET_B and d.n_lags == 1)})
    dm = pd.DataFrame(dm)
    dm["p_bh"] = np.nan
    for _, idx in dm.groupby("reference").groups.items():
        p = dm.loc[idx, "p"].to_numpy(); ok = np.isfinite(p); q = np.full(len(p), np.nan)
        if ok.any():
            pv = p[ok]; n = len(pv); o = np.argsort(pv)
            adj = np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1]
            tmp = np.empty(n); tmp[o] = np.minimum(adj, 1); q[ok] = tmp
        dm.loc[idx, "p_bh"] = q
    summ = []
    for (ref, g), s in dm.groupby(["reference", "granularity"]):
        for scope, ss in (("all", s), ("distinct", s[s.distinct])):
            r = ss.rmse_arlrx / ss.rmse_ref
            summ.append({"reference": ref, "granularity": g, "scope": scope, "n": len(ss),
                         "arlrx_wins": int((r < 1).sum()),
                         "sig_wins_bh": int(((ss.DM < 0) & (ss.p_bh < 0.05)).sum()),
                         "sig_losses_bh": int(((ss.DM > 0) & (ss.p_bh < 0.05)).sum()),
                         "pooled_pct": 100 * (np.exp(np.log(r).mean()) - 1)})
    return dm, pd.DataFrame(summ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-lstm", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    rows, data = run(args.skip_lstm)
    dm, summ = analyse(rows, data)
    name = EXPERIMENT + ("_nolstm" if args.skip_lstm else "")
    P.save_results(rows, name)
    dm.to_csv(P.RESULTS_DIR / f"{name}_dm.csv", index=False)
    summ.to_csv(P.RESULTS_DIR / f"{name}_summary.csv", index=False)
    pd.set_option("display.width", 200)
    show = summ[summ.scope == "distinct"].copy(); show["pooled_pct"] = show.pooled_pct.round(2)
    print("\nAR-LRX [linear] against each reference, 20 distinct configurations "
          "(negative pooled_pct = AR-LRX better):")
    print(show.drop(columns="scope").to_string(index=False))
    print(f"\nTotal runtime {(time.time()-t0)/60:.1f} min -> results/{name}*.csv")


if __name__ == "__main__":
    main()
