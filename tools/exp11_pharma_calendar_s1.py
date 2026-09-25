"""
exp11 -- Calendar-keyed structural first stage on PharmaSales.

Question
--------
IJIES 20265893, Section 5.4 and the Conclusion, leave one question open: the
structural first stage that turned the retail result around needs categorical
keys, and the pharmaceutical series have none. Constructing calendar keys
(day of week, month) makes the structural first stage applicable. Does
structure alignment then improve accuracy on the medicine series, as it did on
Rossmann, or does the pharmaceutical regime stay a noise-residual regime?

Predictions written BEFORE running (reported whatever the outcome)
-----------------------------------------------------------------
  P11-1  Daily series: a calendar-aware first stage (struct-linear) beats the
         lag-only linear first stage of exp05c in most of the 16 daily
         configurations, because pharmacy sales have a weekday profile that
         lag features encode only indirectly.
  P11-2  Weekly series: no material gain (|pooled change| < 1%), because a
         weekly total has no weekday structure and month effects are weak.
  P11-3  The residual regime does not change: the second-stage validation R^2
         stays <= 0.05 in most configurations and the gate still closes
         often, i.e. any gain comes from the first stage, not from S2.

Design (contract C1-C8 unchanged)
---------------------------------
* Same 32 configurations as exp05c: {daily, weekly} x 8 ATC x {A_lag1, B_rich}.
  Rows, lag count k (PACF on the training block) and train/val/test dates are
  IDENTICAL to exp05c, because calendar columns introduce no missing values.
* Calendar columns added to every configuration:
      daily : one-hot day of week (6) + one-hot month (11); keys (DayOfWeek, Month)
      weekly: one-hot month (11);                           key  (Month)
  Integer key columns are used ONLY by the structural estimator (group means)
  and by tree learners; linear components use the one-hot columns instead,
  so no ordinal assumption is imposed on a linear model.
* First stages (hierarchical fallback identical to arlrx._hierarchical_group_predict):
      linear        LR on lags (+ one-hot calendar when present)
      structural    group mean over (DayOfWeek x Month) -> DayOfWeek -> global (daily)
                    group mean over Month -> global (weekly)
      struct_linear structural, then LR on its residual
* Residuals cross-fitted (5 chronological folds), gate w in {0, 0.1, ..., 1}
  chosen on validation, GRID_XGB_PHARMA -- all exactly as exp05c.
* The reference is S1 [linear] WITHOUT calendar, recomputed here and checked
  against results/exp05c_crossfit_crossfit.csv (the published baseline).

Models per configuration
------------------------
  Naive                           previous observed value
  S1 [linear] (exp05c)            reference, no calendar
  S1 [linear+cal]                 LR with calendar dummies (separates "calendar
                                  information" from "structure alignment")
  S1 [structural]                 calendar group means alone
  S1 [struct_linear]              calendar group means + LR
  XGBoost [+cal]                  XGBoost on lags + calendar
  AR-LRX [linear+cal]             gated
  AR-LRX [struct_linear]          gated
  AR-LRX-Aug [struct_linear]      augmented (S1 fed to S2)

Run (Windows, from repo root; about 20-30 min on CPU):
    python tools/exp11_pharma_calendar_s1.py            # full run
    python tools/exp11_pharma_calendar_s1.py --quick    # 2 categories, smoke test
Outputs: results/exp11_pharma_calendar.csv (+ .meta.json), _dm.csv, _summary.csv,
         results/exp11_pharma_calendar_predictions.npz (git-ignored).

Platform note: GRID_XGB_PHARMA uses subsample=0.8, so XGBoost numbers can differ
slightly between Linux and Windows. The canonical run is the Windows one.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import protocol as P      # noqa: E402
from src.experiments import arlrx as A          # noqa: E402

EXPERIMENT = "exp11_pharma_calendar"
CATEGORIES = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
GRANS = [("daily", ROOT / "data/raw/pharma-sales/salesdaily.csv", 7),
         ("weekly", ROOT / "data/raw/pharma-sales/salesweekly.csv", 52)]
KEY_COLS = ("DayOfWeek", "Month")          # structural keys, in hierarchy order


# --------------------------------------------------------------------------- #
# Calendar features
# --------------------------------------------------------------------------- #

def add_calendar(d: P.Dataset, granularity: str) -> P.Dataset:
    """Return a copy of `d` with calendar columns appended to the feature list.
    Rows and split points are untouched (no NaN is introduced)."""
    f = d.frame.copy()
    ds = pd.to_datetime(f["ds"])
    new = []
    f["Month"] = ds.dt.month.astype(float)
    if granularity == "daily":
        f["DayOfWeek"] = ds.dt.dayofweek.astype(float)
        for k in range(1, 7):
            f[f"dow_{k}"] = (ds.dt.dayofweek == k).astype(float)
            new.append(f"dow_{k}")
        keys = ["DayOfWeek", "Month"]
    else:
        keys = ["Month"]
    for m in range(2, 13):
        f[f"month_{m}"] = (ds.dt.month == m).astype(float)
        new.append(f"month_{m}")
    return P.Dataset(name=d.name, feature_set=d.feature_set + "+cal",
                     feature_names=list(d.feature_names) + new + keys, frame=f,
                     i_train_end=d.i_train_end, i_val_end=d.i_val_end,
                     n_lags=d.n_lags, seasonal_period=d.seasonal_period,
                     lag_rule=d.lag_rule)


# --------------------------------------------------------------------------- #
# First stages with calendar keys (replaces arlrx.make_stage1 for this script)
# --------------------------------------------------------------------------- #

def make_stage1_cal(kind: str, feature_names):
    names = list(feature_names)
    key_idx = [names.index(k) for k in KEY_COLS if k in names]
    lin_idx = [i for i, n in enumerate(names) if n not in KEY_COLS]

    def linear(X_fit, y_fit, X_eval):
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(X_fit[:, lin_idx], y_fit)
        return m.predict(X_fit[:, lin_idx]), m.predict(X_eval[:, lin_idx])

    if kind == "linear":
        return linear
    if not key_idx:
        raise ValueError("structural first stage needs calendar key columns")

    def structural(X_fit, y_fit, X_eval):
        return A._hierarchical_group_predict(X_fit[:, key_idx], y_fit, X_eval[:, key_idx])

    if kind == "structural":
        return structural

    def struct_linear(X_fit, y_fit, X_eval):
        from sklearn.linear_model import LinearRegression
        b_fit, b_eval = structural(X_fit, y_fit, X_eval)
        m = LinearRegression().fit(X_fit[:, lin_idx], y_fit - b_fit)
        return b_fit + m.predict(X_fit[:, lin_idx]), b_eval + m.predict(X_eval[:, lin_idx])

    if kind == "struct_linear":
        return struct_linear
    raise ValueError(kind)


# run_arlrx / run_stage1_only / run_arlrx_segmented look up `make_stage1` in the
# arlrx module namespace at call time, so this swap routes them to the calendar
# version without touching arlrx.py (exp05-exp08 remain bit-reproducible).
A.make_stage1 = make_stage1_cal


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def run(quick: bool):
    P.set_global_seed()
    cats = CATEGORIES[:2] if quick else CATEGORIES
    grid = P.GRID_XGB_PHARMA
    rows, data_by_key = [], {}
    t0 = time.time()
    for gran, path, sp in GRANS:
        raw = pd.read_csv(path)
        for cat in cats:
            for fs in P.FEATURE_SETS:
                base = P.build_pharma_dataset(raw, cat, fs, seasonal_period=sp,
                                              lag_rule="pacf_train")
                cal = add_calendar(base, gran)
                key = (gran, cat, fs)
                data_by_key[key] = base
                n_val = int(base.describe()["n_val"])
                n_folds = int(min(5, max(2, n_val // 30)))
                ex = {"granularity": gran, "config_feature_set": fs}

                def add(r):
                    r.update(ex)
                    rows.append(r)

                for r in P.naive_rows(base):
                    if r["model"] == "Naive":
                        add(r)
                add(A.run_stage1_only("S1 [linear] (exp05c)", base, "linear"))
                add(A.run_stage1_only("S1 [linear+cal]", cal, "linear"))
                add(A.run_stage1_only("S1 [structural]", cal, "structural"))
                add(A.run_stage1_only("S1 [struct_linear]", cal, "struct_linear"))
                add(P.run_model("XGBoost [+cal]", P.fp_xgboost, cal, grid))
                add(A.run_arlrx("AR-LRX [linear+cal]", cal, "linear", grid))
                add(A.run_arlrx("AR-LRX [struct_linear]", cal, "struct_linear", grid))
                add(A.run_arlrx_segmented("AR-LRX-Aug [struct_linear]", cal, "struct_linear",
                                          grid, schemes=("global",), n_folds=n_folds,
                                          augment_stage1=True))
                print(f"  {gran:6s} {cat:6s} {fs:7s} done ({(time.time()-t0)/60:.1f} min)",
                      flush=True)
    return rows, data_by_key


def analyse(rows, data_by_key):
    res = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    by = {(r["granularity"], r["category"], r["config_feature_set"], r["model"]): r for r in rows}
    REF = "S1 [linear] (exp05c)"

    # 1. reproduction check against the published exp05c baseline
    pub = pd.read_csv(ROOT / "results/exp05c_crossfit_crossfit.csv")
    pub = pub[pub.model == "S1 [linear]"].set_index(["granularity", "category", "feature_set"])
    diffs = []
    for (g, c, f, m), r in by.items():
        if m == REF and (g, c, f) in pub.index:
            diffs.append(abs(r["test_RMSE"] - pub.loc[(g, c, f), "test_RMSE"]))
    max_diff = max(diffs) if diffs else float("nan")
    print(f"\nReproduction of exp05c S1 [linear]: max |dRMSE| = {max_diff:.3e} over {len(diffs)} configs")

    # 2. every model against the reference and against S1 [linear+cal]
    dm_rows = []
    for (g, c, f), d in data_by_key.items():
        for m in sorted({k[3] for k in by if k[:3] == (g, c, f)}):
            for ref in (REF, "S1 [linear+cal]"):
                if m == ref or (g, c, f, ref) not in by:
                    continue
                a, b = by[(g, c, f, m)], by[(g, c, f, ref)]
                t = P.diebold_mariano(d.y_test, a["_test_pred"], b["_test_pred"])
                dm_rows.append({"granularity": g, "category": c, "feature_set": f,
                                "model": m, "reference": ref,
                                "rmse_model": a["test_RMSE"], "rmse_ref": b["test_RMSE"],
                                "delta_pct": 100 * (a["test_RMSE"] / b["test_RMSE"] - 1),
                                "DM": t["DM"], "p": t["p_value"]})
    dm = pd.DataFrame(dm_rows)
    # Benjamini-Hochberg within each (model, reference) family
    dm["p_bh"] = np.nan
    for _, idx in dm.groupby(["model", "reference"]).groups.items():
        p = dm.loc[idx, "p"].to_numpy()
        ok = np.isfinite(p)
        q = np.full(len(p), np.nan)
        if ok.any():
            pv = p[ok]; n = len(pv); order = np.argsort(pv)
            adj = pv[order] * n / np.arange(1, n + 1)
            adj = np.minimum.accumulate(adj[::-1])[::-1]
            tmp = np.empty(n); tmp[order] = np.minimum(adj, 1.0); q[ok] = tmp
        dm.loc[idx, "p_bh"] = q
    dm["distinct"] = True
    # configurations where k == 1 make A_lag1 and B_rich the same model (see IJIES 2.4)
    for (g, c, f), d in data_by_key.items():
        if f == P.FEATURE_SET_B and d.n_lags == 1:
            dm.loc[(dm.granularity == g) & (dm.category == c) & (dm.feature_set == f), "distinct"] = False

    summ = []
    for (m, ref, g), s in dm.groupby(["model", "reference", "granularity"]):
        for scope, ss in (("all", s), ("distinct", s[s.distinct])):
            ratio = ss.rmse_model / ss.rmse_ref
            summ.append({"model": m, "reference": ref, "granularity": g, "scope": scope,
                         "n": len(ss), "wins": int((ratio < 1).sum()),
                         "ties(|d|<0.1%)": int((ss.delta_pct.abs() < 0.1).sum()),
                         "sig_wins_bh": int(((ss.DM < 0) & (ss.p_bh < 0.05)).sum()),
                         "sig_losses_bh": int(((ss.DM > 0) & (ss.p_bh < 0.05)).sum()),
                         "pooled_pct": 100 * (np.exp(np.log(ratio).mean()) - 1),
                         "median_pct": float(ss.delta_pct.median())})
    summ = pd.DataFrame(summ)

    # 3. regime diagnostics for the gated models
    gate = res[res.model.str.startswith("AR-LRX")][
        ["granularity", "category", "config_feature_set", "model", "gate_w",
         "resid_val_r2", "stage1_only_test_RMSE", "test_RMSE"]]
    print("\nGate and residual regime:")
    for (m, g), s in gate.groupby(["model", "granularity"]):
        print(f"  {m:28s} {g:6s} gate closed {int((s.gate_w == 0).sum()):2d}/{len(s)} | "
              f"resid_val_r2 median {s.resid_val_r2.median():+.3f}, "
              f"> 0.05 in {int((s.resid_val_r2 > 0.05).sum())}/{len(s)}")
    return res, dm, summ, max_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    rows, data_by_key = run(args.quick)
    res, dm, summ, max_diff = analyse(rows, data_by_key)
    name = EXPERIMENT + ("_quick" if args.quick else "")
    P.save_results(rows, name)
    dm.to_csv(P.RESULTS_DIR / f"{name}_dm.csv", index=False)
    summ.to_csv(P.RESULTS_DIR / f"{name}_summary.csv", index=False)
    np.savez_compressed(
        P.RESULTS_DIR / f"{name}_predictions.npz",
        **{f"{r['granularity']}|{r['category']}|{r['config_feature_set']}|{r['model']}|test_pred":
           np.asarray(r["_test_pred"], dtype=np.float32) for r in rows})
    pd.set_option("display.width", 220)
    show = summ[(summ.scope == "all")].copy()
    show["pooled_pct"] = show.pooled_pct.round(2)
    show["median_pct"] = show.median_pct.round(2)
    print("\nSummary (negative pooled_pct = better than the reference):")
    print(show.drop(columns="scope").to_string(index=False))
    print(f"\nexp05c reproduction max |dRMSE| = {max_diff:.3e}")
    print(f"Total runtime {(time.time()-t0)/60:.1f} min -> results/{name}*.csv")


if __name__ == "__main__":
    main()
