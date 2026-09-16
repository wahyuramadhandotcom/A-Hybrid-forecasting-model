"""
exp08 - is the regime diagnostic a property of the residual, or of XGBoost?
(reviewer point R2-2, IJIES Paper ID 20265893, round 2)

The paper assigns a configuration to the "structured-residual" or the
"noise-residual" regime by the sign of `resid_val_r2`: the validation R^2 of the
XGBoost residual learner selected by the protocol. A negative value shows that
THIS learner failed; it does not show that no learner could succeed. This script
fits a range of residual learners of different inductive bias and capacity to
exactly the same cross-fitted training residuals and reports, per configuration,

  * the validation R^2 of every learner and hyperparameter setting;
  * the best validation R^2 over all of them (the most optimistic reading, which
    is biased upwards by the search itself);
  * for the setting that is best on validation within each learner, the R^2 on
    the TEST residuals after refitting on train+validation (never used for any
    choice).

Regime agreement: the XGBoost regime (sign of its best validation R^2) is
compared with the best validation R^2 of the five OTHER learners, both at 0 and
at a practical threshold of 0.01 (a best-of-many value below 1% of residual
variance is not exploitable signal). Ridge is included as a linear control; on
the residual of a linear first stage it can find nothing by construction.

Learners (scikit-learn / LightGBM / XGBoost):
  XGBoost       the protocol grid of the paper
  LightGBM      num_leaves {15, 63} x n_estimators {100, 400}, learning_rate 0.05
  RandomForest  max_depth {6, None} x min_samples_leaf {5, 50}, 200 trees
  kNN           k {10, 50, 200}, standardised inputs
  Ridge         alpha {0.1, 10, 1000}, standardised inputs
  MLP           hidden (32,), (64, 32); early stopping; standardised inputs

Rossmann: 3 feature variants x 3 first stages. Alternative learners are fitted on
a random subsample of 100,000 training rows (seed 42) to keep the run
tractable; XGBoost uses the full block, exactly as in exp05b. PharmaSales: the
20 distinct configurations (B_rich duplicates of A_lag1 are skipped), linear
first stage, full blocks.

Usage (repository root):
    python tools/exp08_learner_stability.py --part pharma      # ~10 min
    python tools/exp08_learner_stability.py --part rossmann    # ~1-2 h
    python tools/exp08_learner_stability.py --part summary
    --quick  smoke test into results/_exp08_quick/

Outputs: results/exp08_learner_stability.csv (one row per configuration x
learner x setting), results/exp08_summary.csv (one row per configuration).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from src.experiments import arlrx as A          # noqa: E402
from src.experiments import protocol as P       # noqa: E402

CATEGORIES = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
GRANS = [("daily", "data/raw/pharma-sales/salesdaily.csv", 7),
         ("weekly", "data/raw/pharma-sales/salesweekly.csv", 52)]
VARIANTS = ["V1_customers_dropped", "V2_customers_lagged", "V3_sales_lagged"]
ROSS_SUBSAMPLE = 100_000
RES = {"dir": os.path.join(ROOT, "results")}


def out(name):
    os.makedirs(RES["dir"], exist_ok=True)
    return os.path.join(RES["dir"], name)


def r2(y, p):
    y = np.asarray(y, float)
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - np.asarray(p, float)) ** 2)) / ss if ss > 0 else np.nan


# --------------------------------------------------------------------------- #
# learners: name -> list of (setting-label, factory)
# --------------------------------------------------------------------------- #

def learners(xgb_grid, quick):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    S = P.SEED
    L = {"XGBoost": [(json.dumps(p, sort_keys=True), (lambda p=p: P.make_xgb(p)))
                     for p in P.param_grid_list(xgb_grid)]}

    def lgbm(leaves, n):
        import lightgbm as lgb
        return lgb.LGBMRegressor(num_leaves=leaves, n_estimators=n, learning_rate=0.05,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 random_state=S, n_jobs=-1, verbose=-1)
    L["LightGBM"] = [(f"leaves={a},n={b}", (lambda a=a, b=b: lgbm(a, b)))
                     for a, b in product([15, 63], [100, 400] if not quick else [100])]
    L["RandomForest"] = [(f"depth={a},leaf={b}",
                          (lambda a=a, b=b: RandomForestRegressor(
                              n_estimators=200 if not quick else 30, max_depth=a,
                              min_samples_leaf=b, max_features=0.5, random_state=S, n_jobs=-1)))
                         for a, b in product([6, None], [5, 50])]
    L["kNN"] = [(f"k={k}", (lambda k=k: make_pipeline(StandardScaler(),
                                                      KNeighborsRegressor(n_neighbors=k))))
                for k in [10, 50, 200]]
    L["Ridge"] = [(f"alpha={a}", (lambda a=a: make_pipeline(StandardScaler(), Ridge(alpha=a))))
                  for a in [0.1, 10, 1000]]
    L["MLP"] = [(f"hidden={h}", (lambda h=h: make_pipeline(
        StandardScaler(), MLPRegressor(hidden_layer_sizes=h, early_stopping=True,
                                       max_iter=300 if not quick else 50,
                                       random_state=S))))
                for h in [(32,), (64, 32)]]
    return L


def run_config(key, d, stage1_kind, xgb_grid, subsample, quick, done):
    """All learners on one configuration; returns rows."""
    if key in done:
        print(f"    {key} already done - skipped", flush=True)
        return []
    st1 = A.make_stage1(stage1_kind, d.feature_names)
    Xtr, ytr, Xva, yva = d.X_train, d.y_train, d.X_val, d.y_val
    Xtv, ytv, Xte, yte = d.X_trainval, d.y_trainval, d.X_test, d.y_test
    _, s1_va = st1(Xtr, ytr, Xva)
    r_tr = ytr - A._cross_fitted_stage1(st1, Xtr, ytr)
    r_va = yva - s1_va
    _, s1_te = st1(Xtv, ytv, Xte)
    r_tv = ytv - A._cross_fitted_stage1(st1, Xtv, ytv)
    r_te = yte - s1_te

    rng = np.random.default_rng(P.SEED)
    sub_tr = (np.sort(rng.choice(len(r_tr), subsample, replace=False))
              if subsample and len(r_tr) > subsample else slice(None))
    sub_tv = (np.sort(rng.choice(len(r_tv), subsample, replace=False))
              if subsample and len(r_tv) > subsample else slice(None))

    rows = []
    for lname, settings in learners(xgb_grid, quick).items():
        use_sub = lname != "XGBoost"
        best = (-np.inf, None, None, None)
        for label, make in settings:
            t0 = time.time()
            idx = sub_tr if use_sub else slice(None)
            m = make().fit(Xtr[idx], r_tr[idx])
            v = r2(r_va, m.predict(Xva))
            rows.append({"config": key, "stage1": stage1_kind, "learner": lname, "setting": label,
                         "subsampled": bool(use_sub and subsample and len(r_tr) > subsample),
                         "val_r2": v, "test_r2_refit": np.nan, "is_best_on_val": False,
                         "runtime_s": round(time.time() - t0, 2)})
            if v > best[0]:
                best = (v, label, make, len(rows) - 1)
        if best[1] is not None:
            idx = sub_tv if use_sub else slice(None)
            m = best[2]().fit(Xtv[idx], r_tv[idx])
            rows[best[3]]["test_r2_refit"] = r2(r_te, m.predict(Xte))
            rows[best[3]]["is_best_on_val"] = True
        print(f"    {key} {lname:12s} best val R2 {best[0]:+.4f}", flush=True)
    return rows


def append(rows, name):
    f = out(name)
    prev = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
    pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(f + ".tmp", index=False)
    os.replace(f + ".tmp", f)


def done_set(name):
    f = out(name)
    if not os.path.exists(f):
        return set()
    d = pd.read_csv(f)
    return set(d.config)


def part_pharma(quick):
    name = "exp08_learner_stability.csv"
    done = done_set(name)
    grid = ({"n_estimators": [100], "max_depth": [3], "learning_rate": [0.1],
             "subsample": [0.8], "colsample_bytree": [0.8]} if quick else P.GRID_XGB_PHARMA)
    cats = CATEGORIES[:2] if quick else CATEGORIES
    for gran, path, sp in GRANS:
        data = pd.read_csv(os.path.join(ROOT, path))
        for cat in cats:
            dA = P.build_pharma_dataset(data, cat, "A_lag1", seasonal_period=sp)
            dB = P.build_pharma_dataset(data, cat, "B_rich", seasonal_period=sp)
            pairs = [("A_lag1", dA)]
            if dB.feature_names != dA.feature_names:        # skip exact duplicates
                pairs.append(("B_rich", dB))
            for fs, d in pairs:
                key = f"pharma|{gran}|{cat}|{fs}"
                rows = run_config(key, d, "linear", grid, None, quick, done)
                if rows:
                    append(rows, name)


def part_rossmann(quick):
    name = "exp08_learner_stability.csv"
    done = done_set(name)
    frame = P.build_rossmann_frame(os.path.join(ROOT, "data/raw/rossmann/train.csv"),
                                   os.path.join(ROOT, "data/raw/rossmann/store.csv"))
    grid = A.GRID_XGB_ARLRX
    if quick:
        keep = np.sort(frame["Store"].unique())[:80]
        frame = frame[frame["Store"].isin(keep)].reset_index(drop=True)
        grid = {"n_estimators": [300], "max_depth": [6], "learning_rate": [0.1],
                "subsample": [0.8], "colsample_bytree": [0.8], "max_bin": [256]}
    for v in VARIANTS:
        d = P.build_rossmann_dataset(frame, v, target="log1p")
        for kind in A.STAGE1_KINDS:
            key = f"rossmann|{v}|{kind}"
            rows = run_config(key, d, kind, grid, ROSS_SUBSAMPLE if not quick else 20_000,
                              quick, done)
            if rows:
                append(rows, name)


def part_summary(threshold=0.01):
    f = out("exp08_learner_stability.csv")
    d = pd.read_csv(f)
    rows = []
    for key, g in d.groupby("config", sort=False):
        best = g[g.is_best_on_val].set_index("learner")
        xgb = float(g[g.learner == "XGBoost"].val_r2.max())
        other = g[g.learner != "XGBoost"]
        o_max = float(other.val_r2.max())
        o_arg = other.loc[other.val_r2.idxmax(), "learner"]
        o_test = float(best.drop(index="XGBoost", errors="ignore").test_r2_refit.max())
        row = {"config": key, "dataset": key.split("|")[0], "stage1": g.stage1.iloc[0],
               "xgb_best_val_r2": xgb,
               "xgb_test_r2": float(best.loc["XGBoost", "test_r2_refit"]) if "XGBoost" in best.index else np.nan,
               "other_best_val_r2": o_max, "other_best_learner": o_arg,
               "other_best_test_r2": o_test,
               "n_learners_val_r2_gt0": int((best.val_r2 > 0).sum()),
               "n_learners_val_r2_gt_thr": int((best.val_r2 > threshold).sum()),
               "n_learners": int(len(best))}
        for ln, b in best.iterrows():
            row[f"val_r2 | {ln}"] = b.val_r2
            row[f"test_r2 | {ln}"] = b.test_r2_refit
        row["regime_xgb"] = "structured" if xgb > 0 else "noise"
        row["agree_at_0"] = (xgb > 0) == (o_max > 0)
        row["agree_at_thr"] = (xgb > threshold) == (o_max > threshold)
        row["agree_on_test_at_thr"] = (row["xgb_test_r2"] > threshold) == (o_test > threshold)
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(out("exp08_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["config", "regime_xgb", "xgb_best_val_r2", "other_best_val_r2", "other_best_learner",
            "xgb_test_r2", "other_best_test_r2", "n_learners_val_r2_gt0", "agree_at_0",
            "agree_at_thr", "agree_on_test_at_thr"]
    print(s[cols].round(4).to_string(index=False))
    for ds, g in s.groupby("dataset"):
        print(f"\n{ds}: {len(g)} configurations | XGBoost 'structured' in "
              f"{int((g.regime_xgb == 'structured').sum())} | best other learner > 0 in "
              f"{int((g.other_best_val_r2 > 0).sum())}, > {threshold} in "
              f"{int((g.other_best_val_r2 > threshold).sum())} (max {g.other_best_val_r2.max():.4f}) | "
              f"regime agrees at 0: {int(g.agree_at_0.sum())}, at {threshold}: "
              f"{int(g.agree_at_thr.sum())}, on test at {threshold}: {int(g.agree_on_test_at_thr.sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["pharma", "rossmann", "summary"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        RES["dir"] = os.path.join(ROOT, "results", "_exp08_quick")
    P.set_global_seed()
    print("Environment:", P.environment_stamp(), "| part", a.part, "| quick", a.quick, flush=True)
    t0 = time.time()
    {"pharma": lambda: part_pharma(a.quick), "rossmann": lambda: part_rossmann(a.quick),
     "summary": part_summary}[a.part]()
    print(f"\nfinished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
