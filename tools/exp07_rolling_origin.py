"""
exp07 - rolling-origin and multi-seed replications (reviewer points R1-6, R2-1,
R2-4; IJIES Paper ID 20265893, round 2).

The main results rest on one chronological train/validation/test split per
dataset and one random seed. This script repeats the tree-based part of the
study

  (a) over several forecast origins (expanding training window, fixed-length
      validation and test windows that move back in time and never overlap),
  (b) over several random seeds on the primary split,

under exactly the protocol of the paper (cross-fitted residuals, tuning on
validation only, refit on train+validation, test touched once). The neural
baselines are not repeated (their cost is two orders of magnitude higher and
they are not bit-reproducible on CPU); the tree baselines are.

It also answers R2-4 directly: for every AR-LRX run it records, next to the
validation diagnostic `resid_val_r2` that the gate is selected on, the same
quantity on the TEST block (`resid_test_r2`: R^2 of the fitted residual
learner on test residuals), which is never used for any selection.

Parts (run in any order; each part checkpoints row by row and skips rows that
are already in its result file, so an interrupted run can be resumed):

    python tools/exp07_rolling_origin.py --part pharma-origins      # ~1 h
    python tools/exp07_rolling_origin.py --part pharma-seeds        # ~1 h
    python tools/exp07_rolling_origin.py --part rossmann-origins    # ~2 h
    python tools/exp07_rolling_origin.py --part rossmann-seeds      # ~2 h
    python tools/exp07_rolling_origin.py --part summary             # seconds

    --redo-model NAME [NAME ...]   discard the saved rows of these models and
              re-run only them (used after the LightGBM bagging fix:
              --redo-model "LightGBM (Zeng)")
    --quick   smoke test (80 stores / 2 categories, one small grid); writes to
              results/_exp07_quick/ and never touches the real files.

Design
  Rossmann (V3, log1p target): 4 origins; test windows of 42 days ending
    2015-07-31, 2015-06-19, 2015-05-08, 2015-03-27; validation = the 84 days
    before each test window; training = everything before that.
  PharmaSales: 4 origins per series; test = last 90 days (daily) / 13 weeks
    (weekly) of the truncated series, validation = the 180 days / 26 weeks
    before it; the lag count is selected on that origin's training block only.
  Seeds: 42 (the paper), 1, 2, 3, 4. Seed 42 on the primary split is not
    re-run for Rossmann (it is exp05b/exp05d); for PharmaSales it is re-run
    and must reproduce exp05c exactly (a built-in consistency check).

Outputs (results/):
  exp07_rossmann_origins.csv, exp07_rossmann_seeds.csv,
  exp07_pharma_origins.csv, exp07_pharma_seeds.csv,
  exp07_*_predictions/ (one npz per origin or seed),
  exp07_summary_*.csv
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
warnings.filterwarnings("ignore")

from src.experiments import arlrx as A          # noqa: E402
from src.experiments import baselines as B      # noqa: E402
from src.experiments import protocol as P       # noqa: E402

VARIANT = "V3_sales_lagged"
ROSS_TEST_DAYS, ROSS_VAL_DAYS, ROSS_ORIGINS = 42, 84, 4
PH_TEST = {"daily": 90, "weekly": 13}
PH_VAL = {"daily": 180, "weekly": 26}
PH_ORIGINS = 4
SEEDS = [42, 1, 2, 3, 4]
CATEGORIES = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
GRANS = [("daily", "data/raw/pharma-sales/salesdaily.csv", 7),
         ("weekly", "data/raw/pharma-sales/salesweekly.csv", 52)]


class Ctx:
    quick = False
    redo = []          # model names whose existing rows are discarded and re-run
    res = os.path.join(ROOT, "results")

    @classmethod
    def path(cls, name):
        os.makedirs(cls.res, exist_ok=True)
        return os.path.join(cls.res, name)


def set_seed(seed: int) -> None:
    """All tree learners read protocol.SEED at call time."""
    P.SEED = int(seed)
    P.set_global_seed(int(seed))


# --------------------------------------------------------------------------- #
# bookkeeping
# --------------------------------------------------------------------------- #

def resid_test_r2(row, y_test):
    """R^2 of the fitted residual learner on TEST residuals (never used for selection)."""
    if "_stage1_test_raw" not in row or "_test_correction" not in row:
        return np.nan
    r = np.asarray(y_test, float) - np.asarray(row["_stage1_test_raw"], float)
    c = np.asarray(row["_test_correction"], float)
    ss = float(np.sum((r - r.mean()) ** 2))
    return 1.0 - float(np.sum((r - c) ** 2)) / ss if ss > 0 else np.nan


class Store:
    """Append-only result csv + one npz of test predictions per group."""

    def __init__(self, name):
        self.csv = Ctx.path(f"{name}.csv")
        self.pdir = Ctx.path(f"{name}_predictions")
        os.makedirs(self.pdir, exist_ok=True)
        self.done = set()
        if os.path.exists(self.csv):
            d = pd.read_csv(self.csv)
            if Ctx.redo and d.model.isin(Ctx.redo).any():
                n = int(d.model.isin(Ctx.redo).sum())
                d = d[~d.model.isin(Ctx.redo)]
                d.to_csv(self.csv, index=False)
                print(f"  --redo-model: removed {n} existing rows of {Ctx.redo} from "
                      f"{os.path.basename(self.csv)}", flush=True)
            self.done = set(zip(d.group.astype(str), d.config.astype(str), d.model.astype(str)))

    def has(self, group, config, model):
        return (str(group), str(config), str(model)) in self.done

    def add(self, row, group, config, y_test):
        row = dict(row)
        row.update({"group": str(group), "config": str(config),
                    "resid_test_r2": resid_test_r2(row, y_test),
                    "cross_fit_folds_protocol": A.N_CROSS_FIT_FOLDS})
        arrays = {k[1:]: np.asarray(v, dtype=np.float32) for k, v in row.items()
                  if k.startswith("_") and isinstance(v, np.ndarray)}
        flat = {k: v for k, v in row.items() if not k.startswith("_")}
        # rows of different model types carry different columns: rewrite the whole
        # file with the union of columns (a plain append would misalign them)
        prev = pd.read_csv(self.csv) if os.path.exists(self.csv) else pd.DataFrame()
        tmp = self.csv + ".tmp"
        pd.concat([prev, pd.DataFrame([flat])], ignore_index=True).to_csv(tmp, index=False)
        os.replace(tmp, self.csv)
        f = os.path.join(self.pdir, f"{group}.npz")
        old = dict(np.load(f)) if os.path.exists(f) else {}
        old[f"{config}|y_test"] = np.asarray(y_test, dtype=np.float32)
        for k, v in arrays.items():
            old[f"{config}|{row['model']}|{k}"] = v
        np.savez_compressed(f, **old)
        self.done.add((str(group), str(config), str(row["model"])))


# --------------------------------------------------------------------------- #
# Rossmann
# --------------------------------------------------------------------------- #

def rossmann_frame():
    frame = P.build_rossmann_frame(os.path.join(ROOT, "data/raw/rossmann/train.csv"),
                                   os.path.join(ROOT, "data/raw/rossmann/store.csv"))
    if Ctx.quick:
        keep = np.sort(frame["Store"].unique())[:80]
        frame = frame[frame["Store"].isin(keep)].reset_index(drop=True)
    return frame


def rossmann_origin_dataset(frame, k):
    """Origin k (0 = latest): expanding train, 84-day validation, 42-day test."""
    base = P.build_rossmann_dataset(frame, VARIANT, target="log1p")
    dates = pd.to_datetime(base.frame["Date"])
    last = dates.max()
    test_end = last - pd.Timedelta(days=ROSS_TEST_DAYS * k)
    test_start = test_end - pd.Timedelta(days=ROSS_TEST_DAYS - 1)
    val_start = test_start - pd.Timedelta(days=ROSS_VAL_DAYS)
    keep = (dates <= test_end).to_numpy()
    fr = base.frame.loc[keep].reset_index(drop=True)
    dd = pd.to_datetime(fr["Date"])
    ds = dataclasses.replace(base, frame=fr,
                             i_train_end=int((dd < val_start).sum()),
                             i_val_end=int((dd < test_start).sum()))
    return ds, {"origin": k, "train_end": str((val_start - pd.Timedelta(days=1)).date()),
                "val_start": str(val_start.date()), "test_start": str(test_start.date()),
                "test_end": str(test_end.date())}


def rossmann_models(quick):
    g = ({"n_estimators": [300], "max_depth": [6], "learning_rate": [0.1],
          "subsample": [0.8], "colsample_bytree": [0.8], "max_bin": [256]}
         if quick else A.GRID_XGB_ARLRX)
    gl = {"n_estimators": [300], "num_leaves": [31], "learning_rate": [0.1],
          "subsample": [0.8], "colsample_bytree": [0.8]} if quick else B.GRID_LIGHTGBM
    gz = g if quick else B.GRID_XGB_ZHAO
    inv = np.expm1
    return [
        ("SeasonalNaive(store x dow x promo median)", False,
         lambda d: P.rossmann_seasonal_naive(d, inverse_transform=inv)),
        ("S1 [structural]", False,
         lambda d: A.run_stage1_only("S1 [structural]", d, "structural", inverse_transform=inv)),
        ("AR-LRX [struct_linear]", True,
         lambda d: A.run_arlrx("AR-LRX [struct_linear]", d, "struct_linear", g, inverse_transform=inv)),
        ("AR-LRX-Aug [structural]", True,
         lambda d: A.run_arlrx_segmented("AR-LRX-Aug [structural]", d, "structural", g,
                                         schemes=("global",), augment_stage1=True,
                                         inverse_transform=inv)),
        ("AR-LRX-Aug [struct_linear]", True,
         lambda d: A.run_arlrx_segmented("AR-LRX-Aug [struct_linear]", d, "struct_linear", g,
                                         schemes=("global",), augment_stage1=True,
                                         inverse_transform=inv)),
        ("LR-XGB (residual, tanpa gerbang)", True,
         lambda d: P.run_model("LR-XGB (residual, tanpa gerbang)", P.fp_lr_xgb_residual, d, g,
                               inverse_transform=inv,
                               extra={"cross_fit_folds": P.N_CROSS_FIT_FOLDS})),
        ("XGBoost", True,
         lambda d: P.run_model("XGBoost", P.fp_xgboost, d, g, inverse_transform=inv)),
        ("LightGBM (Zeng)", True,
         lambda d: P.run_model("LightGBM (Zeng)", B.fp_lightgbm, d, gl, inverse_transform=inv)),
        ("XGBoost (grid Zhao)", True,
         lambda d: P.run_model("XGBoost (grid Zhao)", B.fp_xgboost_zhao, d, gz,
                               inverse_transform=inv)),
    ]


def part_rossmann_origins():
    frame = rossmann_frame()
    st = Store("exp07_rossmann_origins")
    set_seed(42)
    for k in range(ROSS_ORIGINS):
        d, meta = rossmann_origin_dataset(frame, k)
        print(f"origin {k}: train <= {meta['train_end']}, val {meta['val_start']}.., "
              f"test {meta['test_start']}..{meta['test_end']} | n_test {len(d.y_test)}", flush=True)
        for name, _, fn in rossmann_models(Ctx.quick):
            if st.has(f"origin{k}", VARIANT, name):
                continue
            t0 = time.time()
            r = fn(d)
            r.update({f"origin_{a}": b for a, b in meta.items()})
            st.add(r, f"origin{k}", VARIANT, d.y_test)
            print(f"  {name:36s} orig RMSE {r.get('orig_RMSE', np.nan):9.2f}  "
                  f"w* {r.get('gate_w', np.nan)}  [{(time.time()-t0)/60:.1f} min]", flush=True)


def part_rossmann_seeds():
    frame = rossmann_frame()
    d = P.build_rossmann_dataset(frame, VARIANT, target="log1p")
    st = Store("exp07_rossmann_seeds")
    for seed in SEEDS:
        if seed == 42 and not Ctx.quick:
            continue                       # = exp05b / exp05d (the paper)
        set_seed(seed)
        for name, stochastic, fn in rossmann_models(Ctx.quick):
            if not stochastic or st.has(f"seed{seed}", VARIANT, name):
                continue
            t0 = time.time()
            r = fn(d)
            st.add(r, f"seed{seed}", VARIANT, d.y_test)
            print(f"seed {seed} {name:36s} orig RMSE {r['orig_RMSE']:9.2f}  "
                  f"[{(time.time()-t0)/60:.1f} min]", flush=True)
    set_seed(42)


# --------------------------------------------------------------------------- #
# PharmaSales
# --------------------------------------------------------------------------- #

def pharma_models(d, n_folds, quick):
    g = ({"n_estimators": [100], "max_depth": [3], "learning_rate": [0.1],
          "subsample": [0.8], "colsample_bytree": [0.8]} if quick else P.GRID_XGB_PHARMA)
    return [
        ("Naive", False, None),
        ("S1 [linear]", False, lambda: A.run_stage1_only("S1 [linear]", d, "linear")),
        ("AR-LRX [linear]", True, lambda: A.run_arlrx("AR-LRX [linear]", d, "linear", g)),
        ("AR-LRX-g [linear]", True,
         lambda: A.run_arlrx_segmented("AR-LRX-g [linear]", d, "linear", g, schemes=("global",),
                                       n_folds=n_folds, augment_stage1=False)),
        ("AR-LRX-Aug [linear]", True,
         lambda: A.run_arlrx_segmented("AR-LRX-Aug [linear]", d, "linear", g, schemes=("global",),
                                       n_folds=n_folds, augment_stage1=True)),
        ("XGBoost", True, lambda: P.run_model("XGBoost", P.fp_xgboost, d, g)),
        ("LR-XGB (residual, tanpa gerbang)", True,
         lambda: P.run_model("LR-XGB (residual, tanpa gerbang)", P.fp_lr_xgb_residual, d, g,
                             extra={"cross_fit_folds": P.N_CROSS_FIT_FOLDS})),
    ]


def pharma_primary_datasets():
    out = {}
    cats = CATEGORIES[:2] if Ctx.quick else CATEGORIES
    for gran, path, sp in GRANS:
        data = pd.read_csv(os.path.join(ROOT, path))
        for cat in cats:
            for fs in P.FEATURE_SETS:
                out[(gran, cat, fs)] = P.build_pharma_dataset(data, cat, fs, seasonal_period=sp,
                                                              lag_rule="pacf_train")
    return out


def pharma_origin_datasets(k):
    """Origin k: series truncated by k test windows; lag count from its training block."""
    out = {}
    cats = CATEGORIES[:2] if Ctx.quick else CATEGORIES
    for gran, path, sp in GRANS:
        data = pd.read_csv(os.path.join(ROOT, path))
        test_len, val_len = PH_TEST[gran], PH_VAL[gran]
        for cat in cats:
            n_raw = len(data)
            trunc = data.iloc[: n_raw - k * test_len].reset_index(drop=True)
            ser = (trunc[["datum", cat]].assign(ds=lambda x: pd.to_datetime(
                x["datum"], format="mixed", dayfirst=False, errors="coerce"))
                   .sort_values("ds")[cat].to_numpy(dtype=float))
            n_train_raw = len(ser) - test_len - val_len
            lag = max(1, P.select_lag(ser[:n_train_raw], rule="pacf_train",
                                      ratios=(1.0, 0.0, 0.0)))
            for fs in P.FEATURE_SETS:
                d = P.build_pharma_dataset(trunc, cat, fs, seasonal_period=sp,
                                           lag_rule="pacf_train", n_lags_override=lag)
                n = len(d.frame)
                d = dataclasses.replace(d, i_train_end=n - test_len - val_len,
                                        i_val_end=n - test_len)
                out[(gran, cat, fs)] = d
    return out


def run_pharma_group(st, group, datasets):
    for (gran, cat, fs), d in datasets.items():
        cfg = f"{gran}|{cat}|{fs}"
        n_val = int(d.describe()["n_val"])
        n_folds = int(min(5, max(2, n_val // 30)))
        extra = {"granularity": gran, "n_folds": n_folds, "test_start": d.describe()["test_start"],
                 "test_end": d.describe()["test_end"]}
        for name, _, fn in pharma_models(d, n_folds, Ctx.quick):
            if st.has(group, cfg, name):
                continue
            if name == "Naive":
                for r in P.naive_rows(d):
                    st.add({**r, **extra}, group, cfg, d.y_test)
                continue
            st.add({**fn(), **extra}, group, cfg, d.y_test)
        print(f"  {group} {cfg} done", flush=True)


def part_pharma_origins():
    st = Store("exp07_pharma_origins")
    set_seed(42)
    for k in range(PH_ORIGINS):
        run_pharma_group(st, f"origin{k}", pharma_origin_datasets(k))


def part_pharma_seeds():
    st = Store("exp07_pharma_seeds")
    ds = pharma_primary_datasets()
    for seed in SEEDS:
        set_seed(seed)
        run_pharma_group(st, f"seed{seed}", ds)
    set_seed(42)


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #

def _rel(a, b):
    return 100.0 * (a / b - 1.0)


def summarise_rossmann(name):
    f = Ctx.path(f"{name}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    if name.endswith("seeds") and not Ctx.quick:
        # seed 42 = the paper's run (exp05b / exp05d)
        refs = []
        for src in ("exp05b_rossmann_arlrx_audit", "exp05d_rossmann_arlrx_dev",
                    "exp06b_rossmann_baselines_strong"):
            p = Ctx.path(f"{src}.csv")
            if os.path.exists(p):
                x = pd.read_csv(p)
                if "feature_set" in x:
                    x = x[x.feature_set == VARIANT]
                refs.append(x)
        ref = pd.concat(refs, ignore_index=True)
        keep = set(d.model)
        ref = ref[ref.model.isin(keep)].drop_duplicates("model").assign(group="seed42", config=VARIANT)
        d = pd.concat([d, ref], ignore_index=True)
    rows = []
    for grp, g in d.groupby("group"):
        base = g.set_index("model")["orig_RMSE"]
        best = base.idxmin()
        prop = base.get("AR-LRX-Aug [structural]", np.nan)
        row = {"group": grp, "best_model": best,
               "rank_AR-LRX-Aug_structural": int((base < prop).sum() + 1)}
        for m, v in base.items():
            row[f"RMSE | {m}"] = v
            if m != "AR-LRX-Aug [structural]":
                row[f"AugStruct vs {m} (%)"] = _rel(prop, v)
        for m in ("AR-LRX-Aug [structural]", "AR-LRX-Aug [struct_linear]", "AR-LRX [struct_linear]"):
            s = g[g.model == m]
            if len(s):
                row[f"w* | {m}"] = s.gate_w.iloc[0]
                row[f"resid_val_r2 | {m}"] = s.resid_val_r2.iloc[0]
                row[f"resid_test_r2 | {m}"] = s.get("resid_test_r2", pd.Series([np.nan])).iloc[0]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(Ctx.path(f"exp07_summary_{name.split('_', 1)[1]}.csv"), index=False)
    return out


def summarise_pharma(name):
    f = Ctx.path(f"{name}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    rows = []
    for grp, g in d.groupby("group"):
        k = g.config.str.split("|", expand=True)
        g = g.assign(gran=k[0], cat=k[1], fs=k[2])
        piv = g.pivot_table(index="config", columns="model", values="test_RMSE")
        ar = g[g.model == "AR-LRX [linear]"].set_index("config")
        # duplicate configurations (B_rich identical to A_lag1)
        dup = set()
        for cfg in piv.index:
            if cfg.endswith("|B_rich"):
                a = cfg.replace("|B_rich", "|A_lag1")
                if a in piv.index and np.allclose(piv.loc[cfg].values, piv.loc[a].values,
                                                  equal_nan=True, rtol=0, atol=0):
                    dup.add(cfg)
        s1 = piv["S1 [linear]"]
        row = {"group": grp, "n_configs": len(piv), "n_duplicate_configs": len(dup),
               "gate_closed": int((ar.gate_w.round(3) == 0).sum()),
               "mean_w": float(ar.gate_w.mean()),
               "resid_val_r2_negative": int((ar.resid_val_r2 < 0).sum()),
               "resid_val_r2_median": float(ar.resid_val_r2.median()),
               "resid_test_r2_negative": int((ar.resid_test_r2 < 0).sum()),
               "resid_test_r2_median": float(ar.resid_test_r2.median()),
               "regime_agree_val_test": int(((ar.resid_val_r2 < 0) == (ar.resid_test_r2 < 0)).sum())}
        for m, lab in [("LR-XGB (residual, tanpa gerbang)", "ungated"),
                       ("AR-LRX [linear]", "gated"), ("AR-LRX-Aug [linear]", "gated_aug")]:
            v = _rel(piv[m], s1)
            row[f"{lab}_worse_than_S1"] = int((v > 1e-9).sum())
            row[f"{lab}_mean_pct"] = float(v.mean())
            row[f"{lab}_worst_pct"] = float(v.max())
        for m in ("Naive", "XGBoost", "LR-XGB (residual, tanpa gerbang)"):
            v = _rel(piv["AR-LRX [linear]"], piv[m])
            row[f"AR-LRX_beats_{m}"] = int((v < 0).sum())
            row[f"AR-LRX_vs_{m}_geomean_pct"] = float(100 * (np.exp(np.log(1 + v / 100).mean()) - 1))
        vd = piv.drop(index=list(dup))
        row["distinct_gate_closed"] = int((ar.drop(index=list(dup)).gate_w.round(3) == 0).sum())
        row["distinct_gated_worse_than_S1"] = int((_rel(vd["AR-LRX [linear]"], vd["S1 [linear]"]) > 1e-9).sum())
        row["distinct_ungated_worse_than_S1"] = int(
            (_rel(vd["LR-XGB (residual, tanpa gerbang)"], vd["S1 [linear]"]) > 1e-9).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(Ctx.path(f"exp07_summary_{name.split('_', 1)[1]}.csv"), index=False)
    return out


def check_seed42_pharma():
    """The seed-42 pharma run must reproduce exp05c exactly."""
    f, ref = Ctx.path("exp07_pharma_seeds.csv"), Ctx.path("exp05c_pharma_arlrx.csv")
    if Ctx.quick or not (os.path.exists(f) and os.path.exists(ref)):
        return None
    d = pd.read_csv(f)
    d = d[d.group == "seed42"]
    k = d.config.str.split("|", expand=True)
    d = d.assign(granularity=k[0], category=k[1], feature_set=k[2])
    r = pd.read_csv(ref)
    m = d.merge(r, on=["granularity", "category", "feature_set", "model"], suffixes=("", "_ref"))
    return float((m.test_RMSE - m.test_RMSE_ref).abs().max()), len(m)


def part_summary():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    for name in ("exp07_rossmann_origins", "exp07_rossmann_seeds"):
        s = summarise_rossmann(name)
        if s is not None:
            print(f"\n== {name} ==")
            print(s.round(3).T.to_string())
    for name in ("exp07_pharma_origins", "exp07_pharma_seeds"):
        s = summarise_pharma(name)
        if s is not None:
            print(f"\n== {name} ==")
            print(s.round(3).T.to_string())
    chk = check_seed42_pharma()
    if chk:
        print(f"\npharma seed 42 vs exp05c: max |test_RMSE diff| = {chk[0]:.2e} over {chk[1]} rows "
              f"-> {'REPRODUCED' if chk[0] < 1e-9 else 'DIFFERS'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["rossmann-origins", "rossmann-seeds", "pharma-origins",
                             "pharma-seeds", "summary"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--redo-model", nargs="+", default=[],
                    help="discard existing rows of these models and run them again "
                         "(e.g. after a baseline fix); other rows are kept")
    a = ap.parse_args()
    Ctx.redo = list(a.redo_model)
    if a.quick:
        Ctx.quick = True
        Ctx.res = os.path.join(ROOT, "results", "_exp07_quick")
    print("Environment:", P.environment_stamp(), "| part", a.part, "| quick", a.quick, flush=True)
    t0 = time.time()
    {"rossmann-origins": part_rossmann_origins, "rossmann-seeds": part_rossmann_seeds,
     "pharma-origins": part_pharma_origins, "pharma-seeds": part_pharma_seeds,
     "summary": part_summary}[a.part]()
    print(f"\npart {a.part} finished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
