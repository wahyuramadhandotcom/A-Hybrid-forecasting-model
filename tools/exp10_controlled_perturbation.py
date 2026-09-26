"""
exp10 -- Controlled perturbation tests: injected noise and demand shocks.

Why
---
The approved proposal planned a robustness test under controlled noise and demand
spikes; it had not been run. It also answers the main limitation of the regime
analysis: each residual regime is represented by one dataset, so the regime was
never manipulated WITHIN a dataset. Injecting noise into Rossmann moves it, under
our control, from the structured-residual regime towards the noise regime, with
everything else held fixed. Pre-registered in PREREG_exp10_exp13_exp14.md
(predictions P6-P9) before any full run of this script.

Design (contract C1-C8 unchanged; the perturbation is applied to the RAW demand
series before feature construction, so lag and rolling features see it exactly as
a forecaster would)
---------------------------------------------------------------------------------
Noise (measurement / recording error; whole series: train, validation and test)
  PharmaSales  y' = max(0, y + e),  e ~ N(0, (alpha * sd_train(y))^2)
               alpha in {0.10, 0.25, 0.50}, 2 independent realisations each;
               the lag count k is held at its clean value so that only the data change.
  Rossmann     Sales' = Sales * exp(e),  e ~ N(0, (alpha * sd_train(log1p Sales))^2)
               on open days with positive sales; alpha in {0.10, 0.25, 0.50}, 1 realisation.
Shocks (test block only; the models and the gate are fitted on unperturbed data)
  spike        demand x 1.5 over a window: PharmaSales 14 days starting 30 days after
               the test start (weekly: 4 weeks starting 8 weeks after); Rossmann 14 days
               starting 28 days after the test start, in a random 25% of the stores.
  level        demand x 1.3 from the middle of the test block to its end
               (Rossmann: the same 25% of the stores).
A clean run (alpha = 0, no shock) is included and must reproduce the main results
(PharmaSales = exp05c cross-fitted; Rossmann = exp05d/exp06b seed 42).

Reported per row, next to the usual metrics on the OBSERVED (perturbed) target:
  clean_*    the same prediction scored against the unperturbed target (noise runs);
  shock_*    RMSE on the affected test rows only; outside_* on the others (shock runs).

Models
  PharmaSales  Naive, S1 [linear], AR-LRX [linear] (gated), LR-XGB residual without
               gate, XGBoost   -- the exp05c model set used in Chapter V.
  Rossmann V3  S1 [structural], AR-LRX-Aug [structural], LR-XGB residual without gate,
               XGBoost, LightGBM (Zeng)   -- the main model and its strongest references.

Parts (each checkpoints row by row and resumes after an interruption):
    python tools/exp10_controlled_perturbation.py --part pharma      # ~1-1.5 h
    python tools/exp10_controlled_perturbation.py --part rossmann    # ~3 h
    python tools/exp10_controlled_perturbation.py --part summary     # seconds
    --quick   smoke test (2 categories / 80 stores, one small grid); writes to
              results/_exp10_quick/ and prints no comparison.

Outputs (results/): exp10_pharma.csv, exp10_rossmann.csv, exp10_*_predictions/,
                    exp10_summary_pharma.csv, exp10_summary_rossmann.csv
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
warnings.filterwarnings("ignore")

from src.experiments import arlrx as A          # noqa: E402
from src.experiments import baselines as B      # noqa: E402
from src.experiments import protocol as P       # noqa: E402
import exp07_rolling_origin as E7               # noqa: E402  (Store, Ctx)

ALPHAS = [0.10, 0.25, 0.50]
PH_REPS = 2
SPIKE, LEVEL = 1.5, 1.3
ROSS_STORE_SHARE = 0.25
SHOCK_SEED = 2026
VARIANT = "V3_sales_lagged"
CATEGORIES = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
GRANS = [("daily", "data/raw/pharma-sales/salesdaily.csv", 7),
         ("weekly", "data/raw/pharma-sales/salesweekly.csv", 52)]


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else np.nan


def scenarios():
    out = [("clean", None, None, 0)]
    for a in ALPHAS:
        for r in range(1, PH_REPS + 1):
            out.append((f"noise_a{a:.2f}_r{r}", "noise", a, r))
    out += [("spike", "spike", SPIKE, 0), ("level", "level", LEVEL, 0)]
    return out


# --------------------------------------------------------------------------- #
# PharmaSales
# --------------------------------------------------------------------------- #

def pharma_models(d, quick):
    g = ({"n_estimators": [100], "max_depth": [3], "learning_rate": [0.1],
          "subsample": [0.8], "colsample_bytree": [0.8]} if quick else P.GRID_XGB_PHARMA)
    return [
        ("Naive", lambda: next(r for r in P.naive_rows(d) if r["model"] == "Naive")),
        ("S1 [linear]", lambda: A.run_stage1_only("S1 [linear]", d, "linear")),
        ("AR-LRX [linear]", lambda: A.run_arlrx("AR-LRX [linear]", d, "linear", g)),
        ("LR-XGB (residual, tanpa gerbang)",
         lambda: P.run_model("LR-XGB (residual, tanpa gerbang)", P.fp_lr_xgb_residual, d, g,
                             extra={"cross_fit_folds": P.N_CROSS_FIT_FOLDS})),
        ("XGBoost", lambda: P.run_model("XGBoost", P.fp_xgboost, d, g)),
    ]


def pharma_perturbed(raw, cat, fs, sp, d0, kind, param, rep, gran):
    """Rebuild one PharmaSales dataset from a perturbed raw series (k held fixed)."""
    dates = pd.to_datetime(raw["datum"], format="mixed", dayfirst=False, errors="coerce")
    raw2 = raw.copy()
    y = raw2[cat].to_numpy(dtype=float)
    mask_rows = np.zeros(len(raw2), bool)
    t0 = pd.Timestamp(d0.frame["ds"].iloc[d0.i_val_end])
    t_end = pd.Timestamp(d0.frame["ds"].iloc[-1])
    if kind == "noise":
        rng = np.random.default_rng(seed_of(gran, cat, param, rep))   # shared by both feature sets
        y = np.maximum(0.0, y + rng.normal(0.0, param * float(np.std(d0.y_train)), len(y)))
    elif kind == "spike":
        if gran == "daily":
            a, b = t0 + pd.Timedelta(days=30), t0 + pd.Timedelta(days=44)
        else:
            a, b = t0 + pd.Timedelta(weeks=8), t0 + pd.Timedelta(weeks=12)
        mask_rows = ((dates >= a) & (dates < b)).to_numpy()
        y = np.where(mask_rows, y * param, y)
    elif kind == "level":
        mid = t0 + (t_end - t0) / 2
        mask_rows = (dates >= mid).to_numpy()
        y = np.where(mask_rows, y * param, y)
    raw2[cat] = y
    d = P.build_pharma_dataset(raw2, cat, fs, seasonal_period=sp, lag_rule="pacf_train",
                               n_lags_override=d0.n_lags)
    assert (d.i_train_end, d.i_val_end, len(d.frame)) == (d0.i_train_end, d0.i_val_end, len(d0.frame))
    assert (d.frame["ds"].to_numpy() == d0.frame["ds"].to_numpy()).all()
    if kind in ("spike", "level"):
        tds = pd.to_datetime(d.dates_test)
        if kind == "spike":
            affected = ((tds >= a) & (tds < b))
        else:
            affected = (tds >= mid)
        affected = np.asarray(affected)
    else:
        affected = None
    return d, affected


def part_pharma(quick):
    st = E7.Store("exp10_pharma")
    P.set_global_seed()
    cats = CATEGORIES[:2] if quick else CATEGORIES
    t_start = time.time()
    for gran, path, sp in GRANS:
        raw = pd.read_csv(os.path.join(ROOT, path))
        for cat in cats:
            for fs in P.FEATURE_SETS:
                d0 = P.build_pharma_dataset(raw, cat, fs, seasonal_period=sp, lag_rule="pacf_train")
                config = f"{gran}|{cat}|{fs}"
                for scen, kind, param, rep in scenarios():
                    if kind is None:
                        d, affected = d0, None
                    else:
                        d, affected = pharma_perturbed(raw, cat, fs, sp, d0, kind, param, rep, gran)
                    for name, fn in pharma_models(d, quick):
                        if st.has(scen, config, name):
                            continue
                        r = dict(fn())
                        pred = r["_test_pred"]
                        r.update({"granularity": gran, "scenario": scen, "kind": kind or "clean",
                                  "param": param if param is not None else 0.0, "rep": rep,
                                  "clean_test_RMSE": rmse(d0.y_test, pred)})
                        if affected is not None:
                            r["shock_RMSE"] = rmse(d.y_test[affected], pred[affected])
                            r["outside_RMSE"] = rmse(d.y_test[~affected], pred[~affected])
                            r["n_affected"] = int(affected.sum())
                        st.add(r, scen, config, d.y_test)
                print(f"  {config:24s} done ({(time.time()-t_start)/60:.1f} min)", flush=True)


# --------------------------------------------------------------------------- #
# Rossmann
# --------------------------------------------------------------------------- #

def rossmann_models(quick):
    g = ({"n_estimators": [300], "max_depth": [6], "learning_rate": [0.1],
          "subsample": [0.8], "colsample_bytree": [0.8], "max_bin": [256]}
         if quick else A.GRID_XGB_ARLRX)
    gl = ({"n_estimators": [300], "num_leaves": [31], "learning_rate": [0.1],
           "subsample": [0.8], "colsample_bytree": [0.8]} if quick else B.GRID_LIGHTGBM)
    inv = np.expm1
    return [
        ("S1 [structural]",
         lambda d: A.run_stage1_only("S1 [structural]", d, "structural", inverse_transform=inv)),
        ("AR-LRX-Aug [structural]",
         lambda d: A.run_arlrx_segmented("AR-LRX-Aug [structural]", d, "structural", g,
                                         schemes=("global",), augment_stage1=True,
                                         inverse_transform=inv)),
        ("LR-XGB (residual, tanpa gerbang)",
         lambda d: P.run_model("LR-XGB (residual, tanpa gerbang)", P.fp_lr_xgb_residual, d, g,
                               inverse_transform=inv,
                               extra={"cross_fit_folds": P.N_CROSS_FIT_FOLDS})),
        ("XGBoost", lambda d: P.run_model("XGBoost", P.fp_xgboost, d, g, inverse_transform=inv)),
        ("LightGBM (Zeng)",
         lambda d: P.run_model("LightGBM (Zeng)", B.fp_lightgbm, d, gl, inverse_transform=inv)),
    ]


def rossmann_raw(quick):
    tr = pd.read_csv(os.path.join(ROOT, "data/raw/rossmann/train.csv"), low_memory=False)
    if quick:
        keep = np.sort(tr["Store"].unique())[:80]
        tr = tr[tr["Store"].isin(keep)].reset_index(drop=True)
    return tr


def rossmann_build(train_df, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
    f = os.path.join(tmpdir, "train_perturbed.csv")
    train_df.to_csv(f, index=False)
    frame = P.build_rossmann_frame(f, os.path.join(ROOT, "data/raw/rossmann/store.csv"))
    return P.build_rossmann_dataset(frame, VARIANT, target="log1p")


def part_rossmann(quick):
    st = E7.Store("exp10_rossmann")
    P.set_global_seed()
    tmp = E7.Ctx.path("_exp10_tmp")
    raw = rossmann_raw(quick)
    d0 = rossmann_build(raw, tmp)
    f0 = d0.frame
    test_dates = pd.to_datetime(f0["Date"].iloc[d0.i_val_end:])
    t0, t_end = test_dates.min(), test_dates.max()
    train_log = np.log1p(f0["Sales"].iloc[:d0.i_train_end].to_numpy(float))
    sd_log = float(np.std(train_log))
    stores = np.sort(raw["Store"].unique())
    rng = np.random.default_rng(SHOCK_SEED)
    hit = set(rng.choice(stores, size=int(round(ROSS_STORE_SHARE * len(stores))), replace=False).tolist())
    clean_orig_test = f0["Sales"].iloc[d0.i_val_end:].to_numpy(float)
    rdates = pd.to_datetime(raw["Date"])
    print(f"Rossmann: test {t0.date()}..{t_end.date()}, sd_train(log1p Sales) = {sd_log:.4f}, "
          f"{len(hit)} shocked stores", flush=True)

    scen = [("clean", None, None)] + [(f"noise_a{a:.2f}_r1", "noise", a) for a in ALPHAS] + \
           [("spike", "spike", SPIKE), ("level", "level", LEVEL)]
    for name_s, kind, param in scen:
        todo = [m for m, _ in rossmann_models(quick) if not st.has(name_s, VARIANT, m)]
        if not todo:
            continue
        affected = None
        if kind is None:
            d = d0
        else:
            r2 = raw.copy()
            s = r2["Sales"].to_numpy(float)
            if kind == "noise":
                g = np.random.default_rng(seed_of("rossmann", param, 1))
                ok = ((r2["Open"] == 1) & (r2["Sales"] > 0)).to_numpy()
                s = np.where(ok, s * np.exp(g.normal(0.0, param * sd_log, len(s))), s)
            else:
                in_store = r2["Store"].isin(hit).to_numpy()
                if kind == "spike":
                    a, b = t0 + pd.Timedelta(days=28), t0 + pd.Timedelta(days=42)
                    win = ((rdates >= a) & (rdates < b)).to_numpy()
                else:
                    mid = t0 + (t_end - t0) / 2
                    win = (rdates >= mid).to_numpy()
                s = np.where(in_store & win, s * param, s)
            r2["Sales"] = s
            d = rossmann_build(r2, tmp)
            assert len(d.frame) == len(f0) and d.i_val_end == d0.i_val_end
            assert (d.frame["Store"].to_numpy() == f0["Store"].to_numpy()).all()
            if kind in ("spike", "level"):
                ft = d.frame.iloc[d.i_val_end:]
                fd = pd.to_datetime(ft["Date"])
                w = ((fd >= a) & (fd < b)) if kind == "spike" else (fd >= mid)
                affected = (ft["Store"].isin(hit) & w).to_numpy()
        y_orig = np.expm1(d.y_test)
        for mname, fn in rossmann_models(quick):
            if st.has(name_s, VARIANT, mname):
                continue
            t1 = time.time()
            r = dict(fn(d))
            p_orig = np.expm1(r["_test_pred"])
            r.update({"scenario": name_s, "kind": kind or "clean",
                      "param": param if param is not None else 0.0,
                      "clean_orig_RMSE": rmse(clean_orig_test, p_orig)})
            if affected is not None:
                r["shock_orig_RMSE"] = rmse(y_orig[affected], p_orig[affected])
                r["outside_orig_RMSE"] = rmse(y_orig[~affected], p_orig[~affected])
                r["n_affected"] = int(affected.sum())
            st.add(r, name_s, VARIANT, d.y_test)
            print(f"  {name_s:16s} {mname:34s} orig RMSE {r.get('orig_RMSE', np.nan):9.2f} "
                  f"[{(time.time()-t1)/60:.1f} min]", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Summary (the statistics named in the pre-registration)
# --------------------------------------------------------------------------- #

def pooled(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r) & (r > 0)]
    return 100 * (np.exp(np.log(r).mean()) - 1) if len(r) else np.nan


def summary_pharma(df):
    out = []
    key = ["config"]
    for scen_group, s in df.groupby(df.scenario.str.replace(r"_r\d+$", "", regex=True)):
        piv = {c: s.pivot_table(index=["config", "rep"], columns="model", values=c)
               for c in ["test_RMSE", "clean_test_RMSE", "shock_RMSE"] if c in s}
        t = piv["test_RMSE"]
        ar = s[s.model == "AR-LRX [linear]"]
        ch_g = 100 * (t["AR-LRX [linear]"] / t["S1 [linear]"] - 1)
        ch_u = 100 * (t["LR-XGB (residual, tanpa gerbang)"] / t["S1 [linear]"] - 1)
        row = {"scenario": scen_group, "n_rows": len(t),
               "gate_closed": int((ar.gate_w == 0).sum()), "gate_closed_share": float((ar.gate_w == 0).mean()),
               "gate_w_mean": ar.gate_w.mean(),
               "resid_val_r2_median": ar.resid_val_r2.median(),
               "gated_mean_change_vs_S1_pct": ch_g.mean(), "gated_n_worse": int((ch_g > 1e-9).sum()),
               "gated_worst_pct": ch_g.max(),
               "ungated_mean_change_vs_S1_pct": ch_u.mean(), "ungated_n_worse": int((ch_u > 1e-9).sum()),
               "ungated_worst_pct": ch_u.max(),
               "pooled_arlrx_vs_S1_pct": pooled(t["AR-LRX [linear]"] / t["S1 [linear]"]),
               "pooled_arlrx_vs_xgb_pct": pooled(t["AR-LRX [linear]"] / t["XGBoost"]),
               "pooled_arlrx_vs_naive_pct": pooled(t["AR-LRX [linear]"] / t["Naive"])}
        if "clean_test_RMSE" in piv:
            c = piv["clean_test_RMSE"]
            row["clean_pooled_arlrx_vs_S1_pct"] = pooled(c["AR-LRX [linear]"] / c["S1 [linear]"])
            row["clean_pooled_arlrx_vs_xgb_pct"] = pooled(c["AR-LRX [linear]"] / c["XGBoost"])
        if "shock_RMSE" in piv and piv["shock_RMSE"].notna().any().any():
            k = piv["shock_RMSE"]
            row.update({"shock_pooled_arlrx_vs_xgb_pct": pooled(k["AR-LRX [linear]"] / k["XGBoost"]),
                        "shock_pooled_arlrx_vs_S1_pct": pooled(k["AR-LRX [linear]"] / k["S1 [linear]"]),
                        "shock_pooled_ungated_vs_S1_pct": pooled(
                            k["LR-XGB (residual, tanpa gerbang)"] / k["S1 [linear]"]),
                        "shock_share_arlrx_le_xgb": float((k["AR-LRX [linear]"] <= k["XGBoost"]).mean())})
        out.append(row)
    order = {"clean": 0, "noise_a0.10": 1, "noise_a0.25": 2, "noise_a0.50": 3, "spike": 4, "level": 5}
    return pd.DataFrame(out).sort_values("scenario", key=lambda x: x.map(order))


def summary_rossmann(df):
    out = []
    ref = "AR-LRX-Aug [structural]"
    for scen, s in df.groupby("scenario"):
        m = s.set_index("model")
        row = {"scenario": scen}
        for col in ["orig_RMSE", "clean_orig_RMSE", "shock_orig_RMSE"]:
            if col not in m or m[col].isna().all():
                continue
            for other in ["S1 [structural]", "XGBoost", "LightGBM (Zeng)", "LR-XGB (residual, tanpa gerbang)"]:
                if other in m.index and ref in m.index:
                    row[f"{col}:{ref} vs {other} (%)"] = 100 * (m.loc[ref, col] / m.loc[other, col] - 1)
            for mm in m.index:
                row[f"{col}:{mm}"] = m.loc[mm, col]
        if ref in m.index:
            row["gate_w"] = m.loc[ref].get("gate_w", np.nan)
            row["resid_val_r2"] = m.loc[ref].get("resid_val_r2", np.nan)
        out.append(row)
    order = {"clean": 0, "noise_a0.10_r1": 1, "noise_a0.25_r1": 2, "noise_a0.50_r1": 3, "spike": 4, "level": 5}
    return pd.DataFrame(out).sort_values("scenario", key=lambda x: x.map(order))


def part_summary():
    pd.set_option("display.width", 220)
    for name, fn in [("pharma", summary_pharma), ("rossmann", summary_rossmann)]:
        f = E7.Ctx.path(f"exp10_{name}.csv")
        if not os.path.exists(f):
            print(f"(no results/exp10_{name}.csv yet)")
            continue
        s = fn(pd.read_csv(f))
        s.to_csv(E7.Ctx.path(f"exp10_summary_{name}.csv"), index=False)
        print(f"\n=== exp10 {name} ===")
        print(s.round(3).T.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["pharma", "rossmann", "summary"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        E7.Ctx.res = os.path.join(ROOT, "results", "_exp10_quick")
    t0 = time.time()
    if a.part == "pharma":
        part_pharma(a.quick)
    elif a.part == "rossmann":
        part_rossmann(a.quick)
    else:
        part_summary()
    if a.quick:
        print("quick smoke test finished (no comparison printed)")
    print(f"runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
