"""
Dependence-aware and multiplicity-aware inference for reviewer points R1-4 and
R2-5 (IJIES Paper ID 20265893, round 2).

Nothing is retrained. The script reads the saved test predictions of the
cross-fitted rerun (results/*_predictions.npz, git-ignored, produced by
exp05b, exp05c, exp05d and exp06b) and writes machine-readable summaries.

R1-4  Rossmann V3 panel. The DM tests of Table 4 treat the 129,333 store-day
      loss differentials as independent. Here the proposed model
      (AR-LRX-Aug [structural]) is compared with every reference under four
      procedures, all on the original sales scale (squared-error loss) and on
      the log scale:
        obs      the original observation-level DM test (reference only);
        date-DM  DM on the loss differential aggregated to one value per
                 calendar date (removes cross-store dependence on a date), with
                 a Newey-West (Bartlett) long-run variance (temporal dependence);
        date-MBB moving-block bootstrap over calendar dates (blocks of 7 and
                 14 days): whole dates move together, so every store sharing a
                 date shock is resampled jointly;
        store-CB cluster bootstrap over stores (all dates of a store move
                 together), which is robust to within-store dependence of any
                 length;
        two-way  pigeonhole bootstrap over stores AND date blocks at once
                 (independent multinomial weights on both dimensions).
      Each bootstrap reports a 95% percentile interval for the relative RMSE
      difference 100 * (RMSE_AR-LRX / RMSE_ref - 1) and a two-sided bootstrap
      p-value.

R2-5  Pharmaceutical series, 32 configurations x 4 references.
        * the per-configuration DM p-values of Table 9, with Benjamini-Hochberg
          FDR control over all valid tests jointly and within each reference;
        * a per-configuration 95% interval for the relative RMSE difference from
          a moving-block bootstrap over test time (block 7 daily, 4 weekly);
        * a pooled summary across the 32 configurations: the geometric-mean
          RMSE ratio AR-LRX / reference, with a 95% interval from a cluster
          bootstrap over the 8 ATC categories (configurations of the same
          category share data and are not independent), overall and per
          granularity.
        * DUPLICATES: when the lag count selected on the training block is 1,
          the "rich" feature set degenerates to the reference set and the two
          configurations are the same model with the same predictions (12 of
          the 16 category x granularity pairs in the cross-fitted rerun). Every
          summary is therefore also reported over the DISTINCT configurations
          only (B_rich rows that duplicate A_lag1 are dropped).

Usage (repository root):
    python tools/groupB_inference.py            # B = 2000 bootstrap draws
    python tools/groupB_inference.py --B 500    # quicker check

Outputs (results/):
    groupB_rossmann_dependence.csv
    groupB_pharma_tests_bh.csv
    groupB_pharma_config_ci.csv
    groupB_pharma_pooled.csv
    groupB_summary.json
"""
from __future__ import annotations

import argparse
import json
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

from src.experiments import protocol as P       # noqa: E402

RES = os.path.join(ROOT, "results")
SEED = P.SEED
VARIANT = "V3_sales_lagged"
PROPOSED = "AR-LRX-Aug [structural]"


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #

def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values); NaN kept as NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def newey_west_dm(d: np.ndarray, lag: int):
    """DM statistic with a Bartlett-kernel long-run variance; t(T-1) reference."""
    d = np.asarray(d, dtype=float)
    T = len(d)
    dbar = d.mean()
    u = d - dbar
    lrv = u @ u / T
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / T
    if lrv <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(lrv / T)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=T - 1))


def nw_bandwidth(T: int) -> int:
    """Newey-West (1994) rule of thumb, floor(4 (T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100) ** (2 / 9)))


def block_starts(rng, n_units: int, block: int, n_draw: int) -> np.ndarray:
    """Index matrix (n_draw, n_units) of a circular moving-block bootstrap."""
    n_blocks = int(np.ceil(n_units / block))
    starts = rng.integers(0, n_units, size=(n_draw, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n_units
    return idx.reshape(n_draw, -1)[:, :n_units]


def boot_summary(ratio_pct: np.ndarray, point: float) -> dict:
    lo, hi = np.percentile(ratio_pct, [2.5, 97.5])
    # two-sided bootstrap p-value for H0: no difference (ratio = 0 %)
    p = 2 * min((ratio_pct >= 0).mean(), (ratio_pct <= 0).mean())
    return {"point_pct": point, "ci_lo_pct": float(lo), "ci_hi_pct": float(hi),
            "p_boot": float(min(1.0, max(p, 1.0 / len(ratio_pct))))}


# --------------------------------------------------------------------------- #
# R1-4: Rossmann
# --------------------------------------------------------------------------- #

def rossmann(B: int) -> pd.DataFrame:
    t0 = time.time()
    frame = P.build_rossmann_frame(os.path.join(ROOT, "data/raw/rossmann/train.csv"),
                                   os.path.join(ROOT, "data/raw/rossmann/store.csv"))
    d = P.build_rossmann_dataset(frame, VARIANT, target="log1p")
    test = d.frame.iloc[d.i_val_end:]
    dates = pd.to_datetime(test["Date"]).to_numpy()
    stores = test["Store"].to_numpy()
    y_log = d.y_test
    print(f"  Rossmann V3 test block: {len(y_log)} store-days, "
          f"{len(np.unique(dates))} dates, {len(np.unique(stores))} stores "
          f"[{time.time()-t0:.0f}s]", flush=True)

    zb = np.load(os.path.join(RES, "exp05b_rossmann_arlrx_audit_predictions.npz"))
    zd = np.load(os.path.join(RES, "exp05d_rossmann_arlrx_dev_predictions.npz"))
    z6 = np.load(os.path.join(RES, "exp06b_rossmann_baselines_strong_predictions.npz"))
    assert np.allclose(zb[f"{VARIANT}|y_test"], y_log, atol=1e-5), "exp05b y_test mismatch"
    assert np.allclose(z6["y_test"], y_log, atol=1e-5), "exp06b y_test mismatch"

    preds = {PROPOSED: zd[f"{VARIANT}|{PROPOSED}|test_pred"]}
    refs = []
    for k in z6.files:
        if k.endswith("|test_pred"):
            name = k.split("|")[0]
            preds[name] = z6[k]; refs.append((name, "retrained baseline"))
    for name in ["XGBoost", "AR-LRX [struct_linear]", "LR-XGB (residual, tanpa gerbang)",
                 "S1 [structural]", "SeasonalNaive(store x dow x promo median)"]:
        preds[name] = zb[f"{VARIANT}|{name}|test_pred"]; refs.append((name, "internal"))
    preds["AR-LRX-Aug [struct_linear]"] = zd[f"{VARIANT}|AR-LRX-Aug [struct_linear]|test_pred"]
    refs.append(("AR-LRX-Aug [struct_linear]", "internal"))
    preds = {k: np.asarray(v, dtype=float) for k, v in preds.items()}

    # unit codes
    d_codes, d_idx = np.unique(dates, return_inverse=True)
    s_codes, s_idx = np.unique(stores, return_inverse=True)
    nD, nS = len(d_codes), len(s_codes)
    count_ds = np.zeros((nS, nD)); np.add.at(count_ds, (s_idx, d_idx), 1)
    L = nw_bandwidth(nD)

    rng_master = np.random.default_rng(SEED)
    # the same resampling draws are used for every comparison (paired design)
    draws = {
        "date_mbb7": block_starts(rng_master, nD, 7, B),
        "date_mbb14": block_starts(rng_master, nD, 14, B),
        "store_cb": rng_master.integers(0, nS, size=(B, nS)),
    }
    # two-way pigeonhole: multinomial weights for stores and for date blocks
    w_store = np.stack([np.bincount(r, minlength=nS) for r in rng_master.integers(0, nS, size=(B, nS))])
    w_date = np.stack([np.bincount(r, minlength=nD) for r in block_starts(rng_master, nD, 7, B)])

    rows = []
    for scale in ("orig", "log"):
        y = np.expm1(y_log) if scale == "orig" else y_log
        tr = (lambda p: np.expm1(p)) if scale == "orig" else (lambda p: p)
        sq = {k: (y - tr(v)) ** 2 for k, v in preds.items()}
        sse_ds = {}
        for k, e2 in sq.items():
            m = np.zeros((nS, nD)); np.add.at(m, (s_idx, d_idx), e2); sse_ds[k] = m
        cnt_d = count_ds.sum(0); cnt_s = count_ds.sum(1); N = count_ds.sum()
        a = PROPOSED
        sa_d, sa_s = sse_ds[a].sum(0), sse_ds[a].sum(1)
        rmse_a = np.sqrt(sse_ds[a].sum() / N)
        for ref, kind in refs:
            sr_d, sr_s = sse_ds[ref].sum(0), sse_ds[ref].sum(1)
            rmse_r = np.sqrt(sse_ds[ref].sum() / N)
            point = 100 * (rmse_a / rmse_r - 1)
            row = {"scale": scale, "reference": ref, "reference_type": kind,
                   "rmse_proposed": rmse_a, "rmse_reference": rmse_r,
                   "rel_diff_pct": point, "n_obs": int(N), "n_dates": nD, "n_stores": nS}
            # (1) observation-level DM, as in Table 4
            t = P.diebold_mariano(y, tr(preds[a]), tr(preds[ref]))
            row.update({"obs_DM": t["DM"], "obs_p": t["p_value"]})
            # (2) date-aggregated DM with Newey-West variance
            dd = (sa_d - sr_d) / (N / nD)          # mean over dates == overall mean differential
            for lag in sorted({L, 7}):
                s, p = newey_west_dm(dd, lag)
                row[f"dateDM_NW{lag}"] = s; row[f"dateDM_NW{lag}_p"] = p
            row["nw_lag_rule"] = L
            # (3)-(5) bootstraps
            def rel(num_a, num_r):
                return 100 * (np.sqrt(num_a) / np.sqrt(num_r) - 1)   # counts cancel
            for key in ("date_mbb7", "date_mbb14"):
                idx = draws[key]
                r = rel(sa_d[idx].sum(1), sr_d[idx].sum(1))
                for kk, vv in boot_summary(r, point).items():
                    if kk != "point_pct":
                        row[f"{key}_{kk}"] = vv
            idx = draws["store_cb"]
            r = rel(sa_s[idx].sum(1), sr_s[idx].sum(1))
            for kk, vv in boot_summary(r, point).items():
                if kk != "point_pct":
                    row[f"store_cb_{kk}"] = vv
            num_a = (w_store * (sse_ds[a] @ w_date.T).T).sum(1)
            num_r = (w_store * (sse_ds[ref] @ w_date.T).T).sum(1)
            r = rel(num_a, num_r)
            for kk, vv in boot_summary(r, point).items():
                if kk != "point_pct":
                    row[f"twoway_{kk}"] = vv
            rows.append(row)
    print(f"  Rossmann done [{time.time()-t0:.0f}s]", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# R2-5: pharmaceutical series
# --------------------------------------------------------------------------- #

PH_REFS = ["Naive", "S1 [linear]", "XGBoost", "LR-XGB (residual, tanpa gerbang)"]
PH_PROP = "AR-LRX [linear]"


def pharma(B: int):
    t0 = time.time()
    z = np.load(os.path.join(RES, "exp05c_pharma_arlrx_predictions.npz"))
    configs = sorted({tuple(k.split("|")[:3]) for k in z.files})
    duplicate = {}
    for g, c, f in configs:
        models = [k.split("|")[3] for k in z.files
                  if k.startswith(f"{g}|{c}|{f}|") and k.endswith("|test_pred")]
        duplicate[(g, c, f)] = f == "B_rich" and all(
            np.array_equal(z[f"{g}|{c}|B_rich|{m}|test_pred"], z[f"{g}|{c}|A_lag1|{m}|test_pred"])
            for m in models)
    print(f"  pharma: {len(configs)} configurations, "
          f"{sum(duplicate.values())} exact duplicates (B_rich == A_lag1)", flush=True)
    rng = np.random.default_rng(SEED + 1)
    tests, cis = [], []
    logratio = {}
    for g, c, f in configs:
        y = np.asarray(z[f"{g}|{c}|{f}|y_test"], dtype=float)
        pa = np.asarray(z[f"{g}|{c}|{f}|{PH_PROP}|test_pred"], dtype=float)
        n = len(y)
        block = 7 if g == "daily" else 4
        idx = block_starts(rng, n, block, B)
        ea2 = (y - pa) ** 2
        for ref in PH_REFS:
            pr = np.asarray(z[f"{g}|{c}|{f}|{ref}|test_pred"], dtype=float)
            er2 = (y - pr) ** 2
            same = bool(np.array_equal(pa, pr))
            t = P.diebold_mariano(y, pa, pr)
            ra, rr = np.sqrt(ea2.mean()), np.sqrt(er2.mean())
            point = 100 * (ra / rr - 1)
            tests.append({"granularity": g, "category": c, "feature_set": f, "reference": ref,
                          "duplicate_of_A_lag1": duplicate[(g, c, f)], "n_test": n, "identical_predictions": same,
                          "rmse_arlrx": ra, "rmse_reference": rr, "rel_diff_pct": point,
                          "DM": t["DM"], "p": t["p_value"]})
            if same:
                ci = {"point_pct": 0.0, "ci_lo_pct": 0.0, "ci_hi_pct": 0.0, "p_boot": 1.0}
            else:
                r = 100 * (np.sqrt(ea2[idx].mean(1)) / np.sqrt(er2[idx].mean(1)) - 1)
                ci = boot_summary(r, point)
            cis.append({"granularity": g, "category": c, "feature_set": f, "reference": ref,
                        "duplicate_of_A_lag1": duplicate[(g, c, f)], "block": block, **ci,
                        "ci_excludes_zero": bool(ci["ci_lo_pct"] > 0 or ci["ci_hi_pct"] < 0)})
            logratio[(g, c, f, ref)] = np.log(ra / rr)
    tests = pd.DataFrame(tests)
    tests["valid"] = tests.p.notna()
    tests["q_BH_all"] = bh_adjust(tests.p.to_numpy())
    tests["q_BH_within_ref"] = np.nan
    for ref in PH_REFS:
        m = tests.reference == ref
        tests.loc[m, "q_BH_within_ref"] = bh_adjust(tests.loc[m, "p"].to_numpy())
    dist = ~tests.duplicate_of_A_lag1
    tests["q_BH_distinct"] = np.nan
    tests.loc[dist, "q_BH_distinct"] = bh_adjust(tests.loc[dist, "p"].to_numpy())
    for col, pcol in [("sig_win_raw", "p"), ("sig_win_BH_all", "q_BH_all"),
                      ("sig_win_BH_within", "q_BH_within_ref"),
                      ("sig_win_BH_distinct", "q_BH_distinct")]:
        tests[col] = (tests.DM < 0) & (tests[pcol] < 0.05)
        tests[col.replace("win", "loss")] = (tests.DM > 0) & (tests[pcol] < 0.05)
    cis = pd.DataFrame(cis)

    # pooled geometric-mean ratio with a cluster bootstrap over ATC categories
    cats = sorted({c for _, c, _ in configs})
    pooled = []
    for ref in PH_REFS:
        for subset in ("all32", "distinct"):
            for gran in ("all", "daily", "weekly"):
                keys = [k for k in configs if (gran == "all" or k[0] == gran)
                        and (subset == "all32" or not duplicate[k])]
                lr = np.array([logratio[k + (ref,)] for k in keys])
                cat_of = np.array([k[1] for k in keys])
                point = 100 * (np.exp(lr.mean()) - 1)
                boots = []
                for _ in range(B):
                    pick = rng.choice(cats, size=len(cats), replace=True)
                    vals = np.concatenate([lr[cat_of == cc] for cc in pick])
                    boots.append(100 * (np.exp(vals.mean()) - 1))
                s = boot_summary(np.array(boots), point)
                pooled.append({"reference": ref, "subset": subset, "granularity": gran,
                               "n_configs": len(keys),
                               "geo_mean_rel_diff_pct": point,
                               "ci_lo_pct": s["ci_lo_pct"], "ci_hi_pct": s["ci_hi_pct"],
                               "p_boot": s["p_boot"],
                               "configs_better": int((lr < 0).sum()),
                               "configs_equal": int((lr == 0).sum()),
                               "configs_worse": int((lr > 0).sum())})
    print(f"  pharma done [{time.time()-t0:.0f}s]", flush=True)
    return tests, cis, pd.DataFrame(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    print("Environment:", P.environment_stamp(), "| B =", args.B, flush=True)

    ross = rossmann(args.B)
    ross.to_csv(os.path.join(RES, "groupB_rossmann_dependence.csv"), index=False)
    tests, cis, pooled = pharma(args.B)
    tests.to_csv(os.path.join(RES, "groupB_pharma_tests_bh.csv"), index=False)
    cis.to_csv(os.path.join(RES, "groupB_pharma_config_ci.csv"), index=False)
    pooled.to_csv(os.path.join(RES, "groupB_pharma_pooled.csv"), index=False)

    summ = {"B": args.B, "seed": SEED, "rossmann_proposed": PROPOSED,
            "pharma_valid_tests": int(tests.valid.sum()),
            "pharma_undefined_tests": int((~tests.valid).sum()),
            "pharma_duplicate_configs": int(tests.drop_duplicates(
                ["granularity", "category", "feature_set"]).duplicate_of_A_lag1.sum()),
            "pharma_valid_tests_distinct": int(tests[~tests.duplicate_of_A_lag1].valid.sum())}
    for ref in PH_REFS:
        g = tests[tests.reference == ref]
        c = cis[cis.reference == ref]
        summ[ref] = {k: int(g[k].sum()) for k in
                     ["sig_win_raw", "sig_win_BH_all", "sig_win_BH_within",
                      "sig_loss_raw", "sig_loss_BH_all", "sig_loss_BH_within"]}
        summ[ref]["ci_excludes_zero_better"] = int((c.ci_hi_pct < 0).sum())
        summ[ref]["ci_excludes_zero_worse"] = int((c.ci_lo_pct > 0).sum())
        gd, cd = g[~g.duplicate_of_A_lag1], c[~c.duplicate_of_A_lag1]
        summ[ref]["distinct"] = {
            "n_configs": int(len(gd)), "wins": int((gd.rel_diff_pct < 0).sum()),
            "ties": int(gd.identical_predictions.sum()), "valid_tests": int(gd.valid.sum()),
            "sig_win_raw": int(gd.sig_win_raw.sum()), "sig_win_BH_distinct": int(gd.sig_win_BH_distinct.sum()),
            "sig_loss_raw": int(gd.sig_loss_raw.sum()), "sig_loss_BH_distinct": int(gd.sig_loss_BH_distinct.sum()),
            "ci_excludes_zero_better": int((cd.ci_hi_pct < 0).sum()),
            "ci_excludes_zero_worse": int((cd.ci_lo_pct > 0).sum())}
    json.dump(summ, open(os.path.join(RES, "groupB_summary.json"), "w"), indent=2)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    o = ross[ross.scale == "orig"]
    print("\nR1-4 Rossmann (original scale), AR-LRX-Aug [structural] vs reference:")
    print(o[["reference", "rel_diff_pct", "obs_DM", f"dateDM_NW{int(o.nw_lag_rule.iloc[0])}",
             f"dateDM_NW{int(o.nw_lag_rule.iloc[0])}_p", "date_mbb7_ci_lo_pct", "date_mbb7_ci_hi_pct",
             "store_cb_ci_lo_pct", "store_cb_ci_hi_pct", "twoway_ci_lo_pct", "twoway_ci_hi_pct",
             "twoway_p_boot"]].round(4).to_string(index=False))
    print("\nR2-5 pharma, counts per reference:")
    print(json.dumps({k: v for k, v in summ.items() if k in PH_REFS}, indent=1))
    print("\nR2-5 pharma, pooled geometric-mean relative RMSE difference (category-cluster bootstrap):")
    print(pooled.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
