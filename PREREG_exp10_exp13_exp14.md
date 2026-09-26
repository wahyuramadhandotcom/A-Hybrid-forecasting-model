# Pre-registration: exp10, exp13 and exp14

Written 26 September 2026, before any full run of the three scripts below. The commit that adds this file, together with the scripts, fixes the design and the predictions. Results are committed afterwards in separate commits.

This file follows the pre-registered predictions P1 to P3 of exp05c (Chapter V of the dissertation). Its predictions are numbered P4 to P9 so that they continue that series.

## 1. Purpose

| Experiment | Script | Question |
|---|---|---|
| exp13 | `tools/exp13_pharma_lstm_wide.py` | Does the LSTM reference on PharmaSales improve when the width grid is widened beyond {64, 128, 256}? In exp12 the selected width sat on the grid edge in 23 of 32 configurations. |
| exp14 | `tools/exp14_pharma_ets.py` | How does AR-LRX compare with exponential smoothing (ETS / Holt-Winters), the most common family in pharmaceutical demand forecasting? |
| exp10 | `tools/exp10_controlled_perturbation.py` | How do AR-LRX and its references behave under controlled noise and demand shocks? Does noise injected into Rossmann move it from the structured-residual regime towards the noise regime, with the gate following? |

## 2. Fixed elements

- Contract C1 to C8 unchanged: chronological 70/15/15 split, tuning on validation only, refit on train+validation, test touched once, seed 42, PACF lag count from the training block, rolling one-step-ahead forecasts.
- AR-LRX is not modified in any way: same stage-1 kinds, same XGBoost grids (`P.GRID_XGB_PHARMA`, `A.GRID_XGB_ARLRX`), same gate grid, same cross-fitting (5 chronological folds).
- Design parameters are fixed in the scripts and are not tuned:
  - LSTM widths {32, 64, 128, 256, 512};
  - the five ETS specifications;
  - noise levels alpha in {0.10, 0.25, 0.50};
  - 2 noise realisations for PharmaSales and 1 for Rossmann;
  - spike factor 1.5 and level-shift factor 1.3;
  - shock windows as given in the exp10 docstring;
  - 25% of Rossmann stores shocked (seed 2026).
- Runs on the Windows reference machine, like exp11 and exp12.

## 3. Statistics used for the predictions

- **Pooled change:** `100 * (exp(mean(log(RMSE_a / RMSE_b))) - 1)`, the geometric-mean ratio used throughout Chapters IV and V. Negative means model a is better.
- **exp13 and exp14:** the statistics are the `pooled_pct` column of the `_summary.csv` files, scope `distinct` (20 configurations), per granularity. They are produced by the exp12 analysis code, which is unchanged.
- **exp10 PharmaSales:** the statistics come from `results/exp10_summary_pharma.csv`. There are 32 configurations per scenario; noise statistics pool the two realisations (64 rows).
- **exp10 Rossmann:** the statistics come from `results/exp10_summary_rossmann.csv` (main model AR-LRX-Aug [structural]).

## 4. Predictions

### P4 (exp13, LSTM with a wider grid)

- **P4a.** The pooled change in LSTM test RMSE relative to exp12 lies within ±2% (all 32 configurations; printed by the script).
- **P4b.** AR-LRX [linear] is not worse than the wide-grid LSTM by more than 1% pooled, on both daily and weekly data (`pooled_pct <= +1.0`).

### P5 (exp14, ETS)

- **P5a.** AR-LRX [linear] is not worse than rolling ETS by more than 1% pooled, on both daily and weekly data (`pooled_pct <= +1.0`).
- **P5b.** Rolling ETS is not more accurate than rolling ARIMA(5,1,0) on daily data. The pooled ETS/ARIMA ratio, derived from the two AR-LRX comparisons in the same summary, is ≥ 1.

### P6 (exp10, PharmaSales, noise)

- **P6a.** The share of configurations with a closed gate (`gate_closed_share`, w* = 0) at alpha = 0.50 is at least the share in the clean run.
- **P6b.** At every alpha, the mean change of gated AR-LRX relative to S1 [linear] (`gated_mean_change_vs_S1_pct`) is ≤ +0.5%. It is also smaller than the same quantity for the hybrid without gate (`ungated_mean_change_vs_S1_pct`).

### P7 (exp10, Rossmann, noise as a controlled regime shift)

- **P7a.** `resid_val_r2` of AR-LRX-Aug [structural] decreases monotonically over clean, 0.10, 0.25 and 0.50.
- **P7b.** The advantage of AR-LRX-Aug over S1 [structural] on the observed test target (`orig_RMSE: ... vs S1 [structural] (%)`) shrinks monotonically over the same sequence, that is, the value becomes less negative.
- **P7c.** At every alpha, AR-LRX-Aug is not worse than S1 [structural] by more than 0.5%.
- **P7d.** At alpha ≤ 0.25, AR-LRX-Aug remains more accurate than both XGBoost and LightGBM (Zeng) on the observed test target.

### P8 (exp10, PharmaSales, shocks in the test block)

For both the spike and the level-shift scenario, on the affected test rows (`shock_RMSE`):

- **P8a.** The pooled change of AR-LRX [linear] relative to XGBoost is negative (`shock_pooled_arlrx_vs_xgb_pct < 0`).
- **P8b.** The pooled change of AR-LRX [linear] relative to S1 [linear] is ≤ +2% (`shock_pooled_arlrx_vs_S1_pct <= 2`).

### P9 (exp10, Rossmann, shocks in the test block)

For both the spike and the level-shift scenario, on the affected rows (`shock_orig_RMSE`):

- **P9a.** AR-LRX-Aug [structural] is more accurate than S1 [structural].
- **P9b.** AR-LRX-Aug is not worse than XGBoost by more than 2%.

## 5. Reporting rules

1. Every prediction is reported as confirmed or failed, with the observed value, in the dissertation and in the repository README. Chapter IV covers exp13 and exp14; Chapter V covers exp10.
2. All rows produced by the three scripts are committed to `results/` whatever they show. No result is withheld because it is unfavourable to AR-LRX.
3. No design parameter in Section 2 is changed after a full run. If a coding error is found:
   - fix it and re-run the affected part;
   - list the error, the fix and both commit hashes in Section 7;
   - keep the output of the faulty run in `results/_superseded/`.
4. Statistics not named in Section 4 are reported as exploratory.

## 6. Smoke tests run before this commit

Before this file was committed, the scripts were executed only in `--quick` mode to check the code paths:

- exp10 PharmaSales: 2 categories and a one-point XGBoost grid;
- exp10 Rossmann: synthetic stand-in data, because the Rossmann file is not in the cloud workspace;
- exp13: 2 categories and widths {32, 512};
- exp14: 2 categories, plus one configuration checked for finite output.

Quick-mode output was inspected for its structure only: row counts, affected-row counts and column names. The one exception is the gate-closure shares of the exp10 PharmaSales quick run, which were displayed during the structural check. That run uses the reduced grid and 2 of 8 categories, and it is not the pre-registered analysis. No comparison between models from exp13 or exp14 was viewed.

## 7. Deviations

(none yet)
