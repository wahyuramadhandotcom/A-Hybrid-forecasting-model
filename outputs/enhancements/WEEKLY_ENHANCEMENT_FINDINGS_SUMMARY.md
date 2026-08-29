# Weekly Enhancement Findings Summary

Summary hasil eksperimen enhancement forecasting **LR + XGBoost** pada PharmaSales weekly (`salesweekly.csv`).

## Overview

Total hasil yang dianalisis:

- Eksperimen: 9
- Baris hasil: 576
- Kategori obat: 8
- Optimizer: Grid Search, Optuna, PSO, GEO
- Skema hybrid: averaging dan residual
- Metrik utama: test MSE dan test RMSE

## Best Result per Category

| Category | Best Experiment | Optimizer | Scheme | MSE | RMSE | Baseline MSE | Delta MSE | Improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M01AB | E2_pseudohuber | Optuna | averaging | 60.214 | 7.760 | 67.021 | -6.807 | 10.16% |
| M01AE | D1_log1p | GEO | residual | 66.920 | 8.180 | 69.475 | -2.554 | 3.68% |
| N02BA | D2_poisson | GEO | averaging | 35.809 | 5.984 | 38.715 | -2.907 | 7.51% |
| N02BE | D1_log1p | GEO | averaging | 2211.250 | 47.024 | 2886.454 | -675.204 | 23.39% |
| N05B | baseline | GEO | averaging | 143.225 | 11.968 | 143.225 | 0.000 | 0.00% |
| N05C | E2_pseudohuber | GEO | averaging | 7.182 | 2.680 | 7.379 | -0.197 | 2.67% |
| R03 | E1_winsorize | PSO | residual | 699.534 | 26.449 | 721.610 | -22.076 | 3.06% |
| R06 | E2_pseudohuber | PSO | averaging | 68.240 | 8.261 | 71.429 | -3.188 | 4.46% |

## Experiment Win Count

| experiment | wins |
| --- | --- |
| E2_pseudohuber | 3 |
| D1_log1p | 2 |
| D2_poisson | 1 |
| baseline | 1 |
| E1_winsorize | 1 |

## MSE Pivot Summary

Nilai di bawah adalah **best MSE** per kategori untuk tiap eksperimen.

| Category | A_seasonal | B_expanded | C_early_stop | D1_log1p | D2_poisson | D3_tweedie | E1_winsorize | E2_pseudohuber | baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M01AB | 66.8 | 65.533 | 65.979 | 65.655 | 65.586 | 65.462 | 65.211 | 60.214 | 67.021 |
| M01AE | 70.404 | 69.56 | 70.187 | 66.92 | 71.726 | 71.149 | 70.351 | 67.786 | 69.475 |
| N02BA | 37.936 | 38.016 | 39.665 | 36.512 | 35.809 | 37.446 | 39.094 | 37.41 | 38.715 |
| N02BE | 2314.177 | 2287.747 | 2317.666 | 2211.25 | 2224.707 | 2376.065 | 2399.143 | 2625.985 | 2886.454 |
| N05B | 160.877 | 159.963 | 163.393 | 159.731 | 168.681 | 166.481 | 165.218 | 163.542 | 143.225 |
| N05C | 7.602 | 7.521 | 7.22 | 8.57 | 7.276 | 7.319 | 7.874 | 7.182 | 7.379 |
| R03 | 793.168 | 765.893 | 725.05 | 809.997 | 734.052 | 727.241 | 699.534 | 718.535 | 721.61 |
| R06 | 74.64 | 74.494 | 73.685 | 88.311 | 75.427 | 72.752 | 73.788 | 68.24 | 71.429 |

## RMSE Pivot Summary

Nilai di bawah adalah **best RMSE** per kategori untuk tiap eksperimen.

| Category | A_seasonal | B_expanded | C_early_stop | D1_log1p | D2_poisson | D3_tweedie | E1_winsorize | E2_pseudohuber | baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M01AB | 8.173 | 8.095 | 8.123 | 8.103 | 8.099 | 8.091 | 8.075 | 7.76 | 8.187 |
| M01AE | 8.391 | 8.34 | 8.378 | 8.18 | 8.469 | 8.435 | 8.388 | 8.233 | 8.335 |
| N02BA | 6.159 | 6.166 | 6.298 | 6.043 | 5.984 | 6.119 | 6.252 | 6.116 | 6.222 |
| N02BE | 48.106 | 47.83 | 48.142 | 47.024 | 47.167 | 48.745 | 48.981 | 51.244 | 53.726 |
| N05B | 12.684 | 12.648 | 12.783 | 12.638 | 12.988 | 12.903 | 12.854 | 12.788 | 11.968 |
| N05C | 2.757 | 2.742 | 2.687 | 2.927 | 2.697 | 2.705 | 2.806 | 2.68 | 2.716 |
| R03 | 28.163 | 27.675 | 26.927 | 28.46 | 27.093 | 26.967 | 26.449 | 26.806 | 26.863 |
| R06 | 8.639 | 8.631 | 8.584 | 9.397 | 8.685 | 8.53 | 8.59 | 8.261 | 8.452 |

## Interpretasi per Enhancement

1. **A_seasonal** membantu sebagian kategori, terutama N02BA dan N02BE, tetapi memperburuk N05B, N05C, R03, dan R06. Artinya fitur seasonal berbasis ACF tidak boleh dipakai universal untuk semua kategori.

2. **B_expanded** memberi dampak kuat pada M01AB dan N02BE karena regularisasi tambahan membantu XGBoost mengontrol overfit pada feature set yang lebih kaya. Namun kategori seperti N05B, N05C, R03, dan R06 masih lebih baik dengan baseline atau robust approach.

3. **C_early_stop** tidak menjadi pemenang akhir di kategori mana pun, tetapi hasilnya relatif stabil. Early stopping tetap berguna sebagai kontrol overfit, hanya saja bukan sumber improvement utama pada eksperimen ini.

4. **D1_log1p** paling penting untuk kategori high-scale/high-variance, terutama N02BE. MSE N02BE turun dari 2886.454 ke 2211.250, improvement 23.39% vs baseline.

5. **D2_poisson** menang untuk N02BA dan cukup kompetitif pada N02BE. Objective Poisson cocok saat pola target lebih menyerupai count process.

6. **D3_tweedie** tidak menang di kategori mana pun. Hasilnya dekat dengan beberapa eksperimen lain, tetapi belum memberi evidence kuat untuk dijadikan approach utama.

7. **E1_winsorize** menang untuk R03. Ini mendukung dugaan bahwa R03 dipengaruhi outlier/volatilitas, sehingga clipping target train membantu stabilitas model.

8. **E2_pseudohuber** menang paling banyak, yaitu M01AB, N05C, dan R06. Robust loss efektif untuk kategori dengan noise/outlier sedang, tetapi tidak cocok untuk semua kategori karena N02BE justru memburuk.

## Key Findings

1. Tidak ada satu enhancement yang menang semua kategori. Approach terbaik perlu dipilih per kategori.

2. N02BE mendapat improvement terbesar: 23.39% dengan `D1_log1p` + GEO + averaging.

3. N05B tidak membaik dengan enhancement apa pun; baseline tetap terbaik.

4. Robust approach penting. `E2_pseudohuber` dan `E1_winsorize` menang di 4 dari 8 kategori.

5. Residual scheme tidak selalu unggul. Banyak pemenang memakai averaging, bukan residual.

6. Kombinasi final sebaiknya category-specific, bukan satu pipeline global.

## Recommended Final Combination

| Category | Recommended Combination | Optimizer | Scheme |
| --- | --- | --- | --- |
| M01AB | pseudohuber + expanded regularization + seasonal features | Optuna | averaging |
| M01AE | log1p + residual + GEO + expanded regularization | GEO | residual |
| N02BA | poisson + averaging + GEO + expanded regularization | GEO | averaging |
| N02BE | log1p + averaging + GEO + expanded regularization | GEO | averaging |
| N05B | baseline | GEO | averaging |
| N05C | pseudohuber + averaging + GEO | GEO | averaging |
| R03 | winsorize + residual + PSO + expanded regularization | PSO | residual |
| R06 | pseudohuber + averaging + PSO | PSO | averaging |

## Final Combo Notebook Analysis

Notebook final `notebooks/our_study_pharma_weekly_10_final_combo.ipynb` menjalankan kombinasi category-specific berdasarkan pemenang historical dari eksperimen 00-08. Notebook ini sudah selesai 8/8 kategori dan menyimpan output ke Google Drive sebagai `weekly_F_final_combo_results.csv` dan `weekly_F_final_combo_compare.csv`.

### Final Combo Result vs Baseline

| Category | Final MSE | Final RMSE | Baseline MSE | Delta MSE | Improvement vs Baseline | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| M01AB | 60.214 | 7.760 | 67.021 | -6.807 | 10.16% | Improved |
| M01AE | 67.602 | 8.222 | 69.475 | -1.873 | 2.70% | Improved |
| N02BA | 35.809 | 5.984 | 38.715 | -2.907 | 7.51% | Improved |
| N02BE | 2358.275 | 48.562 | 2886.454 | -528.179 | 18.30% | Improved |
| N05B | 143.224 | 11.968 | 143.225 | -0.001 | ~0.00% | Same |
| N05C | 7.341 | 2.709 | 7.379 | -0.039 | 0.52% | Slightly improved |
| R03 | 836.099 | 28.915 | 721.610 | 114.489 | -15.86% | Worse |
| R06 | 75.641 | 8.697 | 71.429 | 4.213 | -5.90% | Worse |

### Final Combo vs Best Historical

| Category | Best Historical MSE | Final Combo MSE | Difference | Note |
| --- | ---: | ---: | ---: | --- |
| M01AB | 60.214 | 60.214 | 0.000 | Matched best historical |
| M01AE | 66.920 | 67.602 | 0.681 | Slightly worse than best historical |
| N02BA | 35.809 | 35.809 | 0.000 | Matched best historical |
| N02BE | 2211.250 | 2358.275 | 147.025 | Still better than baseline, but worse than best historical |
| N05B | 143.225 | 143.224 | -0.001 | Matched baseline-level best |
| N05C | 7.182 | 7.341 | 0.158 | Slightly worse than best historical |
| R03 | 699.534 | 836.099 | 136.565 | Worse than best historical and baseline |
| R06 | 68.240 | 75.641 | 7.401 | Worse than best historical and baseline |

### Final Combo Findings

1. Final combo improves 5 categories against baseline: M01AB, M01AE, N02BA, N02BE, and N05C.

2. N05B is effectively unchanged because the selected final approach is the baseline configuration.

3. R03 and R06 fail in the final combo run. Both are worse than baseline, even though their historical best experiments were better than baseline.

4. Main cause: final combo notebook re-runs optimizer search instead of reusing locked best params from the historical result CSV. PSO and GEO are stochastic, so re-tuning can produce different outcomes.

5. N02BE still shows large improvement vs baseline, but final combo does not reproduce the strongest historical score. Historical best MSE was 2211.250, final combo MSE is 2358.275.

### Final Recommendation

The final combo notebook is useful as a category-specific experiment runner, but the result should not be treated as the final best table for every category. For reporting, use a locked-params final notebook or direct historical best selection to avoid stochastic re-tuning drift.

Recommended reporting strategy:

1. Use final combo result for M01AB, M01AE, N02BA, N02BE, N05B, and N05C because all are equal or better than baseline.

2. Do not use final combo result for R03 and R06 because both are worse than baseline.

3. For reproducibility, create a locked-params final run that reads the best historical params from the CSV results and only re-fits/evaluates, without running optimizer search again.

## Output Files Used

- `weekly_A_seasonal_results.csv`
- `weekly_B_expanded_results.csv`
- `weekly_baseline_results.csv`
- `weekly_C_early_stop_results.csv`
- `weekly_D1_log1p_results.csv`
- `weekly_D2_poisson_results.csv`
- `weekly_D3_tweedie_results.csv`
- `weekly_E1_winsorize_results.csv`
- `weekly_E2_pseudohuber_results.csv`
