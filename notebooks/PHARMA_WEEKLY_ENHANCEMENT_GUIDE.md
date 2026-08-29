# Pharma Weekly Enhancement — Panduan Notebook

Eksperimen peningkatan forecasting **LR + XGBoost** pada `salesweekly.csv`.
Semua notebook memakai fondasi sama: **4 optimizer** (Grid Search, Optuna, PSO, GEO)
× **2 skema hybrid** (averaging & residual) × **8 kategori**, tuning di validation,
dievaluasi MSE/RMSE di test.

## Tujuan Analisis per Notebook

1. **`00_baseline`** — Titik acuan. Fitur ACF dasar (lag signifikan + rolling mean) + XGBoost 3-HP. Menghasilkan `weekly_baseline_results.csv` sebagai pembanding semua enhancement. Versi bersih tanpa leakage (tuning di validation, bukan test).

2. **`01_acf_seasonal`** (Poin 4) — Menambah fitur musiman: Fourier terms periode 52, week-of-year, month, quarter, semua lag ACF signifikan, rolling std/min/max, dan diff. Tujuan: cek apakah fitur musiman menekan error, khususnya kategori musiman (R03, R06, N02BE).

3. **`02_expanded_regularization`** (Poin 5) — Memperluas search space XGBoost dari 3 HP jadi 8 HP (`reg_alpha, reg_lambda, subsample, colsample_bytree, min_child_weight`). Tujuan: cek apakah regularisasi lebih kaya mengurangi overfit di kategori bervariasi tinggi (N02BE, R03, N05B).

4. **`03_early_stopping`** (Poin 6) — XGBoost pakai `eval_set` + `early_stopping_rounds`. Tujuan: cek apakah menghentikan jumlah pohon otomatis (bukan fixed n_estimators) memperbaiki generalisasi.

5. **`04_transform_log1p`** (Poin 7.1) — Target di-`log1p`, prediksi di-`expm1`. Tujuan: cek apakah stabilisasi variance membantu kategori berskala besar/skew (N02BE, R03).

6. **`05_transform_poisson`** (Poin 7.2) — Objective `count:poisson`. Tujuan: cek apakah objective khusus data hitungan (count) lebih cocok daripada `reg:squarederror` default.

7. **`06_transform_tweedie`** (Poin 7.3) — Objective `reg:tweedie`. Tujuan: sama seperti Poisson tapi untuk target non-negatif condong skew; bandingkan mana yang lebih baik.

8. **`07_robust_winsorize`** (Poin 8.1) — Clip target train di persentil 1%/99% (test tidak disentuh). Tujuan: cek apakah meredam outlier memperbaiki kategori dengan outlier tinggi (R03, N05C).

9. **`08_robust_pseudohuber`** (Poin 8.2) — Objective `reg:pseudohubererror`. Tujuan: cek apakah loss robust terhadap outlier mengalahkan MSE default di kategori bervariasi tinggi.

10. **`09_summary`** — Tidak menjalankan optimizer. Membaca semua CSV hasil, membuat pivot best RMSE/MSE per kategori per section, menentukan pemenang tiap kategori, dan menampilkan konfigurasi pemenang. Bahan keputusan kombinasi.

## Urutan Run (Colab)

1. Jalankan `00_baseline` dulu (menghasilkan CSV acuan).
2. Jalankan notebook eksperimen mana pun (`01`–`08`) secara terpisah.
3. Jalankan `09_summary` untuk menggabungkan hasil dan melihat pemenang per kategori.
