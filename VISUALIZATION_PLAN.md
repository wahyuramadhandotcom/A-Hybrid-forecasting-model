# Notebook Visualization Plan

Tujuan: tambah visualisasi `actual vs predicted` di semua notebook eksperimen dalam `notebooks/`, dengan format konsisten seperti `notebooks/our_study_pharma_daily.ipynb`.

## Format Standar

- PharmaSales 8 kategori: subplot grid `4x2`, `figsize=(18, 15)`.
- Rossmann multi-model: subplot per model atau satu plot per model sesuai struktur section.
- Actual: garis merah solid, label `Test Actual` atau `Actual`.
- Predicted: garis hijau putus-putus, label `Test Predicted` atau `Predicted`.
- Title memuat category/model, metric utama, dan parameter model jika tersedia.
- Legend di `upper left`.
- Ikuti format `notebooks/our_study_pharma_daily.ipynb`: `plt.subplots_adjust(wspace=0.1, hspace=0.3)`, tanpa memaksa grid.
- Hindari dependency baru.

## Output Cell Policy

- Tambahkan code cell visualisasi dulu tanpa memaksa rerun full training.
- Simpan output gambar hanya setelah rerun selective notebook disetujui.
- Jika prediction variable belum tersedia, ubah cell model seminimal mungkin agar menyimpan `y_test`, `y_pred`, `dates`, `model`, `category`, dan metric ke list/dict hasil.

## Step 1: Audit Struktur Notebook

- [ ] Cek section model dan result variable di semua notebook.
- [ ] Tandai notebook yang sudah punya visualisasi.
- [ ] Tandai notebook yang butuh prediction storage tambahan.

## Step 2: PharmaSales Daily/Weekly

- [x] `notebooks/our_study_pharma_daily.ipynb`
  - Rujukan format visualisasi. Do not modify.
- [x] `notebooks/our_study_pharma_weekly.ipynb`
  - Sudah punya visualisasi dengan format PharmaSales.
- [x] `notebooks/rathipriya_pharma_daily.ipynb`
  - Tambah visualisasi Reference best per category.
  - Tambah visualisasi Our Preprocessing best per category.
- [x] `notebooks/ramadhan_pharma_daily.ipynb`
  - Tambah visualisasi LR-XGBoost per ATC category.
- [x] `notebooks/fourkiotis_pharma_weekly.ipynb`
  - Review visualisasi existing Reference/Our.
  - Samakan style/title.
  - Tambah plot jika section belum lengkap.
- [x] `notebooks/zdravkovic_pharma_weekly.ipynb`
  - Tambah visualisasi Reference best per category.
  - Tambah visualisasi Our Preprocessing best per category.

## Step 3: Rossmann Benchmark Notebooks

- [x] `notebooks/diamantini_2024_rossmann.ipynb`
  - Tambah visualisasi tiap model Reference.
  - Tambah visualisasi tiap model Our Preprocessing.
- [x] `notebooks/malik_rossmann_daily.ipynb`
  - Tambah visualisasi ARIMA, FB Prophet, XGBoost scaled/unscaled untuk Reference.
  - Tambah visualisasi model yang sama untuk Our Preprocessing.
- [x] `notebooks/qureshi_rossmann_daily.ipynb`
  - Tambah visualisasi LSTM dan GRU Reference.
  - Tambah visualisasi LSTM dan GRU Our Preprocessing.
- [x] `notebooks/zeng_rossmann_daily.ipynb`
  - Tambah visualisasi XGBoost, LightGBM, TS-XGBoost, TS-LGBM Reference.
  - Tambah visualisasi model yang sama untuk Our Preprocessing.
- [x] `notebooks/zhaoweijie_rossmann_daily.ipynb`
  - Tambah visualisasi XGBRegressor unadjusted, XGBoost baseline, XGBoost tuned v1 Reference.
  - Tambah visualisasi model yang sama untuk Our Preprocessing.

## Step 4: Our Study Rossmann

- [x] `notebooks/our_study_rosman.ipynb`
  - Review visualisasi transformed sales.
  - Review visualisasi non-transformed sales.
  - Sudah punya visualisasi actual vs predicted untuk section test/load transformed dan non-transformed.

## Step 5: Template Notebook

- [x] `notebooks/format.ipynb`
  - Skip. Tidak diperlukan untuk target visualisasi hasil eksperimen.

## Step 6: Validation

- [x] Validasi semua notebook JSON valid.

```bash
python -c "import json, pathlib; [json.loads(p.read_text(encoding='utf-8')) for p in pathlib.Path('notebooks').glob('*.ipynb')]; print('ok')"
```

- [x] Grep semua heading/cell visualisasi.

```bash
rg -n "Visualisasi|Visualization|actual vs predicted|Actual|Predicted" notebooks
```

- [ ] Review diff notebook supaya tidak ada perubahan output besar yang tidak disengaja.
- [ ] Jika user setuju, rerun selective notebook untuk menyimpan output gambar.

## Commit Plan

- Commit 1: PharmaSales visualizations.
- Commit 2: Rossmann visualizations.
- Commit 3: template + cleanup + validation.
