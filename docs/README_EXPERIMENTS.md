# Protokol eksperimen terpadu (revisi pasca-review IJIES)

Dokumen ini mendeskripsikan pipeline eksperimen yang menggantikan perbandingan
lintas-notebook pada versi naskah sebelumnya. Isinya dirancang untuk dapat disalin
menjadi bagian *Experimental Setup* dan *Reproducibility* pada naskah.

---

## 1. Mengapa pipeline lama tidak dapat dipertahankan

Audit kode menemukan enam cacat yang masing-masing cukup untuk membatalkan klaim
komparatif. Kolom terakhir menunjuk ke penanganannya di pipeline baru.

| # | Temuan | Bukti di repo lama | Penanganan |
|---|---|---|---|
| 1 | **Perbandingan tidak apple-to-apple.** Baris "Reference" pada Tabel 2 dilatih hanya dengan `lag_1` dan split 70/15/15; model usulan dilatih dengan 2-16 fitur dan split 60/20/20. | `rathipriya_pharma_daily.ipynb` cell 6 vs `our_study_pharma_daily.ipynb` cell 18 | Set fitur dijadikan **faktor eksperimen**; semua model dijalankan pada `A_lag1` dan `B_rich` dengan split identik (exp01/exp02) |
| 2 | **Tuning pada test set.** Cabang "Our Preprocessing" pada notebook baseline memakai split 80/20 tanpa validation dan memilih hyperparameter dengan menilai test. | `rathipriya_pharma_daily.ipynb` cell 26 | Kontrak C3: seleksi hyperparameter **hanya** dari RMSE validation; test disentuh sekali |
| 3 | **Kebocoran fitur `Customers` pada Rossmann.** `Customers` hari-H dipakai sebagai prediktor `Sales` hari-H, padahal kolom itu tidak ada pada `test.csv` kompetisi. | `our_study_rosman.ipynb` cell 5 + 52 | Ablasi V0/V1/V2/V3 (exp03); V0 hanya dilaporkan sebagai *leakage upper bound* |
| 4 | **Nondeterminisme.** 14 instansiasi `XGBRegressor` tanpa `random_state` pada tiga notebook utama - termasuk model final yang angkanya masuk ke naskah. | `our_study_pharma_daily.ipynb` cells 22, 24, 27, 35, 44, 53; `our_study_pharma_weekly.ipynb` cells 23 (x2), 25, 32, 41, 50; `our_study_rosman.ipynb` cell 42 | Semua sudah di-patch; konstruktor tunggal `P.make_xgb()` memasang `random_state=SEED`; sel pemeriksaan determinisme di setiap notebook baru |
| 5 | **Train/predict scaling mismatch.** Model di-`fit` pada `X_train` mentah lalu `predict` pada `X_test_scaled`. | `our_study_rosman.ipynb` cell 42 | Sudah di-patch; pipeline baru tidak memakai scaler untuk model pohon sama sekali |
| 6 | **Kebocoran pada desain fitur.** ACF/PACF untuk memilih jumlah lag dihitung pada seluruh deret termasuk test; dan dua notebook memakai statistik berbeda (PACF vs ACF) untuk pipeline yang seharusnya dibandingkan. | `our_study_pharma_daily.ipynb` cell 13 vs `rathipriya_pharma_daily.ipynb` cell 25 | Kontrak C5: statistik dihitung pada blok training saja; aturan seleksi lag menjadi ablasi tersendiri |

Cacat tambahan yang ditemukan dan ikut diperbaiki:

* **Grid bandwidth kernel terpotong.** Pada grid lama (`sigma` maksimum 5,0) optimum
  validation jatuh persis di batas atas untuk sebagian besar kategori. Grid yang
  memotong optimum secara sistematis merugikan GRNN dan P_NN, sehingga selisih
  terhadap metode usulan menjadi berlebihan. Grid diperlebar sampai `sigma = 50`.
* **Nilai grid yang tidak valid.** `subsample: [0.8, 2.0]` dan
  `colsample_bytree: [0.8, 2.0]` - kedua parameter terdefinisi pada (0, 1].
* **Fitur duplikat.** Untuk kategori dengan k = 1, `rolling_mean_1` identik dengan
  `lag_1` (kolinearitas sempurna pada regresi linier). Kolom duplikat dibuang, dan
  konsekuensinya dilaporkan apa adanya.
* **Split memotong tanggal.** Pada Rossmann, split berbasis indeks baris pada panel
  1.115 toko menempatkan sebagian baris tanggal batas di train dan sisanya di test.
  Split baru berbasis tanggal.
* **Tidak ada baseline naif.** Ditambahkan Naive, Seasonal Naive (PharmaSales), dan
  median per (Toko x Hari x Promo) (Rossmann).
* **Tidak ada uji signifikansi.** Ditambahkan uji Diebold-Mariano dengan koreksi
  Harvey-Leybourne-Newbold.

---

## 2. Kontrak eksperimen

Dikodekan di `src/experiments/protocol.py`. Semua notebook memakai modul ini,
sehingga protokolnya tidak dapat menyimpang antar eksperimen.

| Kode | Ketentuan |
|---|---|
| **C1** | `SEED = 42` dipasang ke `random`, NumPy, dan `PYTHONHASHSEED`. Setiap `XGBRegressor` dibuat lewat `make_xgb()` yang selalu memasang `random_state=SEED`. `KMeans` pada RBFNN juga di-seed. |
| **C2** | Split kronologis 70 / 15 / 15 tanpa pengacakan, identik untuk semua model dan semua set fitur. |
| **C3** | Hyperparameter dipilih hanya dari RMSE validation. Test split diprediksi tepat satu kali oleh model final. |
| **C4** | Set fitur adalah faktor eksperimen: `A_lag1` dan `B_rich`. |
| **C5** | Statistik pemilihan lag dihitung pada blok training saja. |
| **C6** | Scaler di-fit ulang pada blok training aktif (train untuk tuning, train+val untuk refit), tidak pernah pada test. |
| **C7** | Setiap baris hasil menyimpan konfigurasi lengkap dan ditulis sebagai CSV + JSON. |

Setelah tuning, model final **di-refit pada train+val**. Ini disengaja: model final
memakai seluruh data yang tersedia sebelum test, dan berlaku sama untuk setiap model,
sehingga anggaran datanya setara.

---

## 3. Struktur berkas

```
src/experiments/protocol.py          kontrak eksperimen (satu-satunya sumber kebenaran)
notebooks/
  exp01_pharma_daily_unified.ipynb   PharmaSales harian, semua model x 2 set fitur
  exp02_pharma_weekly_unified.ipynb  PharmaSales mingguan, idem
  exp03_rossmann_leakage_ablation.ipynb  ablasi kebocoran Customers (V0-V3)
  exp04_paper_tables.ipynb           merakit tabel naskah dari results/ (tanpa melatih model)
results/
  <eksperimen>.csv                   satu baris per (kategori, set fitur, model)
  <eksperimen>.meta.json             versi pustaka, seed, definisi protokol
  <eksperimen>_splits.csv            tanggal dan ukuran setiap blok
  <eksperimen>_dm_test.csv           hasil uji Diebold-Mariano
  paper_tables/                      tabel siap-tempel (.csv/.md/.tex)
```

Urutan menjalankan: exp01 -> exp02 -> exp03 -> exp04.

---

## 4. Model yang dibandingkan

Seluruh model di bawah menerima matriks fitur, split, dan anggaran tuning yang sama.

| Model | Tuning | Catatan |
|---|---|---|
| Naive (`y_{t-1}`) | - | baseline wajib |
| Seasonal Naive (`y_{t-s}`) | - | s = 7 (harian), 52 (mingguan) |
| ARIMA(5,1,0) | - | univariat, ditandai eksplisit karena tidak memakai fitur |
| Linear Regression | - | komponen tren dari metode usulan |
| GRNN | `sigma` (16 nilai) | fitur distandardisasi |
| P_NN | `sigma` (16 nilai) | fitur distandardisasi |
| RBFNN | `n_centers` x `gamma` x `alpha` (60) | pusat via KMeans ber-seed |
| XGBoost | `GRID_XGB_PHARMA` (12) | - |
| LR+XGB (rata-rata) | `GRID_XGB_PHARMA` (12) | baseline hibrida terdahulu |
| **LR-XGB (residual)** | `GRID_XGB_PHARMA` (12) | **metode usulan** |

Pada Rossmann: Seasonal Naive per toko, LR, XGBoost, LR+XGB (rata-rata), dan
LR-XGB (residual), dengan `GRID_XGB_ROSSMANN` (12 konfigurasi).

---

## 5. Pelaporan metrik

* `compute_metrics` menghitung MSE lebih dahulu, lalu `RMSE = sqrt(MSE)`, sehingga
  keduanya konsisten secara eksak menurut konstruksi. exp04 memuat pemeriksaan yang
  gagal secara eksplisit bila konsistensi ini dilanggar.
* Untuk Rossmann, metrik skala log (`test_*`) dan skala asli (`orig_*`) selalu
  dilaporkan pada blok kolom terpisah dengan nama yang menyebut skalanya. Keduanya
  tidak pernah dicampur dalam satu kolom.
* RMSPE disertakan untuk Rossmann karena itu metrik resmi kompetisinya.
* Tidak ada sel metrik yang boleh kosong; exp04 memverifikasinya.

---

## 6. Cara menjalankan

```bash
# lingkungan
pip install numpy pandas scikit-learn scipy statsmodels xgboost matplotlib jupyter

# eksperimen
jupyter lab notebooks/exp01_pharma_daily_unified.ipynb
jupyter lab notebooks/exp02_pharma_weekly_unified.ipynb
jupyter lab notebooks/exp03_rossmann_leakage_ablation.ipynb   # QUICK_RUN=True untuk uji cepat
jupyter lab notebooks/exp04_paper_tables.ipynb
```

exp01 dan exp02 selesai dalam hitungan menit. exp03 pada data penuh memerlukan
beberapa jam pada CPU multi-core; set `QUICK_RUN = True` untuk memvalidasi pipeline
lebih dulu pada subset 100 toko.

---

## 7. Cara membingkai hasilnya di naskah

Judul baru - *Evaluating the Empirical Robustness of a Residual-Based Linear
Regression-XGBoost Framework* - sudah tepat, dan pipeline ini mendukungnya secara
langsung. Klaim yang dapat dipertahankan adalah klaim tentang **kondisi**: di bawah
protokol tunggal, pada set fitur mana dan pada granularitas mana kerangka residual
memberi keunggulan, dan di mana ia tidak. Bila hasilnya menunjukkan keunggulan yang
kecil atau tidak signifikan, itu tetap merupakan kontribusi yang layak terbit sebagai
studi robustness - dan jauh lebih kuat daripada klaim superioritas yang bersandar pada
perbandingan yang tidak setara.

Tiga hal yang sebaiknya dinyatakan terbuka:

1. Angka Rossmann pada versi sebelumnya berasal dari konfigurasi yang memakai
   `Customers` kontemporer, dan harus diganti dengan hasil bebas-kebocoran.
2. Aturan pemilihan lag pada pipeline lama tidak stabil; sensitivitas hasil terhadap
   pilihan itu dilaporkan sebagai ablasi, bukan disembunyikan.
3. Bila metode usulan tidak berbeda signifikan dari baseline pada sebagian kategori,
   nyatakan demikian dan batasi klaim pada kondisi di mana keunggulannya bertahan.
