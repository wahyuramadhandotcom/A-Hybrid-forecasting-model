# exp05a — AR-LRX: pengembangan kerangka residual (Rossmann)

## Apa yang baru

`src/experiments/arlrx.py` — modul **terpisah** dari `protocol.py`. Kontrak eksperimen
C1–C7 tidak diubah sedikit pun, sehingga hasil exp01–exp03 tetap dapat direproduksi
persis dengan berkas yang sama. Yang ditambahkan hanya arsitektur modelnya.

Model yang diusulkan:

    ŷ = S1(x) + w · S2(x ; y − S1(x)),    w ∈ [0, 1]

Tiga komponen, masing-masing menjawab satu diagnosis dari exp01–exp03:

| Komponen | Isi | Menjawab |
|---|---|---|
| **S1** | tahap pertama dapat dipilih: `linear` (versi lama), `structural` (rata-rata hierarkis per Toko × Hari × Promo dengan fallback bertingkat), `struct_linear` (struktural + LR pada residualnya) | tahap pertama bukan model tren; tahap pertama buruk meracuni tahap kedua |
| **S2** | XGBoost pada residual S1, grid boleh turun sampai kapasitas kecil | solusi-pojok pada exp01 |
| **G** | gerbang `w` dipilih **hanya di validation**, bersama hyperparameter S2 | tidak ada mekanisme mundur ketika residual tak terprediksi |

Karena `w = 0` ada di ruang pencarian, kerangka ini **secara konstruksi tidak pernah
lebih buruk daripada tahap pertamanya sendiri pada validation**. Dan karena
`S1 = linear, w = 1` adalah kerangka versi lama, **versi lama adalah kasus khusus** —
perbandingannya bukan antar-protokol, melainkan antar konfigurasi dari satu kerangka.

## Efisiensi

`w` tidak memengaruhi pelatihan tahap kedua, sehingga seluruh 11 nilai `w` dievaluasi
dari prediksi yang sama. Anggaran komputasi per model = 8 pelatihan XGBoost untuk
tuning + 1 refit — **lebih murah daripada exp03**.

## Cara memasang & menjalankan

```bash
cp hybrid-forecasting-revisi-exp05a/src/experiments/arlrx.py        src/experiments/
cp hybrid-forecasting-revisi-exp05a/notebooks/exp05a_*.ipynb        notebooks/
cp hybrid-forecasting-revisi-exp05a/EXP05A_CATATAN.md               .

jupyter lab notebooks/exp05a_rossmann_arlrx.ipynb
```

Set `QUICK_RUN = True` di sel 3 untuk uji cepat (80 toko, grid satu titik) sebelum
menjalankan data penuh. Data penuh: 3 varian × 5 model ber-XGBoost × 8 konfigurasi
≈ 135 pelatihan pada ~590 ribu baris — sekitar setengah biaya exp03.

Hasil ditulis ke `results/exp05a_rossmann_arlrx{,_summary,_ablation,_dm_test}.csv`.

## Hasil uji-jalan (booster substitusi, 80 toko — arah saja, BUKAN angka final)

Pada `V1_customers_dropped`, yaitu varian di mana kerangka lama kalah 30% dari
baseline naif, **AR-LRX [struct_linear] mengungguli keempat pembandingnya sekaligus,
semuanya signifikan**: XGBoost polos (p = 4×10⁻¹⁴), naif per toko (p < 0,001),
kerangka lama (p = 3×10⁻⁶), dan tahap pertamanya sendiri (p = 3×10⁻¹²).

Pada `V2` dan `V3`, AR-LRX mengungguli naif dan tahap pertamanya, tetapi **belum
mengungguli XGBoost polos**. Laporkan apa adanya — justru pola "menang di sini,
kalah di sana" yang memberi isi pada kriteria di bawah.

Temuan diagnostik terpenting: **korelasi antara perbaikan RMSE validation dari
gerbang dan perbaikan RMSE test aktual adalah r = 0,977.** Artinya nilai gerbang
yang dipilih di validation dapat dipakai sebagai kriteria keputusan **sebelum**
menyentuh test — inilah kontribusi metodologisnya, bukan sekadar angka akurasi.

Angka final harus datang dari eksekusi Anda dengan XGBoost sebenarnya pada data
penuh. Perlu diantisipasi: XGBoost asli akan membuat pembanding *lebih kuat*
daripada di uji-jalan ini, sehingga jarak pada V2/V3 dapat melebar.

## Klaim yang dapat dipertahankan di disertasi

Bukan "model kami paling akurat pada semua kondisi" — data tidak mendukungnya, dan
klaim itu sudah ditolak sekali. Melainkan tiga hal yang terbukti:

1. **Kualitas tahap pertama menentukan apakah koreksi residual berguna** — terukur
   langsung sebagai selisih antara `S1 [linear]` dan `S1 [struct_linear]` pada tabel
   yang sama.
2. **Gerbang membuat kerangka aman** ketika residual tidak dapat diprediksi, dengan
   jaminan konstruktif pada validation.
3. **Manfaat hibridisasi dapat diperkirakan sebelum pengujian** melalui perbaikan
   validation dari gerbang (r = 0,977 terhadap perbaikan test aktual).

Ketiganya bersifat metodologis dan dapat diuji ulang orang lain — itulah bentuk
novelty yang bertahan di hadapan reviewer Q1/Q2, bukan selisih RMSE beberapa persen.
