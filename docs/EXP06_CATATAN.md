# exp06 — baseline rujukan di bawah protokol terpadu

## Mengapa ini eksperimen yang menentukan

Naskah lama mengadu `RMSE 577,63` melawan CNN 775,28, MLP 825,32, Transformer 881,28.
Klaim itu tidak sah — bukan hanya karena 577,63 berasal dari `V0` yang bocor, tetapi
karena **angka rujukan itu pun direproduksi lewat pipeline yang sama**, pipeline dengan
kebocoran `Customers` dan pembagian berbasis indeks.

Model Anda memburuk 577,63 → 1004,69 (**+74%**) begitu protokolnya dibetulkan, lalu
turun ke 945,42 setelah exp05d. Tidak ada alasan baseline deep learning kebal terhadap
koreksi yang sama. exp06 menjalankan ulang mereka semua di bawah kontrak yang identik.

## Cakupan

Varian tunggal **V3_sales_lagged** — paling dekat dengan keadaan yang diasumsikan
naskah rujukan (penjualan historis tersedia sebagai prediktor) dan paling kaya
informasi. Delapan baseline:

| Model | Sumber |
|---|---|
| MLP, CNN, RNN, LSTM, Transformer | Diamantini et al. |
| GRU | Qureshi et al. |
| LightGBM | Zeng et al. |
| XGBoost (grid Zhao) | Zhao et al. |

Semua menerima matriks fitur yang sama, split kronologis yang sama, seed yang sama,
penyetelan hanya pada validation, refit pada train+val, test disentuh satu kali, dan
evaluasi dikembalikan ke skala asli lewat `expm1`.

## Tiga keputusan desain yang harus dinyatakan di naskah

**1. Representasi model berurut.** CNN, RNN, LSTM, GRU, Transformer menerima matriks
yang sama, dibentuk ulang menjadi `(n, p, 1)`. Ini bukan kompromi yang merugikan
mereka: informasi temporal memang sudah berada di dalam fitur (`lag_1` penjualan,
rata-rata bergulir). Membangun jendela waktu terpisah **hanya** untuk model berurut
justru memberi mereka informasi yang tidak dimiliki pembanding, dan itulah yang akan
merusak kesahihan perbandingan.

**2. Anggaran pelatihan ditetapkan, bukan early stopping.** Jumlah epoch sama antara
fase penyetelan dan fase refit — syarat agar refit pada train+val sah tanpa menyentuh
test. Early stopping akan memberi model neural akses ke sinyal yang tidak dimiliki
model pohon.

**3. Yang tidak disertakan, dan mengapa.** TS-XGBoost dan TS-LGBM (Zeng) runtuh
menjadi model pohon biasa di bawah protokol ini, karena rekayasa fitur temporalnya
kini milik bersama semua model — memasukkannya berarti menduplikasi XGBoost. ARIMA dan
Prophet adalah model deret tunggal, sedangkan Rossmann adalah panel 1.115 toko.

## Uji-cepat sebelum membuang berjam-jam

Bagian 1 notebook membangun, melatih satu epoch, dan memprediksi dengan **setiap**
arsitektur pada 256 baris sintetis. Bila ada galat API Keras, ia muncul dalam 30 detik,
bukan setelah dua jam. **Jalankan sel itu lebih dulu dan pastikan lulus** sebelum
melanjutkan — saya tidak dapat menguji jalur Keras di lingkungan saya karena PyPI
diblokir, jadi sel inilah pengganti dry-run untuk bagian tersebut. Sisa alurnya
(integrasi `run_model`, penskalaan, grid, perakitan tabel, uji DM, penyimpanan
prediksi) sudah saya uji end-to-end dengan stub.

## Keluaran

| Berkas | Isi |
|---|---|
| `exp06_..._table_utama.csv` | **Pengganti Tabel 3 naskah lama** — semua model, satu protokol |
| `exp06_..._lama_vs_adil.csv` | Pembelaan metodologis: berapa besar tiap angka menggelembung |
| `exp06_..._dm.csv` | Uji Diebold-Mariano dua skala terhadap setiap baseline |
| `exp06_..._predictions.npz` | Prediksi test seluruh baseline |

Tabel `lama_vs_adil` adalah yang paling penting secara retoris. Bila baseline ikut
memburuk dengan proporsi sebanding dengan model usulan, itu bukti langsung bahwa
perbandingan lama tidak sah — sekaligus menjelaskan mengapa 577,63 tidak boleh
dipakai lagi, tanpa membuat model Anda tampak lemah.

## Bila hasilnya tidak sesuai harapan

Ada kemungkinan nyata CNN atau MLP tetap unggul walau protokolnya diadilkan.
Bila itu terjadi: laporkan apa adanya, dan arahkan klaim ke sumbangan metodologis yang
sudah kokoh dari exp05b/exp05d — **kapan dan mengapa** hibridisasi residual bekerja,
bukan klaim akurasi tertinggi. Itu tetap paper Q1/Q2 yang sah. Yang tidak akan
selamat adalah klaim akurasi yang dibangun di atas perbandingan lintas-protokol.

## Cara menjalankan

1. Salin `src/experiments/baselines.py` (berkas baru) ke `src/experiments/`.
2. Salin `notebooks/exp06_rossmann_baselines_unified.ipynb` ke `notebooks/`.
3. Pastikan `keras`, `tensorflow`, `lightgbm` terpasang (sudah ada di `pyproject.toml`).
4. **Restart Kernel**, jalankan bagian 1 (uji-cepat) lebih dulu.
5. Bila lulus: `QUICK_RUN = True` sekali untuk memastikan pipeline penuh jalan
   (80 toko, 3 epoch, beberapa menit), baru `QUICK_RUN = False`.
6. `EPOCHS` default 12 — turunkan ke 8 bila terlalu lama, naikkan ke 20 bila cepat.
7. Kirim balik keempat CSV dan notebook ber-output.

## Keterbatasan determinisme

Model neural di CPU tidak selalu identik bit-per-bit antar-jalan walaupun seed
disetel (multithreading BLAS). Seluruh model pohon dan AR-LRX sudah diverifikasi
identik. Nyatakan keterbatasan ini di naskah.
