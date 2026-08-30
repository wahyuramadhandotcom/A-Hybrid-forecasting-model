# exp06b — memperkuat baseline rujukan

## Mengapa memperkuat lawan, bukan metode sendiri

exp06 menempatkan AR-LRX-Aug di peringkat 1 dari 15 dan menang signifikan 8/8.
Tetapi baseline neuralnya tidak layak dipertahankan:

| Gejala | Angka |
|---|---|
| Model neural yang kalah dari **baseline naif musiman** (1274,63) | **6 dari 6** |
| Transformer `val_R2` | **0,153** |
| Memilih `units=64`, yaitu **batas atas grid** | **5 dari 6** |
| `val_R2` rata-rata neural vs pohon | 0,562 vs 0,809 |

MLP yang dilatih layak tidak mungkin kalah dari "median penjualan toko ini di hari
yang sama". Dan solusi-pojok pada grid adalah cacat yang **sudah Anda dokumentasikan
sendiri di exp01** — naskah akan tidak konsisten dengan standar auditnya sendiri.

**Menang 7% atas baseline yang terlatih baik jauh lebih berharga daripada menang 65%
atas baseline yang rusak.**

## Tiga perbaikan, menurut besar pengaruhnya

**1. Target distandarkan — ini akar masalahnya.** Jaringan harus memanjat dari sekitar
0 menuju rata-rata `log1p(sales)` ≈ 8,5 dengan loss MSE, sehingga sebagian besar
anggaran epoch habis hanya untuk mencocokkan intercept. Model pohon tidak punya
masalah itu — itulah sebab utama jaraknya tampak begitu lebar. Rata-rata dan simpangan
baku dihitung **hanya** dari blok pelatihan aktif; prediksi dikembalikan ke skala semula.

**2. Early stopping.** Maksimum 30 epoch, `patience=4`, dinilai terhadap **15% terakhir
secara kronologis** dari blok pelatihan aktif, dengan pemulihan bobot terbaik dan
`ReduceLROnPlateau`. Blok test tidak pernah tersentuh, dan prosedurnya identik antara
fase penyetelan dan fase refit — kontrak protokol tetap utuh.

**3. Grid dilebarkan** ke `units ∈ {64, 128, 256}`. Bagian 6 notebook memeriksa apakah
pemenangnya masih di batas grid, dan mencetak perbandingannya dengan exp06 (5 dari 6).

## Perlindungan untuk run panjang

Perkiraan **5–9 jam** di CPU. Hasil ditulis ke cakram **setiap kali satu model
selesai** (`exp06b_..._partial.json` dan `_predictions.npz`), sehingga gangguan pada
jam keempat tidak menghapus tiga jam pertama. Untuk melanjutkan run yang terputus,
persempit daftar `MODELS_TO_RUN` di sel bagian 2 ke model yang belum selesai.

Urutan biaya dari exp06 (12 epoch): Transformer 51,5 mnt ≫ GRU 17,7 ≫ LSTM 15,5 ≫
RNN 4,6 ≫ CNN 3,8 ≫ MLP 1,0. Dengan grid tiga tingkat dan early stopping, Transformer
adalah risiko waktu terbesar — bila terlalu lama, keluarkan ia dari `MODELS_TO_RUN`
dan laporkan sebagai keterbatasan.

## Yang akan terjadi, dan mengapa itu baik

Baseline akan membaik, dan keunggulan Anda akan **menyempit**. Itu bukan kemunduran —
itu justru yang membuat klaimnya sah. Tabel bagian 4 (exp06 lemah vs exp06b kuat)
adalah bukti bahwa Anda sengaja memperkuat lawan sebelum mengklaim menang. Reviewer
sangat jarang melihat penulis melakukan ini.

Yang harus dinyatakan terbuka di naskah: baseline neural mendapat standardisasi target,
early stopping, dan grid tiga tingkat, sedangkan AR-LRX tidak mendapat perlakuan setara
karena ia berbasis pohon dan tidak memerlukannya.

## Bila keunggulan menjadi tipis atau hilang

Arahkan klaim ke sumbangan metodologis exp05b/exp05d — **kapan dan mengapa**
hibridisasi residual bekerja: keselarasan tahap pertama dengan struktur data,
gerbang terpelajar yang mengungguli gerbang tabel, dan monotonisitas manfaat terhadap
kekayaan informasi tahap kedua. Itu tetap paper Q1/Q2 yang sah, dan tidak bergantung
pada peringkat akurasi.

## Cara menjalankan

1. Ganti `src/experiments/baselines.py` dengan berkas di paket ini
   (`make_keras_fp` lama tidak disentuh, exp06 tetap tereproduksi).
2. Salin `notebooks/exp06b_rossmann_baselines_strong.ipynb` ke `notebooks/`.
3. **Restart Kernel**, jalankan bagian 1 (uji-cepat) lebih dulu — ia memverifikasi
   bahwa standardisasi target bekerja, dengan melatih 3 epoch pada data bertaraf 8,5
   dan memeriksa apakah prediksinya langsung berada di level itu.
4. Jalankan semalaman. Pantau waktu per model yang dicetak.
5. Kirim balik: `exp06b_..._table_utama.csv`, `_exp06_vs_exp06b.csv`, `_dm.csv`,
   `exp06b_rossmann_baselines_strong.csv`, dan notebook ber-output.

## Sisa pekerjaan setelah ini

**exp05c (PharmaSales)** masih tertunda. Itu yang menutup argumen gerbang — satu-satunya
tempat sifat pengaman `w = 0` dapat dibuktikan benar-benar bekerja — dan memberi
dataset kedua untuk disertasi.
