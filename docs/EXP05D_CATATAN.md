# exp05d — pengembangan AR-LRX: gerbang terpelajar vs gerbang tabel

## Ide, dan dari mana datangnya

Dua fakta dari exp05b yang menjadi titik tolak:

1. Gerbang skalar praktis **tidak menyumbang apa pun** (rata-rata +0,075% untuk tahap
   pertama yang baik). Jadi masalahnya bukan penyetelan, melainkan **bentuk gerbangnya
   terlalu kaku**.
2. Tabel segmen menunjukkan pola yang sangat tidak seragam. Pada V3, AR-LRX kalah dari
   XGBoost di Selasa (+12,5%), Kamis (+13,7%), promo (+1,8%), toko kecil (+5,6%),
   tetapi menang besar di Sabtu (−17,9%), Senin (−10,6%), Minggu (−7,2%).

Dua cara memperbaikinya, diuji berdampingan dalam desain faktorial 2×2:

| | Gagasan | Bentuk gerbang |
|---|---|---|
| **Seg** | `w` dipilih per segmen (promo / hari / hari×promo / kuartil toko) | tabel pencarian, konstan-sepotong |
| **Aug** | prediksi tahap pertama dijadikan **fitur** bagi tahap kedua | terpelajar, kontinu, bergantung fitur |

**Aug** secara teoretis lebih kuat: bila tahap kedua mengetahui `S1(x)`, ia dapat
mempelajari sendiri di mana tahap pertama lemah dan seberapa besar koreksi yang pantas.
Itu gerbang terpelajar yang secara ketat **lebih umum** daripada tabel `w(s)`.

## Hasil uji-jalan (stub XGBoost, 40 toko — arah saja, bukan angka final)

| Komponen | Efek rata-rata vs gerbang skalar |
|---|---:|
| **+Aug** | **−1,01%** (dan −3,4% pada V3) |
| +Seg | +0,26% (netral sampai sedikit merugikan) |
| +Aug+Seg | −0,66% (lebih buruk daripada Aug sendirian) |

Segmen yang berbalik dari kalah menjadi menang atas XGBoost: **3** — persis segmen
tengah-pekan yang bermasalah pada V3 (Selasa +2,47 → −1,19; Rabu +5,45 → +0,60;
Kamis +7,78 → −0,01).

Jadi hipotesisnya terdukung: yang menentukan bukan **banyaknya** parameter gerbang,
melainkan apakah gerbang itu dapat **bergantung pada fitur secara kontinu**. Kontras
"gerbang tabel gagal, gerbang terpelajar berhasil" justru yang membuat temuannya
bermakna — laporkan keduanya, jangan buang yang gagal.

## Penjinakan overfitting seleksi (wajib disebut di naskah)

Versi pertama gerbang per segmen memilih `w` langsung dari RMSE validation. Gejalanya
langsung muncul: RMSE validation turun, RMSE test **naik sampai 3,1%**. Ini persoalan
yang sama yang sudah terlihat pada exp05b (V3 dengan `S1 = linear` memburuk 7,5%
semata-mata karena `w` ikut dipilih).

Perbaikannya: skema segmentasi dan hyperparameter dipilih dari **RMSE validasi-silang
5 lipatan kronologis di dalam validation**, bukan dari RMSE validation langsung. Peta
`w` akhir baru dipasang pada seluruh validation setelah skemanya terpilih. Efeknya
terukur: kasus terburuk turun dari +3,1% menjadi +1,0%, dan skema jatuh kembali ke
`global` (setara gerbang skalar) pada separuh konfigurasi — persis perilaku yang
diinginkan ketika segmentasi memang tidak membantu.

Test tetap disentuh tepat satu kali.

## Efisiensi

Pembanding (naif, XGBoost, kerangka lama, `S1` sendirian, AR-LRX skalar) **tidak
dilatih ulang** — prediksinya dibaca dari `exp05b_..._predictions.npz`. Inilah gunanya
menyimpan prediksi pada exp05b. Yang dilatih hanya 18 model baru
(3 varian × 2 tahap pertama × 3 jenis pengembangan). `S1 = linear` sengaja tidak
disertakan karena exp05b sudah menunjukkan ia kalah di ketiga varian.

Perkiraan waktu: sekitar 1–1,5 jam.

## Perubahan pada `arlrx.py`

Penambahan murni; `run_arlrx` **tidak disentuh**, sehingga exp05a dan exp05b tetap
tereproduksi persis. Yang baru: `run_arlrx_segmented(...)` dengan parameter `schemes`,
`n_folds`, dan `augment_stage1`. Dengan `schemes=("global",)` dan
`augment_stage1=False`, fungsi itu mereproduksi `run_arlrx` bit-per-bit — sudah
diverifikasi dalam uji-jalan.

## Keterbatasan yang harus dinyatakan

`S1` pada blok fit adalah nilai **in-sample**. Untuk penaksir struktural nilai itu
sedikit optimistis karena rata-rata kelompok memuat titiknya sendiri (bobot
≈ 1/n_kelompok). Konvensi ini sama persis dengan yang sudah dipakai membentuk residual
pada kerangka aslinya, tetapi karena `S1` kini menjadi fitur eksplisit, sebaiknya
dinyatakan terbuka di bagian metodologi.

## Cara menjalankan

1. Ganti `src/experiments/arlrx.py` dengan berkas di paket ini.
2. Salin `notebooks/exp05d_rossmann_arlrx_dev.ipynb` ke folder `notebooks/`.
3. Pastikan `results/exp05b_rossmann_arlrx_audit.csv` dan
   `results/exp05b_rossmann_arlrx_audit_predictions.npz` masih ada — notebook ini
   membacanya.
4. **Restart Kernel**, Run All, `QUICK_RUN` biarkan `False`.
5. Kirim balik: `exp05d_rossmann_arlrx_dev.csv`, `_main.csv`, `_ablation_2x2.csv`,
   `_dm.csv`, `_segments.csv`, dan notebook ber-output.

## Setelah ini

**exp06** — jalankan ulang seluruh baseline rujukan (CNN, MLP, Transformer, RNN, LSTM,
GRU, TS-XGBoost, LightGBM) di bawah protokol terpadu. Itu satu-satunya jalan sah menuju
klaim "mengungguli semua model rujukan", dan sekarang perbandingannya akan memakai
versi model Anda yang terbaik.
