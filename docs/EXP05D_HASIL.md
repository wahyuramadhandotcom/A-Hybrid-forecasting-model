# exp05d — hasil: gerbang terpelajar menang, gerbang tabel gagal

Lingkungan identik exp05a/b (xgboost 3.2.0, seed 42, data penuh 838.760 baris /
1.115 toko). Pembanding dibaca dari `exp05b_..._predictions.npz` tanpa dilatih ulang.

---

## 1. Hasil ablasi 2×2 — hipotesisnya terbukti

| Komponen | Efek rata-rata | Menang | Rentang |
|---|---:|---:|---|
| **+Aug** (gerbang terpelajar) | **−2,72%** | **5 dari 6** | −6,52% … +1,11% |
| +Seg (gerbang tabel) | +1,58% | 0 dari 6 | +0,32% … +2,33% |
| +Aug+Seg | −1,66% | 4 dari 6 | −4,17% … +0,40% |

Rinciannya:

| Varian | S1 | skalar (exp05b) | +Seg | **+Aug** | +Aug+Seg |
|---|---|---:|---:|---:|---:|
| V1 | structural | 1206,44 | 1220,76 | 1219,85 | 1211,22 |
| V1 | struct_linear | 1204,75 | 1208,61 | **1201,36** | 1207,44 |
| V2 | structural | 1062,76 | 1082,16 | **1026,73** | 1043,41 |
| V2 | struct_linear | 1054,97 | 1073,59 | 1029,21 | 1046,98 |
| V3 | structural | 1011,38 | 1034,95 | **945,42** | 969,24 |
| V3 | struct_linear | 1004,69 | 1025,28 | 956,27 | 966,11 |

Kesimpulan yang dapat dipertahankan: **yang menentukan bukan banyaknya parameter
gerbang, melainkan apakah gerbang itu dapat bergantung pada fitur secara kontinu.**
Gerbang tabel `w(s)` menambah sampai 12 parameter dan gagal di **seluruh** konfigurasi;
gerbang terpelajar tidak menambah satu pun parameter gerbang dan menang di 5 dari 6.

## 2. Hasil akhir — model terbaik per varian

| Varian | Model | RMSE | vs naif | vs XGBoost | vs kerangka lama | vs S1 | vs exp05b |
|---|---|---:|---:|---:|---:|---:|---:|
| V1 | AR-LRX-Aug [struct_linear] | **1201,36** | −5,75% | −27,25% | −27,43% | −2,60% | −0,28% |
| V2 | AR-LRX-Aug [structural] | **1026,73** | −19,45% | −7,62% | −10,74% | −17,24% | −3,39% |
| V3 | AR-LRX-Aug [structural] | **945,42** | −25,83% | −7,11% | −10,71% | −23,80% | −6,52% |

**Seluruh 15 uji Diebold-Mariano menang signifikan pada KEDUA skala**, sepakat arah
15/15, p terbesar 1,01 × 10⁻⁸. Termasuk melawan AR-LRX skalar exp05b: DM −5,73 (V1),
−22,51 (V2), −30,73 (V3). Jadi pengembangan ini **nyata**, bukan selisih dalam derau.

Perbaikan terhadap exp05b pada V3: 1004,69 → 945,42 (**−5,90%**).

## 3. Mekanismenya terukur, bukan diklaim

`resid_val_r2` (seberapa terprediksi residual tahap pertama oleh tahap kedua):

| Varian / S1 | skalar (exp05b) | +Aug | perubahan |
|---|---:|---:|---|
| V3 structural | 0,1013 | **0,2316** | 2,3× |
| V3 struct_linear | 0,0981 | 0,1966 | 2,0× |
| V2 structural | 0,0765 | 0,1637 | 2,1× |
| V2 struct_linear | 0,0855 | 0,1499 | 1,8× |
| V1 structural | −0,0247 | −0,0185 | tetap negatif |
| V1 struct_linear | −0,0296 | −0,0300 | tetap negatif |

Dengan mengetahui `S1(x)`, tahap kedua benar-benar menemukan struktur yang sebelumnya
tak terlihat. `gate_val_gain_pct` ikut naik (V3 structural: 9,40 → 15,64).

## 4. Temuan tak terduga yang justru memperkuat argumen

Manfaat augmentasi **membesar seiring kekayaan informasi** yang dimiliki tahap kedua:

| Varian | Informasi yang tersedia bagi S2 | Efek Aug (structural / struct_linear) |
|---|---|---:|
| V1 | tanpa lag apa pun (`Customers` dibuang) | **+1,11% / −0,28%** |
| V2 | `Customers` ter-lag | −3,39% / −2,44% |
| V3 | penjualan ter-lag | **−6,52% / −4,82%** |

Monoton. Penjelasannya koheren: augmentasi memberi tahap kedua kemampuan
**mengkondisikan** besaran koreksi pada keadaan; kemampuan itu tidak berguna bila tahap
kedua tidak punya informasi lain untuk dikondisikan. Pada V1, `resid_val_r2` bahkan
tetap negatif — dan di situlah satu-satunya kasus Aug merugikan.

Ini bukan sekadar catatan kaki: ia mengubah klaim dari "metode kami lebih akurat"
menjadi **"kami tahu kapan dan mengapa metode ini lebih akurat"** — jenis klaim yang
jauh lebih tahan di hadapan reviewer Q1/Q2.

## 5. Kelemahan segmen exp05b hampir tertutup seluruhnya

Jumlah segmen (dari 14) yang kalah dari XGBoost polos:

| Varian | exp05b | exp05d |
|---|---:|---:|
| V1 | 1 | **0** |
| V2 | 3 | **0** |
| V3 | 6 | **2** |

Delapan segmen berbalik dari kalah menjadi menang. Dua yang tersisa menyusut drastis:
V3 Selasa +12,92% → +2,54%, V3 Kamis +14,44% → +1,94%.

## 6. Kegagalan yang HARUS dilaporkan apa adanya

Gerbang per segmen gagal di seluruh konfigurasi, **walaupun** skemanya sudah dipilih
lewat validasi-silang 5 lipatan di dalam validation. Lebih dari itu: skema `dow_promo`
(12 segmen) terpilih pada **6 dari 6** konfigurasi — mekanisme mundur ke `global` tidak
pernah aktif pada data penuh, padahal aktif pada uji-jalan kecil.

Penjelasannya bukan derau, melainkan **pergeseran distribusi**. Lipatan validasi-silang
seluruhnya berada di dalam blok validation (Okt 2014 – Mar 2015), sehingga pola
hari × promo di sana saling menegaskan. Blok test (Mar – Jul 2015) adalah musim yang
berbeda, dan bobot per segmen itu tidak ikut berpindah. Validasi-silang **di dalam**
satu blok waktu tidak dapat mendeteksi ketidakstabilan **antar** blok waktu.

Ini pelajaran metodologis yang layak dilaporkan tersendiri: penjinakan overfitting
seleksi lewat cross-fitting mengurangi kerusakan (uji-jalan: +3,1% → +1,0%) tetapi
**tidak menghilangkannya** ketika masalahnya pergeseran distribusi, bukan varians.

## 7. Posisi terhadap angka rujukan (orientasi saja — protokol masih berbeda)

| Peringkat | RMSE | Model |
|---:|---:|---|
| 1 | 775,28 | CNN (Diamantini) |
| 2 | 825,32 | MLP (Diamantini) |
| 3 | 872,63 | TS-XGBoost (Zeng) |
| 4 | 881,28 | Transformer (Diamantini) |
| 5 | 913,09 | XGBoost Tuned (Zhao) |
| **6** | **945,42** | **AR-LRX-Aug V3 (protokol adil)** |
| 7 | 1004,69 | AR-LRX skalar V3 (exp05b) |
| 8 | 1044,78 | RNN (Diamantini) |

Jarak ke XGBoost Tuned menyempit menjadi 3,5%. **Tetapi angka rujukan itu direproduksi
lewat pipeline lama yang mengandung kebocoran `Customers` dan pembagian berbasis
indeks** — mengadu 945,42 dengan 775,28 sama tidak sahnya dengan mengadu 577,63 dengan
775,28. Perbandingan yang sah hanya mungkin setelah exp06.

## 8. Yang boleh diklaim setelah exp05d

1. Keselarasan tahap pertama dengan struktur data menentukan berhasil-tidaknya
   hibridisasi residual (dari exp05b).
2. **Gerbang yang bergantung fitur secara kontinu mengungguli gerbang tabel maupun
   gerbang skalar** — dan efeknya terukur pada `resid_val_r2`, bukan hanya pada RMSE.
3. **Manfaat hibridisasi membesar seiring kekayaan informasi tahap kedua** — monoton
   di ketiga varian, dengan tanda yang berbalik ketika informasinya nihil.
4. Seluruh klaim bertahan pada dua skala pengukuran, 15/15 uji DM signifikan.

Yang TIDAK boleh diklaim: unggul dari model rujukan (belum sah sampai exp06); bahwa
segmentasi gerbang membantu (gagal 6 dari 6); bahwa Aug selalu membantu (V1 structural
memburuk 1,11%).
