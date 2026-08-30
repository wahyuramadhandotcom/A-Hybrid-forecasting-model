# exp05a — hasil final, dan apa yang exp05b tutup

Lingkungan run: Python 3.10.11 / Windows, numpy 2.2.6, pandas 2.3.3, sklearn 1.7.2,
xgboost 3.2.0, statsmodels 0.14.6, seed 42. `QUICK_RUN = False` (data penuh:
838.760 baris, 1.115 toko). Runtime total 1,37 jam. Determinisme diperiksa di
notebook: prediksi identik bit-per-bit pada run kedua.

---

## 1. Hasil utama exp05a — AR-LRX menang empat arah di SEMUA varian

`AR-LRX [struct_linear]` adalah konfigurasi terbaik pada ketiga varian yang sah.
RMSE skala asli, dan selisih terhadap empat acuan (negatif = AR-LRX lebih baik):

| Varian | RMSE AR-LRX | vs naif | vs XGBoost polos | vs kerangka lama | vs S1 sendirian | 4 arah? |
|---|---:|---:|---:|---:|---:|:--:|
| V1 `customers_dropped` | **1204,75** | −5,48% | −27,04% | −27,23% | −2,33% | ya |
| V2 `customers_lagged`  | **1054,97** | −17,23% | −5,08% | −8,29% | −13,84% | ya |
| V3 `sales_lagged`      | **1004,69** | −21,18% | −1,29% | −5,11% | −17,28% | ya |

Seluruh 12 perbandingan Diebold-Mariano untuk `struct_linear` signifikan pada
α = 0,05. Yang paling tipis adalah V3 vs XGBoost polos (p = 1,04 × 10⁻³) — tetap
signifikan, tetapi harus dilaporkan apa adanya sebagai selisih kecil.

Rekap DM per jenis tahap pertama (36 perbandingan):

| S1 | menang signifikan | kalah signifikan | imbang |
|---|---:|---:|---:|
| `struct_linear` | 12 | 0 | 0 |
| `structural` | 12 | 0 | 0 |
| `linear` | 5 | 6 | 1 |

**Ini temuan intinya:** kerangka residual tidak gagal karena gagasannya; ia gagal
karena tahap pertamanya. Dengan `S1 = linear` (versi lama) hasilnya kalah dari
XGBoost polos di ketiga varian. Ganti tahap pertamanya, dan kerangka yang sama
menang di ketiga varian.

## 2. Bukti struktural: kerangka lama adalah kasus khusus

Pada V1, gerbang memilih `w = 1,0` untuk `S1 = linear`. Hasilnya identik
bit-per-bit dengan kerangka lama: `test_RMSE` 0,21228248 pada keduanya, dan uji DM
mengembalikan NaN karena selisih galatnya nol persis.

Ini bukan kebetulan yang menguntungkan — ini konsekuensi definisi. Nyatakan di
naskah: perbandingan AR-LRX vs kerangka lama **bukan** perbandingan antar-protokol,
melainkan antar-konfigurasi dari satu kerangka yang sama, dan degenerasinya
terbukti secara numerik, bukan sekadar diklaim.

## 3. Sinergi: tidak satu pun komponen cukup sendirian

Selisih terhadap kerangka lama (negatif = lebih baik):

| Varian | gerbang saja | S1 struktural saja | AR-LRX penuh | prediksi aditif | sinergi |
|---|---:|---:|---:|---:|---:|
| V1 | +0,00% | −25,06% | −27,23% | −25,06% | −2,17 pp |
| V2 | +3,71% | +7,85% | **−8,29%** | +11,57% | **−19,85 pp** |
| V3 | +7,48% | +17,17% | **−5,11%** | +24,65% | **−29,76 pp** |

Pada V2 dan V3, **masing-masing komponen sendirian justru lebih buruk** daripada
kerangka lama, tetapi gabungannya menang. Ini efek interaksi yang nyata, bukan
penjumlahan dua perbaikan. Klaim yang tepat bukan "gerbang membantu" atau "tahap
pertama struktural membantu", melainkan: **koreksi residual bergerbang hanya
bermakna bila dibangun di atas tahap pertama yang sepadan dengan strukturnya.**

Mekanismenya terbaca dari `resid_val_r2` (R² tahap kedua terhadap residual validation):

* `S1 = linear` → 0,44–0,59: residualnya masih sangat terstruktur, jadi koreksi
  penuh masuk akal, dan menyusutkan `w` malah merugikan.
* `S1 = structural` → −0,02 sampai 0,10: residualnya nyaris tak terprediksi, jadi
  penyusutan `w` ke 0,8 melindungi dari derau.

## 4. Gerbang dapat memprediksi manfaatnya sendiri

Korelasi antara perbaikan RMSE **validation** akibat gerbang dan perbaikan RMSE
**test** yang sebenarnya, atas 9 konfigurasi:

* Pearson **r = 0,964**
* Spearman **ρ = 0,950** (p = 8,8 × 10⁻⁵)

Artinya: besar manfaat hibridisasi dapat diperkirakan dari validation saja, sebelum
test disentuh. Inilah kriteria keputusan yang layak diklaim sebagai sumbangan
metodologis — lebih tahan kritik daripada klaim "model kami paling akurat".

## 5. Tiga hal yang HARUS diantisipasi dari reviewer

**R1. `S1 [structural]` sangat mirip baseline naif.** Naif = median per
(Toko × Hari × Promo) → RMSE 1274,63. `S1 [structural]` = mean + fallback
bertingkat → 1240,64, hanya −2,67%. Lompatan terbesar dari kerangka lama (1655 →
1240) karena itu berasal dari mengganti LR dengan sesuatu yang pada dasarnya
adalah baseline naif yang sedikit dirapikan. **Jangan sembunyikan ini** — laporkan
terus terang, lalu tunjukkan bahwa AR-LRX penuh tetap mengungguli naif secara
signifikan (−5,48% / −17,23% / −21,18%). Justru kejujuran di titik ini yang
menyelamatkan klaimnya.

**R2. Gerbang tidak pernah memilih `w = 0` pada Rossmann.** Nilai yang terpilih
hanya {0,8; 0,9; 1,0}. Sifat pengaman "tidak pernah lebih buruk dari tahap pertama"
karena itu benar secara konstruksi tetapi **tidak terdemonstrasikan** di dataset
ini. Simulasi PharmaSales sebelumnya menunjukkan gerbang mati (`w = 0`) pada
6 dari 8 kategori. Naskah membutuhkan **kedua** dataset agar gerbang terbukti
sebagai mekanisme adaptif, bukan konstanta yang kebetulan cocok.

**R3. Skala uji tidak sama dengan skala pelaporan.** Tabel utama melaporkan RMSE
skala asli; uji DM exp05a dihitung pada skala log. Reviewer Q1/Q2 berhak menuntut
uji pada skala yang dilaporkan. Ditutup di exp05b.

Catatan tambahan: `AR-LRX [structural]` dan `AR-LRX [struct_linear]` hampir tidak
berbeda (V1: 1206,44 vs 1204,75 = 0,14%). Komponen linier praktis tidak menambah
apa pun setelah struktur toko masuk. Laporkan begitu; jangan mengklaim sumbangan
yang tidak ada.

---

## 6. Yang dilakukan exp05b

Tidak ada satu pun keputusan pemodelan yang berubah. Protokol, seed, split, grid,
dan definisi model identik. Yang ditambahkan:

1. **Verifikasi reproduksi** terhadap `exp05a_rossmann_arlrx.csv` — bila ada satu
   baris yang meleset, analisis berikutnya tidak boleh dipakai.
2. **Uji DM pada dua skala** (log dan asli), disandingkan, dengan kolom `sepakat`
   yang menandai apakah kesimpulannya sama. Menutup R3.
3. **Ablasi lengkap 3 × 2** (tiga `S1` × gerbang mati/hidup). Karena `w` tidak
   memengaruhi pelatihan tahap kedua, sel "gerbang dimatikan" (`w = 1`) dihitung
   dari model yang **sama persis** tanpa melatih ulang — ablasi yang lebih bersih
   daripada melatih ulang dengan `w` dipatok, sebab satu-satunya yang berubah
   benar-benar hanya gerbang.
4. **Kurva sensitivitas gerbang**: RMSE test untuk seluruh `w ∈ [0, 1]`, dengan
   penanda `w` pilihan validation dan `w` terbaik di test. Selisih keduanya =
   "harga" memilih dari validation. Ini **diagnostik post-hoc, bukan pemilihan** —
   nyatakan begitu di naskah.
5. **Pemecahan galat per segmen** (promo, hari, kuartil volume toko): menjelaskan
   *mengapa* kerangka ini bekerja, bukan hanya bahwa angkanya lebih kecil.
6. **Penyimpanan seluruh prediksi test** ke `.npz`, sehingga setiap analisis
   lanjutan (plot residual, uji lain, tabel per toko untuk bab disertasi) tidak
   perlu melatih ulang 1,37 jam lagi.

### Perubahan pada `arlrx.py`

Satu penambahan, murni penyimpanan: `_stage1_test_raw` dan `_test_correction`
disimpan pada baris hasil, sehingga prediksi pada sembarang `w` dapat dihitung
sebagai `S1 + w × koreksi` tanpa melatih ulang. Pemanggilan `final.predict()` yang
sebelumnya inline kini ditampung ke variabel lebih dulu — **tidak ada satu pun
angka yang berubah**, dan bagian 2 exp05b memverifikasinya.

### Cara menjalankan

1. Ganti `src/experiments/arlrx.py` dengan berkas dari paket ini.
2. Salin `notebooks/exp05b_rossmann_arlrx_audit.ipynb` ke folder `notebooks/`.
3. **Restart Kernel**, lalu Run All. `QUICK_RUN` biarkan `False`.
4. Kirim balik: `exp05b_rossmann_arlrx_audit.csv`, `_dm_two_scales.csv`,
   `_ablation_full.csv`, `_gate_curve.csv`, `_segments.csv`, dan notebook ber-output.
   Berkas `.npz` tidak perlu dikirim (besar); cukup simpan di mesin Anda.

Perkiraan waktu: setara exp05a (± 1,4 jam), karena beban pelatihannya identik —
seluruh analisis baru dihitung dari prediksi yang sudah ada.

---

## 7. Langkah berikutnya setelah exp05b

**exp05c — AR-LRX pada PharmaSales.** Ini yang menutup R2 dan melengkapi klaim
disertasi. Rossmann menunjukkan gerbang aktif (`w` 0,8–1,0) karena residualnya
masih memuat struktur; PharmaSales (dari simulasi awal) menunjukkan gerbang mati
(`w = 0`) karena residualnya derau. Dua perilaku itu dari satu kerangka yang sama,
dipilih otomatis dari validation, adalah demonstrasi bahwa gerbang benar-benar
adaptif. Tanpa itu, sifat pengaman AR-LRX hanya klaim di atas kertas.
