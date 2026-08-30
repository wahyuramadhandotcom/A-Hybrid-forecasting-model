# exp05b — hasil, dan koreksi terhadap kesimpulan exp05a

Lingkungan identik exp05a (Python 3.10.11, numpy 2.2.6, pandas 2.3.3, sklearn 1.7.2,
xgboost 3.2.0, seed 42, data penuh 838.760 baris / 1.115 toko).

---

## 0. Verifikasi lulus

Reproduksi terhadap exp05a **persis**: selisih maksimum `test_RMSE` 8,3 × 10⁻¹⁷,
`orig_RMSE` 4,5 × 10⁻¹³, `gate_w` nol mutlak. Seluruh analisis di bawah berdiri di
atas angka yang sama dengan exp05a.

## 1. Celah C1 tertutup bersih — uji skala asli sepakat sepenuhnya

| | log | skala asli |
|---|---:|---:|
| Sepakat arah | 36/36 | 36/36 |
| Menang signifikan | 29/36 | 29/36 |
| `struct_linear` saja | 12/12 | **12/12** |

Untuk konfigurasi utama `AR-LRX [struct_linear]`, p-value terbesar pada skala asli
adalah 1,06 × 10⁻⁵ (V3 vs XGBoost polos). Klaim keunggulan **tidak bergantung pada
skala pengukuran** — ini jawaban lengkap untuk keberatan reviewer yang paling
mudah diajukan.

## 2. KOREKSI PENTING: gerbang bukan sumber keunggulan

Ini membalik sebagian narasi yang saya sampaikan setelah exp05a. Ablasi exp05a
membandingkan hal-hal yang tidak sebanding (tahap pertama berbeda **dan**
hyperparameter terpilih berbeda), sehingga tampak ada "sinergi" besar. Ablasi
exp05b menahan segalanya tetap dan hanya mengubah `w` — dan hasilnya berbeda.

Sumbangan gerbang (negatif = gerbang menolong):

| Varian | S1 | w\* | w=1 (gerbang mati) | w=w\* | sumbangan gerbang |
|---|---|---:|---:|---:|---:|
| V1 | structural | 0,8 | 1211,53 | 1206,44 | **−0,42%** |
| V1 | struct_linear | 0,8 | 1211,14 | 1204,75 | **−0,53%** |
| V2 | structural | 0,8 | 1063,08 | 1062,76 | −0,03% |
| V2 | struct_linear | 0,8 | 1050,32 | 1054,97 | +0,44% |
| V3 | structural | 0,8 | 1008,92 | 1011,38 | +0,24% |
| V3 | struct_linear | 0,8 | 997,30 | 1004,69 | +0,74% |
| V2 | linear | 0,9 | 1150,31 | 1193,03 | +3,71% |
| V3 | linear | 0,8 | 1069,79 | 1138,00 | +6,38% |

Gerbang menolong pada **3 dari 9** konfigurasi. Untuk tahap pertama yang baik,
rata-rata sumbangannya **+0,075%** — praktis nol, condong sedikit merugikan.

Peringkat rata-rata tiga varian:

| RMSE rata-rata | Model |
|---:|---|
| **1086,25** | S1[struct_linear] + koreksi penuh (**tanpa gerbang**) |
| 1088,14 | AR-LRX [struct_linear] (gerbang dari validation) |
| 1093,53 | AR-LRX [structural] |
| 1094,51 | S1[structural] + koreksi penuh (tanpa gerbang) |
| 1224,14 | S1[struct_linear] sendirian |
| 1260,17 | XGBoost polos |
| 1274,63 | Naif per toko |
| 1288,21 | Kerangka lama (LR-XGB) |

**Versi tanpa gerbang sedikit lebih unggul (0,17%).** Reviewer yang menjalankan
ablasi ini akan menemukannya. Karena itu jangan mengklaim gerbang sebagai pendorong
akurasi pada Rossmann.

## 3. Yang SEBENARNYA menjadi sumber keunggulan

Seluruh perolehan berasal dari **mengganti tahap pertama**, bukan dari gerbang:

* kerangka lama (S1 = linear, koreksi penuh) → 1288,21
* S1 struktural + koreksi penuh → 1086,25 (**−15,7%**)

Klaim yang benar dan dapat dipertahankan:

> Keberhasilan hibridisasi residual ditentukan oleh keselarasan tahap pertama
> dengan struktur data, bukan oleh kapasitas tahap kedua. Mengganti tahap pertama
> linier dengan penaksir hierarkis mengubah kerangka yang kalah dari XGBoost polos
> di ketiga varian menjadi menang signifikan di ketiganya, pada kedua skala
> pengukuran.

Itu tetap temuan yang kuat — dan lebih jujur daripada klaim sebelumnya.

## 4. Kurva gerbang: mengapa gerbang tidak berguna di sini

`w` optimum di test selalu jatuh pada **0,70–0,95** — tidak pernah mendekati nol.
Artinya residual pada Rossmann memang masih memuat struktur, sehingga koreksi
hampir penuh adalah jawaban yang benar dan tidak ada yang perlu "dimatikan".

Kurvanya sangat datar di sekitar optimum: harga memilih `w` dari validation alih-alih
dari test rata-rata hanya **0,353%** untuk tahap pertama yang baik. Jadi gerbangnya
bekerja sebagaimana mestinya — persoalannya, di dataset ini tidak ada yang perlu
dikerjakan.

## 5. Efek samping gerbang yang harus dilaporkan

Menambahkan `w` ke ruang pencarian **tidak gratis**: ia memperbesar ruang seleksi
dan dapat memicu overfitting validation. Bukti terkuatnya V3 dengan S1 = linear —
seleksi gabungan memilih (`learning_rate` 0,2 , `w` 0,8) → RMSE test 1138,00,
sedangkan seleksi tanpa gerbang memilih (`learning_rate` 0,1 , `w` 1) → 1058,82.
Selisihnya **7,5% lebih buruk** semata-mata karena gerbang ikut dipilih.

Pada V1 dan V2 hyperparameter terpilih identik dengan kerangka lama, sehingga
`w = 1` mereproduksi kerangka lama persis (selisih 0,000%). Hanya V3 yang menyimpang,
dan penyebabnya terdokumentasi.

## 6. Segmen: di mana keunggulan itu berasal

| Varian | keseluruhan | kalah di segmen |
|---|---:|---|
| V1 | −27,04% | hanya hari 7 (+0,12%, n=640) |
| V2 | −5,08% | hari 2 (+1,46%), hari 4 (+4,46%), toko Q2 (+0,80%) |
| V3 | −1,29% | promo (+1,81%), hari 2 (+12,51%), hari 3 (+4,49%), hari 4 (+13,72%), toko Q1 kecil (+5,63%), toko Q2 (+4,60%) |

Menang terbesar konsisten di ketiga varian: **hari 1 (Senin), hari 6 (Sabtu),
hari 7 (Minggu), tanpa promo, dan toko volume besar**.

Mekanismenya terbaca jelas: penaksir hierarkis unggul ketika pola
toko × hari-dalam-minggu mendominasi (awal pekan, akhir pekan, hari tanpa promo,
toko besar dengan riwayat tebal), dan tertinggal ketika informasi paling berguna
justru ada di lag penjualan (tengah pekan, toko kecil, hari promo pada V3).

**Ini penjelasan mekanistik yang membuat sumbangannya bersifat metodologis, bukan
sekadar angka lebih kecil.** Sertakan tabel ini di naskah; ia menjawab pertanyaan
"mengapa berhasil", bukan hanya "seberapa".

---

## 7. Konsekuensi untuk naskah dan disertasi

**Yang harus diklaim:**
1. Keselarasan tahap pertama dengan struktur data menentukan berhasil-tidaknya
   hibridisasi residual — terukur sebagai selisih S1[linear] vs S1[struct_linear],
   dan sebagai pembalikan hasil dari kalah menjadi menang di ketiga varian.
2. Keunggulan itu bertahan pada kedua skala pengukuran (36/36 sepakat).
3. Sumbernya dapat dilacak ke segmen tertentu, bukan perbaikan menyeluruh yang kabur.

**Yang TIDAK boleh diklaim:**
1. Bahwa gerbang meningkatkan akurasi pada Rossmann — ablasi bersih menunjukkan
   sebaliknya (rata-rata +0,075%, dan versi tanpa gerbang menang 0,17%).
2. Bahwa AR-LRX menang di seluruh segmen — pada V3 ia kalah di enam segmen.
3. Bahwa gerbang gratis — pada V3-linear ia memperburuk 7,5% lewat overfitting seleksi.

**Posisi gerbang yang jujur:** mekanisme pengaman dengan premi terukur (±0,1–0,7%
pada tahap pertama yang baik). Pada Rossmann premi itu dibayar tanpa klaim — karena
residualnya memang masih informatif. Nilainya harus dibuktikan pada dataset yang
residualnya derau, dan simulasi awal menunjukkan PharmaSales adalah dataset itu
(gerbang mati di 6 dari 8 kategori).

**Karena itu exp05c bukan pelengkap, melainkan penutup argumen.** Tanpa dataset
tempat gerbang benar-benar menutup, gerbang hanyalah parameter tambahan yang pada
Rossmann terbukti tidak membayar dirinya sendiri.
