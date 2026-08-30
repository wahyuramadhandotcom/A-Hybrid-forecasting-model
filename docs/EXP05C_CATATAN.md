# exp05c — AR-LRX pada PharmaSales: menguji tiga ramalan

## Ini bukan replikasi

Rangkaian Rossmann sudah selesai dan menghasilkan tiga klaim mekanistik. exp05c
mengujinya pada dataset yang **karakternya berlawanan**: PharmaSales univariat,
berderau, dan tahap keduanya hanya punya fitur lag — keadaan paling miskin informasi
yang bisa ditemui kerangka ini.

Ramalan ditulis **sebelum** hasilnya dilihat, dan bagian 8 notebook memeriksanya satu
per satu. Inilah yang membedakan uji mekanisme dari sekadar "jalankan di dataset lain".

| | Ramalan | Dasarnya |
|---|---|---|
| **P1** | Gerbang akan **menutup** (`w* = 0`) pada sebagian besar kategori | Pada Rossmann `w*` tidak pernah 0 (0,8–1,0) karena residualnya masih berstruktur (`resid_val_r2` 0,08–0,23). Bila di PharmaSales residualnya derau, mekanisme pengaman harus aktif. **Satu-satunya tempat sifat pengaman itu dapat dibuktikan bekerja.** |
| **P2** | Augmentasi akan **hampir tidak membantu** | exp05d: manfaat augmentasi membesar monoton dengan kekayaan informasi tahap kedua — V1 (tanpa lag) +1,11% → V2 −3,39% → V3 −6,52%. PharmaSales ada di ujung termiskin. |
| **P3** | `gate_val_gain_pct` tetap **memprediksi** perbaikan test | Rossmann: r = 0,964 (Spearman 0,950). Kriteria yang benar-benar umum harus bertahan di dataset kedua. |

## Kendala desain yang menentukan bentuk eksperimen

Tahap pertama **struktural tidak dapat dipakai** di sini. PharmaSales hanya punya
fitur lag (`lag_1..lag_k`, `rolling_mean_k`); tidak ada kolom kunci kategorikal
seperti Toko × Hari × Promo. Karena itu `S1 = linear` — dan justru itu yang tepat:
yang diuji di sini adalah **gerbangnya**, bukan tahap pertamanya.

## Cakupan

Harian **dan** mingguan, delapan kategori ATC, dua set fitur (`A_lag1`, `B_rich`) —
32 konfigurasi, kontrak identik exp01/exp02 (aturan lag `pacf_train` bebas kebocoran,
split 70/15/15, penyetelan hanya validation, refit train+val, test sekali). Target
PharmaSales **tidak** ditransformasi log, sehingga metriknya sudah pada skala asli.

Tujuh model per konfigurasi: Naive, SeasonalNaive, `S1 [linear]`, `AR-LRX [linear]`
(kontrak exp05a), `AR-LRX-g [linear]` (kontrol berpasangan untuk P2),
`AR-LRX-Aug [linear]`, XGBoost, dan kerangka lama.

Jumlah lipatan seleksi disesuaikan ukuran validation (`min(5, max(2, n_val // 30))`)
karena validation mingguan hanya ~45 baris.

## Yang terlihat pada uji-jalan (stub XGBoost, 4 kategori — arah saja)

* **P1 tampak akan terbukti**: `w* = 0` pada 6 dari 16 konfigurasi, `w*` rata-rata
  0,212, dan `resid_val_r2` rata-rata **−0,20** (negatif = residual memang derau).
  Kontras tajam dengan Rossmann (`w*` 0,8–1,0, `resid_val_r2` +0,08…+0,23).
* **Sifat pengaman terlihat bekerja**: AR-LRX lebih buruk daripada `S1` sendirian
  hanya pada 2 dari 16 konfigurasi (terburuk +0,70%), sedangkan **kerangka lama yang
  tidak punya gerbang lebih buruk pada 9 dari 16 (terburuk +6,41%)**. Inilah
  demonstrasi paling langsung tentang apa yang sebenarnya dibeli oleh gerbang.
* **P2 tampak akan terbukti**: efek augmentasi rata-rata −0,001%.
* P3 belum dapat dinilai dari uji-jalan karena boosternya distub.

Angka sebenarnya akan berbeda; yang penting arahnya sudah konsisten dengan ramalan.

## Cara menjalankan

1. Pastikan `src/experiments/arlrx.py` sudah versi terbaru (memuat
   `run_arlrx_segmented`) — versi yang sama yang Anda pakai di exp05d.
2. Salin `notebooks/exp05c_pharma_arlrx.ipynb` ke `notebooks/`.
3. **Restart Kernel**, Run All. `QUICK_RUN` biarkan `False`.
4. Perkiraan waktu: **puluhan menit**, bukan jam — PharmaSales jauh lebih kecil
   daripada Rossmann.
5. Kirim balik: `exp05c_pharma_arlrx.csv`, `_gate.csv`, `_aug_effect.csv`,
   `_comparison.csv`, `_dm.csv`, dan notebook ber-output.

## Mengapa ini melengkapi disertasi

Bila P1 terbukti, Anda memperoleh kalimat yang tidak dapat dibuat hanya dengan satu
dataset:

> Satu kerangka, dua perilaku. Pada Rossmann gerbang membuka lebar (`w*` 0,8–1,0)
> karena residual tahap pertama masih memuat struktur; pada PharmaSales ia menutup
> (`w* = 0`) karena residualnya derau. Keduanya dipilih otomatis dari validation,
> tanpa campur tangan, dan tanpa pernah menyentuh test.

Itulah klaim adaptivitas yang sesungguhnya — dan uji Diebold-Mariano di PharmaSales
berkuasa rendah (test hanya ratusan baris harian, puluhan mingguan), jadi kekuatan
argumennya memang harus bertumpu pada mekanisme, bukan pada peringkat akurasi.
