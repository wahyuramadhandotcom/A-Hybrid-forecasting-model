# exp06b — hasil final: klaim keunggulan bertahan di hadapan baseline yang kuat

Total runtime **19,05 jam** (perkiraan saya 5–9 jam — saya meleset jauh, dan
Transformer sendirian memakan 14,3 jam dari itu).

---

## 1. Baseline benar-benar menguat

| Model | exp06 (lemah) | exp06b (kuat) | perbaikan | val R² lama → baru |
|---|---:|---:|---:|---|
| CNN | 1970,26 | 1288,28 | **−34,61%** | 0,623 → 0,759 |
| MLP | 1764,32 | 1257,25 | **−28,74%** | 0,718 → 0,754 |
| LSTM | 1757,61 | 1253,76 | **−28,67%** | 0,469 → 0,762 |
| Transformer | 2738,56 | 2319,54 | −15,30% | 0,153 → 0,390 |
| RNN | 1424,57 | 1242,73 | −12,76% | 0,716 → 0,773 |
| GRU | 1398,12 | 1275,29 | −8,79% | 0,696 → 0,760 |
| LightGBM / XGBoost | tidak berubah | tidak berubah | 0,00% | tidak berubah |

Rata-rata perbaikan enam baseline neural **−21,48%**; `val_R2` rata-rata seluruh
delapan baseline naik 0,624 → 0,727. Model pohon identik sampai 10⁻¹⁴ — sebagaimana
seharusnya, karena tidak ada yang diubah pada keduanya.

Dua kritik terbesar terhadap exp06 hilang atau menyusut drastis:

* kalah dari baseline naif: **6 dari 6 → 3 dari 6**
* solusi-pojok grid: **5 dari 6 → 1 dari 6** (tinggal CNN di 256)

## 2. Klaim keunggulan bertahan

| # | Model | RMSE | MAE | R² |
|---:|---|---:|---:|---:|
| **1** | **AR-LRX-Aug [structural]** | **945,42** | 645,64 | **0,9084** |
| 2 | AR-LRX-Aug [struct_linear] | 956,27 | 652,97 | 0,9063 |
| 3 | AR-LRX [struct_linear] | 1004,69 | 690,89 | 0,8966 |
| 4 | XGBoost (grid Zhao) | 1014,27 | 692,14 | 0,8946 |
| 5 | LightGBM (Zeng) | 1015,73 | 697,14 | 0,8943 |
| 9–14 | RNN, LSTM, MLP, GRU, CNN | 1242–1288 | | 0,830–0,842 |
| 15 | Transformer | 2319,54 | 1630,59 | 0,4487 |

**Peringkat 1 dari 15. Baseline yang mengungguli: 0 dari 8. Uji Diebold-Mariano
menang signifikan 8 dari 8 pada KEDUA skala, sepakat 8/8, seluruhnya p = 0.**

Unggul **6,79%** atas baseline terkuat (XGBoost grid Zhao) dan **25,17%** atas
rata-rata lima arsitektur neural inti (1263,46).

## 3. Temuan yang layak masuk naskah

Lima arsitektur neural inti menyatu ke rentang yang sangat sempit:

* RMSE 1242,73 – 1288,28 → rentang hanya **3,7%**
* `val_R2` 0,7543 – 0,7729 → rentang hanya **0,019**

Artinya: **di bawah protokol adil dengan fitur yang identik, perbedaan arsitektur di
antara baseline neural praktis tidak bermakna.** MLP, CNN, RNN, LSTM, dan GRU semuanya
mendarat di sekitar 1250; gradient boosting mencapai ~1015; AR-LRX-Aug 945,42.

Ini observasi yang berdiri sendiri dan menarik: yang menentukan pada masalah ini bukan
kecanggihan arsitektur, melainkan **bagaimana struktur data dimodelkan** — persis tesis
yang sudah Anda bangun lewat exp05b dan exp05d.

## 4. Satu titik lemah yang tersisa: Transformer

`val_R2` 0,3902, RMSE 2319,54, dan **14,3 jam** waktu komputasi. Ia tetap outlier jauh.

Penyebabnya kemungkinan besar kombinasi representasi (17 token berdimensi 1) dengan
Adam laju tetap 0,001 tanpa warmup — kombinasi yang memang buruk untuk transformer,
dan bukan sesuatu yang dapat diperbaiki tanpa mengubah arsitekturnya secara substansial.

**Rekomendasi: jangan dijalankan ulang.** Biayanya 14 jam dengan peluang perbaikan
kecil. Laporkan sebagai keterbatasan, misalnya:

> *Transformer sebagaimana diadaptasi ke representasi fitur ini tidak konvergen ke
> solusi kompetitif di bawah anggaran pelatihan terpadu. Hasilnya dilaporkan demi
> kelengkapan dan tidak diklaim mewakili potensi arsitektur tersebut.*

Kalimat itu jujur, dan justru memperkuat kredibilitas — jauh lebih baik daripada
mengklaim "kami mengungguli Transformer sebesar 59%".

## 5. Cara melaporkan

* **Tabel bagian 2 adalah Tabel 3 versi final.** Buang tabel exp06.
* **Tabel bagian 1 adalah bukti itikad baik Anda** — bahwa Anda sengaja memperkuat
  lawan sebelum mengklaim menang, dan berapa besar mereka menguat. Reviewer sangat
  jarang melihat penulis melakukan ini.
* **Nyatakan perbedaan perlakuan terbuka:** baseline neural memperoleh standardisasi
  target, early stopping, dan grid tiga tingkat; AR-LRX tidak memerlukannya karena
  berbasis pohon.
* **Nyatakan CNN masih memilih batas atas grid** (1 dari 6) sebagai keterbatasan.
* **Nyatakan keterbatasan determinisme** model neural di CPU.
* Angka lama Rossmann 577,63 tetap tidak boleh dipakai.

## 6. Status Rossmann: selesai

Rangkaian eksperimen Rossmann sudah lengkap dan saling mengunci:

| Eksperimen | Yang dibuktikan |
|---|---|
| exp03 | kebocoran `Customers`, kerangka lama kalah dari naif |
| exp05a/b | tahap pertama struktural membalikkan hasil; ablasi gerbang yang jujur |
| exp05d | gerbang terpelajar mengungguli gerbang tabel; manfaat monoton terhadap informasi |
| exp06/06b | keunggulan bertahan atas baseline rujukan yang terlatih layak |

Yang tersisa satu: **exp05c (PharmaSales)** — dataset kedua, dan satu-satunya tempat
sifat pengaman gerbang (`w = 0`) dapat dibuktikan benar-benar aktif.
