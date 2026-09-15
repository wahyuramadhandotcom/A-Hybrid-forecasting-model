"""
AR-LRX -- Adaptive Residual Linear Regression-XGBoost.

Pengembangan dari kerangka LR-XGBoost residual pada naskah sebelumnya. Modul ini
SENGAJA terpisah dari `protocol.py`: kontrak eksperimen (split, tuning, seed,
pelaporan) tidak diubah sedikit pun, sehingga hasil exp01-exp03 tetap dapat
direproduksi persis. Yang ditambahkan hanya arsitektur modelnya.

Diagnosis yang melatarbelakangi pengembangan ini (dari hasil exp01-exp03):

  D1. Tahap pertama pada versi lama BUKAN model tren. Ia menerima matriks fitur
      yang sama persis dengan tahap kedua, sehingga yang terjadi bukan dekomposisi
      melainkan penumpukan dua peramal. Residualnya karena itu tidak memuat
      "komponen nonlinier", melainkan sisa derau.
  D2. Tahap pertama yang buruk meracuni tahap kedua. Pada Rossmann, LR hanya
      mencapai R2 = 0,234 pada skala log sementara prediktor struktural per toko
      mencapai 0,842. Kerangka lama memulai dari basis yang lebih buruk daripada
      baseline naif, lalu meminta XGBoost menambalnya.
  D3. Tidak ada mekanisme mundur. Ketika residual memang tidak dapat diprediksi,
      kerangka lama tetap memaksakan koreksi dan menambah galat. Pada exp01,
      71% pemilihan hyperparameter jatuh di pojok grid paling konservatif -
      tanda bahwa tahap kedua "ingin" tidak belajar apa pun.

Tiga komponen pengembangan yang menjawabnya:

  S1  Tahap pertama dapat dipilih (`stage1_kind`):
        "linear"       - LinearRegression pada seluruh fitur (replikasi versi lama)
        "structural"   - prediktor struktural hierarkis per (Toko x Hari x Promo)
                         dengan fallback bertingkat; menjawab D2
        "struct_linear"- struktural, lalu LR pada residualnya; menangkap efek
                         linier sisa (tren, hari libur) di atas struktur toko
  S2  Tahap residual XGBoost, grid boleh turun sampai kapasitas sangat kecil.
  G   Gerbang residual adaptif w dalam [0, 1]:

          y_hat = S1(x) + w * S2(residual)

      w dipilih HANYA pada validation, bersama hyperparameter S2. Karena w = 0
      termasuk dalam ruang pencarian, kerangka ini secara konstruksi TIDAK PERNAH
      lebih buruk daripada tahap pertamanya sendiri pada validation - menjawab D3.

Kuantitas diagnostik `resid_val_r2` (R2 tahap kedua terhadap residual validation)
dilaporkan pada setiap baris. Hubungan antara nilai itu dan gerbang w terpilih
adalah kriteria empiris "kapan hibridisasi residual layak diterapkan".

--------------------------------------------------------------------------------
REVISI (R1-1, reviewer IJIES Paper ID 20265893, putaran ke-2): kebocoran residual
in-sample.

Sampai revisi ini, residual yang melatih tahap kedua dibentuk sebagai
`y_fit - stage1(X_fit, y_fit).predict(X_fit)` -- tahap pertama di-fit dan
diprediksi pada BLOK YANG SAMA. Untuk S1 = "linear" ini adalah residual latih
standar (bias kecil, dikenal). Untuk S1 = "structural" / "struct_linear" ini
jauh lebih serius: prediktor struktural adalah rata-rata kelompok (Toko x Hari
x Promo), sehingga y observasi itu sendiri IKUT MEMBENTUK rata-rata kelompok
yang dipakai memprediksinya -- residual yang dihasilkan sistematis terlalu
kecil, sebanding 1/n_kelompok. Pada varian `augment_stage1=True` (skema Aug,
pemenang exp05d/exp06b) prediksi in-sample yang sama itu JUGA disuntikkan
sebagai fitur bagi XGBoost, sehingga tahap kedua bisa "mengintip" y lewat dua
jalur sekaligus.

Perbaikan: `_cross_fitted_stage1` membentuk residual latih lewat K lipatan
kronologis DI DALAM blok yang sedang di-fit (train pada fase tuning; train+val
pada fase refit final) -- setiap baris diprediksi oleh model tahap pertama
yang TIDAK PERNAH melihat baris itu (atau y-nya). Ini berlaku pada seluruh S1
(linear/structural/struct_linear) tanpa mengubah `make_stage1`: fungsi stage1
dipanggil ulang sebagai kotak hitam pada tiap lipatan.

Prediksi tahap pertama pada validation dan test TIDAK diubah -- keduanya sudah
diprediksi oleh model yang di-fit pada blok yang sepenuhnya terpisah (train
untuk validation; train+val untuk test), sehingga tidak pernah bocor.

Parameter `cross_fit_folds=None` mereproduksi perilaku LAMA (in-sample) secara
sengaja, HANYA untuk tabel audit "sebelum vs sesudah" pada response letter --
tidak untuk hasil utama naskah.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import time
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import protocol as P

# Gerbang: w = 0 (tanpa koreksi) sampai w = 1 (koreksi penuh, perilaku versi lama).
GATE_GRID = np.round(np.linspace(0.0, 1.0, 11), 3)

# Grid tahap residual. Dibanding GRID_XGB_ROSSMANN, batas bawah n_estimators
# diturunkan karena bukti solusi-pojok pada exp01.
GRID_XGB_ARLRX = {
    "n_estimators": [300, 900],
    "max_depth": [6, 9],
    "learning_rate": [0.1, 0.2],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "max_bin": [256],
}

STAGE1_KINDS = ("linear", "structural", "struct_linear")

# Jumlah lipatan cross-fitting default untuk pembentukan residual latih (R1-1).
# 5, konsisten dengan N_GATE_FOLDS yang sudah dipakai untuk validasi-silang
# gerbang per segmen (exp05d) -- bukan nilai baru yang perlu dijustifikasi
# terpisah di naskah. Satu sumber kebenaran: protocol.py (dipakai juga oleh
# fp_lr_xgb_residual, sehingga kesetaraan S1=linear, w=1 dengan hibrida tanpa
# gerbang tetap eksak).
N_CROSS_FIT_FOLDS = P.N_CROSS_FIT_FOLDS


# --------------------------------------------------------------------------- #
# Tahap pertama
# --------------------------------------------------------------------------- #

def _hierarchical_group_predict(keys_fit: np.ndarray, y_fit: np.ndarray,
                                keys_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Prediktor struktural hierarkis.

    Rata-rata y pada level kunci terpanjang; kombinasi kunci yang tidak pernah
    muncul di blok fit mundur ke level yang lebih pendek, dan akhirnya ke rerata
    global. Seluruh statistik dihitung HANYA dari blok fit (kontrak C6).

    Implementasi memakai groupby+merge pada DataFrame, bukan MultiIndex.reindex,
    karena penanganan MultiIndex satu level berbeda antar versi pandas dan level
    itu PASTI tercapai pada Rossmann (kombinasi Toko x Hari x Promo yang tidak
    pernah muncul di blok training, mis. promo di hari Minggu).
    """
    n_levels = keys_fit.shape[1]
    pred_fit = np.full(len(keys_fit), np.nan)
    pred_eval = np.full(len(keys_eval), np.nan)
    global_mean = float(np.mean(y_fit))

    cols = [f"k{i}" for i in range(n_levels)]
    df_fit = pd.DataFrame(keys_fit, columns=cols)
    df_fit["_y"] = np.asarray(y_fit, dtype=float)
    df_eval = pd.DataFrame(keys_eval, columns=cols)

    for depth in range(n_levels, 0, -1):
        level = cols[:depth]
        table = (df_fit.groupby(level, sort=False)["_y"].mean()
                 .rename("_m").reset_index())
        for frame, pred in ((df_fit, pred_fit), (df_eval, pred_eval)):
            missing = np.isnan(pred)
            if not missing.any():
                continue
            slot = np.nonzero(missing)[0]
            # merge how="left" mempertahankan urutan baris frame kiri
            merged = frame.loc[missing, level].merge(table, on=level, how="left")
            pred[slot] = merged["_m"].to_numpy()

    pred_fit = np.nan_to_num(pred_fit, nan=global_mean)
    pred_eval = np.nan_to_num(pred_eval, nan=global_mean)
    return pred_fit, pred_eval


def _key_columns(feature_names: Sequence[str],
                 preferred: Sequence[str] = ("Store", "DayOfWeek", "Promo")) -> list:
    """Indeks kolom kunci struktural yang benar-benar tersedia, sesuai urutan."""
    idx = []
    for name in preferred:
        if name in feature_names:
            idx.append(list(feature_names).index(name))
    if not idx:
        raise ValueError("tidak ada kolom kunci struktural pada matriks fitur")
    return idx


def make_stage1(kind: str, feature_names: Sequence[str]) -> Callable:
    """Kembalikan fungsi stage1(X_fit, y_fit, X_eval) -> (pred_fit, pred_eval)."""
    if kind not in STAGE1_KINDS:
        raise ValueError(f"stage1_kind tidak dikenal: {kind}")

    if kind == "linear":
        def stage1(X_fit, y_fit, X_eval):
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X_fit, y_fit)
            return model.predict(X_fit), model.predict(X_eval)
        return stage1

    key_idx = _key_columns(feature_names)

    def stage1_structural(X_fit, y_fit, X_eval):
        return _hierarchical_group_predict(X_fit[:, key_idx], y_fit, X_eval[:, key_idx])

    if kind == "structural":
        return stage1_structural

    def stage1_struct_linear(X_fit, y_fit, X_eval):
        base_fit, base_eval = stage1_structural(X_fit, y_fit, X_eval)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X_fit, y_fit - base_fit)
        return base_fit + model.predict(X_fit), base_eval + model.predict(X_eval)

    return stage1_struct_linear


# --------------------------------------------------------------------------- #
# Cross-fitting tahap pertama (R1-1) -- lihat catatan revisi di atas.
# --------------------------------------------------------------------------- #

# Batas lipatan kronologis kontiguous, gaya sama dengan `_oof_gate_rmse`;
# didefinisikan di protocol.py agar fp_lr_xgb_residual memakai lipatan identik.
_chrono_fold_bounds = P._chrono_fold_bounds


def _cross_fitted_stage1(stage1: Callable, X: np.ndarray, y: np.ndarray,
                         n_folds: int = N_CROSS_FIT_FOLDS) -> np.ndarray:
    """Prediksi tahap pertama YANG DIPAKAI UNTUK MEMBENTUK RESIDUAL, cross-fitted.

    Untuk tiap lipatan kronologis k di dalam blok (X, y) yang diberikan, tahap
    pertama di-fit HANYA pada baris di luar lipatan k lalu memprediksi baris di
    dalam lipatan k. Setiap baris karena itu diprediksi oleh sebuah model yang
    tidak pernah melihat baris itu sendiri (atau y-nya) -- menjawab R1-1:
    kebocoran residual in-sample, paling parah untuk penaksir struktural
    (rata-rata kelompok yang memuat titiknya sendiri).

    `stage1` diperlakukan sebagai kotak hitam (signature identik dengan yang
    dikembalikan `make_stage1`); fungsi ini karena itu berlaku sama untuk
    "linear", "structural", maupun "struct_linear" tanpa perubahan apa pun
    pada `make_stage1`.

    Lipatan kronologis kontiguous (bukan acak): konsisten dengan `_oof_gate_rmse`
    yang sudah dipakai untuk seleksi gerbang per segmen, dan tetap deterministik
    tanpa bergantung pada seed pengacakan.
    """
    n = len(y)
    if n_folds < 2:
        raise ValueError(f"cross_fit_folds harus >= 2, dapat {n_folds}")
    bounds = _chrono_fold_bounds(n, n_folds)
    oof = np.full(n, np.nan)
    for i in range(n_folds):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi <= lo:
            continue
        held = np.zeros(n, dtype=bool)
        held[lo:hi] = True
        fit = ~held
        if not fit.any():
            raise ValueError("lipatan cross-fit mencakup seluruh blok yang diberikan")
        _, pred_held = stage1(X[fit], y[fit], X[held])
        oof[held] = pred_held
    if np.isnan(oof).any():
        raise RuntimeError("sebagian baris tidak tercakup oleh lipatan cross-fit")
    return oof


# --------------------------------------------------------------------------- #
# Runner AR-LRX (mengikuti kontrak C3/C6/C7 pada protocol.py)
# --------------------------------------------------------------------------- #

def run_arlrx(model_name: str,
              dataset,
              stage1_kind: str,
              param_grid: Optional[Dict[str, Sequence]] = None,
              gate_grid: Sequence[float] = GATE_GRID,
              clip_nonnegative: bool = True,
              inverse_transform: Optional[Callable] = None,
              extra: Optional[Dict] = None,
              cross_fit_folds: Optional[int] = N_CROSS_FIT_FOLDS) -> Dict:
    """Satu baris hasil AR-LRX, skema kolom identik dengan `protocol.run_model`.

    Kontrak yang ditegakkan:
      * hyperparameter tahap kedua DAN gerbang w dipilih hanya dari RMSE validation;
      * tahap pertama di-fit ulang pada blok training aktif (train untuk tuning,
        train+val untuk refit), tidak pernah melihat test;
      * test diprediksi tepat satu kali oleh konfigurasi final;
      * (R1-1) residual yang melatih tahap kedua dibentuk dari prediksi tahap
        pertama CROSS-FITTED di dalam blok yang sedang di-fit -- lihat catatan
        revisi di kepala berkas dan `_cross_fitted_stage1`.

    `cross_fit_folds`: jumlah lipatan cross-fitting (>=2). `None` mereproduksi
    perilaku LAMA (residual in-sample, leaky) -- disediakan hanya untuk tabel
    audit "sebelum vs sesudah" pada response letter, JANGAN dipakai untuk hasil
    utama naskah.

    Catatan efisiensi: w tidak memengaruhi pelatihan tahap kedua, sehingga untuk
    setiap konfigurasi XGBoost seluruh nilai w dievaluasi tanpa pelatihan ulang.
    Anggaran komputasinya karena itu sama dengan `run_model` biasa, ditambah
    `cross_fit_folds` kali fit tahap pertama (tahap pertama jauh lebih murah
    daripada XGBoost, sehingga tambahan ini kecil).
    """
    t0 = time.time()
    grid = P.param_grid_list(param_grid) if param_grid else [{}]
    gate_grid = list(gate_grid)
    stage1 = make_stage1(stage1_kind, dataset.feature_names)

    X_tr, y_tr = dataset.X_train, dataset.y_train
    X_va, y_va = dataset.X_val, dataset.y_val

    # ---- fase 1: tuning, HANYA validation ---------------------------------
    # s1_va berasal dari SATU fit pada seluruh X_tr, memprediksi X_va -- blok
    # terpisah, tidak bocor. s1_tr (dibuang di sini bila cross-fitted) adalah
    # yang leaky pada versi lama: fit dan prediksi pada X_tr yang sama.
    s1_tr_insample, s1_va = stage1(X_tr, y_tr, X_va)
    if cross_fit_folds is None:
        s1_tr = s1_tr_insample          # LEGACY, leaky -- hanya untuk audit
    else:
        s1_tr = _cross_fitted_stage1(stage1, X_tr, y_tr, cross_fit_folds)
    residual_tr = y_tr - s1_tr
    residual_va = y_va - s1_va

    # RMSE tahap pertama sendirian pada validation: acuan sumbangan gerbang
    s1_va_clipped = np.maximum(s1_va, 0.0) if clip_nonnegative else s1_va
    stage1_val_rmse = float(np.sqrt(np.mean((y_va - s1_va_clipped) ** 2)))

    best = {"rmse": np.inf, "params": None, "w": 0.0, "pred": None, "resid_r2": np.nan}
    for params in grid:
        model = P.make_xgb(params).fit(X_tr, residual_tr)
        correction_va = model.predict(X_va)
        ss_tot = float(np.sum((residual_va - residual_va.mean()) ** 2))
        resid_r2 = (1.0 - float(np.sum((residual_va - correction_va) ** 2)) / ss_tot
                    if ss_tot > 0 else np.nan)
        for w in gate_grid:
            pred = s1_va + w * correction_va
            if clip_nonnegative:
                pred = np.maximum(pred, 0.0)
            rmse = float(np.sqrt(np.mean((y_va - pred) ** 2)))
            if rmse < best["rmse"]:                 # tie-break: konfigurasi pertama menang
                best = {"rmse": rmse, "params": params, "w": float(w),
                        "pred": pred, "resid_r2": resid_r2}

    # ---- fase 2: refit pada train+val, test disentuh sekali ----------------
    # s1_te berasal dari SATU fit pada seluruh X_tv, memprediksi test -- blok
    # terpisah, tidak pernah bocor. Residual yang melatih model FINAL (s1_tv)
    # dibentuk cross-fitted dengan alasan yang sama seperti fase tuning.
    X_tv, y_tv = dataset.X_trainval, dataset.y_trainval
    s1_tv_insample, s1_te = stage1(X_tv, y_tv, dataset.X_test)
    if cross_fit_folds is None:
        s1_tv = s1_tv_insample          # LEGACY, leaky -- hanya untuk audit
    else:
        s1_tv = _cross_fitted_stage1(stage1, X_tv, y_tv, cross_fit_folds)
    final = P.make_xgb(best["params"]).fit(X_tv, y_tv - s1_tv)
    correction_te = final.predict(dataset.X_test)
    test_pred = s1_te + best["w"] * correction_te
    if clip_nonnegative:
        test_pred = np.maximum(test_pred, 0.0)

    # tahap pertama sendirian pada test, untuk mengukur sumbangan gerbang
    s1_only = np.maximum(s1_te, 0.0) if clip_nonnegative else s1_te

    row = {
        "model": model_name,
        **dataset.describe(),
        "params": json.dumps(best["params"], sort_keys=True, default=str),
        "n_grid": len(grid) * len(gate_grid),
        "scaler": "none",
        "seed": P.SEED,
        "stage1": stage1_kind,
        "cross_fit_folds": cross_fit_folds if cross_fit_folds is not None else 0,
        "gate_w": best["w"],
        # Diagnostik utama: berapa persen RMSE validation turun berkat koreksi
        # residual bergerbang, relatif terhadap tahap pertama sendirian.
        # Nilai ini yang dipakai sebagai kriteria "kapan hibridisasi layak",
        # BUKAN resid_val_r2 -- R2 memusatkan data sehingga meremehkan koreksi
        # yang sebagian besar memperbaiki pergeseran level antara train dan val.
        "stage1_val_RMSE": stage1_val_rmse,
        "gate_val_gain_pct": round((stage1_val_rmse - best["rmse"]) / stage1_val_rmse * 100, 4),
        "resid_val_r2": best["resid_r2"],
        "stage1_only_test_RMSE": float(np.sqrt(np.mean((dataset.y_test - s1_only) ** 2))),
        **P.compute_metrics(y_va, best["pred"], prefix="val_"),
        **P.compute_metrics(dataset.y_test, test_pred, prefix="test_"),
        "runtime_s": round(time.time() - t0, 3),
    }
    if inverse_transform is not None:
        row.update(P.compute_metrics(inverse_transform(dataset.y_test),
                                     inverse_transform(test_pred), prefix="orig_"))
    if extra:
        row.update(extra)
    row["_val_pred"] = best["pred"]
    row["_test_pred"] = test_pred
    row["_stage1_test_pred"] = s1_only
    # Disimpan agar kurva sensitivitas gerbang dan ablasi "gerbang dimatikan"
    # dapat dihitung TANPA melatih ulang apa pun: prediksi pada sembarang w
    # adalah s1_te + w * correction_te. Penambahan ini murni penyimpanan,
    # tidak mengubah satu pun angka yang dihitung di atas.
    row["_stage1_test_raw"] = s1_te
    row["_test_correction"] = correction_te
    return row


def run_stage1_only(model_name: str, dataset, stage1_kind: str,
                    clip_nonnegative: bool = True,
                    inverse_transform: Optional[Callable] = None) -> Dict:
    """Tahap pertama sendirian, tanpa koreksi residual. Tidak punya hyperparameter,
    tetapi tetap dievaluasi pada split yang sama persis."""
    t0 = time.time()
    stage1 = make_stage1(stage1_kind, dataset.feature_names)
    _, val_pred = stage1(dataset.X_train, dataset.y_train, dataset.X_val)
    _, test_pred = stage1(dataset.X_trainval, dataset.y_trainval, dataset.X_test)
    if clip_nonnegative:
        val_pred = np.maximum(val_pred, 0.0)
        test_pred = np.maximum(test_pred, 0.0)

    row = {
        "model": model_name, **dataset.describe(),
        "params": "{}", "n_grid": 0, "scaler": "none", "seed": P.SEED,
        "stage1": stage1_kind, "gate_w": 0.0,
        "stage1_val_RMSE": float(np.sqrt(np.mean((dataset.y_val - val_pred) ** 2))),
        "gate_val_gain_pct": 0.0, "resid_val_r2": np.nan,
        "stage1_only_test_RMSE": float(np.sqrt(np.mean((dataset.y_test - test_pred) ** 2))),
        **P.compute_metrics(dataset.y_val, val_pred, prefix="val_"),
        **P.compute_metrics(dataset.y_test, test_pred, prefix="test_"),
        "runtime_s": round(time.time() - t0, 3),
    }
    if inverse_transform is not None:
        row.update(P.compute_metrics(inverse_transform(dataset.y_test),
                                     inverse_transform(test_pred), prefix="orig_"))
    row["_val_pred"] = val_pred
    row["_test_pred"] = test_pred
    return row


# --------------------------------------------------------------------------- #
# Gerbang per segmen (exp05d) -- PENAMBAHAN MURNI, tidak menyentuh `run_arlrx`.
#
# Latar: tabel segmen exp05b menunjukkan satu nilai w global adalah kompromi
# buruk. Pada V3, AR-LRX kalah dari XGBoost polos di Selasa (+12,5%) dan Kamis
# (+13,7%) tetapi menang besar di Sabtu (-17,9%) dan Senin (-10,6%). Pola
# sebeda itu meminta gerbang yang bergantung konteks:
#
#     y_hat = S1(x) + w(s(x)) * S2(x; y - S1(x))
#
# dengan s(x) segmen yang dapat dihitung saat prediksi (promo, hari, kuartil
# volume toko).
#
# MASALAH YANG HARUS DIJINAKKAN. Memilih satu w per segmen memperbesar ruang
# seleksi (sampai 14 parameter, bukan 1). Uji coba awal menunjukkan gejalanya
# jelas: RMSE validation turun, RMSE test justru NAIK sampai 3%. Itu overfitting
# seleksi, gejala yang sama yang sudah terlihat pada exp05b (V3 dengan S1 linear
# memburuk 7,5% semata-mata karena w ikut dipilih).
#
# PENJINAKANNYA: skema segmentasi dan hyperparameter TIDAK dipilih dari RMSE
# validation langsung, melainkan dari RMSE VALIDASI-SILANG DI DALAM validation
# (`n_folds` lipatan kronologis). Untuk tiap lipatan, peta w dipasang pada
# lipatan lain dan dinilai pada lipatan yang ditahan. Dengan begitu skema yang
# hanya mencocokkan derau tidak akan terpilih, dan skema "global" (satu segmen,
# setara gerbang skalar) menang secara wajar bila segmentasi tidak membantu.
# Setelah skema terpilih, peta w akhir dipasang ulang pada SELURUH validation.
#
# Test tetap disentuh tepat satu kali. Kuartil toko dihitung dari blok yang
# sedang dilatih saja -- tidak pernah dari test.
# --------------------------------------------------------------------------- #

SEGMENT_SCHEMES = ("global", "promo", "dow", "dow_promo", "store_q")
N_GATE_FOLDS = 5


def _fit_segmenter(X_fit, y_fit, feature_names, scheme: str, n_bins: int = 4) -> Dict:
    fn = list(feature_names)
    if scheme == "global":
        return {"scheme": scheme}
    if scheme in ("dow", "promo", "dow_promo"):
        need = {"dow": ["DayOfWeek"], "promo": ["Promo"],
                "dow_promo": ["DayOfWeek", "Promo"]}[scheme]
        for c in need:
            if c not in fn:
                raise ValueError(f"skema {scheme} butuh kolom {c}")
        return {"scheme": scheme}
    if scheme == "store_q":
        if "Store" not in fn:
            raise ValueError("skema store_q butuh kolom Store")
        i = fn.index("Store")
        means = (pd.Series(np.asarray(y_fit, dtype=float))
                 .groupby(pd.Series(X_fit[:, i])).mean())
        cuts = np.quantile(means.to_numpy(), np.linspace(0, 1, n_bins + 1)[1:-1])
        binmap = {float(s): int(np.searchsorted(cuts, float(v)))
                  for s, v in means.items()}
        return {"scheme": scheme, "binmap": binmap, "default": n_bins // 2}
    raise ValueError(f"skema segmentasi tidak dikenal: {scheme}")


def _apply_segmenter(seg: Dict, X, feature_names) -> np.ndarray:
    scheme = seg["scheme"]
    if scheme == "global":
        return np.zeros(len(X), dtype=np.int64)
    fn = list(feature_names)
    if scheme == "dow":
        return X[:, fn.index("DayOfWeek")].astype(np.int64)
    if scheme == "promo":
        return X[:, fn.index("Promo")].astype(np.int64)
    if scheme == "dow_promo":
        return (X[:, fn.index("DayOfWeek")].astype(np.int64) * 2
                + X[:, fn.index("Promo")].astype(np.int64))
    if scheme == "store_q":
        st = X[:, fn.index("Store")]
        bm, dflt = seg["binmap"], seg["default"]
        return np.array([bm.get(float(s), dflt) for s in st], dtype=np.int64)
    raise ValueError(scheme)


def _best_global_w(y, s1, corr, gate_grid, clip_nonnegative=True) -> float:
    best_w, best = float(list(gate_grid)[0]), np.inf
    for w in gate_grid:
        pred = s1 + w * corr
        if clip_nonnegative:
            pred = np.maximum(pred, 0.0)
        sse = float(np.sum((y - pred) ** 2))
        if sse < best:
            best, best_w = sse, float(w)
    return best_w


def _choose_w_per_segment(y, s1, corr, seg_ids, gate_grid, clip_nonnegative=True):
    """w yang meminimalkan galat kuadrat DI DALAM tiap segmen."""
    w_map, gate_grid = {}, list(gate_grid)
    for sid in np.unique(seg_ids):
        mask = seg_ids == sid
        w_map[int(sid)] = _best_global_w(y[mask], s1[mask], corr[mask],
                                         gate_grid, clip_nonnegative)
    return w_map


def _apply_w_map(s1, corr, seg_ids, w_map, default_w, clip_nonnegative=True):
    w = np.array([w_map.get(int(s), default_w) for s in seg_ids], dtype=float)
    pred = s1 + w * corr
    if clip_nonnegative:
        pred = np.maximum(pred, 0.0)
    return pred, w


def _oof_gate_rmse(y, s1, corr, seg_ids, gate_grid, n_folds=N_GATE_FOLDS,
                   clip_nonnegative=True) -> float:
    """RMSE validasi-silang kronologis DI DALAM validation.

    Inilah kriteria seleksi yang jujur: peta w dipasang pada lipatan lain dan
    dinilai pada lipatan yang ditahan, sehingga skema yang hanya mencocokkan
    derau validation tidak memperoleh keuntungan palsu.
    """
    n = len(y)
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    sse = 0.0
    for i in range(n_folds):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        held = np.zeros(n, dtype=bool)
        held[lo:hi] = True
        fit = ~held
        if not fit.any():
            continue
        gw = _best_global_w(y[fit], s1[fit], corr[fit], gate_grid, clip_nonnegative)
        w_map = _choose_w_per_segment(y[fit], s1[fit], corr[fit], seg_ids[fit],
                                      gate_grid, clip_nonnegative)
        pred, _ = _apply_w_map(s1[held], corr[held], seg_ids[held], w_map, gw,
                               clip_nonnegative)
        sse += float(np.sum((y[held] - pred) ** 2))
    return float(np.sqrt(sse / n))


def run_arlrx_segmented(model_name: str,
                        dataset,
                        stage1_kind: str,
                        param_grid: Optional[Dict[str, Sequence]] = None,
                        gate_grid: Sequence[float] = GATE_GRID,
                        schemes: Sequence[str] = SEGMENT_SCHEMES,
                        n_folds: int = N_GATE_FOLDS,
                        augment_stage1: bool = False,
                        clip_nonnegative: bool = True,
                        inverse_transform: Optional[Callable] = None,
                        extra: Optional[Dict] = None,
                        cross_fit_folds: Optional[int] = N_CROSS_FIT_FOLDS) -> Dict:
    """AR-LRX dengan gerbang bergantung segmen, diseleksi lewat validasi-silang
    di dalam validation. Skema kolom identik `run_arlrx`, ditambah
    `segment_scheme`, `n_segments`, `gate_w_map`, `val_oof_RMSE`.

    `augment_stage1=True` menambahkan prediksi tahap pertama sebagai satu kolom
    fitur bagi tahap kedua. Motivasinya: gerbang w(s) hanyalah tabel pencarian
    konstan-sepotong, sedangkan dengan mengetahui S1(x) tahap kedua dapat
    mempelajari besaran koreksi yang bergantung fitur secara kontinu -- gerbang
    terpelajar yang secara ketat lebih umum daripada w(s).

    (R1-1) S1 pada blok fit, dipakai untuk membentuk residual DAN sebagai fitur
    augmentasi, adalah cross-fitted -- bukan lagi in-sample seperti versi lama.
    Ini menutup persis jalur kebocoran yang disebut reviewer untuk varian Aug:
    "the augmented variant then also supplies that [in-sample] prediction to
    XGBoost". `cross_fit_folds=None` mereproduksi perilaku lama (leaky), hanya
    untuk audit -- lihat `run_arlrx` dan `_cross_fitted_stage1`.

    Dengan `schemes=("global",)`, `augment_stage1=False`, fungsi ini
    mereproduksi gerbang skalar `run_arlrx` persis (termasuk pilihan
    `cross_fit_folds`).
    """
    t0 = time.time()
    grid = P.param_grid_list(param_grid) if param_grid else [{}]
    gate_grid = list(gate_grid)
    stage1 = make_stage1(stage1_kind, dataset.feature_names)
    fnames = dataset.feature_names

    X_tr, y_tr = dataset.X_train, dataset.y_train
    X_va, y_va = dataset.X_val, dataset.y_val

    # s1_va: satu fit pada X_tr, memprediksi X_va -- blok terpisah, tidak bocor.
    # s1_tr: dipakai membentuk residual_tr DAN (bila augment_stage1) sebagai
    # fitur -- cross-fitted di sini persis seperti pada run_arlrx.
    s1_tr_insample, s1_va = stage1(X_tr, y_tr, X_va)
    if cross_fit_folds is None:
        s1_tr = s1_tr_insample          # LEGACY, leaky -- hanya untuk audit
    else:
        s1_tr = _cross_fitted_stage1(stage1, X_tr, y_tr, cross_fit_folds)
    residual_tr = y_tr - s1_tr
    s1_va_clipped = np.maximum(s1_va, 0.0) if clip_nonnegative else s1_va
    stage1_val_rmse = float(np.sqrt(np.mean((y_va - s1_va_clipped) ** 2)))

    usable, seg_fit = [], {}
    for sc in schemes:
        try:
            seg_fit[sc] = _fit_segmenter(X_tr, y_tr, fnames, sc)
            usable.append(sc)
        except ValueError:
            continue
    if not usable:
        raise ValueError("tidak ada skema segmentasi yang dapat dipakai")

    best = {"oof": np.inf, "params": None, "scheme": "global",
            "w_map": {0: 0.0}, "default_w": 0.0, "pred": None,
            "resid_r2": np.nan, "val_rmse": np.inf}

    # Opsi penambahan fitur: tahap kedua ikut melihat prediksi tahap pertama.
    if augment_stage1:
        X_tr_s2 = np.hstack([X_tr, s1_tr.reshape(-1, 1)])
        X_va_s2 = np.hstack([X_va, s1_va.reshape(-1, 1)])
    else:
        X_tr_s2, X_va_s2 = X_tr, X_va

    for params in grid:
        model = P.make_xgb(params).fit(X_tr_s2, residual_tr)
        corr_va = model.predict(X_va_s2)
        resid_va = y_va - s1_va
        ss_tot = float(np.sum((resid_va - resid_va.mean()) ** 2))
        resid_r2 = (1.0 - float(np.sum((resid_va - corr_va) ** 2)) / ss_tot
                    if ss_tot > 0 else np.nan)
        gw_full = _best_global_w(y_va, s1_va, corr_va, gate_grid, clip_nonnegative)
        for sc in usable:
            ids_va = _apply_segmenter(seg_fit[sc], X_va, fnames)
            # KRITERIA SELEKSI: out-of-fold di dalam validation
            oof = _oof_gate_rmse(y_va, s1_va, corr_va, ids_va, gate_grid,
                                 n_folds, clip_nonnegative)
            if oof < best["oof"]:
                # peta w akhir dipasang pada SELURUH validation
                w_map = _choose_w_per_segment(y_va, s1_va, corr_va, ids_va,
                                              gate_grid, clip_nonnegative)
                pred, _ = _apply_w_map(s1_va, corr_va, ids_va, w_map, gw_full,
                                       clip_nonnegative)
                best = {"oof": oof, "params": params, "scheme": sc,
                        "w_map": w_map, "default_w": gw_full, "pred": pred,
                        "resid_r2": resid_r2,
                        "val_rmse": float(np.sqrt(np.mean((y_va - pred) ** 2)))}

    # ---- refit train+val, test disentuh sekali -----------------------------
    # s1_te: satu fit pada X_tv, memprediksi test -- blok terpisah, tidak bocor.
    # s1_tv: melatih model FINAL dan (bila augment_stage1) fitur augmentasi
    # finalnya -- cross-fitted dengan alasan sama seperti fase tuning.
    X_tv, y_tv = dataset.X_trainval, dataset.y_trainval
    s1_tv_insample, s1_te = stage1(X_tv, y_tv, dataset.X_test)
    if cross_fit_folds is None:
        s1_tv = s1_tv_insample          # LEGACY, leaky -- hanya untuk audit
    else:
        s1_tv = _cross_fitted_stage1(stage1, X_tv, y_tv, cross_fit_folds)
    if augment_stage1:
        X_tv_s2 = np.hstack([X_tv, s1_tv.reshape(-1, 1)])
        X_te_s2 = np.hstack([dataset.X_test, s1_te.reshape(-1, 1)])
    else:
        X_tv_s2, X_te_s2 = X_tv, dataset.X_test
    final = P.make_xgb(best["params"]).fit(X_tv_s2, y_tv - s1_tv)
    corr_te = final.predict(X_te_s2)
    seg_te = _fit_segmenter(X_tv, y_tv, fnames, best["scheme"])
    ids_te = _apply_segmenter(seg_te, dataset.X_test, fnames)
    test_pred, w_used = _apply_w_map(s1_te, corr_te, ids_te, best["w_map"],
                                     best["default_w"], clip_nonnegative)
    s1_only = np.maximum(s1_te, 0.0) if clip_nonnegative else s1_te

    row = {
        "model": model_name,
        **dataset.describe(),
        "params": json.dumps(best["params"], sort_keys=True, default=str),
        "n_grid": len(grid) * len(gate_grid) * len(usable),
        "scaler": "none",
        "seed": P.SEED,
        "stage1": stage1_kind,
        "cross_fit_folds": cross_fit_folds if cross_fit_folds is not None else 0,
        "augment_stage1": bool(augment_stage1),
        "segment_scheme": best["scheme"],
        "n_segments": len(best["w_map"]),
        "gate_w_map": json.dumps({str(k): v for k, v in sorted(best["w_map"].items())}),
        "gate_w": float(np.mean(w_used)),
        "gate_w_min": float(np.min(w_used)),
        "gate_w_max": float(np.max(w_used)),
        "val_oof_RMSE": best["oof"],
        "stage1_val_RMSE": stage1_val_rmse,
        "gate_val_gain_pct": round((stage1_val_rmse - best["val_rmse"]) / stage1_val_rmse * 100, 4),
        "resid_val_r2": best["resid_r2"],
        "stage1_only_test_RMSE": float(np.sqrt(np.mean((dataset.y_test - s1_only) ** 2))),
        **P.compute_metrics(y_va, best["pred"], prefix="val_"),
        **P.compute_metrics(dataset.y_test, test_pred, prefix="test_"),
        "runtime_s": round(time.time() - t0, 3),
    }
    if inverse_transform is not None:
        row.update(P.compute_metrics(inverse_transform(dataset.y_test),
                                     inverse_transform(test_pred), prefix="orig_"))
    if extra:
        row.update(extra)
    row["_val_pred"] = best["pred"]
    row["_test_pred"] = test_pred
    row["_stage1_test_pred"] = s1_only
    row["_stage1_test_raw"] = s1_te
    row["_test_correction"] = corr_te
    row["_test_segment_ids"] = ids_te
    row["_test_w"] = w_used
    return row
