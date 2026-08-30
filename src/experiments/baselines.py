"""
Baseline rujukan di bawah protokol terpadu (exp06).

Seluruh model di sini menerima MATRIKS FITUR YANG SAMA PERSIS dengan AR-LRX:
split kronologis yang sama, seed yang sama, penyetelan hanya pada validation,
refit pada train+val, test disentuh satu kali, dan evaluasi dikembalikan ke skala
asli lewat `expm1`. Itulah satu-satunya cara membuat perbandingan dengan literatur
menjadi sah - angka yang dilaporkan naskah-naskah rujukan diproduksi lewat pipeline
lama yang mengandung kebocoran `Customers` dan pembagian berbasis indeks, sehingga
tidak dapat diadu langsung dengan angka protokol adil.

CATATAN REPRESENTASI. Model berurut (CNN, RNN, LSTM, GRU, Transformer) menerima
matriks yang sama, dibentuk ulang menjadi (n, p, 1): sumbu panjangnya adalah fitur,
bukan waktu. Ini bukan kompromi yang merugikan mereka - informasi temporalnya memang
sudah berada DI DALAM fitur (lag_1 penjualan, rata-rata bergulir, dan seterusnya),
persis seperti yang diterima setiap model lain. Membangun jendela waktu terpisah
untuk model berurut saja justru akan memberi mereka informasi yang tidak dimiliki
pembanding, dan merusak kesahihan perbandingan.

CATATAN TS-XGBoost / TS-LGBM. Di bawah protokol terpadu, varian "time-series aware"
Zeng et al. runtuh menjadi model pohon biasa: rekayasa fitur temporalnya sekarang
menjadi milik bersama semua model. Karena itu yang dijalankan hanya XGBoost (grid
protokol), XGBoost (grid Zhao), dan LightGBM - bukan duplikatnya.

CATATAN ARIMA/Prophet. Tidak disertakan: keduanya model deret tunggal, sedangkan
Rossmann adalah panel 1.115 toko. Menjalankannya per toko bukan lagi baseline yang
sama dan biayanya tidak sebanding dengan informasinya.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from . import protocol as P

BATCH_SIZE = 2048
EPOCHS = 12


def set_keras_seed(seed: int = P.SEED) -> None:
    import os
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        try:
            import keras
            keras.utils.set_random_seed(seed)
        except Exception:
            pass
    except Exception:
        pass


def _seq(X: np.ndarray) -> np.ndarray:
    """(n, p) -> (n, p, 1) untuk model berurut."""
    return np.asarray(X, dtype="float32").reshape(X.shape[0], X.shape[1], 1)


# --------------------------------------------------------------------------- #
# Arsitektur (mengikuti bentuk yang dipakai naskah rujukan)
# --------------------------------------------------------------------------- #

def _build_mlp(p: int, params: Dict):
    from keras import layers, models
    u = int(params.get("units", 64))
    m = models.Sequential([
        layers.Input(shape=(p,)),
        layers.Dense(u, activation="relu"),
        layers.Dense(u // 2, activation="relu"),
        layers.Dense(1),
    ])
    return m


def _build_cnn(p: int, params: Dict):
    from keras import layers, models
    f = int(params.get("units", 64))
    m = models.Sequential([
        layers.Input(shape=(p, 1)),
        layers.Conv1D(f, kernel_size=3, padding="same", activation="relu"),
        layers.Conv1D(f // 2, kernel_size=3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(f // 2, activation="relu"),
        layers.Dense(1),
    ])
    return m


def _build_recurrent(kind: str):
    def build(p: int, params: Dict):
        from keras import layers, models
        u = int(params.get("units", 64))
        cell = {"rnn": layers.SimpleRNN, "lstm": layers.LSTM, "gru": layers.GRU}[kind]
        m = models.Sequential([
            layers.Input(shape=(p, 1)),
            cell(u),
            layers.Dense(u // 2, activation="relu"),
            layers.Dense(1),
        ])
        return m
    return build


def _build_transformer(p: int, params: Dict):
    from keras import layers, models
    d = int(params.get("units", 64))
    heads = int(params.get("heads", 4))
    inp = layers.Input(shape=(p, 1))
    x = layers.Dense(d)(inp)
    attn = layers.MultiHeadAttention(num_heads=heads, key_dim=max(d // heads, 1))(x, x)
    # layers.Add() dipakai, bukan operator "+", agar aman lintas versi Keras 2/3.
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, attn]))
    ffn = layers.Dense(d * 2, activation="relu")(x)
    ffn = layers.Dense(d)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, ffn]))
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(d // 2, activation="relu")(x)
    out = layers.Dense(1)(x)
    return models.Model(inp, out)


_BUILDERS = {
    "mlp": (_build_mlp, False),
    "cnn": (_build_cnn, True),
    "rnn": (_build_recurrent("rnn"), True),
    "lstm": (_build_recurrent("lstm"), True),
    "gru": (_build_recurrent("gru"), True),
    "transformer": (_build_transformer, True),
}


def make_keras_fp(kind: str, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                  verbose: int = 0):
    """Kembalikan fit_predict(X_fit, y_fit, X_eval, params) untuk `protocol.run_model`.

    Jumlah epoch ditetapkan (bukan early stopping) supaya fase penyetelan dan fase
    refit memakai anggaran pelatihan yang identik - syarat agar refit pada train+val
    tetap sah tanpa menyentuh test.
    """
    builder, as_seq = _BUILDERS[kind]

    def fit_predict(X_fit, y_fit, X_eval, params):
        set_keras_seed()
        from keras import optimizers
        Xf = _seq(X_fit) if as_seq else np.asarray(X_fit, dtype="float32")
        Xe = _seq(X_eval) if as_seq else np.asarray(X_eval, dtype="float32")
        model = builder(X_fit.shape[1], params)
        model.compile(optimizer=optimizers.Adam(
            learning_rate=float(params.get("learning_rate", 1e-3))), loss="mse")
        model.fit(Xf, np.asarray(y_fit, dtype="float32"),
                  epochs=int(params.get("epochs", epochs)),
                  batch_size=batch_size, shuffle=True, verbose=verbose)
        return model.predict(Xe, batch_size=batch_size, verbose=0).ravel()

    return fit_predict


# --------------------------------------------------------------------------- #
# Model pohon pembanding
# --------------------------------------------------------------------------- #

def fp_lightgbm(X_fit, y_fit, X_eval, params):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(
        n_estimators=int(params.get("n_estimators", 300)),
        num_leaves=int(params.get("num_leaves", 31)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        random_state=P.SEED, n_jobs=-1, verbose=-1)
    m.fit(X_fit, y_fit)
    return m.predict(X_eval)


def fp_xgboost_zhao(X_fit, y_fit, X_eval, params):
    """XGBoost dengan wilayah hyperparameter yang dilaporkan Zhao et al."""
    return P.fp_xgboost(X_fit, y_fit, X_eval, params)


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #

GRID_NEURAL = {"units": [32, 64], "learning_rate": [0.001]}
GRID_TRANSFORMER = {"units": [32, 64], "heads": [4], "learning_rate": [0.001]}
GRID_LIGHTGBM = {"n_estimators": [300, 900], "num_leaves": [31, 127],
                 "learning_rate": [0.1], "subsample": [0.8], "colsample_bytree": [0.8]}
GRID_XGB_ZHAO = {"n_estimators": [500, 1000], "max_depth": [8, 10],
                 "learning_rate": [0.05, 0.1], "subsample": [0.9],
                 "colsample_bytree": [0.7], "max_bin": [256]}

def smoke_test(n: int = 256, p: int = 17, verbose: bool = True) -> None:
    """Bangun-latih-prediksi setiap arsitektur pada data sintetis kecil.

    Tujuannya menangkap galat API Keras dalam hitungan detik, bukan setelah dua jam
    pelatihan. Angka yang dihasilkan tidak berarti apa-apa.
    """
    rng = np.random.default_rng(P.SEED)
    X = rng.normal(size=(n, p)).astype("float32")
    y = (X[:, 0] * 2 + rng.normal(scale=0.1, size=n)).astype("float32")
    for kind in _BUILDERS:
        params = {"units": 16, "heads": 2, "learning_rate": 1e-3, "epochs": 1}
        fp = make_keras_fp(kind, epochs=1, batch_size=64)
        pred = fp(X, y, X[:32], params)
        assert pred.shape == (32,), (kind, pred.shape)
        assert np.isfinite(pred).all(), f"{kind} menghasilkan nilai tak hingga"
        if verbose:
            print(f"  {kind:12s} OK  (contoh prediksi {pred[0]:+.4f})")
    for name, fp in (("lightgbm", fp_lightgbm), ("xgboost", fp_xgboost_zhao)):
        pred = fp(X, y, X[:32], {"n_estimators": 20, "max_depth": 4})
        assert np.asarray(pred).shape == (32,), (name, np.asarray(pred).shape)
        if verbose:
            print(f"  {name:12s} OK  (contoh prediksi {float(pred[0]):+.4f})")
    if verbose:
        print("Seluruh arsitektur dapat dibangun, dilatih, dan memprediksi.")


NEURAL_MODELS = [
    ("MLP (Diamantini)", "mlp", GRID_NEURAL),
    ("CNN (Diamantini)", "cnn", GRID_NEURAL),
    ("RNN (Diamantini)", "rnn", GRID_NEURAL),
    ("LSTM (Diamantini / Qureshi)", "lstm", GRID_NEURAL),
    ("GRU (Qureshi)", "gru", GRID_NEURAL),
    ("Transformer (Diamantini)", "transformer", GRID_TRANSFORMER),
]


# --------------------------------------------------------------------------- #
# exp06b — baseline neural yang diperkuat. PENAMBAHAN MURNI: `make_keras_fp`
# di atas tidak disentuh, sehingga exp06 tetap dapat direproduksi.
#
# Tiga cacat pada exp06 yang diperbaiki di sini, berurutan menurut besar
# pengaruhnya:
#
#   1. TARGET TIDAK DISTANDARKAN. Jaringan harus memanjat dari sekitar 0 menuju
#      rata-rata log1p(sales) ~ 8,5 dengan loss MSE, sehingga sebagian besar
#      anggaran epoch habis hanya untuk mencocokkan intercept. Model pohon tidak
#      punya masalah ini - itulah sebab utama jaraknya tampak begitu lebar.
#      Perbaikan: target distandarkan memakai rata-rata dan simpangan baku blok
#      pelatihan aktif saja, lalu prediksi dikembalikan ke skala semula.
#
#   2. EPOCH TETAP TANPA EARLY STOPPING. 12 epoch pada batch 2048 hanya ~3.456
#      langkah gradien. Perbaikan: early stopping terhadap 15% terakhir (secara
#      kronologis) dari blok pelatihan aktif, dengan pemulihan bobot terbaik.
#      Blok test tidak pernah tersentuh, dan prosedurnya identik antara fase
#      penyetelan dan fase refit - sehingga kontrak protokol tetap utuh.
#
#   3. GRID TERLALU SEMPIT. Pada exp06, 5 dari 6 model neural memilih `units=64`,
#      yaitu batas atas grid - solusi-pojok, cacat yang sama yang sudah
#      didokumentasikan pada exp01. Perbaikan: {64, 128, 256}.
#
# Ini memperkuat LAWAN, bukan metode usulan. Menang 7% atas baseline yang
# terlatih baik jauh lebih berharga daripada menang 65% atas baseline yang rusak.
# --------------------------------------------------------------------------- #

MAX_EPOCHS = 30
PATIENCE = 4
INNER_VAL_FRAC = 0.15

GRID_NEURAL_V2 = {"units": [64, 128, 256], "learning_rate": [0.001]}
GRID_TRANSFORMER_V2 = {"units": [64, 128, 256], "heads": [4], "learning_rate": [0.001]}

NEURAL_MODELS_V2 = [
    ("MLP (Diamantini)", "mlp", GRID_NEURAL_V2),
    ("CNN (Diamantini)", "cnn", GRID_NEURAL_V2),
    ("RNN (Diamantini)", "rnn", GRID_NEURAL_V2),
    ("LSTM (Diamantini / Qureshi)", "lstm", GRID_NEURAL_V2),
    ("GRU (Qureshi)", "gru", GRID_NEURAL_V2),
    ("Transformer (Diamantini)", "transformer", GRID_TRANSFORMER_V2),
]


def make_keras_fp_v2(kind: str,
                     max_epochs: int = MAX_EPOCHS,
                     patience: int = PATIENCE,
                     inner_val_frac: float = INNER_VAL_FRAC,
                     batch_size: int = BATCH_SIZE,
                     verbose: int = 0):
    """fit_predict untuk `protocol.run_model`, dengan target terstandarkan dan
    early stopping terhadap ekor kronologis blok pelatihan aktif.

    Kontrak yang tetap dijaga: seluruh statistik (rata-rata, simpangan baku,
    pemilihan epoch) dihitung HANYA dari blok yang sedang dilatih; test tidak
    pernah tersentuh; prosedur identik antara fase penyetelan (train -> val) dan
    fase refit (train+val -> test).
    """
    builder, as_seq = _BUILDERS[kind]

    def fit_predict(X_fit, y_fit, X_eval, params):
        set_keras_seed()
        from keras import callbacks, optimizers

        y_fit = np.asarray(y_fit, dtype="float64")
        mu = float(y_fit.mean())
        sd = float(y_fit.std())
        sd = sd if sd > 1e-12 else 1.0
        y_z = ((y_fit - mu) / sd).astype("float32")

        Xf = _seq(X_fit) if as_seq else np.asarray(X_fit, dtype="float32")
        Xe = _seq(X_eval) if as_seq else np.asarray(X_eval, dtype="float32")

        # Ekor kronologis sebagai set early stopping. Baris sudah terurut waktu.
        n = len(Xf)
        cut = max(1, min(n - 1, int(round(n * (1.0 - inner_val_frac)))))

        model = builder(X_fit.shape[1], params)
        model.compile(optimizer=optimizers.Adam(
            learning_rate=float(params.get("learning_rate", 1e-3))), loss="mse")
        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                    restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                        patience=2, min_lr=1e-5, verbose=0),
        ]
        model.fit(Xf[:cut], y_z[:cut],
                  validation_data=(Xf[cut:], y_z[cut:]),
                  epochs=max_epochs, batch_size=batch_size,
                  shuffle=True, verbose=verbose, callbacks=cbs)
        pred_z = model.predict(Xe, batch_size=batch_size, verbose=0).ravel()
        return pred_z * sd + mu

    return fit_predict


def smoke_test_v2(n: int = 512, p: int = 17, verbose: bool = True) -> None:
    """Versi uji-cepat untuk jalur exp06b (target terstandarkan + early stopping)."""
    rng = np.random.default_rng(P.SEED)
    X = rng.normal(size=(n, p)).astype("float32")
    # target dengan level tinggi, meniru log1p(sales) ~ 8,5 -- inilah yang dulu
    # membuat jaringan kehabisan anggaran epoch hanya untuk mengejar intercept.
    y = (8.5 + X[:, 0] * 0.5 + rng.normal(scale=0.1, size=n)).astype("float32")
    for kind in _BUILDERS:
        fp = make_keras_fp_v2(kind, max_epochs=3, patience=2, batch_size=64)
        pred = fp(X, y, X[:32], {"units": 16, "heads": 2, "learning_rate": 1e-3})
        assert pred.shape == (32,), (kind, pred.shape)
        assert np.isfinite(pred).all(), f"{kind} menghasilkan nilai tak hingga"
        # setelah standardisasi, prediksi harus sudah berada di sekitar level y
        assert abs(float(np.mean(pred)) - 8.5) < 3.0, (
            f"{kind}: level prediksi {np.mean(pred):.2f} jauh dari 8,5 - "
            "standardisasi target tampaknya tidak bekerja")
        if verbose:
            print(f"  {kind:12s} OK  (rata-rata prediksi {np.mean(pred):.3f}, target ~8,5)")
    if verbose:
        print("Standardisasi target bekerja: jaringan langsung berada di level yang benar.")
