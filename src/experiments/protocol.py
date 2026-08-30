"""
Unified experimental protocol for
"Evaluating the Empirical Robustness of a Residual-Based Linear Regression-XGBoost
Framework in Pharmaceutical and Retail Demand Forecasting".

Design contract (satu-satunya sumber kebenaran untuk SEMUA notebook eksperimen):

C1  Determinism.  Satu SEED global (42) dipasang ke Python `random`, NumPy, dan
    PYTHONHASHSEED; setiap estimator stokastik (XGBoost, KMeans) menerima
    random_state=SEED secara eksplisit. Tidak ada estimator yang dibuat tanpa seed.

C2  Split.  Kronologis, tanpa shuffle, rasio identik untuk semua model dan semua
    set fitur: 70% train / 15% validation / 15% test (SPLIT_RATIOS).

C3  Tuning.  Hyperparameter DIPILIH HANYA dari performa pada validation split.
    Test split disentuh tepat satu kali, oleh model final. Model final di-refit
    pada train+val memakai konfigurasi terbaik dari validation.
    Tidak ada GridSearchCV yang di-fit pada test, tidak ada seleksi berbasis test.

C4  Fitur.  Set fitur adalah FAKTOR EKSPERIMEN, bukan properti model.
      - FEATURE_SET_A ("lag1")  : [lag_1]                      -> protokol referensi
      - FEATURE_SET_B ("rich")  : [lag_1..lag_k, rolling_mean_k] -> protokol usulan
    k dipilih dari PACF yang dihitung HANYA pada blok training (lihat C5).
    Setiap model dijalankan pada kedua set fitur.

C5  Tidak ada informasi masa depan dalam desain fitur. Pemilihan lag (argmax PACF)
    dihitung pada 70% pertama deret mentah, bukan pada deret penuh. Ini
    memperbaiki kebocoran halus pada notebook lama yang menghitung ACF/PACF pada
    seluruh deret termasuk test.

C6  Penskalaan. Scaler apa pun di-fit ulang pada blok training aktif saja
    (train untuk fase tuning, train+val untuk fase refit) lalu ditransform ke
    blok evaluasi. Tidak pernah di-fit pada test.

C7  Pelaporan. Setiap baris hasil menyimpan: dataset, granularitas, kategori,
    set fitur, nama model, hyperparameter terpilih, metrik validation, metrik
    test, jumlah fitur, ukuran split, seed, dan waktu jalan. Ditulis sebagai
    CSV + JSON machine-readable ke results/.
"""

from __future__ import annotations

import json
import os
import platform
import random
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# C1 - determinism
# --------------------------------------------------------------------------- #

SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Pasang seed global. Panggil di cell pertama setiap notebook."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# C2 - split
# --------------------------------------------------------------------------- #

SPLIT_RATIOS = (0.70, 0.15, 0.15)


def chrono_split_bounds(n: int, ratios: Sequence[float] = SPLIT_RATIOS):
    """Kembalikan (i_train_end, i_val_end) untuk split kronologis."""
    r_train, r_val, _ = ratios
    i_tr = int(n * r_train)
    i_va = i_tr + int(n * r_val)
    return i_tr, i_va


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Percentage Error (metrik resmi kompetisi Rossmann).

    Baris dengan y_true == 0 diabaikan, sesuai aturan kompetisi.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def compute_metrics(y_true, y_pred, prefix: str = "") -> Dict[str, float]:
    """Metrik seragam. RMSE selalu = sqrt(MSE) persis (menjawab kritik reviewer
    soal RMSE/MSE yang tidak konsisten satu sama lain)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    mean_y = float(np.mean(y_true))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - mean_y) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    out = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "nRMSE": rmse / mean_y if mean_y != 0 else float("nan"),
        "SMAPE": smape(y_true, y_pred),
        "RMSPE": rmspe(y_true, y_pred),
    }
    return {f"{prefix}{k}": v for k, v in out.items()}


# --------------------------------------------------------------------------- #
# C5 - PACF dihitung hanya pada blok training
# --------------------------------------------------------------------------- #

def _pacf_levinson_durbin(x: np.ndarray, nlags: int) -> np.ndarray:
    """PACF Yule-Walker (adjusted) via rekursi Levinson-Durbin, murni NumPy.

    Dipakai sebagai fallback deterministik bila statsmodels tidak tersedia;
    hasilnya setara dengan statsmodels.tsa.stattools.pacf(method='ywadjusted')
    hingga presisi numerik.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    nlags = min(nlags, n - 1)
    # autokovarians "adjusted" (pembagi n-k)
    acov = np.empty(nlags + 1)
    for k in range(nlags + 1):
        acov[k] = np.dot(x[: n - k], x[k:]) / (n - k)
    r = acov / acov[0] if acov[0] != 0 else np.zeros_like(acov)

    pacf = np.zeros(nlags + 1)
    pacf[0] = 1.0
    phi = np.zeros((nlags + 1, nlags + 1))
    if nlags >= 1:
        phi[1, 1] = r[1]
        pacf[1] = r[1]
        for k in range(2, nlags + 1):
            num = r[k] - np.sum(phi[k - 1, 1:k] * r[k - 1 : 0 : -1])
            den = 1.0 - np.sum(phi[k - 1, 1:k] * r[1:k])
            phi[k, k] = num / den if den != 0 else 0.0
            phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, k - 1 : 0 : -1]
            pacf[k] = phi[k, k]
    return pacf


def partial_autocorrelation(x: np.ndarray, nlags: int = 26) -> np.ndarray:
    """PACF; memakai statsmodels bila ada, jika tidak memakai fallback NumPy."""
    try:
        from statsmodels.tsa.stattools import pacf as _sm_pacf  # type: ignore

        return np.asarray(_sm_pacf(np.asarray(x, dtype=float), nlags=nlags,
                                   method="ywadjusted"))
    except Exception:
        return _pacf_levinson_durbin(x, nlags)


def autocorrelation(x: np.ndarray, nlags: int = 26) -> np.ndarray:
    """ACF (pembagi adjusted n-k), setara statsmodels.acf(adjusted=True)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    nlags = min(nlags, n - 1)
    denom = np.dot(x, x) / n
    if denom == 0:
        return np.zeros(nlags + 1)
    return np.array([np.dot(x[: n - k], x[k:]) / (n - k) for k in range(nlags + 1)]) / denom


# Aturan pemilihan lag yang diuji. Notebook lama memakai dua aturan BERBEDA untuk
# dua pipeline yang seharusnya dibandingkan (our_study: argmax PACF pada deret
# penuh; rathipriya "our preprocessing": argmax ACF pada deret penuh), dan
# keduanya menghitung statistik pada data termasuk test. Semua varian disediakan
# di sini agar sensitivitas hasil terhadap pilihan ini dapat dilaporkan.
LAG_RULES = (
    "pacf_train",        # DEFAULT: argmax PACF, dihitung pada blok train saja
    "pacf_full",         # legacy our_study (bocor: memakai seluruh deret)
    "acf_train",         # argmax ACF, blok train saja
    "acf_full",          # legacy rathipriya (bocor)
    "pacf_significant",  # lag terbesar dengan |PACF| > 1.96/sqrt(n_train)
    "fixed",             # k tetap, ditentukan lewat n_lags_override
)


def select_lag(series: np.ndarray,
               rule: str = "pacf_train",
               ratios: Sequence[float] = SPLIT_RATIOS,
               nlags: int = 26) -> int:
    """Pilih k (jumlah lag) menurut `rule`.

    Aturan default (pacf_train) faithful terhadap metode paper (argmax PACF)
    tetapi dihitung tanpa melihat validation/test (C5).
    """
    series = np.asarray(series, dtype=float)
    i_tr = int(len(series) * ratios[0])
    train_block = series[:i_tr]

    if rule == "pacf_train":
        return int(np.argmax(partial_autocorrelation(train_block, nlags)[1:]) + 1)
    if rule == "pacf_full":
        return int(np.argmax(partial_autocorrelation(series, nlags)[1:]) + 1)
    if rule == "acf_train":
        return int(np.argmax(autocorrelation(train_block, nlags)[1:]) + 1)
    if rule == "acf_full":
        return int(np.argmax(autocorrelation(series, nlags)[1:]) + 1)
    if rule == "pacf_significant":
        values = partial_autocorrelation(train_block, nlags)[1:]
        threshold = 1.96 / np.sqrt(len(train_block))
        significant = np.nonzero(np.abs(values) > threshold)[0]
        return int(significant[-1] + 1) if len(significant) else 1
    raise ValueError(f"lag rule tidak dikenal: {rule}")


def select_lag_from_train(series, ratios=SPLIT_RATIOS, nlags=26) -> int:
    """Alias kompatibilitas untuk aturan default."""
    return select_lag(series, "pacf_train", ratios, nlags)


# --------------------------------------------------------------------------- #
# C4 - dataset builder (PharmaSales)
# --------------------------------------------------------------------------- #

FEATURE_SET_A = "A_lag1"     # protokol referensi (Rathipriya et al.)
FEATURE_SET_B = "B_rich"     # protokol usulan (lag terpilih + rolling mean)
FEATURE_SETS = (FEATURE_SET_A, FEATURE_SET_B)


@dataclass
class Dataset:
    """Wadah data satu (kategori x set fitur) dengan split yang sudah tetap."""
    name: str
    feature_set: str
    feature_names: List[str]
    frame: pd.DataFrame          # kolom: ds, y, <fitur...>  (sudah dropna)
    i_train_end: int
    i_val_end: int
    n_lags: int
    seasonal_period: int
    lag_rule: str = "pacf_train"

    # ---- views -----------------------------------------------------------
    @property
    def X(self) -> np.ndarray:
        return self.frame[self.feature_names].to_numpy(dtype=float)

    @property
    def y(self) -> np.ndarray:
        return self.frame["y"].to_numpy(dtype=float)

    def _slice(self, a, b):
        return slice(a, b)

    @property
    def X_train(self): return self.X[: self.i_train_end]
    @property
    def y_train(self): return self.y[: self.i_train_end]
    @property
    def X_val(self): return self.X[self.i_train_end : self.i_val_end]
    @property
    def y_val(self): return self.y[self.i_train_end : self.i_val_end]
    @property
    def X_test(self): return self.X[self.i_val_end :]
    @property
    def y_test(self): return self.y[self.i_val_end :]
    @property
    def X_trainval(self): return self.X[: self.i_val_end]
    @property
    def y_trainval(self): return self.y[: self.i_val_end]

    @property
    def dates_test(self):
        return self.frame["ds"].to_numpy()[self.i_val_end :]

    def describe(self) -> Dict[str, object]:
        ds = self.frame["ds"]
        return {
            "category": self.name,
            "feature_set": self.feature_set,
            "lag_rule": self.lag_rule,
            "n_features": len(self.feature_names),
            "n_lags": self.n_lags,
            "n_train": int(self.i_train_end),
            "n_val": int(self.i_val_end - self.i_train_end),
            "n_test": int(len(self.frame) - self.i_val_end),
            "train_start": str(ds.iloc[0].date()),
            "train_end": str(ds.iloc[self.i_train_end - 1].date()),
            "val_start": str(ds.iloc[self.i_train_end].date()),
            "val_end": str(ds.iloc[self.i_val_end - 1].date()),
            "test_start": str(ds.iloc[self.i_val_end].date()),
            "test_end": str(ds.iloc[-1].date()),
        }


def build_pharma_dataset(df: pd.DataFrame,
                         category: str,
                         feature_set: str,
                         seasonal_period: int,
                         date_col: str = "datum",
                         nlags: int = 26,
                         ratios: Sequence[float] = SPLIT_RATIOS,
                         lag_rule: str = "pacf_train",
                         n_lags_override: Optional[int] = None) -> Dataset:
    """Bangun satu Dataset PharmaSales.

    PENTING: k (jumlah lag) dipilih dari blok training saja dan SAMA untuk kedua
    set fitur pada kategori yang sama, sehingga satu-satunya perbedaan antara
    FEATURE_SET_A dan FEATURE_SET_B adalah fiturnya, bukan seleksi lag-nya.
    Baris yang dibuang akibat NaN juga disamakan (keduanya memakai k yang sama),
    sehingga train/val/test berisi TANGGAL YANG PERSIS SAMA untuk kedua set.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set tidak dikenal: {feature_set}")

    dfg = df[[date_col, category]].rename(columns={date_col: "ds", category: "y"}).copy()
    dfg["ds"] = pd.to_datetime(dfg["ds"], format="mixed", dayfirst=False, errors="coerce")
    dfg = dfg.sort_values("ds").reset_index(drop=True)

    k = int(n_lags_override) if n_lags_override is not None else \
        select_lag(dfg["y"].to_numpy(), rule=lag_rule, ratios=ratios, nlags=nlags)
    k = max(k, 1)

    # Bangun SEMUA lag sampai k supaya jumlah baris yang hilang identik antar set.
    for lag in range(1, k + 1):
        dfg[f"lag_{lag}"] = dfg["y"].shift(lag)
    dfg[f"rolling_mean_{k}"] = dfg["y"].shift(1).rolling(window=k).mean()
    dfg = dfg.dropna().reset_index(drop=True)

    if feature_set == FEATURE_SET_A:
        feature_names = ["lag_1"]
    else:
        feature_names = [f"lag_{i}" for i in range(1, k + 1)] + [f"rolling_mean_{k}"]
        # Untuk k == 1, rolling_mean_1 identik dengan lag_1 (kolinearitas sempurna
        # pada LR). Kolom duplikat dibuang; konsekuensinya, untuk kategori dengan
        # argmax PACF = 1 set fitur "rich" memang berdegenerasi menjadi set
        # referensi -- ini dilaporkan apa adanya, bukan ditutupi.
        seen, deduped = [], []
        for name in feature_names:
            col = dfg[name].to_numpy(dtype=float)
            if any(np.array_equal(col, prev) for prev in seen):
                continue
            seen.append(col)
            deduped.append(name)
        feature_names = deduped

    i_tr, i_va = chrono_split_bounds(len(dfg), ratios)
    return Dataset(name=category, feature_set=feature_set,
                   feature_names=feature_names, frame=dfg,
                   i_train_end=i_tr, i_val_end=i_va, n_lags=k,
                   seasonal_period=seasonal_period, lag_rule=lag_rule)


# --------------------------------------------------------------------------- #
# C3/C6 - runner: tuning di validation, refit di train+val, test sekali
# --------------------------------------------------------------------------- #

def param_grid_list(grid: Dict[str, Sequence]) -> List[Dict]:
    """Perluas dict grid menjadi daftar konfigurasi, urutan deterministik."""
    keys = sorted(grid.keys())
    return [dict(zip(keys, combo)) for combo in product(*(grid[k] for k in keys))]


def _fit_scaler(kind: Optional[str], X: np.ndarray):
    if kind is None:
        return None
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    scaler = StandardScaler() if kind == "standard" else MinMaxScaler()
    scaler.fit(X)
    return scaler


def _apply(scaler, X):
    return X if scaler is None else scaler.transform(X)


FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray, Dict], np.ndarray]


def run_model(model_name: str,
              fit_predict: FitPredict,
              dataset: Dataset,
              param_grid: Optional[Dict[str, Sequence]] = None,
              scaler_kind: Optional[str] = None,
              clip_nonnegative: bool = True,
              inverse_transform: Optional[Callable] = None,
              extra: Optional[Dict] = None) -> Dict:
    """Jalankan satu model mengikuti kontrak C3/C6 dan kembalikan satu baris hasil.

    fit_predict(X_fit, y_fit, X_eval, params) -> prediksi untuk X_eval.
    """
    t0 = time.time()
    grid = param_grid_list(param_grid) if param_grid else [{}]

    # ---- fase 1: tuning, HANYA validation -------------------------------
    scaler_tr = _fit_scaler(scaler_kind, dataset.X_train)
    Xtr = _apply(scaler_tr, dataset.X_train)
    Xva = _apply(scaler_tr, dataset.X_val)

    best_params, best_val_rmse, best_val_pred = None, np.inf, None
    for params in grid:
        pred = np.asarray(fit_predict(Xtr, dataset.y_train, Xva, params), dtype=float)
        if clip_nonnegative:
            pred = np.maximum(pred, 0.0)
        val_rmse = float(np.sqrt(np.mean((dataset.y_val - pred) ** 2)))
        if val_rmse < best_val_rmse:            # tie-break: konfigurasi pertama menang
            best_params, best_val_rmse, best_val_pred = params, val_rmse, pred

    # ---- fase 2: refit di train+val, test disentuh sekali ----------------
    scaler_full = _fit_scaler(scaler_kind, dataset.X_trainval)
    Xtv = _apply(scaler_full, dataset.X_trainval)
    Xte = _apply(scaler_full, dataset.X_test)

    test_pred = np.asarray(fit_predict(Xtv, dataset.y_trainval, Xte, best_params), dtype=float)
    if clip_nonnegative:
        test_pred = np.maximum(test_pred, 0.0)

    row = {
        "model": model_name,
        **dataset.describe(),
        "params": json.dumps(best_params, sort_keys=True, default=str),
        "n_grid": len(grid),
        "scaler": scaler_kind or "none",
        "seed": SEED,
        **compute_metrics(dataset.y_val, best_val_pred, prefix="val_"),
        **compute_metrics(dataset.y_test, test_pred, prefix="test_"),
        "runtime_s": round(time.time() - t0, 3),
    }
    if inverse_transform is not None:
        # Evaluasi kedua pada skala asli. Dilaporkan berdampingan, tidak pernah
        # dicampur dengan metrik skala-transformasi (kritik reviewer #2).
        row.update(compute_metrics(inverse_transform(dataset.y_test),
                                   inverse_transform(test_pred), prefix="orig_"))
    if extra:
        row.update(extra)
    row["_val_pred"] = best_val_pred
    row["_test_pred"] = test_pred
    return row


# --------------------------------------------------------------------------- #
# Model library - semua memakai signature fit_predict yang sama
# --------------------------------------------------------------------------- #

def fp_linear_regression(X_fit, y_fit, X_eval, params):
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_fit, y_fit)
    return model.predict(X_eval)


def fp_grnn(X_fit, y_fit, X_eval, params, batch_size: int = 512):
    """General Regression NN (Nadaraya-Watson dengan kernel Gaussian).

    Implementasi identik dengan baseline Rathipriya, TETAPI dijalankan pada
    fitur terstandardisasi (lihat catatan di README_EXPERIMENTS.md): pada skala
    mentah, jarak kuadrat berorde 1e4 membuat semua bobot underflow ke nol dan
    GRNN kolaps menjadi prediktor rata-rata.
    """
    sigma = max(float(params["sigma"]), 1e-12)
    preds = []
    for start in range(0, len(X_eval), batch_size):
        batch = X_eval[start : start + batch_size]
        d2 = ((batch[:, None, :] - X_fit[None, :, :]) ** 2).sum(axis=2)
        w = np.exp(-d2 / (2 * sigma ** 2))
        denom = w.sum(axis=1)
        fallback = np.full(len(batch), float(np.mean(y_fit)))
        preds.append(np.divide(w @ y_fit, denom, out=fallback, where=denom > 1e-12))
    return np.concatenate(preds)


def fp_pnn(X_fit, y_fit, X_eval, params, batch_size: int = 512):
    """Probabilistic NN versi regresi (voting bin), identik dengan baseline."""
    sigma = max(float(params["sigma"]), 1e-12)
    n_bins = int(params.get("n_bins", 20))
    y_min, y_max = float(np.min(y_fit)), float(np.max(y_fit))
    margin = (y_max - y_min) * 0.01 if y_max > y_min else 1.0
    bins = np.linspace(y_min - margin, y_max + margin, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    idx = np.clip(np.digitize(y_fit, bins) - 1, 0, n_bins - 1)

    preds = []
    for start in range(0, len(X_eval), batch_size):
        batch = X_eval[start : start + batch_size]
        d2 = ((batch[:, None, :] - X_fit[None, :, :]) ** 2).sum(axis=2)
        w = np.exp(-d2 / (2 * sigma ** 2))
        out = np.empty(len(batch))
        for j in range(len(batch)):
            out[j] = centers[np.argmax(np.bincount(idx, weights=w[j], minlength=n_bins))]
        preds.append(out)
    return np.concatenate(preds)


def fp_rbfnn(X_fit, y_fit, X_eval, params):
    """RBF network: pusat via KMeans (seeded), output layer Ridge."""
    from sklearn.cluster import KMeans
    from sklearn.linear_model import Ridge

    n_centers = min(int(params["n_centers"]), len(X_fit))
    gamma = float(params["gamma"])
    kmeans = KMeans(n_clusters=n_centers, random_state=SEED, n_init=10)
    centers = kmeans.fit(X_fit).cluster_centers_

    def phi(X):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-gamma * d2)

    model = Ridge(alpha=float(params["alpha"]), random_state=SEED)
    model.fit(phi(X_fit), y_fit)
    return model.predict(phi(X_eval))


def make_xgb(params: Dict):
    """Konstruktor TUNGGAL untuk XGBRegressor. Semua notebook wajib lewat sini,
    sehingga random_state / n_jobs / objective tidak mungkin lupa dipasang (C1)."""
    from xgboost import XGBRegressor

    defaults = dict(
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    defaults.update(params or {})
    return XGBRegressor(**defaults)


def fp_xgboost(X_fit, y_fit, X_eval, params):
    model = make_xgb(params)
    model.fit(X_fit, y_fit)
    return model.predict(X_eval)


def fp_lr_xgb_average(X_fit, y_fit, X_eval, params):
    """Baseline hibrida (Ramadhan et al.): rata-rata sederhana LR dan XGBoost."""
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression().fit(X_fit, y_fit)
    xgb_model = make_xgb(params).fit(X_fit, y_fit)
    return (lr.predict(X_eval) + xgb_model.predict(X_eval)) / 2.0


def fp_lr_xgb_residual(X_fit, y_fit, X_eval, params):
    """METODE USULAN: LR memodelkan tren, XGBoost memodelkan residual LR."""
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression().fit(X_fit, y_fit)
    residual = y_fit - lr.predict(X_fit)
    xgb_model = make_xgb(params).fit(X_fit, residual)
    return lr.predict(X_eval) + xgb_model.predict(X_eval)


def fp_arima(X_fit, y_fit, X_eval, params):
    """ARIMA univariat; mengabaikan X (dilaporkan sekali per kategori)."""
    from statsmodels.tsa.arima.model import ARIMA

    order = tuple(params.get("order", (5, 1, 0)))
    fitted = ARIMA(np.asarray(y_fit, dtype=float), order=order).fit()
    return np.asarray(fitted.forecast(steps=len(X_eval)))


# ---- baseline naif (wajib ada di paper robustness) -------------------------

def naive_rows(dataset: Dataset) -> List[Dict]:
    """Naive (y_{t-1}) dan Seasonal Naive (y_{t-s}) dihitung dari deret mentah.

    Tidak punya hyperparameter, jadi tidak melalui tuning; tetap dievaluasi pada
    split validation dan test yang sama persis.
    """
    y = dataset.y
    s = dataset.seasonal_period
    rows = []
    for label, shift in (("Naive", 1), (f"SeasonalNaive(s={s})", s)):
        pred = np.concatenate([np.full(shift, np.nan), y[:-shift]])
        va = slice(dataset.i_train_end, dataset.i_val_end)
        te = slice(dataset.i_val_end, len(y))
        val_pred = np.nan_to_num(pred[va], nan=float(np.mean(y[: dataset.i_train_end])))
        test_pred = np.nan_to_num(pred[te], nan=float(np.mean(y[: dataset.i_val_end])))
        rows.append({
            "model": label,
            **dataset.describe(),
            "params": "{}",
            "n_grid": 0,
            "scaler": "none",
            "seed": SEED,
            **compute_metrics(dataset.y_val, val_pred, prefix="val_"),
            **compute_metrics(dataset.y_test, test_pred, prefix="test_"),
            "runtime_s": 0.0,
            "_val_pred": val_pred,
            "_test_pred": test_pred,
        })
    return rows


# --------------------------------------------------------------------------- #
# Grid standar (dipakai bersama oleh semua notebook)
# --------------------------------------------------------------------------- #

# Grid sigma sengaja diperlebar sampai 50: pada grid lama (maksimum 5.0) optimum
# validation jatuh tepat di batas atas untuk hampir semua kategori, artinya grid
# lama memotong ruang pencarian dan merugikan baseline berbasis kernel.
GRID_GRNN = {"sigma": [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                       5.0, 7.5, 10.0, 15.0, 25.0, 50.0]}
GRID_PNN = {"sigma": [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                      5.0, 7.5, 10.0, 15.0, 25.0, 50.0]}
GRID_RBFNN = {"n_centers": [5, 10, 20, 40], "gamma": [0.01, 0.05, 0.1, 0.5, 1.0],
              "alpha": [0.01, 0.1, 1.0]}
GRID_XGB_PHARMA = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}
GRID_XGB_ROSSMANN = {
    "n_estimators": [300, 600, 900],
    "max_depth": [6, 9],
    "learning_rate": [0.1, 0.2],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "max_bin": [256],
}


# --------------------------------------------------------------------------- #
# C7 - penulisan hasil
# --------------------------------------------------------------------------- #

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def environment_stamp() -> Dict[str, str]:
    stamp = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seed": str(SEED),
    }
    for mod in ("sklearn", "xgboost", "statsmodels"):
        try:
            stamp[mod] = __import__(mod).__version__
        except Exception:
            stamp[mod] = "unavailable"
    return stamp


def save_results(rows: Iterable[Dict], name: str,
                 results_dir: Optional[Path] = None) -> pd.DataFrame:
    """Tulis results/<name>.csv dan results/<name>.meta.json."""
    results_dir = Path(results_dir or RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in rows])
    df.to_csv(results_dir / f"{name}.csv", index=False)
    (results_dir / f"{name}.meta.json").write_text(
        json.dumps({"experiment": name,
                    "n_rows": len(df),
                    "protocol": {"split_ratios": list(SPLIT_RATIOS),
                                 "tuning": "validation-only, refit on train+val",
                                 "feature_sets": list(FEATURE_SETS)},
                    "environment": environment_stamp()},
                   indent=2), encoding="utf-8")
    return df


# --------------------------------------------------------------------------- #
# Rossmann: rekayasa fitur + ablasi kebocoran `Customers`
# --------------------------------------------------------------------------- #

# `Customers` TIDAK tersedia di test.csv kompetisi Rossmann (kolomnya hanya
# Id, Store, DayOfWeek, Date, Open, Promo, StateHoliday, SchoolHoliday).
# Memakainya sebagai prediktor Sales berarti mengetahui berapa orang yang akan
# datang ke toko pada hari yang diramalkan -- bukan forecasting yang sah.
CUSTOMER_VARIANTS = (
    "V0_customers_contemporaneous",  # legacy notebook (BOCOR - hanya sebagai acuan)
    "V1_customers_dropped",          # Customers dibuang sepenuhnya
    "V2_customers_lagged",           # diganti riwayat Customers per toko (lag 1 + rolling 7)
    "V3_sales_lagged",               # V1 + riwayat Sales per toko (lag 1 + rolling 7)
)


@dataclass
class PanelDataset:
    """Dataset panel (banyak toko) dengan split berbasis TANGGAL."""
    name: str
    feature_set: str
    feature_names: List[str]
    frame: pd.DataFrame
    i_train_end: int
    i_val_end: int
    target_col: str
    date_col: str = "Date"
    lag_rule: str = "n/a"
    n_lags: int = 0

    @property
    def X(self): return self.frame[self.feature_names].to_numpy(dtype=float)
    @property
    def y(self): return self.frame[self.target_col].to_numpy(dtype=float)
    @property
    def X_train(self): return self.X[: self.i_train_end]
    @property
    def y_train(self): return self.y[: self.i_train_end]
    @property
    def X_val(self): return self.X[self.i_train_end : self.i_val_end]
    @property
    def y_val(self): return self.y[self.i_train_end : self.i_val_end]
    @property
    def X_test(self): return self.X[self.i_val_end :]
    @property
    def y_test(self): return self.y[self.i_val_end :]
    @property
    def X_trainval(self): return self.X[: self.i_val_end]
    @property
    def y_trainval(self): return self.y[: self.i_val_end]

    def describe(self) -> Dict[str, object]:
        d = self.frame[self.date_col]
        return {
            "category": self.name,
            "feature_set": self.feature_set,
            "lag_rule": self.lag_rule,
            "n_features": len(self.feature_names),
            "n_lags": self.n_lags,
            "n_train": int(self.i_train_end),
            "n_val": int(self.i_val_end - self.i_train_end),
            "n_test": int(len(self.frame) - self.i_val_end),
            "train_start": str(d.iloc[0].date()),
            "train_end": str(d.iloc[self.i_train_end - 1].date()),
            "val_start": str(d.iloc[self.i_train_end].date()),
            "val_end": str(d.iloc[self.i_val_end - 1].date()),
            "test_start": str(d.iloc[self.i_val_end].date()),
            "test_end": str(d.iloc[-1].date()),
        }


def build_rossmann_frame(train_csv, store_csv) -> pd.DataFrame:
    """Rekayasa fitur Rossmann, mereplikasi notebook asal ditambah fitur riwayat.

    Perbedaan yang disengaja terhadap notebook asal:
      * lag/rolling per toko untuk Customers DAN Sales dihitung SEBELUM
        penyaringan Open/Sales>0, sehingga kalendernya benar;
      * baris NaN akibat lag dibuang SATU KALI, sehingga semua varian ablasi
        berbagi himpunan baris yang persis sama (syarat ablasi yang valid).
    """
    train = pd.read_csv(train_csv, low_memory=False)
    store = pd.read_csv(store_csv, low_memory=False)
    df = train.merge(store, how="left", on="Store")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # --- riwayat per toko (semuanya di-shift: hanya memakai masa lalu) -----
    g = df.groupby("Store", sort=False)
    df["Customers_lag1"] = g["Customers"].shift(1)
    df["Customers_roll7"] = g["Customers"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    df["Sales_lag1"] = g["Sales"].shift(1)
    df["Sales_roll7"] = g["Sales"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)

    # --- fitur promo (identik dengan notebook asal) ------------------------
    df["MonthStr"] = df["Date"].dt.strftime("%b")
    interval = df["PromoInterval"].fillna("")
    in_interval = np.array([m in i.split(",") if i else False
                            for m, i in zip(df["MonthStr"], interval)])
    df["IsPromo2Active"] = ((df["Promo2"] == 1).to_numpy() & in_interval).astype(int)
    df["Promo2DurationWeeks"] = np.where(
        df["Promo2"] == 1,
        (df["Date"].dt.year - df["Promo2SinceYear"]) * 52
        + (df["Date"].dt.isocalendar().week - df["Promo2SinceWeek"]),
        0,
    )
    df["Promo2DurationWeeks"] = df["Promo2DurationWeeks"].clip(lower=0).fillna(0)

    df = pd.get_dummies(df, columns=["StoreType", "Assortment", "StateHoliday"],
                        drop_first=True)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    df = df.drop(columns=["CompetitionDistance", "CompetitionOpenSinceMonth",
                          "CompetitionOpenSinceYear", "Promo2SinceWeek",
                          "Promo2SinceYear", "PromoInterval", "MonthStr"],
                 errors="ignore")

    # Sasaran pemodelan: hari toko buka dengan penjualan positif (sama seperti asal)
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
    df = df.dropna().reset_index(drop=True)
    return df.sort_values(["Date", "Store"]).reset_index(drop=True)


def build_rossmann_dataset(frame: pd.DataFrame,
                           variant: str,
                           target: str = "log1p",
                           ratios: Sequence[float] = SPLIT_RATIOS) -> PanelDataset:
    """Bangun satu varian ablasi. Semua varian memakai baris & split identik."""
    if variant not in CUSTOMER_VARIANTS:
        raise ValueError(f"varian tidak dikenal: {variant}")

    df = frame.copy()
    df["Sales_log"] = np.log1p(df["Sales"])
    target_col = "Sales_log" if target == "log1p" else "Sales"

    history = ["Customers_lag1", "Customers_roll7", "Sales_lag1", "Sales_roll7"]
    base = [c for c in df.columns
            if c not in (["Sales", "Sales_log", "Date", "Customers"] + history)]

    if variant == "V0_customers_contemporaneous":
        features = base + ["Customers"]
    elif variant == "V1_customers_dropped":
        features = base
    elif variant == "V2_customers_lagged":
        features = base + ["Customers_lag1", "Customers_roll7"]
    else:  # V3_sales_lagged
        features = base + ["Sales_lag1", "Sales_roll7"]

    # split berbasis TANGGAL, bukan indeks baris: batas split tidak boleh
    # memotong satu tanggal menjadi dua blok (cacat pada notebook asal).
    dates = np.sort(df["Date"].unique())
    i_tr, i_va = chrono_split_bounds(len(dates), ratios)
    d_tr_end, d_va_end = dates[i_tr], dates[i_va]
    i_train_end = int((df["Date"] < d_tr_end).sum())
    i_val_end = int((df["Date"] < d_va_end).sum())

    # buang kolom konstan (mis. Open == 1 setelah penyaringan) agar tidak
    # menyesatkan analisis feature importance
    features = [c for c in sorted(features) if df[c].nunique(dropna=False) > 1]

    return PanelDataset(name="rossmann", feature_set=variant,
                        feature_names=features, frame=df,
                        i_train_end=i_train_end, i_val_end=i_val_end,
                        target_col=target_col)


def rossmann_seasonal_naive(dataset: PanelDataset,
                            inverse_transform: Optional[Callable] = None) -> Dict:
    """Baseline: median Sales historis per (Store, DayOfWeek, Promo) dari blok
    training aktif. Baseline wajib untuk klaim keunggulan pada data ritel."""
    df = dataset.frame
    key = ["Store", "DayOfWeek", "Promo"]
    y = dataset.y

    def predict(fit_slice, eval_slice):
        table = df.iloc[fit_slice].assign(_y=y[fit_slice]).groupby(key)["_y"].median()
        idx = pd.MultiIndex.from_frame(df.iloc[eval_slice][key])
        pred = table.reindex(idx).to_numpy()
        return np.nan_to_num(pred, nan=float(np.median(y[fit_slice])))

    val_pred = predict(slice(0, dataset.i_train_end),
                       slice(dataset.i_train_end, dataset.i_val_end))
    test_pred = predict(slice(0, dataset.i_val_end),
                        slice(dataset.i_val_end, len(df)))
    row = {
        "model": "SeasonalNaive(store x dow x promo median)",
        **dataset.describe(),
        "params": "{}", "n_grid": 0, "scaler": "none", "seed": SEED,
        **compute_metrics(dataset.y_val, val_pred, prefix="val_"),
        **compute_metrics(dataset.y_test, test_pred, prefix="test_"),
        "runtime_s": 0.0,
    }
    if inverse_transform is not None:
        row.update(compute_metrics(inverse_transform(dataset.y_test),
                                   inverse_transform(test_pred), prefix="orig_"))
    row["_val_pred"] = val_pred
    row["_test_pred"] = test_pred
    return row


# --------------------------------------------------------------------------- #
# Uji signifikansi: Diebold-Mariano
# --------------------------------------------------------------------------- #

def diebold_mariano(y_true, pred_a, pred_b, h: int = 1, power: int = 2) -> Dict[str, float]:
    """Uji Diebold-Mariano untuk akurasi prediksi yang sama (H0: d_bar = 0).

    Statistik memakai koreksi ukuran sampel kecil Harvey-Leybourne-Newbold (1997)
    dan distribusi t dengan n-1 derajat kebebasan.

    Nilai negatif berarti `pred_a` lebih akurat daripada `pred_b`.
    Mengembalikan {'DM', 'p_value', 'n'}; p_value dua sisi.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    e_a = y_true - np.asarray(pred_a, dtype=float).ravel()
    e_b = y_true - np.asarray(pred_b, dtype=float).ravel()
    d = np.abs(e_a) ** power - np.abs(e_b) ** power
    n = len(d)
    d_bar = float(np.mean(d))
    if n < 3 or np.allclose(d, 0):
        return {"DM": float("nan"), "p_value": float("nan"), "n": n}

    def gamma(k):
        return float(np.sum((d[k:] - d_bar) * (d[: n - k] - d_bar)) / n)

    var_d = gamma(0) + 2 * sum(gamma(k) for k in range(1, h))
    if var_d <= 0:
        return {"DM": float("nan"), "p_value": float("nan"), "n": n}
    dm = d_bar / np.sqrt(var_d / n)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction
    try:
        from scipy import stats

        p = float(2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1)))
    except Exception:
        p = float("nan")
    return {"DM": float(dm_hln), "p_value": p, "n": n}
