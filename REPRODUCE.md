# Reproducing the paper

Every table and figure in

> *AR-LRX: Robustness of Adaptive Residual Linear Regression–XGBoost Forecasting
> Across Pharmaceutical and Retail Demand Regimes*

is produced by one of the notebooks in `notebooks/`, and every number in the paper
can be read out of a file in `results/` without retraining anything.

## 1. Run order

The experiments are independent apart from `exp06b`, which supersedes `exp06`.
Runtimes are from the authors' machine (Windows, Python 3.10.11, CPU only).

| # | Notebook | What it establishes | Runtime |
|---|---|---|---|
| 1 | `exp01_pharma_daily_unified.ipynb` | Pharmaceutical daily series under the experimental contract | ~10 min |
| 2 | `exp02_pharma_weekly_unified.ipynb` | Pharmaceutical weekly series under the same contract | ~5 min |
| 3 | `exp03_rossmann_leakage_ablation.ipynb` | Why the `Customers` column cannot be used contemporaneously; defines `V1`/`V2`/`V3` | ~25 min |
| 4 | `exp04_paper_tables.ipynb` | Assembles the tables of the earlier study | ~1 min |
| 5 | `exp05a_rossmann_arlrx.ipynb` | AR-LRX on the retail panel, three first stages × three feature sets | 1.4 h |
| 6 | `exp05b_rossmann_arlrx_audit.ipynb` | Reproduction check, DM on both scales, full ablation, gate curve, segments | 1.4 h |
| 7 | `exp05d_rossmann_arlrx_dev.ipynb` | Augmented and segmented gate variants | 2.1 h |
| 8 | `exp06_rossmann_baselines_unified.ipynb` | First pass at the eight retrained baselines | 6.5 h |
| 9 | `exp06b_rossmann_baselines_strong.ipynb` | Baselines retrained **correctly** — this is the pass reported in the paper | 19.1 h |
| 10 | `exp05c_pharma_arlrx.ipynb` | AR-LRX on the 32 pharmaceutical configurations | ~20 min |
| 11 | `figures_paper.ipynb` | Figures 2–5, read from `results/` only | ~5 s |

`notebooks/` also holds the notebooks of the earlier study, whose file names carry the
name of the work each one reimplements. They are kept for provenance and are not part
of the run order above.

`exp06` is kept deliberately. Section 3.3 of the paper reports how much the neural
baselines improved once their training was corrected, and that claim is only checkable
if the weaker pass is also in the repository.

## 2. Where each paper artefact comes from

| Paper artefact | Notebook | Result file |
|---|---|---|
| Table 1 — datasets and splits | `exp05b`, `exp05c` | `*.meta.json` |
| Table 2 — component ablation | `exp05b`, `exp05d` | `exp05b_..._ablation_full.csv`, `exp05d_rossmann_arlrx_dev.csv` |
| Table 3 — all models, unified protocol | `exp06b`, `exp05d` | `exp06b_..._table_utama.csv` |
| Table 4 — Diebold–Mariano tests | `exp06b` | `exp06b_..._dm.csv` |
| Table 5 — effect of correcting baseline training | `exp06` → `exp06b` | `exp06b_..._exp06_vs_exp06b.csv` |
| Table 6 — test RMSE by segment | `exp05d` | `exp05d_..._segments.csv` |
| Table 7 — selected gate weight | `exp05c` | `exp05c_pharma_arlrx_gate.csv` |
| Table 8 — degradation vs the first stage | `exp05c` | `exp05c_..._comparison.csv`, `exp05c_..._gate.csv` |
| Table 9 — accuracy vs four references | `exp05c` | `exp05c_..._comparison.csv`, `exp05c_..._dm.csv` |
| Figure 1 — framework diagram | drawn by hand | `paper/figures/fig1_framework.png` |
| Figure 2 — gate sensitivity curve | `figures_paper` | `exp05b_..._gate_curve.csv` |
| Figure 3 — regime diagnostic | `figures_paper` | `exp05b_..._audit.csv`, `exp05d_..._dev.csv`, `exp05c_..._gate.csv` |
| Figure 4 — degradation distribution | `figures_paper` | `exp05c_..._comparison.csv`, `exp05c_..._gate.csv` |
| Figure 5 — validation indicator across regimes | `figures_paper` | `exp05b_..._audit.csv`, `exp05c_..._gate.csv` |
| Section 5.3 — comparison with published values | (no run) | `results/published_baseline_values.csv` |

## 3. The experimental contract

Enforced by `src/experiments/protocol.py` and obeyed by every model, including the
retrained baselines.

| | Rule |
|---|---|
| C1 | Chronological 70/15/15 split; no observation from a later block precedes one from an earlier block |
| C2 | Hyperparameters selected by minimising **validation** RMSE; the test block is never consulted during selection |
| C3 | The selected configuration is refitted on training + validation before the test block is predicted once |
| C4 | Seed 42 everywhere; tree-based results verified bit-identical across repeated runs |
| C5 | Autoregressive lag count chosen from the PACF of the **training block alone** |
| C6 | Scalers, group means and every other fitted statistic estimated on the block currently being trained on |
| C7 | Every experiment writes a result file with all metrics, selected hyperparameters, split boundaries and an environment stamp |

## 4. Column glossary

The result files are written by the notebooks in the authors' working language; the
column names are left exactly as the code emits them so that the files and the code
cannot drift apart.

| Column | Meaning |
|---|---|
| `varian` | Rossmann feature configuration (`V1_customers_dropped`, `V2_customers_lagged`, `V3_sales_lagged`) |
| `stage1` / `S1` | First stage: `linear`, `structural`, or `struct_linear` |
| `gate_w`, `w*` | Gate weight selected on the validation fold |
| `resid_val_r2` | R² of the second stage on the **validation** residuals — the regime diagnostic |
| `gate_val_gain_pct` | RMSE improvement attributable to the gate, measured on validation |
| `stage1_only_test_RMSE` | Test RMSE of the first stage alone |
| `orig_RMSE`, `RMSE asli` | RMSE on the original sales scale (after inverse transformation) |
| `test_RMSE` | RMSE on the modelling scale — log(1+y) for Rossmann |
| `pembanding` | Reference model in a Diebold–Mariano comparison |
| `DM (log)` / `DM (asli)` | DM statistic on the log and the original scale; negative favours AR-LRX |
| `sepakat` | Whether the two scales agree in sign |
| `segmen` | Segment of the test block (store quartile, promotion status, day of week) |
| `kerangka lama` | The ungated residual hybrid of the earlier study |
| `sumbangan gerbang (%)` | Contribution of the gate, in per cent of RMSE |

## 5. What `results/` carries

Every file the paper was written from, including the `.meta.json` files that hold the
environment stamp and the split boundaries promised by the availability statement.
No number in the paper requires a notebook to be re-run.

## 6. Data

Both panels are public. The copies in `data/raw/` are as downloaded, unmodified, and
are included here so that a run does not depend on the sources remaining reachable.

| Panel | Source | Files used | Path expected by the notebooks |
|---|---|---|---|
| Rossmann retail | Kaggle competition *Rossmann Store Sales* (`kaggle.com/c/rossmann-store-sales`) | `train.csv`, `test.csv`, `store.csv` | `data/raw/rossmann/` |
| PharmaSales | Kaggle dataset *Pharma sales data* by M. Zdravkovic (`kaggle.com/datasets/milanzdravkovic/pharma-sales-data`) | `salesdaily.csv`, `salesweekly.csv` | `data/raw/pharma-sales/` |

The Rossmann competition requires a Kaggle account and acceptance of the competition
rules before the files can be downloaded.

## 7. Environment

```
Python 3.10.11 (Windows, CPU only)
numpy 2.2.6      pandas 2.3.3       scikit-learn 1.7.2
xgboost 3.2.0    lightgbm 4.7.0     statsmodels 0.14.6
keras 3.12.4     tensorflow 2.21.0
```

Tree-based components — AR-LRX included — reproduce bit for bit across runs, selected
hyperparameters and gate weights included. The neural baselines do not, because TensorFlow's
multithreaded CPU kernels are not deterministic; this is stated as a limitation in the paper.
