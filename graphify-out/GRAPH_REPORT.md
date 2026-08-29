# Graph Report - forecasting-medicine-public  (2026-08-26)

## Corpus Check
- 30 files · ~439,120 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 228 nodes · 265 edges · 31 communities (20 shown, 11 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 45 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- PharmaSales Daily Pipeline & Baselines
- Rossmann Experiment Architecture
- PharmaSales Architecture & Preprocessing
- GRNN Model Implementation
- GRNN Article Variant
- XGBoost Feature Importance
- PharmaSales Daily RMSE Results
- PharmaSales Weekly MSE Results
- Pharma Weekly Residual Distribution
- Rossmann Residual Distribution
- Rossmann Log-Transformed RMSE
- Study 6 Box-Cox Forecast
- Study 6 No-Transform Baseline
- Daily Residual Distribution
- Study 6 Box-Cox Inverse Forecast
- Study 6 Log-Transform Forecast
- Rossmann Original-Scale RMSE
- Study 6 Log-Inverse Forecast
- Model & Feature Persistence
- GEO Optimizer
- Weekly Guide: ACF & Baseline
- External Data
- Guide: Early Stopping
- Guide: Expanded Regularization
- Guide: Pseudo-Huber Loss
- Guide: Winsorize Robustness
- Guide: Poisson Transform
- Guide: Tweedie Transform
- Project Root

## God Nodes (most connected - your core abstractions)
1. `LR-XGBoost (Residual)` - 18 edges
2. `GRNN` - 12 edges
3. `GrnnArticel` - 11 edges
4. `Top 10 Feature Importance - XGBoost Residual Model (Study 6)` - 10 edges
5. `PharmaSales Daily Forecasting Performance (Top 5 RMSE per Category)` - 9 edges
6. `PharmaSales Weekly Forecasting Performance (Top 5 MSE per Category, Log Scale)` - 7 edges
7. `Pharma Weekly Proposed Residual Distribution (Grid Search KDE, 8 drug categories)` - 7 edges
8. `Rossmann Proposed Residual Distribution Chart` - 7 edges
9. `Rossmann Log-Transformed Forecasting Performance (RMSE)` - 7 edges
10. `Target Inverse Log-Transformation (exp y)` - 6 edges

## Surprising Connections (you probably didn't know these)
- `04_transform_log1p` --semantically_similar_to--> `log1p/expm1 Target Transformation`  [INFERRED] [semantically similar]
  notebooks/PHARMA_WEEKLY_ENHANCEMENT_GUIDE.md → README.md
- `Four Optimizers (Grid Search, Optuna, PSO, GEO)` --semantically_similar_to--> `XGBoost Hyperparameter Tuning (Grid Search, Optuna, PSO, GEO)`  [INFERRED] [semantically similar]
  notebooks/PHARMA_WEEKLY_ENHANCEMENT_GUIDE.md → README.md
- `Raw Data` --shares_data_with--> `PharmaSales Dataset`  [INFERRED]
  data/raw/README.md → README.md
- `Notebook Visualization Plan` --references--> `Our Study PharmaSales Daily Notebook`  [INFERRED]
  VISUALIZATION_PLAN.md → README.md
- `Pharma Weekly Enhancement Guide` --references--> `Our Study PharmaSales Weekly Notebook`  [INFERRED]
  notebooks/PHARMA_WEEKLY_ENHANCEMENT_GUIDE.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **LR-XGBoost Residual Forecasting Pipeline** — readme_lr_xgboost_residual, readme_linear_regression, readme_xgboost, readme_residual_correction, readme_feature_engineering [EXTRACTED 0.90]
- **Pharma Weekly Enhancement Experiment Set** — notebooks_pharma_weekly_enhancement_guide_baseline, notebooks_pharma_weekly_enhancement_guide_acf_seasonal, notebooks_pharma_weekly_enhancement_guide_expanded_regularization, notebooks_pharma_weekly_enhancement_guide_early_stopping, notebooks_pharma_weekly_enhancement_guide_summary [EXTRACTED 0.85]

## Communities (31 total, 11 thin omitted)

### Community 0 - "PharmaSales Daily Pipeline & Baselines"
Cohesion: 0.07
Nodes (33): Immutable Raw Data Policy, src/data/preprocess.py, Raw Data, Two Hybrid Schemes (Averaging & Residual), Four Optimizers (Grid Search, Optuna, PSO, GEO), Pharma Weekly Enhancement Guide, 04_transform_log1p, ATC Sales Categories (+25 more)

### Community 1 - "Rossmann Experiment Architecture"
Cohesion: 0.09
Nodes (26): Rossmann Experiment Architecture Diagram, Autocorrelation Analysis, Categorical One-Hot Encoding, Data Preprocessing & Exploratory Analysis, Domain-Specific Feature Construction, Evaluation, Exploratory Data Analysis (EDA), Feature Engineering (+18 more)

### Community 2 - "PharmaSales Architecture & Preprocessing"
Cohesion: 0.15
Nodes (21): Adoption of the ATC Classification System, Analysis Autocorrelation, Analysis seasonality, Cleaning Data, Collecting Data, Create Lags, Create Rolling Mean, PharmaSales Experiment Architecture Diagram (+13 more)

### Community 5 - "XGBoost Feature Importance"
Cohesion: 0.18
Nodes (11): Assortment_b (importance ~0.063), Assortment_c (importance ~0.041), Top 10 Feature Importance - XGBoost Residual Model (Study 6), CompetitionDistance (importance ~0.067), CompetitionOpenSinceYear (importance ~0.061), Customers (importance ~0.086), Promo2 (importance ~0.105), Store (importance ~0.062) (+3 more)

### Community 6 - "PharmaSales Daily RMSE Results"
Cohesion: 0.24
Nodes (10): ARIMA, ATC Category, PharmaSales Daily Forecasting Performance (Top 5 RMSE per Category), GR_NN, LR+XGB baseline, P_NN, PharmaSales Daily Sales Data, Proposed (+2 more)

### Community 7 - "PharmaSales Weekly MSE Results"
Cohesion: 0.32
Nodes (8): ATC Category (M01AB, M01AE, N02BA, N02BE, N05B, N05C, R03, R06), Baseline Forecasting Models (Seasonal Naive, ARIMA, ARIMA Rolling, Auto-ARIMA, Single/Double/Triple Exp. Smoothing, FB Prophet, XGBoost, Stacked LSTM), PharmaSales Weekly Forecasting Performance (Top 5 MSE per Category, Log Scale), Proposed model achieves competitive low MSE across categories; MSE varies widely by ATC category (N05C lowest ~7-8, N02BE/R03 highest ~10^3), MSE (Mean Squared Error), PharmaSales Weekly Dataset, Proposed Model, Weekly Sales Forecasting

### Community 8 - "Pharma Weekly Residual Distribution"
Cohesion: 0.36
Nodes (8): Drug categories: M01AB, M01AE, N02BA, N02BE, N05B, N05C, R03, R06, Pharma Weekly Proposed Residual Distribution (Grid Search KDE, 8 drug categories), Residuals centered near zero; LR and LR-XGBoost distributions largely overlap, RMSE differences small, Residual density (KDE) per drug category, RMSE per model (LR vs LR-XGBoost), Linear Regression (LR) residual model, LR-XGBoost hybrid residual model, Grid Search weekly pharma forecast (proposed model)

### Community 9 - "Rossmann Residual Distribution"
Cohesion: 0.36
Nodes (8): Rossmann Proposed Residual Distribution Chart, Insight: LR-XGBoost residuals tighter around zero, lower RMSE, Original vs Log Scale Comparison, Linear Regression (LR) Residual, LR-XGBoost Residual, Residual Distribution (KDE Density), RMSE Metric (LR=3097.23/0.2084, LR-XGBoost=577.63/0.0702), Rossmann Dataset (Proposed Model)

### Community 10 - "Rossmann Log-Transformed RMSE"
Cohesion: 0.32
Nodes (8): ARIMA (RMSE 1.3094), Rossmann Log-Transformed Forecasting Performance (RMSE), Rossmann log-transformed dataset, Proposed model achieves lowest RMSE, outperforming XGBoost, Prophet, and ARIMA, RMSE metric, Prophet (RMSE 0.5874), Proposed model (RMSE 0.0708), XGBoost (RMSE 0.4437)

### Community 11 - "Study 6 Box-Cox Forecast"
Cohesion: 0.32
Nodes (8): Actual vs Predicted Sales Time Series (2015-05 to 2015-08), Box-Cox Transformation, Study 6 Box-Cox Hybrid Model (LR+XGBoost) Last 90 Days Result Figure, Hybrid Model (Linear Regression + XGBoost), Linear Regression, RMSE 0.166 (90-Day Test), Sales Forecasting, XGBoost

### Community 12 - "Study 6 No-Transform Baseline"
Cohesion: 0.32
Nodes (8): Study 6 Result Figure: Hybrid Model (LR + XGBoost) Last 90 Days, No Log Transform (Baseline), Hybrid Model (Linear Regression + XGBoost), Linear Regression, No Log Transform Baseline, RMSE = 661.599 (Last 90 Days), Sales Forecasting (Actual vs Predicted), Study 6 (3 Months Forecast Horizon), XGBoost

### Community 13 - "Daily Residual Distribution"
Cohesion: 0.38
Nodes (7): Daily Proposed Residual Distribution (Grid Search), Drug Categories (M01AB, M01AE, N02BA, N02BE, N05B, N05C, R03, R06), Grid Search Hyperparameter Tuning, LR Residual, LR-XGBoost Residual, LR vs LR-XGBoost near-identical residuals, minimal RMSE gain, RMSE per Drug Category

### Community 14 - "Study 6 Box-Cox Inverse Forecast"
Cohesion: 0.47
Nodes (6): Box-Cox Transform, Study 6: Hybrid Model (LR + XGBoost) Box-Cox Inverse Sales Forecast, Last 90 Days (RMSE 560.803), Sales Forecast Result (RMSE 560.803), Hybrid Model (LR + XGBoost), Inverse Transform, RMSE Evaluation Metric

### Community 15 - "Study 6 Log-Transform Forecast"
Cohesion: 0.53
Nodes (6): Actual vs Predicted Sales Time Series (2015-05-01 to 2015-08-01), Study 6 Result Figure: Hybrid Model (LR + XGBoost) Last 90 Days, Log-Transformed Sales, 90-Day (3 Months) Forecast Horizon, Hybrid Model (Linear Regression + XGBoost), Log Transform of Sales Target, RMSE 0.070 (Last 90 Days, log scale)

### Community 16 - "Rossmann Original-Scale RMSE"
Cohesion: 0.60
Nodes (5): Rossmann Original-Scale Forecasting Performance (Top 10 RMSE), Rossmann dataset (original scale), Proposed lowest RMSE; LSTM/XGBoost Baseline worst (~1150), RMSE (original scale), Proposed model (RMSE 577.63, best)

### Community 17 - "Study 6 Log-Inverse Forecast"
Cohesion: 0.60
Nodes (5): Study 6 Hybrid LR+XGBoost Log-Inverse Forecast (Last 90 Days, RMSE 577.633), Hybrid Model (Linear Regression + XGBoost), Inverse Log Transform to Sales Scale, Log Transform of Sales, RMSE = 577.633 (Last 90 Days)

### Community 18 - "Model & Feature Persistence"
Cohesion: 0.40
Nodes (4): Save a list of features used by the model. File name: features.pkl Stored in:…, Save a model to the 'models/' folder using dynamic path based on notebook…, save_features(), save_model()

### Community 20 - "Weekly Guide: ACF & Baseline"
Cohesion: 0.67
Nodes (3): 01_acf_seasonal (Fourier/Seasonal Features), 00_baseline Notebook, 09_summary

## Knowledge Gaps
- **73 isolated node(s):** `forecasting-medicine-public`, `Hybrid Residual Forecasting`, `Linear Regression`, `Rossmann Store Sales Dataset`, `ATC Sales Categories` (+68 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `forecasting-medicine-public`, `Hybrid Residual Forecasting`, `Linear Regression` to the rest of the system?**
  _73 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `PharmaSales Daily Pipeline & Baselines` be split into smaller, more focused modules?**
  _Cohesion score 0.06628787878787878 - nodes in this community are weakly interconnected._
- **Should `Rossmann Experiment Architecture` be split into smaller, more focused modules?**
  _Cohesion score 0.08923076923076922 - nodes in this community are weakly interconnected._
- **Should `PharmaSales Architecture & Preprocessing` be split into smaller, more focused modules?**
  _Cohesion score 0.14761904761904762 - nodes in this community are weakly interconnected._