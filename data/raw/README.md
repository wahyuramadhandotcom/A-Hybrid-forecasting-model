# Raw Data

This folder stores raw data in its original form — exactly as it was received from the source.

> ⚠️ **Never modify or delete files in this folder directly.**

## Important Rules

- Files in this folder are *immutable* and must not be changed
- All preprocessing is handled by `src/data/preprocess.py`, with results saved to `data/processed/`
- If the data is updated, add a new versioned file without removing the old one

## Data Source

| Attribute | Detail |
|---|---|
| Dataset name | [fill in dataset name] |
| Source | [Kaggle / internal / API / etc.] |
| Download date | [YYYY-MM-DD] |
| Size | [e.g., 500MB, 1.2M rows] |
| License | [CC BY / MIT / proprietary / etc.] |

## File Naming Convention

```
dataset_name_YYYYMMDD.csv
dataset_name_YYYYMMDD_v2.csv   # if updated
```