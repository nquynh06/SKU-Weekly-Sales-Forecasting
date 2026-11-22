
# SKU-Weekly-Sales-Forecasting

## 1. Project Overview

This project focuses on weekly sales quantity forecasting at the SKU level, combined with data quality assessment and improvement.

The project aims to:

- Analyze and assess data quality.
- Construct SKU-week ground truth datasets.
- Engineer time-series features.
- Develop and compare forecasting models.
- Evaluate the impact of data quality on forecasting performance.

## 2. Dataset

**Source:** [Electronics Sales Dataset – B-Tech Style (Kaggle)](https://www.kaggle.com/datasets/fahdelatiby/electronics-sales-dataset-b-tech-style)

The raw dataset is stored in Google Drive and is not included in this repository.

## 3. Methods

The project uses the following forecasting methods:

- Moving Average (Baseline)
- Random Forest Regressor
- HistGradientBoosting Regressor

## 4. Project Structure

```text
sku-weekly-sales-forecasting/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
├── notebooks/
├── src/
│   ├── data/
│   ├── quality/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── utils/
│
├── tests/
├── reports/
│   ├── tables/
│   └── figures/
│
├── docs/
└── app/
```

### Directory Description

| Directory | Description |
|---|---|
| `configs/` | Configuration files for data paths, features, and models. |
| `notebooks/` | Notebooks for EDA and data processing in Google Colab. |
| `src/data/` | Data loading, joining, and preprocessing. |
| `src/quality/` | Data quality assessment and D0–D4 experiments. |
| `src/features/` | SKU-week dataset construction and feature engineering. |
| `src/models/` | Forecasting models (MA4, RF, HGB) and training scripts. |
| `src/evaluation/` | Model evaluation using MAE, RMSE, WAPE, and comparison. |
| `src/utils/` | Shared utilities and Google Drive configuration. |
| `tests/` | Unit tests and data validation tests. |
| `reports/` | Experimental results, tables, and figures. |
| `docs/` | Project documentation, workflow, and data descriptions. |
| `app/` | Dashboard or demo application. |

## 5. Technology Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Development Environment:** VS Code, Google Colab
- **Data Storage:** Google Drive
- **Version Control:** Git, GitHub

## 6. Data Storage

Google Drive is used to store:

- Raw datasets.
- Preprocessed datasets.
- Data quality experiment datasets (D0–D4).
- SKU-week feature datasets.
- Trained models and experimental results.

GitHub is used to manage source code, notebooks, configuration files, documentation, and selected experimental reports.

Large datasets and model artifacts are not stored in this repository.

## 7. Project Status

**Status:** In Progress
