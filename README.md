
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
├── models/
│   └── random forest/       # Implemented standalone quantity H=1 pipeline
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

## 8. Random Forest pipeline

The implemented RF pipeline lives in [models/random forest](<models/random forest/README.md>)
as requested for this model assignment. The broader structure above remains the project plan;
this implementation does not silently relocate or replace other models under `src/models/`.

- [Pipeline documentation: input tables, label rules, features, evaluation and outputs](<docs/random_forest_pipeline.md>)
- [Workflow diagram](<docs/figures/random_forest_workflow.svg>)
- [Local verification results and review checklist](<docs/random_forest_review.md>)

Run from the repository root (quote the path containing a space):

```bash
python "models/random forest/run.py" run \
  --config "models/random forest/configs/random_forest.json" \
  --data-dir /path/to/raw \
  --output-dir reports/random_forest/my_run
```

The target is **gross completed sales units per SKU in the next calendar week (H=1)**,
not revenue, price or unconstrained demand. Raw CSV files stay outside Git.
The local config explicitly assumes complete weekly feeds if no coverage table is supplied;
this is an unverified assumption awaiting group review, not a data-quality finding.
The implementation is prepared on `feat/random-forest-weekly-pipeline` for a pull request
into `main`. GitHub publication and the secretary's review are tracked separately;
the pipeline does not push, merge, or request reviews automatically.
