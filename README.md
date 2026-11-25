
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

## 9. Reproducing Lab 1–4 data outputs

Raw and processed datasets are intentionally excluded from Git. After cloning,
download the 27 source CSV files into `data/raw/`, create the Python environment,
and run the notebooks in order:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install jupyter
code .
```

In VS Code, select the `.venv` kernel and run:

1. `data/notebooks/00_data_collection.ipynb`
2. `data/notebooks/01_data_understanding.ipynb`
3. `data/notebooks/02_data_quality_assessment.ipynb`
4. `data/notebooks/03_data_preprocessing.ipynb`

This recreates:

- `reports/data_inventory.xlsx`
- `reports/relationship_candidates.xlsx`
- `reports/data_understanding_report.xlsx`
- `reports/data_quality_report.xlsx`
- `reports/preprocessing_report.xlsx`
- `plots/*.png`
- `data/processed/*_cleaned.csv`
- `data/processed/joined_sales_data.csv`
- duplicate and negative-quantity quarantine folders

## 10. Controlled preprocessing

The repository also provides a reproducible preprocessing entry point that
reports missing values, exact duplicates, IQR outliers and join-key conflicts
without silently imputing or deleting uncertain records:

```powershell
python src/data/preprocess_lab.py --raw-dir data/raw --processed-dir data/processed --report-dir reports/tables
```

Outputs include `missing_values_report.csv`, `inconsistency_report.csv`,
`join_log.csv`, cleaned per-table CSVs and quarantine folders.

Current invariant results:

- 556,094 valid completed sales lines.
- 23,478 canceled-order lines excluded and listed in `excluded_sales_lines.csv`.
- 845 duplicate order rows removed before cleaning.
- No surviving non-completed status, nonpositive Quantity or duplicate line ID.

## 11. Recommended rerun order after cloning

The repository does not track raw or processed datasets. A new clone should
restore/download `data/raw/27-CSV-snapshot` first.

Minimal order:

1. Install dependencies and restore 27 raw CSV files.
2. Run the 4 Lab notebooks only when reports, plots or notebook outputs are needed.
3. Run `src/data/audit_sources.py` to confirm source structure and relationships.
4. Run `src/data/clean_transactions.py` to create the clean Silver transaction file.
5. Run `src/data/preprocess_lab.py` for the notebook-compatible inventory, quarantine and join reports.
6. Run `src/data/sku_lifecycle.py` and `src/data/build_ground_truth.py` for the SKU-week Gold data.
7. Run quality, EDA and validation scripts for the final evidence package.
8. Run the Random Forest pipeline; it consumes the raw contract, not `joined_sales_data.csv`.

Recommended files to preserve locally or on shared storage:

- `data/raw/` — original snapshot.
- `data/processed/clean_sales_transactions.csv` — reusable Silver transaction data.
- `data/processed/sku_week_ground_truth_provisional.csv` — reusable Gold ground truth.
- `reports/tables/` — validation, quality and provenance reports.
- `reports/random_forest/` — model experiments and metrics.

No need to rerun notebook 01–02 or regenerate every table if the preserved
checkpoints above are still valid and their source snapshot is unchanged.

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
