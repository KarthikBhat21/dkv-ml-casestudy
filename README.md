# Credit Card Default Prediction — Azure ML Pipeline

## Overview
End-to-end Azure ML pipeline predicting credit card payment default
using the UCI Credit Card Default dataset (30,000 records, binary classification).

## Implementation

### Pipeline Steps
1. **Validate** — Check raw dataset column names against schema.yaml. Pipeline stops immediately if validation fails.
2. **Preprocess** — Clean data, fix undocumented categories, engineer 5 features, scale, split 80/20
3. **Train** — Compare 7 classifiers (CV ROC-AUC), auto-select best, tune with RandomizedSearchCV
4. **Evaluate** — Log accuracy, ROC-AUC, F1, precision, recall + confusion matrix

### Models Compared
Logistic Regression, Decision Tree, Random Forest, KNN, AdaBoost, Gradient Boosting, XGBoost

### Best Model
Gradient Boosting — selected automatically by cross-validated ROC-AUC, tuned with RandomizedSearchCV

### Key Metrics (Test Set)
| Metric | Score |
|--------|-------|
| Accuracy | 0.82 |
| ROC-AUC | 0.78 |
| F1 Score | 0.45 |
| Precision | 0.67 |
| Recall | 0.34 |

## Azure ML Setup
- **Workspace:** dkv-ml-workspace (West Europe)
- **Compute:** dkv-cpu-cluster (Standard_DS11_v2)
- **Tracking:** MLflow (built-in Azure ML integration)
- **Registry:** Model auto-registered on pipeline completion

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Local test — run in order
python src/validate.py
python src/preprocess.py
python src/train.py
python src/evaluate.py

# Submit to Azure ML (runs all 4 steps on cloud)
python pipeline/run_pipeline.py
```

## Project Structure
├── data/
│   └── credit_card_default_raw.xls  # Raw dataset (not committed to GitHub)
├── src/
│   ├── validate.py          # Step 1 — Column validation against schema.yaml
│   ├── preprocess.py        # Step 2 — Data cleaning, feature engineering, train/test split
│   ├── train.py             # Step 3 — Multi-model training with hyperparameter tuning
│   └── evaluate.py          # Step 4 — Model evaluation and metric logging
├── pipeline/
│   └── run_pipeline.py      # Azure ML SDK v2 pipeline definition and submission
├── environments/
│   └── conda.yaml           # Conda environment definition
├── notebooks/
│   └── eda.ipynb            # Exploratory Data Analysis
├── screenshots/             # Azure ML pipeline run and model registry screenshots
├── schema.yaml              # Expected column names for data validation
├── params.yaml              # Hyperparameter search space for RandomizedSearchCV
├── .amlignore               # Files excluded from Azure ML upload
└── README.md

## Notes
- Data validation step stops the pipeline immediately if column names don't match schema.yaml
- Class imbalance handled via `scale_pos_weight` (XGBoost) and `class_weight='balanced'` (sklearn models)
- Hyperparameter search space defined in `params.yaml` — no hardcoded values in code
- `StandardScaler` fitted on training data only — no data leakage
- 7 models compared by 5-fold cross-validated ROC-AUC — best model selected automatically
- RandomizedSearchCV (n_iter=20, cv=5) applied to winning model only — faster than tuning all models
- Model metadata (winner, CV scores, best params) saved as `model_metadata.json`
- Model auto-registered in Azure ML Model Registry on pipeline completion

## Screenshots
See `screenshots/` folder for:
- Successful Azure ML pipeline run (all 4 steps green)
- Registered model in Azure ML workspace