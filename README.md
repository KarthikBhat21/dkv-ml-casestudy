# Credit Card Default Prediction — Azure ML Pipeline

## Overview
End-to-end Azure ML pipeline predicting credit card payment default
using the UCI Credit Card Default dataset (30,000 records, binary classification).

## Implementation

### Pipeline Steps
1. **Preprocess** — Clean data, fix undocumented categories, engineer 5 features, scale, split 80/20
2. **Train** — Compare 7 classifiers (CV ROC-AUC), auto-select best, tune with RandomizedSearchCV
3. **Evaluate** — Log accuracy, ROC-AUC, F1, precision, recall + confusion matrix

### Models Compared
Logistic Regression, Decision Tree, Random Forest, KNN, AdaBoost, Gradient Boosting, XGBoost

### Best Model
Gradient Boosting — selected automatically by cross-validated ROC-AUC, tuned with RandomizedSearchCV

### Key Metrics (Test Set)
| Metric | Score |
|--------|-------|
| Accuracy | 0.77 |
| ROC-AUC | 0.79 |
| F1 Score | 0.54 |
| Recall | 0.61 |

## Azure ML Setup
- **Workspace:** dkv-ml-workspace (West Europe)
- **Compute:** dkv-cpu-cluster (Standard_DS11_v2)
- **Tracking:** MLflow (built-in Azure ML integration)
- **Registry:** Model auto-registered on pipeline completion

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Local test
python src/preprocess.py
python src/train.py
python src/evaluate.py

# Submit to Azure ML
python pipeline/run_pipeline.py
```

## Project Structure

├── src/
│   ├── preprocess.py        # Data cleaning, feature engineering, train/test split
│   ├── train.py             # Multi-model training with hyperparameter tuning
│   └── evaluate.py          # Model evaluation and metric logging
├── pipeline/
│   └── run_pipeline.py      # Azure ML SDK v2 pipeline definition and submission
├── environments/
│   └── conda.yaml           # Conda environment definition
├── research/
│   └── 01_EDA.ipynb         # Exploratory Data Analysis
├── params.yaml              # Hyperparameter search space
├── .amlignore               # Files excluded from Azure ML upload
└── README.md

## Notes
- Class imbalance handled via `scale_pos_weight` and `class_weight='balanced'`
- Hyperparameter search space defined in `params.yaml`
- `StandardScaler` fitted on training data only — no data leakage
- Model metadata (winner, CV scores, best params) saved as `model_metadata.json`

## Screenshots
See `screenshots/` folder for:
- Successful Azure ML pipeline run
- Registered model in Azure ML workspace