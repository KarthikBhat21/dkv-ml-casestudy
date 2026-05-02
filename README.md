# Credit Card Default Prediction — Azure ML Pipeline

## Overview
End-to-end Azure ML pipeline predicting credit card payment default
using the UCI Credit Card Default dataset (30,000 records, binary classification).

## Pipeline Steps
1. **Validate** — Check raw dataset column names against schema.yaml. Pipeline stops if validation fails.
2. **Preprocess** — Clean data, fix undocumented categories, engineer 5 features, scale, split 80/20
3. **Train** — Compare 7 classifiers (CV ROC-AUC), auto-select best, tune with RandomizedSearchCV
4. **Evaluate** — Log accuracy, ROC-AUC, F1, precision, recall + confusion matrix

## Models Compared
Logistic Regression, Decision Tree, Random Forest, KNN, AdaBoost, Gradient Boosting, XGBoost

## Best Model
Gradient Boosting — selected automatically by cross-validated ROC-AUC, tuned with RandomizedSearchCV

## Key Metrics (Test Set)
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
# Local test — run in order
python src/validate.py
python src/preprocess.py
python src/train.py
python src/evaluate.py

# Submit to Azure ML
python pipeline/run_pipeline.py
```

## Notes
- Data validation stops the pipeline if column names don't match schema.yaml
- Class imbalance handled via `scale_pos_weight` and `class_weight='balanced'`
- Hyperparameter search space defined in `params.yaml` — no hardcoded values in code
- `StandardScaler` fitted on training data only — no data leakage
- RandomizedSearchCV applied to winning model only — faster than tuning all models
- Model auto-registered in Azure ML Model Registry on pipeline completion

## Screenshots
See `screenshots/` folder for:
- Successful Azure ML pipeline run (all 4 steps green)
- Registered model in Azure ML workspace