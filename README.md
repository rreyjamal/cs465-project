# Explainable Credit Risk Prediction Using Machine Learning

A CS465 machine learning course project implementing a complete end-to-end pipeline for credit risk prediction on the UCI German Credit Dataset, with emphasis on model explainability and interpretability.

## Project Overview

This project predicts whether a loan applicant belongs to a good or bad credit risk class using structured tabular data. Four classical machine learning models are compared under a unified preprocessing and evaluation setup, with emphasis on both predictive performance and interpretability.

### Research Question

Which classical machine learning model provides the best balance between predictive performance and interpretability for credit risk classification on the UCI German Credit dataset?

## Dataset

- **Source**: UCI German Credit Dataset
- **Format**: Space-separated values (german.data)
- **Samples**: 1000 credit applications
- **Features**: 20 mixed categorical and numerical features
- **Target**: Binary classification (Good Credit = 0, Bad Credit = 1)
- **Class distribution**: 70% Good Credit, 30% Bad Credit

### Features

- `status_checking_account`: Status of existing checking account
- `duration_months`: Credit duration in months
- `credit_history`: Credit history
- `purpose`: Purpose of credit
- `credit_amount`: Credit amount
- `savings_account`: Savings account/bonds
- `employment_since`: Present employment since
- `installment_rate`: Installment rate in percentage of disposable income
- `personal_status_sex`: Personal status and sex
- `other_debtors_guarantors`: Other debtors/guarantors
- `present_residence_since`: Present residence since
- `property`: Property
- `age_years`: Age in years
- `other_installment_plans`: Other installment plans
- `housing`: Housing
- `existing_credits`: Number of existing credits at this bank
- `job`: Job
- `num_dependents`: Number of people liable for maintenance
- `telephone`: Telephone
- `foreign_worker`: Foreign worker
- `target`: Credit risk (0 = Good, 1 = Bad)

## Project Structure

```
cs465-credit-risk-ml-project/
├── data/
│   ├── raw/                    # Original dataset
│   │   └── german.data
│   └── processed/              # Processed data (if needed)
├── notebooks/
│   └── 01_german_credit_experiments.ipynb  # Main analysis notebook
├── src/
│   ├── data_preprocessing.py   # Data loading and preprocessing utilities
│   ├── feature_engineering.py  # Feature engineering pipelines
│   ├── train.py                # Model training functions
│   ├── evaluate.py             # Evaluation and metrics
│   ├── explainability.py       # Model explainability utilities
│   └── utils.py                # General utilities
├── results/
│   ├── figures/                # Plots and visualizations
│   ├── tables/                 # Result tables
│   └── metrics/                # Model performance metrics
├── paper/
│   └── paper_notes.md          # Paper writing guidance
├── presentation/
│   └── presentation_outline.md # Presentation structure
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore file
```

## Models Used

1. **Logistic Regression** - Interpretable linear baseline with coefficient analysis
2. **Decision Tree** - Non-linear model with explicit decision rules
3. **Random Forest** - Ensemble method with tree-based feature importance
4. **Gradient Boosting** - Powerful sequential ensemble method

## Evaluation Metrics

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- 5-fold Stratified Cross-Validation (scored by ROC-AUC)

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rreyjamal/cs465-project.git
   cd cs465-project
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure the dataset is in place**:
   - `data/raw/german.data` is included in the repository

## Running the Analysis

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/01_german_credit_experiments.ipynb
```

Run all cells top to bottom. The notebook handles all path setup automatically.

### Option 2: Python Modules

```bash
# From project root
python -c "from src.data_preprocessing import load_german_credit; df = load_german_credit(); print(df.head())"
```

## Key Features

- **Reproducible**: Fixed random seed (42), stratified splits, deterministic preprocessing
- **Modular**: Clean separation of concerns across reusable `src/` modules
- **Comprehensive**: Full pipeline from EDA to explainability analysis
- **Robust paths**: Notebook uses `pathlib` for reliable cross-platform file saving
- **Academic**: Paper notes and presentation outline included

## Results Summary

The complete pipeline has been executed successfully. All outputs are saved in the `results/` directory:

- **Model comparison**: `results/metrics/model_comparison.csv`
- **Classification reports**: `results/metrics/{model}_classification_report.csv`
- **Cross-validation results**: `results/metrics/cross_validation_results.csv`
- **Feature importance**: `results/metrics/random_forest_feature_importance.csv`
- **Coefficients**: `results/metrics/logistic_regression_coefficients.csv`
- **Figures**: `results/figures/`

## Final Results

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.750 | 0.764 | 0.558 | 0.800 | 0.658 | 0.806 |
| Random Forest | 0.775 | 0.692 | 0.674 | 0.483 | 0.563 | 0.800 |
| Gradient Boosting | 0.755 | 0.682 | 0.612 | 0.500 | 0.551 | 0.764 |
| Decision Tree | 0.630 | 0.593 | 0.405 | 0.500 | 0.448 | 0.575 |

### Main Finding

Logistic Regression achieved the best overall balance between predictive performance and interpretability, with the highest ROC-AUC (0.806), highest balanced accuracy (0.764), and strongest recall for the bad-credit class (0.800). Random Forest achieved the highest raw accuracy (0.775), but lower recall for risky applicants.

## Reproducibility Notes

- Random seed set to 42 for all experiments
- Stratified train-test split (80/20) preserves class distribution
- All preprocessing steps are deterministic
- Results are saved automatically on each run

## Dependencies

See `requirements.txt` for exact versions. Core libraries:

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Visualization
- `seaborn` - Statistical plots
- `jupyter` - Interactive notebooks
- `shap` - Model explainability (optional, graceful fallback if unavailable)

## Academic Context

This project is part of the CS465 Machine Learning course and follows undergraduate research standards:

- Clean, documented, modular code
- Reproducible experiments with fixed random seeds
- Academic writing support via `paper/paper_notes.md`
- Presentation materials via `presentation/presentation_outline.md`
- Proper evaluation methodology including balanced metrics for imbalanced data

## Potential Extensions

- Hyperparameter optimization with GridSearchCV or Optuna
- Additional models (SVM, XGBoost, Neural Networks)
- Advanced feature engineering and selection
- Fairness and bias analysis
- Cost-sensitive learning for asymmetric misclassification costs

