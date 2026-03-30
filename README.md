# Bank Customer Churn Prediction — MSE 446, Group 3

Predicting customer churn for a bank using seven supervised classification models.
The project covers exploratory data analysis, preprocessing, and a comparative
evaluation of all models across recall, F1, and ROC-AUC.

**Group members:** Karan Champaneri, Ethan Gabriel, Sunny Jiao, Desmond Nixon, Adele Younis

## Dataset

- **Source:** `data/raw/Customer-Churn-Records.csv`
- 10,000 observations, 18 features
- Binary target: `Exited` (1 = churned, 0 = retained)
- ~20% churn / ~80% retained (moderate class imbalance)

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open `notebook.ipynb` in Jupyter and run all cells top-to-bottom.
   The dataset is already in `data/raw/` — no download needed.

## Folder Structure

```
churn_project/
├── notebook.ipynb          # Main notebook (EDA → cleaning → preprocessing → modeling)
├── requirements.txt        # Python dependencies
├── README.md
└── data/
    └── raw/
        └── Customer-Churn-Records.csv
```

## Notebook Structure

### Section 1 — Exploratory Data Analysis
- Dataset overview, missing values & duplicates check
- Class imbalance analysis
- Feature vs. churn visualizations (age, balance, geography, etc.)
- Correlation heatmap (identified `Complain` as near-perfect leakage feature, ρ ≈ 1.0)

### Section 2 — Cleaning
- Dropped identifiers (`RowNumber`, `CustomerId`, `Surname`)
- Dropped data leakage column (`Complain`)

### Section 3 — Preprocessing
- One-hot encoding for categorical features (`Geography`, `Gender`, `Card Type`)
- Stratified 80/20 train/test split
- `StandardScaler` applied to continuous features for Logistic Regression, Neural Network, and SVM (fit on training set only)

### Section 4 — Modeling

All models use `class_weight='balanced'` (or sample weights for MLP) to handle class imbalance.
Hyperparameter tuning is done via `GridSearchCV` with `StratifiedKFold(n_splits=5)`, scoring on recall.

| Model | Recall | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.723 | 0.505 | 0.780 |
| Decision Tree (pruned) | 0.821 | 0.562 | 0.857 |
| Bagging | 0.760 | 0.613 | 0.873 |
| Random Forest | 0.694 | 0.625 | 0.869 |
| Gradient Boosting | 0.762 | 0.603 | 0.874 |
| Support Vector Machine | 0.713 | 0.503 | 0.780 |
| Neural Network | 0.828 | 0.532 | 0.860 |

- **4.1 Logistic Regression** — L1 vs L2 regularization comparison; L2 selected
- **4.2 Decision Tree** — fully grown baseline → cost-complexity pruning (ccp_alpha)
- **4.3 Bagging** — ensemble of pruned decision trees; tuned n_estimators
- **4.4 Random Forest** — bagging with feature randomization; tuned n_estimators, max_features, min_samples_leaf
- **4.5 Gradient Boosting** — sequential boosting; tuned n_estimators, learning_rate, max_depth
- **4.6 SVM** — RBF kernel; tuned C, kernel, gamma
- **4.7 Neural Network** — MLPClassifier with threshold tuning (F2-score sweep → threshold 0.111)
- **4.8 Model Comparison** — side-by-side evaluation table and bar chart

## Reproducibility

All random states are fixed at `random_state=42`. Results are fully reproducible by running
cells in order on the provided dataset.
