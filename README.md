# Bank Customer Churn Prediction — MSE 446, Group 3

Predicting customer churn for a bank using supervised classification models. The project covers exploratory data analysis, preprocessing, and a comparison of five modeling approaches.

## Dataset

- **Source:** `data/raw/Customer-Churn-Records.csv`
- 10,000 observations, 18 features
- Binary target: `Exited` (1 = churned, 0 = retained)
- ~20% churn / ~80% retained (moderate class imbalance)

## Notebook Structure

### Section 1 — Exploratory Data Analysis
- Dataset overview, missing values & duplicates check
- Class imbalance analysis
- Feature vs. churn visualizations (age, balance, geography, etc.)
- Correlation heatmap (identified and removed `Complain` due to near-perfect correlation with target)

### Section 2 — Cleaning
- Dropped identifiers (`RowNumber`, `CustomerId`, `Surname`) and data leakage columns (`Complain`, `Satisfaction Score`)

### Section 3 — Preprocessing
- One-hot encoding for categorical features (`Geography`, `Gender`, `Card Type`)
- Train/test split (80/20, stratified)
- StandardScaler applied for logistic regression only

### Section 4 — Modeling
All models use `class_weight="balanced"` or sample weights to handle class imbalance. Hyperparameter tuning is done via `GridSearchCV` with `StratifiedKFold`.

| Model | Recall | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.765 | 0.545 | 0.855 |
| Decision Tree (pruned) | 0.762 | 0.579 | 0.839 |
| Bagging (pruned trees) | 0.715 | 0.587 | 0.853 |
| Random Forest | 0.740 | 0.601 | 0.868 |
| Gradient Boosting | 0.762 | 0.603 | 0.874 |

- **4.1 Logistic Regression** — baseline with regularization tuning
- **4.2 Decision Tree** — fully grown tree + cost-complexity pruning
- **4.3 Bagging** — ensemble of pruned decision trees
- **4.4 Random Forest** — bagging with feature randomization
- **4.5 Gradient Boosting** — sequential boosting with tuned learning rate, depth, and estimators
- **4.6 Model Comparison** — side-by-side evaluation table and discussion