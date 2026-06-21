#  Football Player Market Value Prediction

A Machine Learning project for predicting football players' market value (in euros) using the **FIFA 22 Complete Player Dataset**, following the **KDD** (Knowledge Discovery in Databases) process.

##  Overview

This project tackles a **regression** (numerical prediction) problem, predicting the `value_eur` variable based on players' sporting, physical, and demographic characteristics. Three models were compared, and the best-performing one was selected as the final model.

##  Dataset

- **Source:** [FIFA 22 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset) (Kaggle, author: stefanoleone992)
- **File:** `players_22.csv`
- **Size:** 19,239 players, 110 attributes
- **Target variable:** `value_eur` (market value in euros)

##  Methodology (KDD)

1. **Data selection** — 21 features initially selected, `release_clause_eur` excluded (data leakage)
2. **Preprocessing** — missing value imputation (median), outlier retention, `log1p` transformation on target
3. **Transformation** — 70/15/15 split (train/validation/test), `StandardScaler` + `OneHotEncoder` via Pipeline
4. **Feature selection** — preliminary Random Forest + aggregated importance → 8 final features
5. **Training** — 3 models with GridSearchCV and cross-validation
6. **Evaluation** — MAE, MSE, RMSE, R² on train/validation/test

##  Models Compared

| Model | Role |
|---|---|
| Linear Regression | Baseline |
| Random Forest Regressor | Parallel ensemble (bagging) |
| Gradient Boosting Regressor | Sequential ensemble (boosting) |

##  Results (test set)

| Model | MAE (EUR) | RMSE (EUR) | R² |
|---|---|---|---|
| **Gradient Boosting** ⭐ | 125,233 | 876,203 | **0.9867** |
| Random Forest | 175,604 | 1,853,679 | 0.9403 |
| Linear Regression | 646,359 | 2,463,095 | 0.8947 |

**Final model:** Gradient Boosting Regressor (`learning_rate=0.1`, `max_depth=5`, `min_samples_leaf=5`, `n_estimators=200`), achieving **R² = 0.9867** on the test set.

## Project Structure
├── BigData_FIFA22_Project.ipynb    # Complete notebook

├── bigdata_fifa22_project.py       # Python script

├── BigData_FIFA22_Report.pdf       # Detailed report

└── README.md


##  Technologies

- Python 3
- scikit-learn, pandas, numpy
- matplotlib, seaborn
- scipy (QQ-Plot, Shapiro-Wilk)

##  Running the Project

The project was developed in **Google Colab**. To run it:

1. Open the notebook in Google Colab
2. The dataset is downloaded automatically via `kagglehub`
3. Run all cells (Runtime → Run all)

##  Authors

- Horătău Darius Cristian
- Varvara Dorin Robert

##  Year

2026
