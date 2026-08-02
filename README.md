# Diabetes Prediction — Classifier Comparison

Fourteen scikit-learn classifiers, run against the same preprocessing pipeline on the same patient-health dataset, to see which one actually predicts diabetes best rather than picking one and assuming it's good enough. A shared project — my part paired kernel PCA with an RBF-kernel SVM; the rest of the comparison (below) ran the other thirteen algorithms through the same pipeline.

## Dataset

The [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — 8 clinical features (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age) and a binary outcome.

The dataset CSV itself isn't committed here. Download it and place it as `diabetes_1.csv` inside whichever algorithm's folder you're running from.

## Pipeline

Every classifier is run through the same preprocessing before it sees the data:

```
diabetes_1.csv
    ↓ mean-impute missing values (columns 1–11)
    ↓ train/test split — separately at test_size = 0.2, 0.4, and 0.6
    ↓ StandardScaler
    ↓ Kernel PCA (RBF kernel, gamma = 15)
    ↓ classifier.fit / .predict
    ↓ confusion matrix, accuracy, full classification report
```

Each classifier is tested at three train/test splits so the comparison isn't tied to one split's luck.

## Classifiers compared

| Folder | Classifier |
| :--- | :--- |
| `KSVM` | Kernel SVM (RBF) — my part of the comparison |
| `LSVM` | Linear SVM |
| `LG` | Logistic Regression |
| `KNN` | K-Nearest Neighbors |
| `DTC` | Decision Tree |
| `RFC` | Random Forest |
| `NB` | Naive Bayes |
| `LDA` | Linear Discriminant Analysis |
| `QDA` | Quadratic Discriminant Analysis |
| `MLPC` | MLP (neural net) Classifier |
| `GBC` | Gradient Boosting |
| `XGB` | XGBoost |
| `LGBM` | LightGBM |
| `Catboost` | CatBoost |
| `adaboost` | AdaBoost |

Each folder has one script per test split — e.g. `KSVM/KSVM_02_testsize_rbf.py` is the 0.2 split, `KSVM_04_testsize_rbf.py` is 0.4, `KSVM_06_testsize_rbf.py` is 0.6 — plus the accuracy/decision-boundary plot that script produces.

## Running a classifier

```bash
pip install numpy pandas matplotlib scikit-learn
# add xgboost / lightgbm / catboost individually if you're running those folders

cd KSVM   # or any other classifier folder
python KSVM_02_testsize_rbf.py
```

Run from inside the folder — each script expects `diabetes_1.csv` in its own working directory, not a shared `data/` path.

## Tech stack

Python · scikit-learn · XGBoost · LightGBM · CatBoost · pandas · NumPy · Matplotlib

## Limitations

- No shared entry point or requirements file — this was built as a side-by-side comparison, not a package, so each folder runs standalone.
- No aggregated results table across all 42 runs (14 classifiers × 3 splits) yet; each script prints its own metrics on run.
