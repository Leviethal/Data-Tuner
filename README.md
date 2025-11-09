# 📊 Comparison of Missing Data Imputation Techniques

This report summarizes the findings from the evaluation of **statistical (Mean, Median)** and **machine learning-based (kNN)** imputation techniques under different levels of missing data.  
The performance was analyzed using **MAE (Mean Absolute Error)**, **L10**, **L25**, and **NOx Regression MSE** metrics.

---

## 🔹 1. Results from MAE, L10, and L25 Plots

### **MAE Trends**
- **Mean** and **Median** imputations show **almost constant MAE** across all levels of missingness, indicating that they are not sensitive to the proportion of missing data.  
- **kNN imputations (k=3, 5, 10)** yield **lower MAE** at low missingness levels (<10%) but MAE gradually increases as more data becomes missing.  
- Among kNN variants, **k=10 performs slightly better** at higher missingness (>50%).

### **L10 Performance**
- **kNN methods** achieve **L10 > 90%** at low missingness (<10%), meaning most imputed values are highly accurate.  
- As missingness increases, **L10 decreases sharply**, reflecting reduced precision in imputation.  
- **Mean** and **Median** have consistently low and flat L10 scores (~40–50%), showing less sensitivity but lower accuracy overall.

### **L25 Performance**
- All **kNN variants** achieve **L25 ≈ 99%** at low missingness, indicating that nearly all imputations are within acceptable error bounds.  
- L25 gradually decreases as missingness increases, but **kNN(10)** performs best, followed by **kNN(5)** and **kNN(3)**.  
- **Mean** and **Median** remain constant (~75–78%), showing stability but relatively low performance.

### ✅ **Inference**
- **kNN imputation consistently outperforms mean and median imputation**, especially as missingness increases.  
- **Statistical imputations (Mean/Median)** are reliable only for **low missingness (<10%)** scenarios.

---

## 🔹 2. Results from NOx Regression MSE Plot

- At **low missingness (<10%)**, **Mean** and **Median** imputations perform competitively, sometimes achieving the lowest MSE.  
- As missingness increases (≥20%), **kNN methods start outperforming** statistical ones in reducing regression MSE.  
- **kNN(3)** shows the **lowest MSE overall** beyond 20–30% missingness, providing a good balance between bias and variance.  
- **Mean** and **Median** MSE values increase steadily beyond 30–40% missingness, showing poor adaptability to higher data loss.

### ✅ **Inference**
- For **<10% missingness**, simple **Mean/Median** imputation is sufficient.  
- For **≥20% missingness**, **kNN (especially k=3–5)** provides more accurate and reliable regression performance.  
- **kNN remains stable** even at **high missingness (>50%)**, while statistical methods degrade noticeably.

---

## 🔹 3. Overall Summary

| Missingness Level | Recommended Method | Reason |
|--------------------|--------------------|--------|
| ≤ 10% | **Mean or Median Imputation** | Simple, efficient, and sufficient at low missingness. |
| ≥ 20% | **kNN (k=5 or k=10)** | Provides higher accuracy, lower MAE/MSE, and better stability. |

**Conclusion:**  
The **kNN imputation method** provides higher fidelity (high L10/L25) and lower prediction error (MSE), making it an ideal choice for handling **real-world datasets with moderate to high missingness**.


# Universal Imputer

> A lightweight, general-purpose imputation library and evaluation demo that automatically chooses sensible imputation strategies (simple statistics, KNN, or ML-based) per-column and provides an easy-to-use API and evaluation notebook.

---

## Table of Contents

1. [Project overview](#project-overview)
2. [Repository structure](#repository-structure)
3. [Quickstart](#quickstart)
4. [API reference (core)](#api-reference-core)
5. [How it works (high level)](#how-it-works-high-level)
6. [Evaluation methodology](#evaluation-methodology)
7. [Design choices, limitations & caveats](#design-choices-limitations--caveats)
8. [Suggested improvements / future work](#suggested-improvements--future-work)
9. [Development / testing / reproducibility](#development--testing--reproducibility)
10. [License & contact](#license--contact)

---

## Project overview

This project implements a **universal imputer** (`univ_imputer.py`) designed to handle both numeric and categorical missing data by automatically selecting an appropriate imputation strategy per column. It supports simple statistical imputers (mean/median/mode), KNN imputation for dense numeric settings, and ML model-based imputations (linear/logistic, random forest, MLP). A companion script (`checking.py`) demonstrates evaluation on the Wine dataset by artificially introducing missing values and reporting MSE / MAE / RMAE metrics. The notebook `imputing.ipynb` is included as a proof-of-concept / extended evaluation notebook (kept in repo but separate from the core package).

---



## Quickstart

### 1) Install dependencies

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install numpy pandas scikit-learn
```

Optionally create `requirements.txt`:

```
numpy
pandas
scikit-learn
```

### 2) Use `universal_imputer` in your own code

Example:

```python
import pandas as pd
from univ_imputer import universal_imputer

df = pd.read_csv("my_data.csv")
df_imputed = universal_imputer(df, model="auto", verbose=True)  # model can be "auto" or choose specific strategy
```

---

## API reference (core)

### `universal_imputer(df, model="auto", columns=None, verbose=True) -> pd.DataFrame`

* **df** (`pd.DataFrame`) — input dataframe containing missing values.
* **model** (`str` or `None`, default `"auto"`) — strategy to use for all columns or `"auto"` to pick per-column strategy. Recognized values:

  * Simple strategies: `"mean"`, `"median"`, `"mode"` (alias `"most_frequent"`), `"knn"`
  * ML strategies: `"random_forest"`, `"linear_regression"`, `"logistic_regression"`, `"mlp"` (MLP regressor/classifier)
  * `"auto"`: uses internal heuristic (see *How it works* below).
* **columns** (`Sequence[str]`, default `None`) — list of columns to impute. If `None`, only columns that contain missing values are imputed.
* **verbose** (`bool`, default `True`) — prints a short progress log and the strategy used per column.

**Returns**: New `pd.DataFrame` with missing values filled.

**Notes from implementation**:

* For categorical pipelines, `OneHotEncoder(handle_unknown="ignore", sparse=False)` is used (hence result dims can grow).
* ML-based imputer trains a separate model per-column using other columns as features. If no features are available, it falls back to mean/mode.
* On training failures, a fallback simple imputation (mean/mode) is applied.

---

## How it works (high level)

### Auto strategy (default decision thresholds)

The `auto` decision is deterministic and based on column type and missing-rate:

* **Numeric columns**

  * missing < 10% → `"mean"`
  * 10% ≤ missing < 30% → `"knn"`
  * missing ≥ 30% → `"random_forest"`
* **Categorical columns**

  * missing < 20% → `"mode"`
  * missing ≥ 20% → `"random_forest"`

### Simple imputer

* `"mean"`, `"median"` or `"most_frequent"` use `sklearn.impute.SimpleImputer`.
* `"knn"` uses `sklearn.impute.KNNImputer` with `n_neighbors=5` — note that when `knn` is selected it imputes all numeric columns (not only the single column).

### ML imputer (per-column model)

* Builds a preprocessing `ColumnTransformer`:

  * numeric features → `SimpleImputer(strategy="mean")`
  * categorical features → pipeline of `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(...)`
* Model choice depends on the target column type and requested `model_type`:

  * Numeric targets: `LinearRegression`, `RandomForestRegressor`, or `MLPRegressor`
  * Categorical targets: `LogisticRegression`, `RandomForestClassifier`, or `MLPClassifier`
* Categorical targets are label-encoded before training; predictions are inverse-transformed and (for MLP/regressor outputs) rounded to nearest class index.
* If model training fails, the method falls back to a simple imputer.

---

## Evaluation methodology

`checking.py` demonstrates a reproducible evaluation workflow:

1. Load a ground-truth DataFrame (`sklearn.datasets.load_wine(as_frame=True)`).
2. Copy and randomly inject missing values (~15% per column) using `np.random.default_rng(42)` for reproducibility.
3. Run `universal_imputer(..., model="auto")`.
4. For numeric columns, compute:

   * **MSE** — mean squared error on the positions that were set to NaN
   * **MAE** — mean absolute error on those positions
   * **RMAE** — normalized MAE (`mae / mean(|y_true|)`) to compare across features
5. Save results and datasets to `imputer_outputs/`.

This method measures **imputation quality relative to known ground truth** (strong evaluation setting because injected missingness is MCAR-like). See `imputing.ipynb` for extended experiments and comparisons (kept separate).

---

## Design choices, limitations & caveats

* **Sequential per-column imputation**: Columns are imputed one-by-one. This is simple and deterministic, but not equivalent to iterative multivariate imputation (which can better account for inter-column missingness structure).
* **KNN behavior**: When `knn` is used, the implementation currently imputes all numeric columns together (intentional because KNN imputation leverages joint numeric space) — be careful if you expect only one column to be changed.
* **Categorical handling**: Label encoding + rounding of predicted values is used for categorical ML predictions. For high-cardinality categorical targets this can be unstable.
* **OneHotEncoder (sparse=False)**: memory and dimensionality can blow up on many categorical features with many levels.
* **Not specialized for time-series**: No handling for time-dependency or forward/backward fill semantics.

---

## Suggested improvements / future work

* Add **Iterative Imputer** (sklearn `IterativeImputer`) as an option to model joint missingness and obtain better consistency.
* Provide a mode that **fits and persists** imputers/models per-column (e.g., return a dict of fitted `Pipeline` objects).
* Add **cross-validation** and **calibration / uncertainty estimates** for ML-based imputations.
* Add **dtype preservation** and strong post-processing to keep categorical dtypes, pandas nullable dtypes, and index integrity.
* Add **parallelization** when imputing independent columns to speed up the pipeline for wide data.
* Improve memory use for one-hot encoding (use sparse matrices where possible) and support category encoders for high-cardinality variables.
---


## Example: Minimal script

```python
import pandas as pd
from univ_imputer import universal_imputer

df = pd.read_csv("data.csv")
# Impute only columns with missing values, auto-decide strategies
df_imputed = universal_imputer(df, model="auto", verbose=True)
df_imputed.to_csv("data_imputed.csv", index=False)
```

---