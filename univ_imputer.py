from typing import Optional, Sequence, Dict, Any
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

__all__ = ["universal_imputer"]


# Identify whether a column is numeric or categorical.
def _get_column_type(df: pd.DataFrame, col: str) -> str:
    return "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"


# Compute percentage of missing values in a column.
def _missing_ratio(df: pd.DataFrame, col: str) -> float:
    return df[col].isna().mean() * 100.0


# Automatically pick imputation strategy based on data type + missing percentage.
def _auto_strategy(df: pd.DataFrame, col: str) -> str:
    col_type = _get_column_type(df, col)
    missing_ratio = _missing_ratio(df, col)
    if col_type == "numeric":
        if missing_ratio < 10:
            return "mean"       
        elif missing_ratio < 25:
            return "median"     
        elif missing_ratio < 40:
            return "knn"
        return "random_forest" 
    else:
        if missing_ratio < 20:
            return "mode"
        return "random_forest"


# Apply simple statistical or KNN-based imputation for one column.
def _simple_imputer(df: pd.DataFrame, col: str, strategy: str = "mean") -> pd.DataFrame:
    if strategy == "mode":
        strategy = "most_frequent"

    if strategy in {"mean", "median", "most_frequent"}:
        imp = SimpleImputer(strategy=strategy)
        df[[col]] = imp.fit_transform(df[[col]])
    elif strategy == "knn":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 1:
            imp = SimpleImputer(strategy="mean")
            df[[col]] = imp.fit_transform(df[[col]])
        else:
            imp = KNNImputer(n_neighbors=5)
            transformed = imp.fit_transform(df[numeric_cols])
            df[numeric_cols] = transformed
    else:
        raise ValueError(f"Unsupported simple imputer strategy: {strategy}")

    return df


# Train a machine learning model (regressor/classifier) to predict missing values.
def _ml_imputer(df: pd.DataFrame, col: str, model_type: str = "random_forest") -> pd.DataFrame:
    df = df.copy()
    df_non_missing = df[df[col].notna()]
    df_missing = df[df[col].isna()]

    # If no missing values, return unchanged.
    if df_missing.empty:
        return df

    X_train = df_non_missing.drop(columns=[col])
    y_train = df_non_missing[col]
    X_pred = df_missing.drop(columns=[col])

    # If no usable features, fall back to simple fill value.
    if X_train.shape[1] == 0:
        if pd.api.types.is_numeric_dtype(y_train):
            fill = float(y_train.mean())
        else:
            fill = y_train.mode().iloc[0] if not y_train.mode().empty else ""
        df.loc[df[col].isna(), col] = fill
        return df

    # Split features by type.
    numeric_features = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_features = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]

    # Preprocessing pipelines.
    numeric_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    # Combine preprocessing steps.
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Select model based on target type.
    col_type = _get_column_type(df, col)
    if col_type == "numeric":
        model = (
            LinearRegression() if model_type == "linear_regression"
            else RandomForestRegressor(n_estimators=100, random_state=42) if model_type == "random_forest"
            else MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        )
    else:
        model = (
            LogisticRegression(max_iter=200) if model_type == "logistic_regression"
            else RandomForestClassifier(n_estimators=100, random_state=42) if model_type == "random_forest"
            else MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        )

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Encode categorical target if needed.
    encoder = None
    if col_type == "categorical":
        encoder = LabelEncoder()
        y_train_enc = encoder.fit_transform(y_train.astype(str))
    else:
        y_train_enc = y_train.astype(float).to_numpy()

    # Train model, fallback to simple imputation on failure.
    try:
        pipe.fit(X_train, y_train_enc)
    except Exception:
        return _simple_imputer(df, col, "mean" if col_type == "numeric" else "mode")

    y_pred = pipe.predict(X_pred)

    # Decode predictions if categorical.
    if encoder is not None:
        y_pred_final = encoder.inverse_transform(np.rint(y_pred).astype(int))
    else:
        y_pred_final = y_pred.astype(float)

    df.loc[df[col].isna(), col] = y_pred_final
    return df


# Main universal imputer that loops through columns and applies the selected strategy.
def universal_imputer(
    df: pd.DataFrame,
    model: Optional[str] = "auto",
    columns: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:

    df = df.copy()
    imputers_used: Dict[str, Any] = {}

    # Default: impute only columns containing missing values.
    if columns is None:
        columns = [c for c in df.columns if df[c].isna().any()]

    if verbose:
        print(f"Columns to impute: {columns}")

    # Loop through each column and impute.
    for col in columns:
        col_type = _get_column_type(df, col)
        missing_ratio = _missing_ratio(df, col)

        chosen_model = model if model != "auto" else _auto_strategy(df, col)

        if verbose:
            print(f"Imputing '{col}' ({col_type}, missing: {missing_ratio:.2f}%) using '{chosen_model}'")

        try:
            if chosen_model in {"mean", "median", "mode", "most_frequent", "knn"}:
                df = _simple_imputer(df, col, chosen_model)
            else:
                df = _ml_imputer(df, col, chosen_model)
            imputers_used[col] = chosen_model
        except Exception as exc:
            fallback = "mean" if col_type == "numeric" else "mode"
            df = _simple_imputer(df, col, fallback)
            imputers_used[col] = f"{chosen_model} -> fallback:{fallback}"

    if verbose:
        print("Imputation complete.")
        print(pd.Series(imputers_used, name="Strategy Used"))

    return df
