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


def _get_column_type(df: pd.DataFrame, col: str) -> str:
    return "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"


def _missing_ratio(df: pd.DataFrame, col: str) -> float:
    return df[col].isna().mean() * 100.0


def _auto_strategy(df: pd.DataFrame, col: str) -> str:
    col_type = _get_column_type(df, col)
    missing_ratio = _missing_ratio(df, col)
    if col_type == "numeric":
        if missing_ratio < 10:
            return "mean"
        if missing_ratio < 30:
            return "knn"
        return "random_forest"
    else:
        if missing_ratio < 20:
            return "mode"
        return "random_forest"


def _simple_imputer(df: pd.DataFrame, col: str, strategy: str = "mean") -> pd.DataFrame:
    if strategy == "mode":
        strategy = "most_frequent"

    if strategy in {"mean", "median", "most_frequent"}:
        imp = SimpleImputer(strategy=strategy)
        df[[col]] = imp.fit_transform(df[[col]])
    elif strategy == "knn":
        # KNNImputer needs numeric data; select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 1:
            # fallback to mean for the column
            imp = SimpleImputer(strategy="mean")
            df[[col]] = imp.fit_transform(df[[col]])
        else:
            imp = KNNImputer(n_neighbors=5)
            # fit_transform returns ndarray; assign back to numeric columns
            transformed = imp.fit_transform(df[numeric_cols])
            df[numeric_cols] = transformed
    else:
        raise ValueError(f"Unsupported simple imputer strategy: {strategy}")

    return df


def _ml_imputer(df: pd.DataFrame, col: str, model_type: str = "random_forest") -> pd.DataFrame:
    df = df.copy()
    df_non_missing = df[df[col].notna()]
    df_missing = df[df[col].isna()]

    if df_missing.empty:
        return df

    X_train = df_non_missing.drop(columns=[col])
    y_train = df_non_missing[col]
    X_pred = df_missing.drop(columns=[col])

    # If there are no features to train on, fallback to simple imputation on the column
    if X_train.shape[1] == 0:
        if pd.api.types.is_numeric_dtype(y_train):
            fill = float(y_train.mean())
        else:
            fill = y_train.mode().iloc[0] if not y_train.mode().empty else ""
        df.loc[df[col].isna(), col] = fill
        return df

    numeric_features = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_features = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]

    numeric_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    col_type = _get_column_type(df, col)
    model = None

    if col_type == "numeric":
        if model_type == "linear_regression":
            model = LinearRegression()
        elif model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "mlp":
            model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        else:
            raise ValueError(f"Unsupported numeric model type: {model_type}")
    else:
        if model_type == "logistic_regression":
            model = LogisticRegression(max_iter=200)
        elif model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "mlp":
            model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        else:
            raise ValueError(f"Unsupported categorical model type: {model_type}")

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    encoder = None
    if col_type == "categorical":
        encoder = LabelEncoder()
        y_train_enc = encoder.fit_transform(y_train.astype(str))
    else:
        y_train_enc = y_train.astype(float).to_numpy()

    # Fit and predict
    try:
        pipe.fit(X_train, y_train_enc)
    except Exception:
        # fallback: simple imputation if training fails
        return _simple_imputer(df, col, "mean" if col_type == "numeric" else "mode")

    y_pred = pipe.predict(X_pred)

    if encoder is not None:
        # predictions may be floats; round and convert to int indices before inverse transform
        y_pred_labels = np.rint(y_pred).astype(int)
        try:
            y_pred_final = encoder.inverse_transform(y_pred_labels)
        except Exception:
            # if inverse_transform fails, coerce to string
            y_pred_final = y_pred_labels.astype(str)
    else:
        y_pred_final = y_pred.astype(float)

    # Assign predictions back preserving index alignment
    df.loc[df[col].isna(), col] = y_pred_final
    return df


def universal_imputer(
    df: pd.DataFrame,
    model: Optional[str] = "auto",
    columns: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Universal imputer that chooses strategies per-column and imputes missing values.
    model: "auto" (choose strategy automatically) or one of:
           "mean", "median", "mode", "knn", "linear_regression", "random_forest", "mlp", "logistic_regression"
    columns: list of columns to impute; if None, all columns with missing values are considered.
    """
    df = df.copy()
    imputers_used: Dict[str, Any] = {}

    if columns is None:
        columns = [c for c in df.columns if df[c].isna().any()]

    if verbose:
        print(f"Columns to impute: {columns}")

    for col in columns:
        if col not in df.columns:
            continue
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
            # fallback to simple strategy on error
            fallback = "mean" if col_type == "numeric" else "mode"
            try:
                df = _simple_imputer(df, col, fallback)
                imputers_used[col] = f"{chosen_model} -> fallback:{fallback}"
                if verbose:
                    print(f"  Warning: imputation with '{chosen_model}' failed; fell back to '{fallback}'. Error: {exc}")
            except Exception:
                if verbose:
                    print(f"  Error: final fallback failed for column '{col}'. Leaving NaNs intact. Error: {exc}")
                imputers_used[col] = f"{chosen_model} -> failed"

    if verbose:
        print("Imputation complete.")
        try:
            print(pd.Series(imputers_used, name="Strategy Used"))
        except Exception:
            pass

    return df