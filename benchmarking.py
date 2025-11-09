import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from univ_imputer import universal_imputer

INPUT_CSV = "your_dataset.csv"          # CSV with no missing values
OUTPUT_CSV = "imputed_dataset.csv"      # Output CSV after imputation
MISSING_RATIO = 0.1                     # Fraction of data to randomly make missing (10%)

def introduce_missingness(df: pd.DataFrame, missing_ratio: float = 0.1):
    """Randomly introduce missing values into the dataframe."""
    np.random.seed(42)
    df_missing = df.copy()

    total_cells = df.size
    n_missing = int(total_cells * missing_ratio)
    rows = np.random.randint(0, df.shape[0], n_missing)
    cols = np.random.randint(0, df.shape[1], n_missing)

    for r, c in zip(rows, cols):
        df_missing.iat[r, c] = np.nan

    return df_missing


def compute_mae(original: pd.DataFrame, imputed: pd.DataFrame):
    """Compute Mean Absolute Error for numeric columns only."""
    mae_scores = {}
    for col in original.select_dtypes(include=[np.number]).columns:
        mask = imputed[col].notna()
        mae = mean_absolute_error(original.loc[mask, col], imputed.loc[mask, col])
        mae_scores[col] = mae
    return mae_scores


def compute_accuracy(original: pd.DataFrame, imputed: pd.DataFrame, df_missing: pd.DataFrame):
    """
    Compute imputation accuracy for categorical columns.
    Accuracy = (Number of correctly imputed values) / (Number of missing values introduced)
    """
    acc_scores = {}
    categorical_cols = original.select_dtypes(exclude=[np.number]).columns

    for col in categorical_cols:
        # Identify which rows were originally missing
        missing_mask = df_missing[col].isna()

        if missing_mask.sum() == 0:
            continue  # no missing values introduced in this column

        true_vals = original.loc[missing_mask, col].astype(str).values
        pred_vals = imputed.loc[missing_mask, col].astype(str).values

        correct = np.sum(true_vals == pred_vals)
        acc = correct / len(true_vals)
        acc_scores[col] = acc * 100  # convert to %
    return acc_scores


if __name__ == "__main__":
    # Step 1: Read the input CSV
    df_original = pd.read_csv(INPUT_CSV)
    print(f"Loaded dataset: {df_original.shape}")

    # Step 2: Introduce missing values
    df_missing = introduce_missingness(df_original, MISSING_RATIO)
    print(f"Introduced missing values (≈{MISSING_RATIO*100}% cells).")

    # Step 3: Impute using the universal imputer (you may edit this part later)
    df_imputed = universal_imputer(df_missing, model="auto", verbose=True)

    # Step 4: Compute metrics
    mae_results = compute_mae(df_original, df_imputed)
    acc_results = compute_accuracy(df_original, df_imputed, df_missing)

    print("\n=== Mean Absolute Error (MAE) per Numeric Column ===")
    for col, mae in mae_results.items():
        print(f"{col}: {mae:.4f}")

    print("\n=== Accuracy (%) per Categorical Column ===")
    if acc_results:
        for col, acc in acc_results.items():
            print(f"{col}: {acc:.2f}%")
    else:
        print("No categorical columns found or no missing values introduced.")

    # Step 5: Compute mean of each column in the imputed dataset
    print("\n=== Mean values after imputation ===")
    mean_values = df_imputed.mean(numeric_only=True)
    print(mean_values)

    # Step 6: Save the imputed CSV
    df_imputed.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved imputed dataset to '{OUTPUT_CSV}'")
