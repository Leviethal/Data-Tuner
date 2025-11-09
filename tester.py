import pandas as pd
from univ_imputer import universal_imputer

INPUT_CSV = "data.csv"
COLUMNS_TO_IMPUTE = ['LotFrontage']
OUTPUT_CSV = "data_imputed.csv"
STRATEGY = "auto"
if __name__ == "__main__":
    
    df = pd.read_csv(INPUT_CSV)
    
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}\n")
    
    # Apply imputation
    print(f"Imputing columns: {COLUMNS_TO_IMPUTE if COLUMNS_TO_IMPUTE else 'ALL'}")
    print(f"Strategy: {STRATEGY}\n")

    df_cleaned = universal_imputer(
        df=df,
        model=STRATEGY,
        columns=COLUMNS_TO_IMPUTE,
        verbose=True
    )
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"\nMissing values after: {missing_after}")
    
    # Save cleaned CSV
    df_cleaned.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned dataset to: {OUTPUT_CSV}")