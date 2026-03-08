"""
Data loading and preprocessing utilities for house price prediction.
Works with the Kaggle Ames Housing dataset by default, but is adaptable
to any tabular dataset with similar structure.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


# ─────────────────────────────────────────────
# Column Definitions (Ames Housing Dataset)
# ─────────────────────────────────────────────

NUMERIC_FEATURES = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
    "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
    "YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold", "MoSold",
    "OverallQual", "OverallCond",
]

CATEGORICAL_FEATURES = [
    "MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour",
    "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1",
    "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
    "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond",
    "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical",
    "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence",
    "MiscFeature", "SaleType", "SaleCondition",
]

TARGET = "SalePrice"
ID_COL = "Id"

# Ordinal mappings for quality/condition columns
ORDINAL_MAPS = {
    "ExterQual":   {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond":   {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual":    {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond":    {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtExposure":{"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "HeatingQC":   {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageFinish":{"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "GarageQual":  {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond":  {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "PoolQC":      {"NA": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "Fence":       {"NA": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
}


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

def load_data(
    train_path: str = "data/train.csv",
    test_path: Optional[str] = "data/test.csv",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load train and (optionally) test CSVs."""
    train = pd.read_csv(train_path, index_col=ID_COL)
    test = pd.read_csv(test_path, index_col=ID_COL) if test_path else None
    print(f"Train shape: {train.shape}")
    if test is not None:
        print(f"Test  shape: {test.shape}")
    return train, test


# ─────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values using domain knowledge:
    - "NA" sentinel → 0 or "None" depending on dtype
    - LotFrontage → median per Neighborhood
    - GarageYrBlt → YearBuilt
    """
    df = df.copy()

    # Features where NaN means "None/not present"
    none_cols = [
        "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType",
        "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence",
        "MiscFeature",
    ]
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("NA")

    zero_cols = [
        "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
        "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "GarageArea",
        "GarageCars",
    ]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # LotFrontage: fill with neighborhood median
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )

    # GarageYrBlt: use YearBuilt as proxy
    if "GarageYrBlt" in df.columns and "YearBuilt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

    # Remaining numeric NaN → median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Remaining categorical NaN → mode
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def apply_ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ordered categorical columns to integers."""
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode remaining categorical columns."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def remove_outliers(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Remove known outliers (based on Ames Housing data analysis)."""
    if "GrLivArea" in df.columns and target in df.columns:
        mask = ~((df["GrLivArea"] > 4000) & (df[target] < 300_000))
        removed = (~mask).sum()
        if removed:
            print(f"Removed {removed} outlier(s) based on GrLivArea vs {target}.")
        df = df[mask]
    return df


# ─────────────────────────────────────────────
# Full Preprocessing Pipeline
# ─────────────────────────────────────────────

def preprocess(
    train: pd.DataFrame,
    test: Optional[pd.DataFrame] = None,
    remove_train_outliers: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
    """
    Full preprocessing: clean → encode → align train/test columns.

    Returns
    -------
    X_train : pd.DataFrame
    y_train : pd.Series  (log1p-transformed SalePrice)
    X_test  : pd.DataFrame or None
    """
    # Separate target
    y_train = np.log1p(train[TARGET])
    train = train.drop(columns=[TARGET])

    # Optionally remove outliers before encoding
    if remove_train_outliers:
        combined_for_outlier = train.copy()
        combined_for_outlier[TARGET] = np.expm1(y_train)
        combined_for_outlier = remove_outliers(combined_for_outlier)
        y_train = y_train[combined_for_outlier.index]
        train = combined_for_outlier.drop(columns=[TARGET])

    # Combine for consistent encoding
    n_train = len(train)
    combined = pd.concat([train, test], axis=0) if test is not None else train.copy()

    combined = handle_missing(combined)
    combined = apply_ordinal_encoding(combined)
    combined = encode_categoricals(combined)

    X_train = combined.iloc[:n_train]
    X_test  = combined.iloc[n_train:] if test is not None else None

    print(f"Processed X_train: {X_train.shape}, y_train: {y_train.shape}")
    if X_test is not None:
        print(f"Processed X_test : {X_test.shape}")

    return X_train, y_train, X_test


def align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Ensure X_test has the same columns as X_train (fill missing with 0)."""
    missing = set(X_train.columns) - set(X_test.columns)
    extra   = set(X_test.columns) - set(X_train.columns)
    if missing:
        print(f"Adding {len(missing)} missing columns to X_test.")
        for col in missing:
            X_test[col] = 0
    if extra:
        print(f"Dropping {len(extra)} extra columns from X_test.")
        X_test = X_test.drop(columns=list(extra))
    return X_test[X_train.columns]
