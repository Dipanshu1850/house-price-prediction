"""
House Price Prediction Model
Core machine learning pipeline with preprocessing, training, and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Custom Transformers
# ─────────────────────────────────────────────

class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p transform to skewed numeric features."""

    def __init__(self, skew_threshold=0.75):
        self.skew_threshold = skew_threshold
        self.skewed_cols_ = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            skewness = X.apply(lambda col: col.skew())
            self.skewed_cols_ = skewness[abs(skewness) > self.skew_threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            for col in self.skewed_cols_:
                if col in X.columns:
                    X[col] = np.log1p(X[col].clip(lower=0))
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Generate domain-specific features for house pricing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Total square footage
        sf_cols = [c for c in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"] if c in df.columns]
        if sf_cols:
            df["TotalSF"] = df[sf_cols].sum(axis=1)

        # Total bathrooms
        bath_cols = {
            "FullBath": 1, "HalfBath": 0.5,
            "BsmtFullBath": 1, "BsmtHalfBath": 0.5
        }
        available = {k: v for k, v in bath_cols.items() if k in df.columns}
        if available:
            df["TotalBaths"] = sum(df[col] * weight for col, weight in available.items())

        # House age at sale
        if "YearBuilt" in df.columns and "YrSold" in df.columns:
            df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
            df["HouseAge"] = df["HouseAge"].clip(lower=0)

        # Years since remodel
        if "YearRemodAdd" in df.columns and "YrSold" in df.columns:
            df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]
            df["YearsSinceRemodel"] = df["YearsSinceRemodel"].clip(lower=0)

        # Has pool / garage / basement flags
        if "PoolArea" in df.columns:
            df["HasPool"] = (df["PoolArea"] > 0).astype(int)
        if "GarageArea" in df.columns:
            df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
        if "TotalBsmtSF" in df.columns:
            df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)

        # Quality × Condition interaction
        if "OverallQual" in df.columns and "OverallCond" in df.columns:
            df["QualCond"] = df["OverallQual"] * df["OverallCond"]

        # Lot ratio
        if "LotArea" in df.columns and "GrLivArea" in df.columns:
            df["LotRatio"] = df["LotArea"] / (df["GrLivArea"] + 1)

        return df


# ─────────────────────────────────────────────
# Stacking Ensemble
# ─────────────────────────────────────────────

class StackingEnsemble:
    """
    Two-layer stacking ensemble:
      Layer 1: XGBoost, LightGBM, GradientBoosting, Ridge
      Layer 2: ElasticNet meta-learner
    """

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.base_models_ = []
        self.meta_model_ = None
        self._base_models_config = self._build_base_models()

    def _build_base_models(self):
        return [
            ("xgb", xgb.XGBRegressor(
                n_estimators=3000, learning_rate=0.05, max_depth=4,
                min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
                gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                random_state=self.random_state, verbosity=0
            )),
            ("lgb", lgb.LGBMRegressor(
                n_estimators=3000, learning_rate=0.05, max_depth=5,
                num_leaves=40, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.5,
                random_state=self.random_state, verbose=-1
            )),
            ("gbr", GradientBoostingRegressor(
                n_estimators=3000, learning_rate=0.05, max_depth=4,
                max_features="sqrt", min_samples_leaf=15,
                min_samples_split=10, loss="huber",
                random_state=self.random_state
            )),
            ("ridge", Ridge(alpha=10.0)),
        ]

    def _fit_meta_features(self, X, y):
        """Generate out-of-fold predictions for meta-learning."""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        meta_train = np.zeros((X.shape[0], len(self._base_models_config)))

        for idx, (name, model) in enumerate(self._base_models_config):
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr = y[train_idx]
                clone = model.__class__(**model.get_params())
                clone.fit(X_tr, y_tr)
                meta_train[val_idx, idx] = clone.predict(X_val)

        return meta_train

    def fit(self, X, y):
        meta_train = self._fit_meta_features(X, y)

        # Fit each base model on full training data
        self.base_models_ = []
        for name, model in self._base_models_config:
            fitted = model.__class__(**model.get_params())
            fitted.fit(X, y)
            self.base_models_.append((name, fitted))

        # Fit meta-learner
        self.meta_model_ = ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=10000)
        self.meta_model_.fit(meta_train, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            model.predict(X) for _, model in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)


# ─────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────

def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for numeric + categorical features."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ], remainder="drop")


def build_full_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """Compose feature engineering + preprocessing into a single Pipeline."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("log_transform", LogTransformer(skew_threshold=0.75)),
        ("preprocessor", preprocessor),
    ])


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Log Error."""
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
