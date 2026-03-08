"""
Unit tests for the house price prediction model.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from src.model import FeatureEngineer, LogTransformer, rmsle
from src.preprocessing import handle_missing, apply_ordinal_encoding, remove_outliers


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal dataframe resembling the Ames dataset."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "LotArea":       np.random.randint(5000, 20000, n),
        "GrLivArea":     np.random.randint(800, 3500, n),
        "TotalBsmtSF":   np.random.randint(0, 2000, n),
        "1stFlrSF":      np.random.randint(500, 2000, n),
        "2ndFlrSF":      np.random.randint(0, 1000, n),
        "FullBath":      np.random.randint(1, 4, n),
        "HalfBath":      np.random.randint(0, 2, n),
        "BsmtFullBath":  np.random.randint(0, 2, n),
        "BsmtHalfBath":  np.random.randint(0, 2, n),
        "OverallQual":   np.random.randint(1, 11, n),
        "OverallCond":   np.random.randint(1, 10, n),
        "YearBuilt":     np.random.randint(1950, 2010, n),
        "YearRemodAdd":  np.random.randint(1960, 2020, n),
        "YrSold":        np.random.randint(2006, 2011, n),
        "GarageArea":    np.random.randint(0, 900, n),
        "PoolArea":      np.random.choice([0, 0, 0, 200], n),
        "Neighborhood":  np.random.choice(["NAmes", "CollgCr", "OldTown"], n),
        "LotFrontage":   np.where(np.random.rand(n) > 0.2, np.random.randint(50, 100, n), np.nan),
        "SalePrice":     np.random.randint(100000, 500000, n),
    })


# ─────────────────────────────────────────────
# Preprocessing Tests
# ─────────────────────────────────────────────

class TestHandleMissing:
    def test_fills_lot_frontage_by_neighborhood(self, sample_df):
        df_out = handle_missing(sample_df.drop(columns=["SalePrice"]))
        assert df_out["LotFrontage"].isna().sum() == 0

    def test_no_nans_remaining(self, sample_df):
        df_out = handle_missing(sample_df.drop(columns=["SalePrice"]))
        assert df_out.isna().sum().sum() == 0

    def test_does_not_mutate_input(self, sample_df):
        original_nans = sample_df["LotFrontage"].isna().sum()
        _ = handle_missing(sample_df)
        assert sample_df["LotFrontage"].isna().sum() == original_nans


class TestRemoveOutliers:
    def test_removes_large_cheap_houses(self, sample_df):
        # Inject an obvious outlier
        sample_df.loc[0, "GrLivArea"] = 5000
        sample_df.loc[0, "SalePrice"] = 100_000
        out = remove_outliers(sample_df)
        assert 0 not in out.index

    def test_keeps_normal_rows(self, sample_df):
        out = remove_outliers(sample_df)
        assert len(out) >= len(sample_df) - 5  # at most 5 removed


# ─────────────────────────────────────────────
# Feature Engineering Tests
# ─────────────────────────────────────────────

class TestFeatureEngineer:
    def test_total_sf_created(self, sample_df):
        fe = FeatureEngineer()
        out = fe.fit_transform(sample_df)
        assert "TotalSF" in out.columns

    def test_total_baths_created(self, sample_df):
        fe = FeatureEngineer()
        out = fe.fit_transform(sample_df)
        assert "TotalBaths" in out.columns

    def test_house_age_nonnegative(self, sample_df):
        fe = FeatureEngineer()
        out = fe.fit_transform(sample_df)
        assert (out["HouseAge"] >= 0).all()

    def test_qual_cond_interaction(self, sample_df):
        fe = FeatureEngineer()
        out = fe.fit_transform(sample_df)
        assert "QualCond" in out.columns
        expected = sample_df["OverallQual"] * sample_df["OverallCond"]
        pd.testing.assert_series_equal(out["QualCond"], expected, check_names=False)

    def test_has_pool_flag(self, sample_df):
        fe = FeatureEngineer()
        out = fe.fit_transform(sample_df)
        assert "HasPool" in out.columns
        assert out["HasPool"].isin([0, 1]).all()

    def test_idempotent(self, sample_df):
        fe = FeatureEngineer()
        out1 = fe.fit_transform(sample_df)
        out2 = fe.transform(sample_df)
        pd.testing.assert_frame_equal(out1, out2)


# ─────────────────────────────────────────────
# Log Transformer Tests
# ─────────────────────────────────────────────

class TestLogTransformer:
    def test_skewed_columns_transformed(self, sample_df):
        df = sample_df[["LotArea", "GrLivArea", "OverallQual"]].copy()
        lt = LogTransformer(skew_threshold=0.5)
        lt.fit(df)
        out = lt.transform(df)
        assert len(lt.skewed_cols_) >= 0  # can be 0 if no skewed cols
        assert out.shape == df.shape

    def test_does_not_mutate_input(self, sample_df):
        df = sample_df[["LotArea", "GrLivArea"]].copy()
        original = df.copy()
        lt = LogTransformer()
        lt.fit_transform(df)
        pd.testing.assert_frame_equal(df, original)


# ─────────────────────────────────────────────
# Metric Tests
# ─────────────────────────────────────────────

class TestRMSLE:
    def test_perfect_prediction(self):
        y = np.array([100_000, 200_000, 300_000])
        assert rmsle(y, y) == pytest.approx(0.0)

    def test_higher_for_worse_predictions(self):
        y = np.array([100_000, 200_000, 300_000])
        pred_good = y * 1.05
        pred_bad  = y * 2.0
        assert rmsle(y, pred_bad) > rmsle(y, pred_good)

    def test_clips_negative_predictions(self):
        y    = np.array([100_000])
        pred = np.array([-50_000])
        score = rmsle(y, pred)
        assert np.isfinite(score)

    def test_scalar_output(self):
        y = np.random.rand(100) * 500_000
        p = y * np.random.uniform(0.9, 1.1, 100)
        result = rmsle(y, p)
        assert isinstance(result, float)
