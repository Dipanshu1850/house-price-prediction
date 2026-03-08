"""
train.py — Train and evaluate the house price prediction model.

Usage
-----
    python train.py                          # default paths
    python train.py --train data/train.csv  # custom path
    python train.py --help
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer

from src.preprocessing import preprocess, align_columns
from src.model import StackingEnsemble, rmsle


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train House Price Prediction Model")
    parser.add_argument("--train", default="data/train.csv", help="Path to training CSV")
    parser.add_argument("--test",  default="data/test.csv",  help="Path to test CSV (optional)")
    parser.add_argument("--output-dir", default="models",    help="Directory to save model + results")
    parser.add_argument("--folds",  type=int, default=5,     help="Number of CV folds")
    parser.add_argument("--seed",   type=int, default=42,    help="Random seed")
    parser.add_argument("--no-test", action="store_true",    help="Skip test inference")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Cross-Validation
# ─────────────────────────────────────────────

def cross_validate_model(X: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 42):
    """Run k-fold CV and report RMSLE for each base model + ensemble."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    models = {
        "XGBoost": XGBRegressor(
            n_estimators=3000, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=seed
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=3000, learning_rate=0.05, max_depth=5,
            num_leaves=40, subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=seed
        ),
        "GBR": GradientBoostingRegressor(
            n_estimators=3000, learning_rate=0.05, max_depth=4,
            max_features="sqrt", loss="huber", random_state=seed
        ),
        "Ridge": Ridge(alpha=10.0),
    }

    scorer = make_scorer(
        lambda yt, yp: -rmsle(np.expm1(yt), np.expm1(yp)),
        greater_is_better=True
    )

    print("\n── Cross-Validation Results ──────────────────")
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring=scorer)
        mean_rmsle = -scores.mean()
        std_rmsle  = scores.std()
        results[name] = {"mean_rmsle": round(mean_rmsle, 5), "std": round(std_rmsle, 5)}
        print(f"  {name:<15}  RMSLE = {mean_rmsle:.5f}  (±{std_rmsle:.5f})")

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & preprocess ──────────────────────
    train_df = pd.read_csv(args.train, index_col="Id")
    test_df  = None
    if not args.no_test and Path(args.test).exists():
        test_df = pd.read_csv(args.test, index_col="Id")

    X_train, y_train, X_test = preprocess(train_df, test_df)

    if X_test is not None:
        X_test = align_columns(X_train, X_test)

    X_arr = X_train.values.astype(np.float32)
    y_arr = y_train.values.astype(np.float32)

    # ── Cross-validation ──────────────────────
    cv_results = cross_validate_model(X_arr, y_arr, n_folds=args.folds, seed=args.seed)

    # ── Train full ensemble ───────────────────
    print("\n── Training Stacking Ensemble ────────────────")
    ensemble = StackingEnsemble(n_folds=args.folds, random_state=args.seed)
    ensemble.fit(X_arr, y_arr)

    # In-sample RMSLE (for sanity check)
    train_preds = np.expm1(ensemble.predict(X_arr))
    train_rmsle = rmsle(np.expm1(y_arr), train_preds)
    print(f"  Train RMSLE: {train_rmsle:.5f}")

    # ── Save model ────────────────────────────
    model_path = output_dir / "stacking_ensemble.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": ensemble, "feature_names": X_train.columns.tolist()}, f)
    print(f"\n  Model saved → {model_path}")

    # ── Save CV results ───────────────────────
    metrics = {**cv_results, "ensemble_train_rmsle": round(train_rmsle, 5)}
    metrics_path = output_dir / "cv_results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  CV results  → {metrics_path}")

    # ── Test inference ────────────────────────
    if X_test is not None:
        print("\n── Generating Test Predictions ───────────────")
        X_test_arr = X_test.values.astype(np.float32)
        test_preds = np.expm1(ensemble.predict(X_test_arr))
        submission = pd.DataFrame({"Id": X_test.index, "SalePrice": test_preds})
        sub_path = output_dir / "submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"  Submission  → {sub_path}")
        print(f"  Predictions: min={test_preds.min():,.0f}  max={test_preds.max():,.0f}  mean={test_preds.mean():,.0f}")

    print("\n✓ Done.\n")


if __name__ == "__main__":
    main()
