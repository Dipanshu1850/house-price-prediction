"""
predict.py — Run inference with a saved model on new data.

Usage
-----
    python predict.py --input data/test.csv
    python predict.py --input data/test.csv --model models/stacking_ensemble.pkl
    python predict.py --help
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import handle_missing, apply_ordinal_encoding, encode_categoricals


def parse_args():
    parser = argparse.ArgumentParser(description="House Price Prediction — Inference")
    parser.add_argument("--input",  required=True,                       help="Path to input CSV")
    parser.add_argument("--model",  default="models/stacking_ensemble.pkl", help="Saved model pickle")
    parser.add_argument("--output", default="predictions.csv",           help="Output CSV path")
    return parser.parse_args()


def load_model(path: str):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_names"]


def prepare_features(df: pd.DataFrame, feature_names: list) -> np.ndarray:
    """Clean and align input data to match training feature set."""
    df = handle_missing(df)
    df = apply_ordinal_encoding(df)
    df = encode_categoricals(df)

    # Align columns
    missing = set(feature_names) - set(df.columns)
    for col in missing:
        df[col] = 0
    df = df[feature_names]

    return df.values.astype(np.float32)


def main():
    args = parse_args()

    # Load model
    print(f"Loading model from: {args.model}")
    model, feature_names = load_model(args.model)

    # Load input
    df = pd.read_csv(args.input)
    ids = df["Id"] if "Id" in df.columns else pd.RangeIndex(len(df))
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    if "SalePrice" in df.columns:
        df = df.drop(columns=["SalePrice"])

    # Preprocess & predict
    print(f"Preparing {len(df)} samples...")
    X = prepare_features(df, feature_names)
    preds = np.expm1(model.predict(X))

    # Save
    out = pd.DataFrame({"Id": ids, "SalePrice": np.round(preds, 2)})
    out.to_csv(args.output, index=False)
    print(f"Predictions saved → {args.output}")
    print(f"  min={preds.min():,.0f}  max={preds.max():,.0f}  mean={preds.mean():,.0f}")


if __name__ == "__main__":
    main()
