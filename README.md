# 🏠 House Price Prediction Model

A production-grade machine learning pipeline for predicting residential property prices using a **stacking ensemble** of XGBoost, LightGBM, Gradient Boosting, and Ridge regression.

Built for the [Kaggle Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), but adaptable to any tabular housing dataset.

---

## 📊 Model Architecture

```
Raw CSV Data
     │
     ▼
┌─────────────────────────────┐
│  Feature Engineering        │  ← TotalSF, TotalBaths, HouseAge, QualCond, etc.
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Log Transform (skew fix)   │  ← log1p on skewed numeric features
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Missing Value Imputation   │  ← Domain-aware: median, mode, neighborhood fill
│  Ordinal Encoding           │  ← Quality/condition columns → integers
│  One-Hot Encoding           │  ← Remaining categoricals
└────────────┬────────────────┘
             │
     ┌───────┴───────┐
     │  Base Models  │  (trained via 5-fold OOF)
     ├───────────────┤
     │  XGBoost      │
     │  LightGBM     │
     │  GradBoost    │
     │  Ridge        │
     └───────┬───────┘
             │
     ┌───────▼───────┐
     │  Meta-Learner │  ElasticNet
     └───────────────┘
             │
             ▼
     log-scale predictions → expm1 → SalePrice ($)
```

**Target transformation:** `SalePrice` is log-transformed (`np.log1p`) during training; predictions are inverse-transformed (`np.expm1`) at output.

**Evaluation metric:** RMSLE (Root Mean Squared Log Error)

---

## 📁 Project Structure

```
house-price-prediction/
├── src/
│   ├── __init__.py
│   ├── model.py           # Stacking ensemble + custom transformers
│   └── preprocessing.py   # Data loading, cleaning, encoding
├── tests/
│   └── test_model.py      # Pytest unit tests
├── data/                  # Place your CSV files here
│   ├── train.csv
│   └── test.csv
├── models/                # Saved model artifacts (auto-created)
├── notebooks/             # Jupyter notebooks for EDA
├── train.py               # Training script
├── predict.py             # Inference script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

### 2. Get the Data

Download from Kaggle and place CSVs in the `data/` folder:

```
data/
├── train.csv
└── test.csv
```

Or use the [Kaggle CLI](https://github.com/Kaggle/kaggle-api):

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip -d data/
```

### 3. Train

```bash
python train.py
```

This will:
- Preprocess the data
- Run 5-fold cross-validation on each base model
- Train the full stacking ensemble
- Save the model to `models/stacking_ensemble.pkl`
- Generate `models/submission.csv` (if test.csv exists)
- Save CV results to `models/cv_results.json`

**Custom paths:**

```bash
python train.py --train data/train.csv --test data/test.csv --folds 10
```

### 4. Predict on New Data

```bash
python predict.py --input data/new_houses.csv --output my_predictions.csv
```

---

## ⚙️ Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--train` | `data/train.csv` | Training data path |
| `--test` | `data/test.csv` | Test data path |
| `--output-dir` | `models/` | Where to save model & results |
| `--folds` | `5` | Number of CV folds |
| `--seed` | `42` | Random seed |
| `--no-test` | `False` | Skip test inference |

---

## 🧪 Tests

```bash
pytest tests/ -v
```

Tests cover preprocessing correctness, feature engineering logic, metric computation, and transformer idempotency.

---

## 📈 Engineered Features

| Feature | Description |
|---------|-------------|
| `TotalSF` | Sum of basement + 1st + 2nd floor area |
| `TotalBaths` | Weighted sum of all bathroom types |
| `HouseAge` | Years between build year and sale year |
| `YearsSinceRemodel` | Years since last remodel |
| `QualCond` | OverallQual × OverallCond interaction |
| `LotRatio` | LotArea / GrLivArea |
| `HasPool` | Binary flag: pool present |
| `HasGarage` | Binary flag: garage present |
| `HasBasement` | Binary flag: basement present |

---

## 🏆 Expected Performance

| Model | CV RMSLE (5-fold) |
|-------|-------------------|
| XGBoost | ~0.1180 |
| LightGBM | ~0.1160 |
| GradientBoosting | ~0.1200 |
| Ridge | ~0.1350 |
| **Stacking Ensemble** | **~0.1120** |

*Results on the Ames Housing dataset. Actual performance may vary.*

---

## 🔧 Adapting to Your Dataset

1. Update `NUMERIC_FEATURES` and `CATEGORICAL_FEATURES` in `src/preprocessing.py`
2. Update or clear `ORDINAL_MAPS` to match your ordinal columns
3. Modify `FeatureEngineer` in `src/model.py` to add domain-specific features
4. Set `TARGET` to your target column name
