"""
predict.py
----------
Loads the saved churn model and generates churn predictions for new data.

Usage:
    # predict from a CSV file
    python src/predict.py --input data/raw/new_customers.csv --output reports/predictions.csv

    # quick sanity-check on the test split
    python src/predict.py --test
"""

import os
import pickle
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "churn_model.pkl")
DATA_PATH  = os.path.join("data", "processed", "cleaned_churn_dataset.csv")

# Columns dropped during feature engineering
DROP_COLS = [
    "total_day_charge",
    "total_eve_charge",
    "total_night_charge",
    "total_intl_charge",
    "state",
    "area_code",
]


def load_model(path: str = MODEL_PATH):
    """Deserialise the trained model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. Run `python src/train_model.py` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature-engineering steps used during training."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if "churn" in df.columns:
        cols_to_drop.append("churn")
    return df.drop(columns=cols_to_drop)


def predict(input_path: str, output_path: str | None = None) -> pd.DataFrame:
    """Load new data, run inference, and optionally save results."""
    model = load_model()

    raw = pd.read_csv(input_path)
    X   = prepare_features(raw)

    preds       = model.predict(X)
    probs       = model.predict_proba(X)[:, 1]  # P(churn=1)

    results = raw.copy()
    results["churn_prediction"] = preds
    results["churn_probability"] = probs.round(4)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Predictions saved → {output_path}")
    else:
        print(results[["churn_prediction", "churn_probability"]].to_string())

    return results


def run_test_mode():
    """Quick sanity-check: re-score the held-out test split."""
    from sklearn.metrics import accuracy_score, classification_report

    model = load_model()
    df    = pd.read_csv(DATA_PATH)
    X     = prepare_features(df)
    y     = df["churn"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    print(f"Test Accuracy : {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn Predictor")
    parser.add_argument("--input",  type=str, help="Path to input CSV with customer data")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save predictions CSV (optional)")
    parser.add_argument("--test",   action="store_true",
                        help="Run evaluation on the held-out test split")
    args = parser.parse_args()

    if args.test:
        run_test_mode()
    elif args.input:
        predict(args.input, args.output)
    else:
        parser.print_help()
