"""
train_model.py
--------------
Trains a Random Forest classifier on the processed telecom churn dataset,
evaluates it, and serialises the model to models/churn_model.pkl.

Usage:
    python src/train_model.py
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "processed", "cleaned_churn_dataset.csv")
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "churn_model.pkl")
FIGURES_DIR = os.path.join("reports", "figures")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH)

# Drop redundant charge columns (perfectly correlated with minutes)
df = df.drop(columns=[
    "total_day_charge",
    "total_eve_charge",
    "total_night_charge",
    "total_intl_charge",
    "state",
    "area_code"
])

X = df.drop("churn", axis=1)
y = df["churn"]

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")

# ── Train Random Forest ────────────────────────────────────────────────────────
print("\nTraining Random Forest …")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

# ── Confusion matrix figure ────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"])
plt.title("Confusion Matrix – Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print(f"\nConfusion matrix saved → {FIGURES_DIR}/confusion_matrix.png")

# ── Feature importance figure ──────────────────────────────────────────────────
feat_imp = pd.DataFrame({
    "feature":    X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp["feature"], feat_imp["importance"], color="steelblue")
plt.title("Feature Importance – Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
plt.close()
print(f"Feature importance saved  → {FIGURES_DIR}/feature_importance.png")

# ── Serialise model ────────────────────────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved → {MODEL_PATH}")
