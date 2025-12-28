import os
import joblib
from preprocessing import load_data, clean_data, encode_data
from feature_engineering import add_features

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# -----------------------------
# Load & preprocess data
# -----------------------------
df = load_data(r"D:\Nivi\Python\EmployeeAttrition\Employee-Attrition - Employee-Attrition.csv")
df = clean_data(df)
df = encode_data(df)
df = add_features(df)

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Logistic Regression (baseline)
# -----------------------------
log_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("log_model", LogisticRegression(max_iter=1000))
])

log_pipeline.fit(X_train, y_train)

log_auc = roc_auc_score(
    y_test,
    log_pipeline.predict_proba(X_test)[:, 1]
)

# -----------------------------
# Random Forest (final model)
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_auc = roc_auc_score(
    y_test,
    rf_model.predict_proba(X_test)[:, 1]
)

# -----------------------------
# Compare & save best model
# -----------------------------
print(f"Logistic Regression AUC : {log_auc:.4f}")
print(f"Random Forest AUC       : {rf_auc:.4f}")

best_model = rf_model if rf_auc >= log_auc else log_pipeline

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/attrition_model.pkl")

print("✅ Best model saved successfully")
