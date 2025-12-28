import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)

from preprocessing import load_data, clean_data, encode_data
from feature_engineering import add_features


def evaluate_model():

    # --------------------------------------------------
    # 1. Load & preprocess data
    # --------------------------------------------------
    df = load_data(
        r"D:\Nivi\Python\EmployeeAttrition\Employee-Attrition - Employee-Attrition.csv"
    )

    df = clean_data(df)
    df = encode_data(df)
    df = add_features(df)

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    # --------------------------------------------------
    # 2. Train-test split (NO data leakage)
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # 3. Load trained model
    # --------------------------------------------------
    model_path = r"D:\Nivi\Python\EmployeeAttrition\models\attrition_model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "❌ Model file not found. Run model_training.py first."
        )

    model = joblib.load(model_path)

    # --------------------------------------------------
    # 4. Predictions
    # --------------------------------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # 5. Metrics
    # --------------------------------------------------
    print("\n📊 MODEL EVALUATION RESULTS")
    print("-" * 40)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))
    print("AUC-ROC  :", roc_auc_score(y_test, y_prob))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # --------------------------------------------------
    # 6. Confusion Matrix
    # --------------------------------------------------
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # --------------------------------------------------
    # 7. ROC Curve
    # --------------------------------------------------
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve")
    plt.show()

    # --------------------------------------------------
    # 8. Feature Importance (only for Random Forest)
    # --------------------------------------------------
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        importances.head(10).plot(kind="barh")
        plt.title("Top 10 Feature Importances")
        plt.gca().invert_yaxis()
        plt.show()
    else:
        print("ℹ️ Feature importance not available for Logistic Regression model.")


if __name__ == "__main__":
    evaluate_model()
