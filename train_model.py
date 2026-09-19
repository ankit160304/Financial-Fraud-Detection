import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE


# ==============================
# 1. Load Dataset
# ==============================

data = pd.read_csv("dataset/transactions.csv")

print("Dataset loaded successfully!")
print("Total records:", len(data))


# ==============================
# 2. Select Features
# ==============================

features = [
    "Amount",
    "Transaction_Hour",
    "Previous_Transaction_Amount",
    "Transaction_Frequency",
    "Account_Age",
    "Is_New_Device",
    "Is_International"
]

X = data[features]
y = data["Fraud"]


# ==============================
# 3. Split Dataset
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records before SMOTE:", len(X_train))
print("Testing records:", len(X_test))


# ==============================
# 4. Apply SMOTE
# ==============================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train,
    y_train
)

print("Training records after SMOTE:", len(X_train_balanced))
print("Fraud cases after SMOTE:", sum(y_train_balanced == 1))
print("Normal cases after SMOTE:", sum(y_train_balanced == 0))


# ==============================
# 5. Create Random Forest Model
# ==============================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# ==============================
# 6. Train Model
# ==============================

print("\nTraining model...")

model.fit(
    X_train_balanced,
    y_train_balanced
)

print("Model training completed!")


# ==============================
# 7. Make Predictions
# ==============================

y_pred = model.predict(X_test)


# ==============================
# 8. Evaluate Model
# ==============================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ==============================
# 9. Display Results
# ==============================

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(cm)


# ==============================
# 10. Save Model + Metrics
# ==============================

model_package = {
    "model": model,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "confusion_matrix": cm
}

joblib.dump(
    model_package,
    "model/fraud_model.pkl"
)

print("\nModel and performance metrics saved successfully!")
print("Location: model/fraud_model.pkl")