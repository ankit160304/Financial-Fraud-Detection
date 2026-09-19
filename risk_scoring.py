
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/fraud_model.pkl")

# Example transaction
transaction = pd.DataFrame([{
    "Amount": 15000,
    "Transaction_Hour": 2,
    "Previous_Transaction_Amount": 2000,
    "Transaction_Frequency": 9,
    "Account_Age": 1,
    "Is_New_Device": 1,
    "Is_International": 1
}])

# Get fraud probability
fraud_probability = model.predict_proba(transaction)[0][1]

# Convert probability into risk score (0-100)
risk_score = fraud_probability * 100

# Determine risk level
if risk_score <= 30:
    risk_level = "Low Risk"
elif risk_score <= 70:
    risk_level = "Medium Risk"
else:
    risk_level = "High Risk"

# Display results
print("----- Transaction Risk Analysis -----")
print("Fraud Probability:", round(fraud_probability * 100, 2), "%")
print("Risk Score:", round(risk_score, 2), "/ 100")
print("Risk Level:", risk_level)

