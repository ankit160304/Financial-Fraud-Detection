import pandas as pd
import numpy as np

np.random.seed(42)

# Number of transactions
n = 10000

# Generate transaction data
data = pd.DataFrame({
    "Transaction_ID": [f"T{i:05d}" for i in range(1, n + 1)],

    "Amount": np.random.exponential(scale=3000, size=n).round(2),

    "Transaction_Hour": np.random.randint(0, 24, n),

    "Previous_Transaction_Amount":
        np.random.exponential(scale=2500, size=n).round(2),

    "Transaction_Frequency":
        np.random.poisson(lam=3, size=n),

    "Account_Age":
        np.random.randint(1, 11, n),

    "Is_New_Device":
        np.random.choice([0, 1], n, p=[0.8, 0.2]),

    "Is_International":
        np.random.choice([0, 1], n, p=[0.85, 0.15])
})


# ==========================================
# Create stronger fraud patterns
# ==========================================

fraud_probability = np.full(n, 0.01)

# High transaction amount
fraud_probability += np.where(
    data["Amount"] > 10000, 0.25, 0
)

# Very unusual transaction time
fraud_probability += np.where(
    data["Transaction_Hour"].isin([0, 1, 2, 3, 4]), 0.25, 0
)

# New device
fraud_probability += np.where(
    data["Is_New_Device"] == 1, 0.20, 0
)

# International transaction
fraud_probability += np.where(
    data["Is_International"] == 1, 0.15, 0
)

# Very high transaction frequency
fraud_probability += np.where(
    data["Transaction_Frequency"] > 7, 0.20, 0
)

# Very new account
fraud_probability += np.where(
    data["Account_Age"] <= 1, 0.15, 0
)

# Large difference from previous transaction
fraud_probability += np.where(
    data["Amount"] > data["Previous_Transaction_Amount"] * 3,
    0.20,
    0
)

# Keep probability between 0 and 0.95
fraud_probability = np.clip(
    fraud_probability,
    0,
    0.95
)


# Generate fraud labels
data["Fraud"] = np.random.binomial(
    1,
    fraud_probability
)


# Save dataset
data.to_csv(
    "dataset/transactions.csv",
    index=False
)


print("Dataset generated successfully!")
print("Total transactions:", len(data))
print("Fraud transactions:", data["Fraud"].sum())
print(
    "Normal transactions:",
    (data["Fraud"] == 0).sum()
)
print("File saved at: dataset/transactions.csv")