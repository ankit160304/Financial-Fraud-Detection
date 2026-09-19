import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="AI Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ==============================
# Load Model and Dataset
# ==============================

model_package = joblib.load(
    "model/fraud_model.pkl"
)

model = model_package["model"]

data = pd.read_csv(
    "dataset/transactions.csv"
)


# ==============================
# Feature List
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


# ==============================
# Title
# ==============================

st.title("💳 AI-Based Financial Fraud Detection")

st.write(
    "Machine Learning based transaction risk analysis "
    "and fraud detection system."
)

st.divider()


# ==========================================================
# DASHBOARD STATISTICS
# ==========================================================

total_transactions = len(data)

fraud_transactions = int(
    data["Fraud"].sum()
)

normal_transactions = (
    total_transactions - fraud_transactions
)

fraud_percentage = (
    fraud_transactions /
    total_transactions
) * 100


st.subheader("📊 Transaction Dashboard")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Transactions",
        total_transactions
    )


with col2:

    st.metric(
        "Fraud Transactions",
        fraud_transactions
    )


with col3:

    st.metric(
        "Normal Transactions",
        normal_transactions
    )


with col4:

    st.metric(
        "Fraud Percentage",
        f"{fraud_percentage:.2f}%"
    )


st.divider()


# ==========================================================
# TRANSACTION OVERVIEW
# ==========================================================

st.subheader("📈 Transaction Overview")


col1, col2 = st.columns(2)


# ==========================================================
# FRAUD VS NORMAL CHART
# ==========================================================

with col1:

    st.write("### Fraud vs Normal Transactions")

    chart_data = pd.DataFrame({
        "Transaction Type": [
            "Normal",
            "Fraud"
        ],

        "Count": [
            normal_transactions,
            fraud_transactions
        ]
    })


    fig, ax = plt.subplots()

    ax.bar(
        chart_data["Transaction Type"],
        chart_data["Count"]
    )

    ax.set_xlabel(
        "Transaction Type"
    )

    ax.set_ylabel(
        "Number of Transactions"
    )

    ax.set_title(
        "Fraud vs Normal Transactions"
    )

    st.pyplot(fig)

    plt.close(fig)


# ==========================================================
# TRANSACTIONS BY HOUR
# ==========================================================

with col2:

    st.write("### Transactions by Hour")

    hourly_data = (
        data.groupby("Transaction_Hour")
        .size()
        .reset_index(
            name="Transactions"
        )
    )


    fig, ax = plt.subplots()

    ax.plot(
        hourly_data["Transaction_Hour"],
        hourly_data["Transactions"],
        marker="o"
    )

    ax.set_xlabel(
        "Transaction Hour"
    )

    ax.set_ylabel(
        "Number of Transactions"
    )

    ax.set_title(
        "Transactions by Hour"
    )

    st.pyplot(fig)

    plt.close(fig)


st.divider()


# ==========================================================
# RECENT TRANSACTIONS
# ==========================================================

st.subheader("📋 Recent Transactions")


recent_transactions = data.tail(
    10
).copy()


st.markdown(
    recent_transactions.to_html(
        index=False
    ),
    unsafe_allow_html=True
)


st.divider()


# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

st.subheader("🤖 Model Performance")


accuracy = model_package["accuracy"]

precision = model_package["precision"]

recall = model_package["recall"]

f1 = model_package["f1"]

cm = model_package["confusion_matrix"]


# ==========================================================
# PERFORMANCE METRICS
# ==========================================================

metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with metric2:

    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )


with metric3:

    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )


with metric4:

    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

st.write("### Confusion Matrix")


fig, ax = plt.subplots()


ax.imshow(cm)


ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "Actual"
)

ax.set_title(
    "Confusion Matrix"
)


ax.set_xticks(
    [0, 1]
)

ax.set_yticks(
    [0, 1]
)


ax.set_xticklabels(
    ["Normal", "Fraud"]
)

ax.set_yticklabels(
    ["Normal", "Fraud"]
)


for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


st.pyplot(fig)

plt.close(fig)


st.divider()


# ==========================================================
# SINGLE TRANSACTION RISK ANALYZER
# ==========================================================

st.subheader(
    "🔍 Analyze a New Transaction"
)


col1, col2 = st.columns(2)


# ==========================================================
# TRANSACTION INPUTS
# ==========================================================

with col1:

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=5000.0
    )


    transaction_hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=12
    )


    previous_amount = st.number_input(
        "Previous Transaction Amount",
        min_value=0.0,
        value=3000.0
    )


    transaction_frequency = st.number_input(
        "Transaction Frequency",
        min_value=0,
        max_value=50,
        value=3
    )


# ==========================================================
# MORE INPUTS
# ==========================================================

with col2:

    account_age = st.number_input(
        "Account Age (Years)",
        min_value=1,
        max_value=10,
        value=5
    )


    new_device = st.selectbox(
        "Is New Device?",
        ["No", "Yes"]
    )


    international = st.selectbox(
        "Is International Transaction?",
        ["No", "Yes"]
    )


# ==========================================================
# CONVERT YES / NO TO 0 / 1
# ==========================================================

is_new_device = (
    1 if new_device == "Yes"
    else 0
)


is_international = (
    1 if international == "Yes"
    else 0
)


st.divider()


# ==========================================================
# ANALYZE SINGLE TRANSACTION
# ==========================================================

if st.button(
    "🔍 Analyze Transaction",
    use_container_width=True
):

    transaction = pd.DataFrame([{

        "Amount":
            amount,

        "Transaction_Hour":
            transaction_hour,

        "Previous_Transaction_Amount":
            previous_amount,

        "Transaction_Frequency":
            transaction_frequency,

        "Account_Age":
            account_age,

        "Is_New_Device":
            is_new_device,

        "Is_International":
            is_international
    }])


    # ==========================
    # Fraud Probability
    # ==========================

    fraud_probability = model.predict_proba(
        transaction[features]
    )[0][1]


    # ==========================
    # Risk Score
    # ==========================

    risk_score = (
        fraud_probability * 100
    )


    # ==========================
    # Risk Level
    # ==========================

    if risk_score <= 30:

        risk_level = "Low Risk"

    elif risk_score <= 70:

        risk_level = "Medium Risk"

    else:

        risk_level = "High Risk"


    # ==========================
    # Model Prediction
    # ==========================

    prediction = model.predict(
        transaction[features]
    )[0]


    st.divider()


    # ==========================
    # Results
    # ==========================

    st.subheader(
        "📊 Transaction Risk Analysis"
    )


    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )


    with result_col1:

        st.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )


    with result_col2:

        st.metric(
            "Risk Score",
            f"{risk_score:.2f} / 100"
        )


    with result_col3:

        st.metric(
            "Risk Level",
            risk_level
        )


    # ==========================
    # Risk Score Progress
    # ==========================

    st.write("### 🎯 Risk Score")


    st.progress(
        min(int(risk_score), 100),
        text=f"Risk Score: {risk_score:.2f} / 100"
    )


    # ==========================
    # Risk Message
    # ==========================

    if risk_level == "Low Risk":

        st.success(
            "🟢 Low Risk Transaction"
        )

    elif risk_level == "Medium Risk":

        st.warning(
            "🟡 Medium Risk Transaction"
        )

    else:

        st.error(
            "🔴 High Risk Transaction"
        )


    # ==========================
    # Model Prediction
    # ==========================

    if prediction == 1:

        st.error(
            "⚠️ Model Prediction: Potential Fraud"
        )

    else:

        st.success(
            "✅ Model Prediction: Normal Transaction"
        )


    # ==========================
    # Risk Factors
    # ==========================

    st.subheader(
        "🔍 Risk Factors"
    )


    risk_factors = []


    if amount > 10000:

        risk_factors.append(
            "⚠️ High transaction amount"
        )


    if transaction_hour in [
        0, 1, 2, 3, 4
    ]:

        risk_factors.append(
            "🌙 Transaction made during unusual hours"
        )


    if new_device == "Yes":

        risk_factors.append(
            "📱 New device detected"
        )


    if international == "Yes":

        risk_factors.append(
            "🌍 International transaction"
        )


    if transaction_frequency > 7:

        risk_factors.append(
            "🔄 Very high transaction frequency"
        )


    if account_age <= 1:

        risk_factors.append(
            "🆕 Very new account"
        )


    if (
        previous_amount > 0
        and amount > previous_amount * 3
    ):

        risk_factors.append(
            "💰 Transaction amount is much higher "
            "than previous transaction"
        )


    # ==========================
    # Display Risk Factors
    # ==========================

    if risk_factors:

        for factor in risk_factors:

            st.warning(factor)

    else:

        st.success(
            "✅ No major risk factors detected"
        )


st.divider()


# ==========================================================
# STEP 18
# CSV UPLOAD + BULK FRAUD DETECTION
# ==========================================================

st.subheader(
    "📂 Bulk Transaction Fraud Detection"
)

st.write(
    "Upload a CSV file containing multiple transactions "
    "to analyze them together."
)


# ==========================================================
# REQUIRED COLUMNS
# ==========================================================

st.write("### Required CSV Columns")

st.code(
    """
Amount
Transaction_Hour
Previous_Transaction_Amount
Transaction_Frequency
Account_Age
Is_New_Device
Is_International
""",
    language="text"
)


# ==========================================================
# SAMPLE CSV DOWNLOAD
# ==========================================================

sample_data = pd.DataFrame({

    "Transaction_ID": [
        "T10001",
        "T10002",
        "T10003"
    ],

    "Amount": [
        5000,
        15000,
        800
    ],

    "Transaction_Hour": [
        12,
        2,
        18
    ],

    "Previous_Transaction_Amount": [
        4000,
        2000,
        1000
    ],

    "Transaction_Frequency": [
        3,
        9,
        2
    ],

    "Account_Age": [
        5,
        1,
        7
    ],

    "Is_New_Device": [
        0,
        1,
        0
    ],

    "Is_International": [
        0,
        1,
        0
    ]
})


sample_csv = sample_data.to_csv(
    index=False
)


st.download_button(
    label="⬇️ Download Sample CSV",
    data=sample_csv,
    file_name="sample_transactions.csv",
    mime="text/csv"
)


# ==========================================================
# CSV FILE UPLOADER
# ==========================================================

uploaded_file = st.file_uploader(
    "📤 Upload Transaction CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        uploaded_data = pd.read_csv(
            uploaded_file
        )


        st.write("### Uploaded Data")

        st.markdown(
            uploaded_data.head(10).to_html(
                index=False
            ),
            unsafe_allow_html=True
        )


        # ==================================================
        # CHECK REQUIRED COLUMNS
        # ==================================================

        missing_columns = [
            column
            for column in features
            if column not in uploaded_data.columns
        ]


        if missing_columns:

            st.error(
                "❌ Missing required columns: "
                + ", ".join(missing_columns)
            )


        else:

            # ==============================================
            # BULK PREDICTION
            # ==============================================

            if st.button(
                "🚀 Analyze Uploaded Transactions",
                use_container_width=True
            ):

                analysis_data = uploaded_data.copy()


                # ==========================================
                # Fraud Probability
                # ==========================================

                probabilities = model.predict_proba(
                    analysis_data[features]
                )[:, 1]


                # ==========================================
                # Model Prediction
                # ==========================================

                predictions = model.predict(
                    analysis_data[features]
                )


                # ==========================================
                # Add Results
                # ==========================================

                analysis_data[
                    "Fraud_Probability"
                ] = (
                    probabilities * 100
                ).round(2)


                analysis_data[
                    "Risk_Score"
                ] = (
                    probabilities * 100
                ).round(2)


                analysis_data[
                    "Prediction"
                ] = predictions


                # ==========================================
                # Risk Level
                # ==========================================

                def get_risk_level(score):

                    if score <= 30:

                        return "Low Risk"

                    elif score <= 70:

                        return "Medium Risk"

                    else:

                        return "High Risk"


                analysis_data[
                    "Risk_Level"
                ] = analysis_data[
                    "Risk_Score"
                ].apply(
                    get_risk_level
                )


                # ==========================================
                # Prediction Label
                # ==========================================

                analysis_data[
                    "Prediction"
                ] = analysis_data[
                    "Prediction"
                ].map({

                    0: "Normal",

                    1: "Potential Fraud"
                })


                # ==========================================
                # BULK SUMMARY
                # ==========================================

                total_uploaded = len(
                    analysis_data
                )


                detected_fraud = (
                    analysis_data[
                        "Prediction"
                    ]
                    == "Potential Fraud"
                ).sum()


                detected_normal = (
                    total_uploaded
                    - detected_fraud
                )


                high_risk = (
                    analysis_data[
                        "Risk_Level"
                    ]
                    == "High Risk"
                ).sum()


                medium_risk = (
                    analysis_data[
                        "Risk_Level"
                    ]
                    == "Medium Risk"
                ).sum()


                low_risk = (
                    analysis_data[
                        "Risk_Level"
                    ]
                    == "Low Risk"
                ).sum()


                st.divider()


                st.subheader(
                    "📊 Bulk Analysis Results"
                )


                result1, result2, result3, result4 = (
                    st.columns(4)
                )


                with result1:

                    st.metric(
                        "Total Analyzed",
                        total_uploaded
                    )


                with result2:

                    st.metric(
                        "Potential Fraud",
                        int(detected_fraud)
                    )


                with result3:

                    st.metric(
                        "High Risk",
                        int(high_risk)
                    )


                with result4:

                    st.metric(
                        "Normal",
                        int(detected_normal)
                    )


                # ==========================================
                # RISK LEVEL SUMMARY
                # ==========================================

                st.write(
                    "### 🎯 Risk Level Summary"
                )


                risk_summary = pd.DataFrame({

                    "Risk Level": [
                        "Low Risk",
                        "Medium Risk",
                        "High Risk"
                    ],

                    "Count": [
                        low_risk,
                        medium_risk,
                        high_risk
                    ]
                })


                fig, ax = plt.subplots()


                ax.bar(
                    risk_summary["Risk Level"],
                    risk_summary["Count"]
                )


                ax.set_xlabel(
                    "Risk Level"
                )

                ax.set_ylabel(
                    "Number of Transactions"
                )

                ax.set_title(
                    "Risk Level Distribution"
                )


                st.pyplot(fig)

                plt.close(fig)


                # ==========================================
                # RESULT TABLE
                # ==========================================

                st.write(
                    "### 📋 Detailed Analysis Results"
                )


                st.markdown(
                    analysis_data.to_html(
                        index=False
                    ),
                    unsafe_allow_html=True
                )


                # ==========================================
                # DOWNLOAD RESULTS
                # ==========================================

                result_csv = analysis_data.to_csv(
                    index=False
                )


                st.download_button(
                    label="⬇️ Download Fraud Analysis Results",
                    data=result_csv,
                    file_name="fraud_analysis_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )


    except Exception as e:

        st.error(
            f"❌ Error processing CSV: {e}"
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "AI-Based Financial Fraud Detection System | "
    "Academic Project"
)