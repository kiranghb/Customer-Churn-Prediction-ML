
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = os.path.join("models", "customer_churn_model.pkl")
DATA_PATH = "Telco-Customer-Churn.csv"

st.title("📊 Customer Churn Prediction Dashboard")
st.write(
    "Analyze telecom customer churn and estimate customer risk "
    "using a trained Random Forest machine learning model."
)

# 2. LOAD MODEL AND DATASET
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    data = data.dropna(subset=["TotalCharges"]).copy()
    return data


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.stop()

try:
    model = load_model()
    df = load_data()

except Exception as e:
    st.error("Could not load the model or dataset.")
    st.code(str(e))
    st.info(
        "Ensure both files are in the project folder and that "
        "your scikit-learn version is compatible with the "
        "version used to train the model."
    )
    st.stop()

# 3. CUSTOMER OVERVIEW - KPI CARDS
st.header("📈 Customer Overview")

total_customers = len(df)
churn_count = (df["Churn"] == "Yes").sum()

churn_rate = (
    churn_count / total_customers * 100
    if total_customers > 0 else 0
)

avg_monthly_charges = df["MonthlyCharges"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Customers Churned",
    f"{churn_count:,}"
)

col3.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%"
)

col4.metric(
    "Avg. Monthly Charges",
    f"${avg_monthly_charges:.2f}"
)

st.divider()

# 4. CHURN ANALYTICS
st.header("📊 Churn Analytics")

chart_col1, chart_col2 = st.columns(2)

# Churn distribution pie chart
with chart_col1:
    churn_counts = df["Churn"].value_counts()

    fig1, ax1 = plt.subplots(figsize=(6, 4))

    ax1.pie(
        churn_counts.values,
        labels=churn_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax1.set_title("Churn Distribution")
    ax1.axis("equal")

    st.pyplot(fig1)
    plt.close(fig1)

# Churn by contract type
with chart_col2:
    fig2, ax2 = plt.subplots(figsize=(7, 4))

    sns.countplot(
        data=df,
        x="Contract",
        hue="Churn",
        ax=ax2
    )

    ax2.set_title("Churn by Contract Type")
    ax2.set_xlabel("Contract Type")
    ax2.set_ylabel("Number of Customers")
    ax2.tick_params(axis="x", rotation=15)

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# 5. CHURN RATE BY CUSTOMER TENURE
st.subheader("📉 Churn Rate by Customer Tenure")

tenure_df = df.copy()

tenure_df["Tenure Group"] = pd.cut(
    tenure_df["tenure"],
    bins=[-1, 12, 24, 48, 60, 72],
    labels=[
        "0–12 months",
        "13–24 months",
        "25–48 months",
        "49–60 months",
        "61–72 months"
    ],
    include_lowest=True
)

tenure_churn = (
    tenure_df.groupby(
        "Tenure Group",
        observed=False
    )["Churn"]
    .apply(lambda values: (values == "Yes").mean() * 100)
    .reset_index(name="Churn Rate")
)

fig3, ax3 = plt.subplots(figsize=(10, 4))

sns.barplot(
    data=tenure_churn,
    x="Tenure Group",
    y="Churn Rate",
    ax=ax3
)

ax3.set_title("Churn Rate Across Tenure Groups")
ax3.set_xlabel("Customer Tenure")
ax3.set_ylabel("Churn Rate (%)")
ax3.set_ylim(0, 100)

fig3.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

st.caption(
    "Charts are calculated from the cleaned customer dataset."
)

st.divider()

# 6. CUSTOMER PREDICTION FORM
st.header("🔍 Predict Churn for a Customer")
st.write("Enter the customer's details below.")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    # Customer demographics and basic services
    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Has Partner?",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Has Dependents?",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    # Internet and additional services
    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    # Streaming, contract, and billing
    with col3:
        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )

# 7. PREDICTION AND PROBABILITY RESULTS
if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    try:
        prediction = model.predict(customer)[0]

        if not hasattr(model, "predict_proba"):
            st.error(
                "The loaded model does not support probability estimates."
            )
            st.stop()

        probabilities = model.predict_proba(customer)[0]
        classes = list(model.classes_)

        # Identify the churn class from the model's actual labels.
        churn_class = next(
            (
                cls for cls in classes
                if str(cls).strip().lower()
                in ("1", "yes", "true")
            ),
            None
        )

        if churn_class is None:
            st.error(
                "Could not identify the churn class. "
                f"Model classes: {classes}"
            )
            st.stop()

        churn_index = classes.index(churn_class)

        churn_probability = float(
            probabilities[churn_index]
        )

        retention_probability = 1.0 - churn_probability

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Estimated Churn Probability",
            f"{churn_probability:.1%}"
        )

        result_col2.metric(
            "Estimated Retention Probability",
            f"{retention_probability:.1%}"
        )

        st.progress(churn_probability)

        # Interpret the predicted label
        prediction_text = str(prediction).strip().lower()

        if prediction_text in ("1", "yes", "true"):
            st.warning(
                "⚠️ Prediction: Customer is at risk of churn."
            )
            st.write(
                "Consider reviewing this customer's service "
                "experience and available retention options."
            )

        else:
            st.success(
                "✅ Prediction: Customer is predicted not to churn."
            )
            st.write(
                "The model predicts that this customer is more "
                "likely to remain with the service."
            )

        st.caption(
            "These probabilities are model estimates, not "
            "guarantees of future customer behavior."
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
        st.info(
            "Check that your prediction columns and data types "
            "match those used during model training."
        )

# 8. FOOTER
st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning Project"
)
