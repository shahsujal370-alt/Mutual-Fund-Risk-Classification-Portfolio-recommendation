import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mutual Fund Risk Analyzer", page_icon="📊", layout="wide")

MODEL_PATH = "risk_classifier.joblib"
DATA_PATH = "mutual_funds_clean.csv"
FEATURE_LABELS = {
    "expense_ratio": "Expense ratio (%)",
    "fund_size_cr": "Fund size (₹ crore)",
    "fund_age_yr": "Fund age (years)",
    "sortino": "Sortino ratio",
    "alpha": "Alpha",
    "sd": "Standard deviation (volatility)",
    "beta": "Beta",
    "sharpe": "Sharpe ratio",
    "returns_1yr": "1-year return (%)",
    "returns_3yr": "3-year return (%)",
    "returns_5yr": "5-year return (%)",
    "rating": "Fund rating (0–5)",
}

@st.cache_resource
def load_assets():
    bundle = joblib.load(MODEL_PATH)
    funds = pd.read_csv(DATA_PATH)
    return bundle, funds

try:
    bundle, funds = load_assets()
except Exception as exc:
    st.error(f"Could not load the model or fund data: {exc}")
    st.stop()

pipeline = bundle["pipeline"]
features = bundle["features"]

st.title("📊 Mutual Fund Risk Classification & Portfolio Analytics")
st.caption("Explore fund characteristics, estimate the dataset-defined risk category, and screen funds by risk profile.")

with st.sidebar:
    st.header("About this app")
    st.write("Enter a fund's historical and risk-related metrics to classify it as Low, Medium, or High risk.")
    st.info("This is an educational analytics project, not investment advice. Model predictions reproduce categories learned from the supplied dataset; they do not forecast future losses or guarantee performance.")

# Overview
m1, m2, m3, m4 = st.columns(4)
m1.metric("Funds in dataset", f"{len(funds):,}")
m2.metric("Average 3-year return", f"{pd.to_numeric(funds['returns_3yr'], errors='coerce').mean():.2f}%")
m3.metric("Average expense ratio", f"{pd.to_numeric(funds['expense_ratio'], errors='coerce').mean():.2f}%")
m4.metric("Average fund rating", f"{pd.to_numeric(funds['rating'], errors='coerce').mean():.2f}/5")

st.divider()
st.subheader("1. Classify a mutual fund")
st.write("Enter values below. The initial values are medians from the project dataset; replace them with the fund's actual figures.")

# Dataset medians provide sensible starting values for the form.
medians = funds[features].apply(pd.to_numeric, errors="coerce").median()
with st.form("risk_form"):
    st.markdown("**Fund metrics**")
    cols = st.columns(3)
    values = {}
    for i, feature in enumerate(features):
        with cols[i % 3]:
            default = float(medians.get(feature, 0.0))
            if feature == "rating":
                values[feature] = st.number_input(FEATURE_LABELS[feature], min_value=0.0, max_value=5.0, value=float(np.clip(default, 0, 5)), step=1.0)
            elif feature in ("expense_ratio", "fund_size_cr", "fund_age_yr", "sd"):
                values[feature] = st.number_input(FEATURE_LABELS[feature], min_value=0.0, value=max(0.0, default), step=0.1, format="%.3f")
            else:
                values[feature] = st.number_input(FEATURE_LABELS[feature], value=default, step=0.1, format="%.3f")
    submitted = st.form_submit_button("Predict risk category", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([[values[f] for f in features]], columns=features)
    raw_prediction = pipeline.predict(input_df)[0]
    # The training pipeline encodes text labels as integers; decode them for display.
    prediction = str(bundle["label_encoder"].inverse_transform([int(raw_prediction)])[0]) if isinstance(raw_prediction, (int, np.integer)) else str(raw_prediction)
    probabilities = pipeline.predict_proba(input_df)[0] if hasattr(pipeline, "predict_proba") else None
    st.session_state["prediction"] = prediction
    st.session_state["input_values"] = values
    st.session_state["probabilities"] = dict(zip(bundle.get("classes", []), probabilities)) if probabilities is not None else None

if "prediction" in st.session_state:
    risk = st.session_state["prediction"]
    st.subheader("Prediction result")
    if risk == "Low":
        st.success(f"Predicted risk category: {risk}")
    elif risk == "Medium":
        st.warning(f"Predicted risk category: {risk}")
    else:
        st.error(f"Predicted risk category: {risk}")
    probs = st.session_state.get("probabilities")
    if probs:
        probability_df = pd.DataFrame({"Risk category": list(probs.keys()), "Model probability": list(probs.values())}).set_index("Risk category")
        st.write("Model probability by category")
        st.bar_chart(probability_df)
        st.caption("Probabilities describe the model's classification confidence, not the probability of an investment loss.")

st.divider()
st.subheader("2. Explore funds and recommendations")
all_risks = ["Low", "Medium", "High"]
selected_risk = st.selectbox("Choose a risk category to screen funds", all_risks, index=all_risks.index(st.session_state.get("prediction", "Medium")) if st.session_state.get("prediction", "Medium") in all_risks else 1)

# The project uses its documented screening heuristic: 70% percentile rank of 3Y return + 30% percentile rank of rating.
view = funds[funds["risk_category"].astype(str).str.title() == selected_risk].copy()
if "recommendation_score" in view.columns:
    view["recommendation_score"] = pd.to_numeric(view["recommendation_score"], errors="coerce")
    view = view.sort_values("recommendation_score", ascending=False)
show_cols = [c for c in ["scheme_name", "amc_name", "category", "rating", "returns_1yr", "returns_3yr", "returns_5yr", "sd", "sharpe", "recommendation_score"] if c in view.columns]
st.write(f"Showing {len(view)} funds labelled **{selected_risk} risk** in the project dataset.")
st.dataframe(view[show_cols].head(10), use_container_width=True, hide_index=True)

with st.expander("View risk category distribution"):
    counts = funds["risk_category"].astype(str).str.title().value_counts().reindex(all_risks, fill_value=0)
    st.bar_chart(counts.rename("Number of funds"))

with st.expander("Dataset and methodology notes"):
    st.write("The project dataset contains 814 mutual funds. Risk labels are mapped from the source risk_level field: 1–2 = Low, 3–4 = Medium, and 5–6 = High.")
    st.write("The portfolio screening score is a simple historical ranking: 70% percentile rank of 3-year returns and 30% percentile rank of fund rating. Past returns are not a reliable indicator of future performance.")
    st.write("The reported 88.96% test accuracy is the previous project run's result and may vary if the model is retrained or dependencies change.")

st.caption("Built with Python, XGBoost, scikit-learn, pandas and Streamlit.")
