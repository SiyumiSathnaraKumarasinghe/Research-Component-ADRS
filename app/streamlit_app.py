from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="FMD Early Warning System — Sri Lanka",
    page_icon="cow",
    layout="wide",
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "FMD_model_ready_main refined_final_dataset.csv"
MODEL_DIR = ROOT_DIR / "models"
STAGE1_SHAP_PATH = ROOT_DIR / "data" / "processed" / "stage1_shap_values.csv"
STAGE2_SHAP_PATH = ROOT_DIR / "data" / "processed" / "stage2_shap_values.csv"

DISTRICTS = sorted(
    [
        "Ampara",
        "Anuradhapura",
        "Badulla",
        "Batticaloa",
        "Colombo",
        "Galle",
        "Gampaha",
        "Hambantota",
        "Jaffna",
        "Kalutara",
        "Kandy",
        "Kegalle",
        "Kilinochchi",
        "Kurunegala",
        "Mannar",
        "Matale",
        "Matara",
        "Monaragala",
        "Mullaitivu",
        "Nuwara Eliya",
        "Polonnaruwa",
        "Puttalam",
        "Ratnapura",
        "Trincomalee",
        "Vavuniya",
    ]
)

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_shap_values() -> tuple[pd.DataFrame, pd.DataFrame]:
    stage1 = pd.read_csv(STAGE1_SHAP_PATH)
    stage2 = pd.read_csv(STAGE2_SHAP_PATH)
    return stage1, stage2


@st.cache_resource
def load_models() -> dict:
    return {
        "stage1_model": joblib.load(MODEL_DIR / "stage1_lr_model.pkl"),
        "stage1_scaler": joblib.load(MODEL_DIR / "stage1_scaler.pkl"),
        "stage1_features": joblib.load(MODEL_DIR / "stage1_feature_cols.pkl"),
        "stage2_model": joblib.load(MODEL_DIR / "stage2_rf_model.pkl"),
        "stage2_encoder": joblib.load(MODEL_DIR / "stage2_label_encoder.pkl"),
        "stage2_features": joblib.load(MODEL_DIR / "stage2_feature_cols.pkl"),
    }


def get_feature_row(
    df: pd.DataFrame,
    district: str,
    month_num: int,
    year: int,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, str]:
    exact = df[
        (df["district"] == district)
        & (df["month_num"] == month_num)
        & (df["year"] == year)
    ]
    if not exact.empty:
        return exact.iloc[[0]], "Exact match found"

    district_month = df[(df["district"] == district) & (df["month_num"] == month_num)]
    if not district_month.empty:
        latest = district_month.sort_values("year", ascending=False).iloc[[0]]
        latest_year = int(latest["year"].iloc[0])
        return latest, f"No exact year match. Using latest available year: {latest_year}"

    district_rows = df[df["district"] == district]
    if not district_rows.empty:
        medians = district_rows[feature_cols].median(numeric_only=True)
        medians = medians.reindex(feature_cols).fillna(0.0)
        return pd.DataFrame([medians], columns=feature_cols), "No month-level record found. Using district medians"

    global_medians = df[feature_cols].median(numeric_only=True)
    global_medians = global_medians.reindex(feature_cols).fillna(0.0)
    return pd.DataFrame([global_medians], columns=feature_cols), "District not found in data. Using global medians"


def build_top_shap_chart(shap_df: pd.DataFrame, title: str):
    top = shap_df.sort_values("mean_abs_shap", ascending=False).head(8).copy()
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#E8593C")
    ax.set_title(title)
    ax.set_xlabel("Mean |SHAP| Value")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig


def decode_severity(encoder, pred_value: int) -> str:
    try:
        decoded = encoder.inverse_transform([int(pred_value)])[0]
        return str(decoded).upper()
    except Exception:
        mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        return mapping.get(int(pred_value), "UNKNOWN")


def main() -> None:
    st.sidebar.title("FMD Risk Prediction")
    st.sidebar.markdown("Select a district and time period to predict outbreak risk.")

    selected_district = st.sidebar.selectbox("District", DISTRICTS)
    selected_month_name = st.sidebar.selectbox("Month", MONTH_NAMES)
    month_num = MONTH_NAMES.index(selected_month_name) + 1
    selected_year = st.sidebar.slider("Year", min_value=2017, max_value=2024, value=2024, step=1)

    predict_clicked = st.sidebar.button("Predict FMD Risk", use_container_width=True, type="primary")

    st.title("FMD Early Warning System")
    st.markdown("Climate-Informed Seasonal Disease Forecasting for Sri Lanka")
    st.divider()

    try:
        df = load_data()
        models = load_models()
        stage1_shap_df, stage2_shap_df = load_shap_values()
    except Exception as exc:
        st.error(f"Failed to load required files: {exc}")
        return

    if not predict_clicked:
        st.info("Choose district/month/year from the sidebar and click Predict FMD Risk.")
        return

    stage1_features = list(models["stage1_features"])
    stage2_features = list(models["stage2_features"])

    feature_row, fallback_msg = get_feature_row(
        df=df,
        district=selected_district,
        month_num=month_num,
        year=selected_year,
        feature_cols=stage1_features,
    )

    if fallback_msg != "Exact match found":
        st.info(fallback_msg)

    for col in stage1_features:
        if col not in feature_row.columns:
            feature_row[col] = 0.0

    x_stage1 = feature_row[stage1_features].fillna(0.0).astype(float)
    x_stage1_scaled = models["stage1_scaler"].transform(x_stage1)
    probability = float(models["stage1_model"].predict_proba(x_stage1_scaled)[:, 1][0])

    if probability >= 0.60:
        risk_level = "HIGH"
    elif probability >= 0.35:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    severity = "LOW"
    if probability >= 0.35:
        for col in stage2_features:
            if col not in feature_row.columns:
                feature_row[col] = 0.0
        x_stage2 = feature_row[stage2_features].fillna(0.0).astype(float)
        severity_pred = int(models["stage2_model"].predict(x_stage2)[0])
        severity = decode_severity(models["stage2_encoder"], severity_pred)

    top_meta_col1, top_meta_col2, top_meta_col3 = st.columns(3)
    top_meta_col1.metric("Selected District", selected_district)
    top_meta_col2.metric("Selected Month/Year", f"{selected_month_name} {selected_year}")
    top_meta_col3.metric("Data Source", "DAPH + CHIRPS + NASA POWER")

    st.subheader("Stage 1 — Outbreak Risk Prediction")
    row2_left, row2_right = st.columns([1, 1])

    with row2_left:
        st.metric("Outbreak Probability", f"{probability * 100:.1f}%")
        st.progress(probability)

        if risk_level == "HIGH":
            st.error("🔴 HIGH RISK — Outbreak Likely")
        elif risk_level == "MEDIUM":
            st.warning("🟠 MEDIUM RISK — Elevated Risk")
        else:
            st.success("🟢 LOW RISK — Routine Monitoring")

    with row2_right:
        fig1 = build_top_shap_chart(stage1_shap_df, "Top Climate Risk Drivers")
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    if probability >= 0.35:
        st.subheader("Stage 2 — Severity Prediction")
        row3_left, row3_right = st.columns([1, 1])

        with row3_left:
            st.metric("Predicted Severity", severity)
            if severity == "LOW":
                st.success("🟢 LOW Severity")
                st.write("Minor outbreak expected. Standard monitoring protocols apply.")
            elif severity == "MEDIUM":
                st.warning("🟠 MEDIUM Severity")
                st.write("Moderate outbreak. Targeted vaccination and surveillance recommended.")
            else:
                st.error("🔴 HIGH Severity")
                st.write("Severe outbreak expected. Emergency response protocols required.")

        with row3_right:
            fig2 = build_top_shap_chart(stage2_shap_df, "Top Severity Drivers")
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

    st.subheader("Recommendation")

    if risk_level == "HIGH" and severity == "HIGH":
        st.error(
            """
EMERGENCY RESPONSE REQUIRED
- Immediately notify DAPH Animal Health Division
- Activate emergency vaccination campaign
- Impose movement restrictions on livestock
- Deploy rapid response veterinary teams
- Isolate affected farms within 24 hours
"""
        )
    elif risk_level == "HIGH" and severity == "MEDIUM":
        st.warning(
            """
TARGETED RESPONSE REQUIRED
- Alert district veterinary surgeons
- Begin targeted vaccination in high-risk areas
- Increase farm surveillance frequency
- Prepare movement restriction protocols
"""
        )
    elif risk_level == "HIGH" and severity == "LOW":
        st.warning(
            """
ELEVATED MONITORING REQUIRED
- Increase surveillance frequency
- Prepare vaccination supplies
- Monitor livestock movement
- Alert local veterinary officers
"""
        )
    elif risk_level == "MEDIUM":
        st.info(
            """
INCREASED SURVEILLANCE RECOMMENDED
- Standard monitoring with increased frequency
- Review vaccination records in district
- Monitor climate conditions closely
"""
        )
    else:
        st.success(
            """
ROUTINE MONITORING
- Standard surveillance protocols apply
- No immediate intervention required
- Continue regular farm visits
"""
        )

    st.divider()
    st.caption("Data sources: DAPH Annual Reports | CHIRPS Rainfall | NASA POWER Climate | FAO GLW Livestock")
    st.caption("Model: Two-Stage Logistic Regression + Random Forest | SHAP Explainability")
    st.caption("Research Component — University of Moratuwa | 2026")


if __name__ == "__main__":
    main()
