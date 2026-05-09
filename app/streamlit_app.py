from pathlib import Path
import datetime
import numpy as np
import io

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
BOOTSTRAP_INTERVALS_PATH = ROOT_DIR / "data" / "processed" / "bootstrap_intervals.csv"

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


@st.cache_data
def load_bootstrap_intervals() -> pd.DataFrame:
    if BOOTSTRAP_INTERVALS_PATH.exists():
        return pd.read_csv(BOOTSTRAP_INTERVALS_PATH)
    return pd.DataFrame()


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


def compute_climatological_forecast(
    df: pd.DataFrame,
    models: dict,
    districts: list[str],
    target_month: int,
) -> pd.DataFrame:
    """
    Compute climatological mean forecast for a given month across all districts.
    
    Steps:
    1. Filter data for target_month for each district
    2. Compute mean of all 21 feature columns grouped by district
    3. Override sin_month and cos_month with fixed mathematical values for the month
    4. Override monsoon_phase to correct seasonal pattern
    5. Keep lat/lon as district-level fixed values (from first row)
    6. Run Stage 1 model on all 25 districts
    7. Run Stage 2 for districts where Stage 1 probability >= 0.35
    """
    
    stage1_features = list(models["stage1_features"])
    stage2_features = list(models["stage2_features"])
    
    # Define monsoon phase mappings for Sri Lanka
    monsoon_mappings = {
        1: {"NE": 1, "SW": 0, "FIM": 0, "SIM": 0},  # January - NE Monsoon
        2: {"NE": 1, "SW": 0, "FIM": 0, "SIM": 0},  # February - NE Monsoon
        3: {"NE": 0, "SW": 0, "FIM": 1, "SIM": 0},  # March - First Inter-Monsoon
        4: {"NE": 0, "SW": 0, "FIM": 1, "SIM": 0},  # April - First Inter-Monsoon
        5: {"NE": 0, "SW": 1, "FIM": 0, "SIM": 0},  # May - SW Monsoon
        6: {"NE": 0, "SW": 1, "FIM": 0, "SIM": 0},  # June - SW Monsoon
        7: {"NE": 0, "SW": 1, "FIM": 0, "SIM": 0},  # July - SW Monsoon
        8: {"NE": 0, "SW": 1, "FIM": 0, "SIM": 0},  # August - SW Monsoon
        9: {"NE": 0, "SW": 0, "FIM": 0, "SIM": 1},  # September - Second Inter-Monsoon
        10: {"NE": 0, "SW": 0, "FIM": 0, "SIM": 1}, # October - Second Inter-Monsoon
        11: {"NE": 1, "SW": 0, "FIM": 0, "SIM": 0}, # November - NE Monsoon
        12: {"NE": 1, "SW": 0, "FIM": 0, "SIM": 0}, # December - NE Monsoon
    }
    
    results = []
    
    for district in districts:
        # Filter January data for this district
        district_month_data = df[(df["district"] == district) & (df["month_num"] == target_month)]
        
        if district_month_data.empty:
            continue
        
        # Compute climatological mean from all available years/records for this month
        mean_row = district_month_data[stage1_features].mean(numeric_only=True)
        
        # Create base row with means
        feature_row = pd.DataFrame([mean_row], columns=stage1_features)
        
        # Keep lat/lon from first row (fixed geographic values)
        if "lat" in stage1_features and not district_month_data["lat"].isna().all():
            feature_row["lat"] = district_month_data["lat"].iloc[0]
        if "lon" in stage1_features and not district_month_data["lon"].isna().all():
            feature_row["lon"] = district_month_data["lon"].iloc[0]
        
        # Override sin_month and cos_month with mathematically fixed values
        if "sin_month" in stage1_features:
            feature_row["sin_month"] = np.sin(2 * np.pi * target_month / 12)
        if "cos_month" in stage1_features:
            feature_row["cos_month"] = np.cos(2 * np.pi * target_month / 12)
        
        # Override monsoon_phase to correct seasonal pattern
        monsoon_map = monsoon_mappings.get(target_month, {})
        if "monsoon_phase_First_Inter_Monsoon" in stage1_features:
            feature_row["monsoon_phase_First_Inter_Monsoon"] = monsoon_map.get("FIM", 0)
        if "monsoon_phase_SW_Monsoon" in stage1_features:
            feature_row["monsoon_phase_SW_Monsoon"] = monsoon_map.get("SW", 0)
        if "monsoon_phase_Second_Inter_Monsoon" in stage1_features:
            feature_row["monsoon_phase_Second_Inter_Monsoon"] = monsoon_map.get("SIM", 0)
        if "monsoon_phase_NE_Monsoon" in stage1_features:
            feature_row["monsoon_phase_NE_Monsoon"] = monsoon_map.get("NE", 0)
        
        # Fill any remaining NaN values
        feature_row = feature_row.fillna(0.0)
        
        # Run Stage 1 model
        x_stage1 = feature_row[stage1_features].astype(float)
        x_stage1_scaled = models["stage1_scaler"].transform(x_stage1)
        probability = float(models["stage1_model"].predict_proba(x_stage1_scaled)[:, 1][0])
        
        # Determine risk level
        if probability >= 0.60:
            risk_level = "HIGH"
        elif probability >= 0.35:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Run Stage 2 if probability >= 0.35
        severity = "No Outbreak Predicted"
        if probability >= 0.35:
            for col in stage2_features:
                if col not in feature_row.columns:
                    feature_row[col] = 0.0
            x_stage2 = feature_row[stage2_features].fillna(0.0).astype(float)
            severity_pred = int(models["stage2_model"].predict(x_stage2)[0])
            severity = decode_severity(models["stage2_encoder"], severity_pred)
        
        results.append({
            "District": district,
            "Outbreak Probability (%)": round(probability * 100, 1),
            "Risk Level": risk_level,
            "Predicted Severity": severity,
        })
    
    # Sort by outbreak probability descending
    results_df = pd.DataFrame(results).sort_values("Outbreak Probability (%)", ascending=False)
    return results_df


def render_forecast_table(forecast_df: pd.DataFrame) -> None:
    """
    Render forecast table with color coding for risk levels.
    """
    
    def risk_color(risk_level: str) -> str:
        if risk_level == "HIGH":
            return "background-color: #ffcccc"  # Light red
        elif risk_level == "MEDIUM":
            return "background-color: #ffe6cc"  # Light orange
        else:
            return "background-color: #ccffcc"  # Light green
    
    # Apply styling to Risk Level column
    styled_df = forecast_df.style.map(
        lambda x: risk_color(x) if isinstance(x, str) else "",
        subset=["Risk Level"]
    )
    
    st.dataframe(styled_df, width='stretch', hide_index=True)


def generate_forecast_csv(forecast_df: pd.DataFrame) -> bytes:
    """
    Generate CSV bytes for download.
    """
    csv_buffer = io.StringIO()
    forecast_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode()


def render_forecast_tab(
    df: pd.DataFrame,
    models: dict,
    districts: list[str],
) -> None:
    """
    Render the forecast tab with climatological mean forecast for all districts.
    """
    
    st.subheader("🗓️ January 2025 Forecast — All 25 Districts")
    
    # Get current month and compute next calendar month
    current_month = datetime.datetime.now().month
    target_month = (current_month % 12) + 1
    target_month_name = MONTH_NAMES[target_month - 1]
    
    # Display explanation
    st.info(
        f"""
**Climate inputs estimated using 2017–2024 historical {target_month_name} averages per district** (Climatological Mean Method). 
This is a true forward-looking forecast — {target_month_name} 2025 data has not been seen by the model.

Currently showing forecast for: **{target_month_name} 2025** (next calendar month from last available data)
        """
    )
    
    # Compute forecast
    with st.spinner("Computing climatological forecast for all 25 districts..."):
        forecast_df = compute_climatological_forecast(
            df=df,
            models=models,
            districts=districts,
            target_month=target_month,
        )
    
    if forecast_df.empty:
        st.warning(f"No data available for {target_month_name}. Please check the dataset.")
        return
    
    # Display table
    st.markdown("### Ranked Forecast Table")
    render_forecast_table(forecast_df)
    
    # Summary statistics
    st.markdown("### Summary")
    col1, col2, col3 = st.columns(3)
    
    high_count = len(forecast_df[forecast_df["Risk Level"] == "HIGH"])
    medium_count = len(forecast_df[forecast_df["Risk Level"] == "MEDIUM"])
    low_count = len(forecast_df[forecast_df["Risk Level"] == "LOW"])
    
    col1.metric("🔴 HIGH Risk Districts", high_count)
    col2.metric("🟠 MEDIUM Risk Districts", medium_count)
    col3.metric("🟢 LOW Risk Districts", low_count)
    
    # Download button
    csv_data = generate_forecast_csv(forecast_df)
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv_data,
        file_name=f"fmd_forecast_{target_month_name}_2025.csv",
        mime="text/csv",
        width='stretch',
    )


def main() -> None:
    st.sidebar.title("FMD Risk Prediction")
    st.sidebar.markdown("Select a district and time period to predict outbreak risk.")

    selected_district = st.sidebar.selectbox("District", DISTRICTS)
    selected_month_name = st.sidebar.selectbox("Month", MONTH_NAMES)
    month_num = MONTH_NAMES.index(selected_month_name) + 1
    selected_year = st.sidebar.slider("Year", min_value=2017, max_value=2024, value=2024, step=1)

    predict_clicked = st.sidebar.button("Predict FMD Risk", width='stretch', type="primary")

    st.title("FMD Early Warning System")
    st.markdown("Climate-Informed Seasonal Disease Forecasting for Sri Lanka")
    st.divider()

    try:
        df = load_data()
        models = load_models()
        stage1_shap_df, stage2_shap_df = load_shap_values()
        bootstrap_intervals_df = load_bootstrap_intervals()
    except Exception as exc:
        st.error(f"Failed to load required files: {exc}")
        return

    # Create tabs
    tab1, tab2 = st.tabs(["🔍 Single District Prediction", "🗺️ January 2025 Forecast — All Districts"])
    
    # ─── TAB 1: Single District Prediction ──────────────────────────────────────
    with tab1:
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
            st.pyplot(fig1, width='stretch')
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

                bootstrap_match = pd.DataFrame()
                if not bootstrap_intervals_df.empty:
                    bootstrap_match = bootstrap_intervals_df[
                        (bootstrap_intervals_df["district"] == selected_district)
                        & (bootstrap_intervals_df["year"] == selected_year)
                        & (bootstrap_intervals_df["month_num"] == month_num)
                    ]

                if not bootstrap_match.empty:
                    bootstrap_row = bootstrap_match.iloc[0]
                    confidence_pct = float(bootstrap_row["confidence_pct"])

                    bootstrap_interval_col1, bootstrap_interval_col2 = st.columns(2)
                    bootstrap_interval_col1.metric("95% Prediction Interval", str(bootstrap_row["interval_label"]))
                    bootstrap_interval_col2.metric("Model Confidence", f"{confidence_pct:.0f}%")

                    if confidence_pct >= 70:
                        st.success("High confidence bootstrap estimate")
                    elif confidence_pct >= 50:
                        st.warning("Moderate confidence bootstrap estimate")
                    else:
                        st.error("Low confidence bootstrap estimate")
                else:
                    st.info("Bootstrap interval not available for this selection")

            with row3_right:
                fig2 = build_top_shap_chart(stage2_shap_df, "Top Severity Drivers")
                st.pyplot(fig2, width='stretch')
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
        st.caption("Research Component - IT22221414 - Kumarasinghe S.S | 2026")
    
    # ─── TAB 2: Climatological Forecast ────────────────────────────────────────
    with tab2:
        render_forecast_tab(df, models, DISTRICTS)


if __name__ == "__main__":
    main()
