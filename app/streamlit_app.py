from pathlib import Path
import datetime
import numpy as np
import io

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


st.set_page_config(
    page_title="FMD Early Warning System — Sri Lanka",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
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

        if risk_level == "HIGH":
            risk_accent = "#e63946"
        elif risk_level == "MEDIUM":
            risk_accent = "#f4a261"
        else:
            risk_accent = "#2a9d8f"
        
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

    def row_style(row: pd.Series) -> str:
        if row.get("Risk Level") == "HIGH":
            return "background-color: #ffe5e7;"
        if row.get("Risk Level") == "MEDIUM":
            return "background-color: #fff1e3;"
        return "background-color: #e8fbf4;"

    render_html_table(forecast_df, row_style_func=row_style)


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
    
    render_stat_card("🔴 HIGH Risk Districts", str(high_count), "#e63946")
    render_stat_card("🟠 MEDIUM Risk Districts", str(medium_count), "#f4a261")
    render_stat_card("🟢 LOW Risk Districts", str(low_count), "#2a9d8f")
    
    # Download button
    csv_data = generate_forecast_csv(forecast_df)
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv_data,
        file_name=f"fmd_forecast_{target_month_name}_2025.csv",
        mime="text/csv",
        width='stretch',
    )


def inject_custom_css() -> None:
    """Inject global CSS styling for professional design language."""
    css = """
    <style>
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(230, 57, 70, 0.7); }
        70% { box-shadow: 0 0 0 20px rgba(230, 57, 70, 0); }
        100% { box-shadow: 0 0 0 0 rgba(230, 57, 70, 0); }
    }
    
    * {
        margin: 0;
        padding: 0;
    }
    
    body, [data-testid="stAppViewContainer"] {
        background-color: #f0f4f8;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0a1628 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] button {
        background-color: #0a1628 !important;
        color: #ffffff !important;
        border: 1px solid #00b36b !important;
    }
    
    [data-testid="stSidebar"] button:hover {
        background-color: #00b36b !important;
        color: #0a1628 !important;
    }
    
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 4px solid #00b36b;
    }
    
    .metric-high {
        border-left-color: #e63946 !important;
    }
    
    .metric-medium {
        border-left-color: #f4a261 !important;
    }
    
    .metric-low {
        border-left-color: #2a9d8f !important;
    }
    
    .header-banner {
        background: linear-gradient(90deg, #0a1628 0%, #1a4d2e 100%);
        color: #ffffff;
        padding: 40px 20px;
        border-radius: 12px;
        margin-bottom: 30px;
    }
    
    .section-header {
        color: #0a1628;
        font-weight: 700;
        border-bottom: 3px solid #00b36b;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }
    
    .risk-badge-high {
        display: inline-block;
        background-color: #e63946;
        color: white;
        padding: 12px 24px;
        border-radius: 24px;
        font-weight: bold;
        font-size: 16px;
        animation: pulse 2s infinite;
    }
    
    .risk-badge-medium {
        display: inline-block;
        background-color: #f4a261;
        color: white;
        padding: 12px 24px;
        border-radius: 24px;
        font-weight: bold;
        font-size: 16px;
    }
    
    .risk-badge-low {
        display: inline-block;
        background-color: #2a9d8f;
        color: white;
        padding: 12px 24px;
        border-radius: 24px;
        font-weight: bold;
        font-size: 16px;
    }
    
    .card-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #00b36b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 16px;
    }
    
    .card-finding {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-top: 4px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 16px;
    }
    
    .finding-blue { border-top-color: #0a1628; }
    .finding-green { border-top-color: #00b36b; }
    .finding-red { border-top-color: #e63946; }
    
    .recommendation-high {
        background-color: #fff0f1;
        border-left: 6px solid #e63946;
    }
    
    .recommendation-medium {
        background-color: #fff5f0;
        border-left: 6px solid #f4a261;
    }
    
    .recommendation-low {
        background-color: #f0fef8;
        border-left: 6px solid #2a9d8f;
    }
    
    .nav-button {
        width: 100%;
        padding: 12px 16px;
        background-color: transparent;
        color: #ffffff;
        border: 2px solid rgba(255,255,255,0.3);
        border-radius: 8px;
        margin-bottom: 8px;
        text-align: left;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .nav-button:hover {
        background-color: #00b36b;
        border-color: #00b36b;
        color: #0a1628;
    }
    
    .nav-button-active {
        background-color: #00b36b;
        border-color: #00b36b;
        color: #0a1628;
        font-weight: 700;
    }
    
    .pipeline-step {
        background-color: #ffffff;
        padding: 12px 16px;
        border-radius: 8px;
        text-align: center;
        font-size: 12px;
        border: 2px solid #00b36b;
        flex: 1;
    }
    
    .pipeline-arrow {
        text-align: center;
        font-size: 18px;
        color: #00b36b;
        font-weight: bold;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize session state for multi-page navigation."""
    if "page" not in st.session_state:
        st.session_state.page = "Overview"


def render_sidebar() -> None:
    """Render custom sidebar with navigation and branding."""
    st.sidebar.markdown(
        """
        <div style='padding: 20px; text-align: center;'>
            <div style='font-size: 32px; margin-bottom: 8px;'>🐄</div>
            <h2 style='color: #ffffff; font-size: 18px; margin-bottom: 4px;'>FMD Early Warning</h2>
            <p style='color: #00b36b; font-size: 12px; margin-bottom: 16px;'>Sri Lanka — DAPH</p>
            <hr style='border: none; border-top: 2px solid #00b36b; margin: 12px 0;'>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Navigation buttons
    pages = ["🏠 Overview", "🎯 Risk Prediction", "🗺️ District Forecast", "📊 Model Insights"]
    
    for page_label in pages:
        page_name = page_label.split(" ", 1)[1] if " " in page_label else page_label
        is_active = st.session_state.page == page_name
        
        button_class = "nav-button-active" if is_active else "nav-button"
        button_html = f'<button class="{button_class}">{page_label}</button>'
        
        if st.sidebar.button(page_label, key=f"nav_{page_name}", use_container_width=True):
            st.session_state.page = page_name
            st.rerun()
    
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style='padding: 12px; font-size: 11px; color: #888888; text-align: center;'>
            <p style='margin: 4px 0;'>Data: DAPH · CHIRPS · NASA · FAO</p>
            <p style='margin: 4px 0;'>University of Moratuwa · 2026</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header_banner(title: str, subtitle: str = "", tag: str = "") -> None:
    """Render a gradient header banner."""
    html = f"""
    <div class='header-banner'>
        <h1 style='margin: 0; font-size: 36px; font-weight: 700;'>{title}</h1>
        {f'<p style="margin: 8px 0 0 0; font-size: 16px; opacity: 0.9;">{subtitle}</p>' if subtitle else ''}
        {f'<p style="margin: 8px 0 0 0; font-size: 12px; opacity: 0.7; color: #00b36b;">{tag}</p>' if tag else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_section_header(title: str) -> None:
    """Render a section header with green underline."""
    st.markdown(f"<h2 class='section-header'>{title}</h2>", unsafe_allow_html=True)


def render_stat_card(label: str, value: str, accent: str = "#00b36b") -> None:
    """Render a stat card that does not depend on st.metric."""
    st.markdown(
        f"""
        <div class='card-content' style='border-left-color: {accent}; margin-bottom: 0;'>
            <div style='font-size: 12px; color: #5c677d; margin-bottom: 6px;'>{label}</div>
            <div style='font-size: 26px; font-weight: 800; color: #0a1628;'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_html_table(df: pd.DataFrame, row_style_func=None) -> None:
    """Render a simple HTML table to avoid Streamlit dataframe widgets."""
    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    rows = []
    for _, row in df.iterrows():
        style = row_style_func(row) if row_style_func else ""
        cells = "".join(f"<td>{row[col]}</td>" for col in df.columns)
        rows.append(f"<tr style='{style}'>{cells}</tr>")
    table_html = f"""
    <div style='overflow-x: auto; background: #ffffff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-left: 4px solid #00b36b;'>
        <table style='width: 100%; border-collapse: collapse; font-size: 14px;'>
            <thead style='background: #0a1628; color: #ffffff;'>
                <tr>{headers}</tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def create_gauge_chart(probability: float, district: str) -> go.Figure:
    """Create a circular gauge chart for probability display."""
    probability_pct = probability * 100
    
    if probability >= 0.60:
        gauge_color = "#e63946"
        risk_text = "HIGH RISK"
    elif probability >= 0.35:
        gauge_color = "#f4a261"
        risk_text = "MEDIUM RISK"
    else:
        gauge_color = "#2a9d8f"
        risk_text = "LOW RISK"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability_pct,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Outbreak Probability", "font": {"size": 16, "color": "#0a1628"}},
        number={"suffix": "%", "font": {"size": 32, "color": "#0a1628"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#0a1628"},
            "bar": {"color": gauge_color},
            "steps": [
                {"range": [0, 35], "color": "#f0f4f8"},
                {"range": [35, 60], "color": "#f0f4f8"},
                {"range": [60, 100], "color": "#f0f4f8"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90,
            },
        },
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "sans-serif", "color": "#0a1628"},
    )
    
    return fig


def page_overview(df: pd.DataFrame, models: dict) -> None:
    """Render the Overview page."""
    render_header_banner(
        "FMD Early Warning System",
        "Climate-Informed · Explainable · Uncertainty-Aware",
        "Sri Lanka · 25 Districts · 2017–2024",
    )
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_stat_card("Districts Monitored", "25", "#00b36b")
    with col2:
        render_stat_card("Outbreak Recall", "91.8%", "#00b36b")
    with col3:
        render_stat_card("Model Confidence", "85.7%", "#00b36b")
    with col4:
        render_stat_card("Training Data", "8 Years", "#00b36b")
    
    st.markdown("")
    render_section_header("📊 Two-Stage Prediction Pipeline")
    
    # Pipeline visualization
    pipeline_html = """
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 24px; overflow-x: auto;'>
        <div class='pipeline-step'>🌧️ Climate Data<br><small>Rainfall · Humidity · Temp · Livestock</small></div>
        <div class='pipeline-arrow'>→→</div>
        <div class='pipeline-step'>Stage 1: Outbreak?<br><small>Logistic Regression · Recall 91.8%</small></div>
        <div class='pipeline-arrow'>→→</div>
        <div class='pipeline-step'>Stage 2: Severity?<br><small>Random Forest · LOYO Validated</small></div>
        <div class='pipeline-arrow'>→→</div>
        <div class='pipeline-step'>⚠️ Early Warning<br><small>Risk Level · Severity · Interval</small></div>
    </div>
    """
    st.markdown(pipeline_html, unsafe_allow_html=True)
    
    st.markdown("")
    render_section_header("🔍 Key Findings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class='card-finding finding-blue'>
                <h4 style='margin: 0 0 8px 0; color: #0a1628;'>🌙 Peak Season</h4>
                <p style='margin: 0; font-size: 14px; color: #555;'>
                    NE Monsoon (Dec–Feb) accounts for 50% of all outbreaks. High moisture and cooler temperatures favor virus survival.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            """
            <div class='card-finding finding-green'>
                <h4 style='margin: 0 0 8px 0; color: #0a1628;'>🐃 Livestock Driver</h4>
                <p style='margin: 0; font-size: 14px; color: #555;'>
                    Buffalo density is the strongest severity predictor. Higher stocking rates increase viral transmission.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            """
            <div class='card-finding finding-red'>
                <h4 style='margin: 0 0 8px 0; color: #0a1628;'>📍 High Risk Zones</h4>
                <p style='margin: 0; font-size: 14px; color: #555;'>
                    Batticaloa · Ampara · Anuradhapura · Kurunegala show consistently elevated baseline risk.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("→ Start Risk Prediction", use_container_width=True, type="primary"):
            st.session_state.page = "Risk Prediction"
            st.rerun()
    
    st.markdown("")
    st.markdown(
        "<p style='text-align: center; font-size: 12px; color: #999; margin-top: 40px;'>"
        "Data sources: DAPH Annual Reports | CHIRPS Rainfall | NASA POWER Climate | FAO GLW Livestock<br>"
        "Model: Two-Stage Logistic Regression + Random Forest | SHAP Explainability<br>"
        "Research Component - IT22221414 - Kumarasinghe S.S | 2026"
        "</p>",
        unsafe_allow_html=True,
    )


def page_risk_prediction(df: pd.DataFrame, models: dict, stage1_shap_df: pd.DataFrame, stage2_shap_df: pd.DataFrame, bootstrap_intervals_df: pd.DataFrame) -> None:
    """Render the Risk Prediction page."""
    render_header_banner(
        "🎯 District Risk Prediction",
        "Select a district and time period for outbreak assessment",
    )
    
    # Input controls
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_district = st.selectbox("District", DISTRICTS, label_visibility="collapsed")
    with col2:
        selected_month_name = st.selectbox("Month", MONTH_NAMES, label_visibility="collapsed")
    with col3:
        selected_year = st.selectbox("Year", list(range(2017, 2025)), index=7, label_visibility="collapsed")
    
    month_num = MONTH_NAMES.index(selected_month_name) + 1
    
    if st.button("⚡ Predict FMD Risk", use_container_width=True, type="primary"):
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
            st.info(f"ℹ️ {fallback_msg}")
        
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

        if risk_level == "HIGH":
            risk_accent = "#e63946"
        elif risk_level == "MEDIUM":
            risk_accent = "#f4a261"
        else:
            risk_accent = "#2a9d8f"
        
        severity = "LOW"
        if probability >= 0.35:
            for col in stage2_features:
                if col not in feature_row.columns:
                    feature_row[col] = 0.0
            x_stage2 = feature_row[stage2_features].fillna(0.0).astype(float)
            severity_pred = int(models["stage2_model"].predict(x_stage2)[0])
            severity = decode_severity(models["stage2_encoder"], severity_pred)
        
        st.markdown("")
        render_section_header(f"📍 {selected_district} — {selected_month_name} {selected_year}")
        
        # Stage 1 results
        st.markdown("#### Stage 1: Outbreak Probability")
        
        col_stage1_left, col_stage1_right = st.columns([1, 1.2])
        
        with col_stage1_left:
            # Risk badge
            risk_icon = "🔴" if risk_level == "HIGH" else ("🟠" if risk_level == "MEDIUM" else "🟢")
            risk_text = f"{risk_icon} {risk_level} RISK"
            
            if risk_level == "HIGH":
                st.markdown(f"<div class='risk-badge-high'>{risk_text}</div>", unsafe_allow_html=True)
            elif risk_level == "MEDIUM":
                st.markdown(f"<div class='risk-badge-medium'>{risk_text}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='risk-badge-low'>{risk_text}</div>", unsafe_allow_html=True)
            
            st.markdown("")
            render_stat_card("Outbreak Probability", f"{probability * 100:.1f}%", risk_accent)
        
        with col_stage1_right:
            gauge_fig = create_gauge_chart(probability, selected_district)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        # SHAP chart for Stage 1
        st.markdown("#### Why This Prediction?")
        fig1 = build_top_shap_chart(stage1_shap_df, "Top Climate Risk Drivers")
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
        
        # Stage 2 results (if applicable)
        if probability >= 0.35:
            st.markdown("")
            st.markdown("#### Stage 2: Severity Estimate")
            
            col_stage2_left, col_stage2_right = st.columns([1, 1])
            
            with col_stage2_left:
                severity_icon = "🔴" if severity == "HIGH" else ("🟠" if severity == "MEDIUM" else "🟢")
                st.markdown(f"<h3 style='color: #0a1628; margin: 0;'>{severity_icon} {severity}</h3>", unsafe_allow_html=True)
                
                if severity == "LOW":
                    st.success("Minor outbreak. Standard monitoring applies.")
                elif severity == "MEDIUM":
                    st.warning("Moderate outbreak. Targeted response recommended.")
                else:
                    st.error("Severe outbreak. Emergency response required.")
                
                # Bootstrap interval
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
                    
                    st.markdown("")
                    st.markdown("**95% Prediction Interval**")
                    render_stat_card("95% Prediction Interval", str(bootstrap_row["interval_label"]), "#f4a261")
                    render_stat_card("Model Confidence", f"{confidence_pct:.0f}%", "#2a9d8f" if confidence_pct >= 70 else ("#f4a261" if confidence_pct >= 50 else "#e63946"))
                else:
                    st.info("Bootstrap interval not available for this selection.")
            
            with col_stage2_right:
                fig2 = build_top_shap_chart(stage2_shap_df, "Top Severity Drivers")
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)
        
        # Recommendation box
        st.markdown("")
        st.markdown("#### 🎯 Recommended Action")
        
        if risk_level == "HIGH" and severity == "HIGH":
            st.markdown(
                """
                <div class='card-content recommendation-high' style='border-left-width: 6px;'>
                    <h4 style='color: #e63946; margin: 0 0 12px 0;'>🚨 EMERGENCY RESPONSE REQUIRED</h4>
                    <ul style='margin: 0; padding-left: 20px;'>
                        <li>Immediately notify DAPH Animal Health Division</li>
                        <li>Activate emergency vaccination campaign</li>
                        <li>Impose movement restrictions on livestock</li>
                        <li>Deploy rapid response veterinary teams</li>
                        <li>Isolate affected farms within 24 hours</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif risk_level == "HIGH" and severity == "MEDIUM":
            st.markdown(
                """
                <div class='card-content recommendation-medium' style='border-left-width: 6px;'>
                    <h4 style='color: #f4a261; margin: 0 0 12px 0;'>⚠️ TARGETED RESPONSE REQUIRED</h4>
                    <ul style='margin: 0; padding-left: 20px;'>
                        <li>Alert district veterinary surgeons</li>
                        <li>Begin targeted vaccination in high-risk areas</li>
                        <li>Increase farm surveillance frequency</li>
                        <li>Prepare movement restriction protocols</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif risk_level == "HIGH" and severity == "LOW":
            st.markdown(
                """
                <div class='card-content recommendation-medium' style='border-left-width: 6px;'>
                    <h4 style='color: #f4a261; margin: 0 0 12px 0;'>📋 ELEVATED MONITORING REQUIRED</h4>
                    <ul style='margin: 0; padding-left: 20px;'>
                        <li>Increase surveillance frequency</li>
                        <li>Prepare vaccination supplies</li>
                        <li>Monitor livestock movement</li>
                        <li>Alert local veterinary officers</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif risk_level == "MEDIUM":
            st.markdown(
                """
                <div class='card-content' style='border-left-width: 6px; border-left-color: #f4a261;'>
                    <h4 style='color: #0a1628; margin: 0 0 12px 0;'>📊 INCREASED SURVEILLANCE RECOMMENDED</h4>
                    <ul style='margin: 0; padding-left: 20px;'>
                        <li>Standard monitoring with increased frequency</li>
                        <li>Review vaccination records in district</li>
                        <li>Monitor climate conditions closely</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class='card-content' style='border-left-width: 6px; border-left-color: #2a9d8f;'>
                    <h4 style='color: #0a1628; margin: 0 0 12px 0;'>✅ ROUTINE MONITORING</h4>
                    <ul style='margin: 0; padding-left: 20px;'>
                        <li>Standard surveillance protocols apply</li>
                        <li>No immediate intervention required</li>
                        <li>Continue regular farm visits</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def page_district_forecast(df: pd.DataFrame, models: dict) -> None:
    """Render the District Forecast page."""
    render_header_banner(
        "🗺️ All-District Risk Forecast",
        "Climatological baseline risk for all 25 districts across 2026",
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        selected_month_name = st.selectbox("Month", MONTH_NAMES, index=0, label_visibility="collapsed")
    with col2:
        selected_year = st.selectbox("Year", list(range(2025, 2031)), index=1, label_visibility="collapsed")
    with col3:
        pass
    
    target_month = MONTH_NAMES.index(selected_month_name) + 1
    
    if st.button("🔄 Generate Forecast", use_container_width=True, type="primary"):
        with st.spinner("Computing climatological forecast for all 25 districts..."):
            forecast_df = compute_climatological_forecast(
                df=df,
                models=models,
                districts=DISTRICTS,
                target_month=target_month,
            )
        
        if forecast_df.empty:
            st.warning(f"No data available for {selected_month_name}.")
            return
        
        # Summary cards
        high_count = len(forecast_df[forecast_df["Risk Level"] == "HIGH"])
        medium_count = len(forecast_df[forecast_df["Risk Level"] == "MEDIUM"])
        low_count = len(forecast_df[forecast_df["Risk Level"] == "LOW"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            render_stat_card("🔴 HIGH Risk", str(high_count), "#e63946")
        with col2:
            render_stat_card("🟠 MEDIUM Risk", str(medium_count), "#f4a261")
        with col3:
            render_stat_card("🟢 LOW Risk", str(low_count), "#2a9d8f")
        
        st.markdown("")
        render_section_header("📋 District Risk Ranking")
        
        # Add rank column
        forecast_df_display = forecast_df.copy()
        forecast_df_display.insert(0, "Rank", range(1, len(forecast_df_display) + 1))
        
        render_forecast_table(forecast_df_display)
        
        # Download button
        csv_data = generate_forecast_csv(forecast_df)
        st.download_button(
            label="📥 Download Forecast as CSV",
            data=csv_data,
            file_name=f"fmd_forecast_{selected_month_name}_{selected_year}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        
        # Bar chart
        st.markdown("")
        render_section_header(f"📊 District Risk Ranking — {selected_month_name} {selected_year} Forecast")
        
        fig = go.Figure(
            data=[
                go.Bar(
                    y=forecast_df_display["District"],
                    x=forecast_df_display["Outbreak Probability (%)"],
                    orientation="h",
                    marker=dict(
                        color=forecast_df_display["Risk Level"].map({
                            "HIGH": "#e63946",
                            "MEDIUM": "#f4a261",
                            "LOW": "#2a9d8f",
                        })
                    ),
                )
            ]
        )
        fig.update_layout(
            title="",
            xaxis_title="Outbreak Probability (%)",
            yaxis_title="District",
            height=600,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f0f4f8",
            font=dict(color="#0a1628"),
            hovermode="closest",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select a month and year, then click 'Generate Forecast' to see the all-district risk ranking for that month.")


def page_model_insights(stage1_shap_df: pd.DataFrame, stage2_shap_df: pd.DataFrame, bootstrap_intervals_df: pd.DataFrame) -> None:
    """Render the Model Insights page."""
    render_header_banner(
        "📊 Model Performance & Explainability",
        "Validation results · Feature importance · Uncertainty",
    )
    
    # Performance Metrics Section
    render_section_header("🎯 Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Stage 1 — Outbreak Prediction")
        st.markdown("*Logistic Regression · Walk-Forward Validation*")
        
        stage1_metrics = pd.DataFrame({
            "Year": ["2022", "2023", "2024", "Mean"],
            "Recall": ["88.9%", "91.7%", "94.9%", "91.8%"],
            "ROC-AUC": ["0.692", "0.757", "0.845", "0.765"],
            "F1": ["0.260", "0.115", "0.556", "0.310"],
        })
        render_html_table(stage1_metrics)
        
        st.markdown(
            """
            <div class='card-content' style='margin-top: 12px;'>
                <strong>Key Insight:</strong> Model catches 91.8% of real outbreaks using only climate data. 
                Recall improved from 88.9% to 94.9% — models get better with recent data.
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown("#### Stage 2 — Severity Classification")
        st.markdown("*Random Forest · Leave-One-Year-Out*")
        
        stage2_metrics = pd.DataFrame({
            "Year": ["2018", "2019", "2021", "2022", "Mean"],
            "Accuracy": ["43.8%", "35.6%", "32.3%", "72.2%", "46.5%"],
            "Macro F1": ["0.428", "0.389", "0.244", "0.532", "0.398"],
            "Status": ["✅", "✅", "✅", "✅", "—"],
        })
        render_html_table(stage2_metrics)
        
        st.markdown(
            """
            <div class='card-content' style='margin-top: 12px;'>
                <strong>Key Insight:</strong> Buffalo density is the strongest severity predictor — confirmed by SHAP. 
                Sparse training data (few high-severity cases) limits accuracy.
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # SHAP Section
    st.markdown("")
    render_section_header("🔍 What Drives FMD Predictions?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Outbreak Risk Drivers (Stage 1)")
        fig_s1 = build_top_shap_chart(stage1_shap_df, "")
        st.pyplot(fig_s1, use_container_width=True)
        plt.close(fig_s1)
        
        st.markdown(
            """
            <div class='card-content'>
                <h5 style='color: #0a1628; margin: 0 0 8px 0;'>🌙 cos_month (0.596)</h5>
                <p style='margin: 0; font-size: 13px;'>January and December are peak outbreak months. NE Monsoon season drives highest risk.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='card-content'>
                <h5 style='color: #0a1628; margin: 0 0 8px 0;'>🌧️ r3h (0.556)</h5>
                <p style='margin: 0; font-size: 13px;'>3-month cumulative rainfall. Wet conditions favor virus survival and transmission.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='card-content'>
                <h5 style='color: #0a1628; margin: 0 0 8px 0;'>📍 lat (0.316)</h5>
                <p style='margin: 0; font-size: 13px;'>Northern districts at higher baseline risk. Geographic location is a key factor.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown("#### Severity Drivers (Stage 2)")
        fig_s2 = build_top_shap_chart(stage2_shap_df, "")
        st.pyplot(fig_s2, use_container_width=True)
        plt.close(fig_s2)
        
        st.markdown(
            """
            <div class='card-content'>
                <h5 style='color: #0a1628; margin: 0 0 8px 0;'>🐃 buffalo_density (0.054)</h5>
                <p style='margin: 0; font-size: 13px;'>Districts with more buffalo experience more severe outbreaks. Key transmission vector.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='card-content'>
                <h5 style='color: #0a1628; margin: 0 0 8px 0;'>📍 lat (0.048)</h5>
                <p style='margin: 0; font-size: 13px;'>Northern/Eastern dry zone districts consistently have higher severity outbreaks.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='card-content'>
                <h5 style='color: #0a1628; margin: 0 0 8px 0;'>💨 wind_speed (0.024)</h5>
                <p style='margin: 0; font-size: 13px;'>Wind patterns affect how far the virus spreads aerially between farms.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Bootstrap Uncertainty Section
    st.markdown("")
    render_section_header("📈 Prediction Confidence & Uncertainty")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        render_stat_card("Mean Model Confidence", "85.7%", "#00b36b")
    with col2:
        render_stat_card("Interval Coverage Rate", "63.6%", "#00b36b")
    with col3:
        render_stat_card("High Confidence Predictions", "74.8%", "#00b36b")
    
    st.markdown("")
    
    # Bootstrap interval distribution (pie chart)
    fig_pie = go.Figure(data=[go.Pie(
        labels=["Narrow [X,X]", "Medium [LOW,MED]", "Wide [LOW,HIGH]"],
        values=[76, 171, 59],
        marker=dict(colors=["#2a9d8f", "#f4a261", "#e63946"]),
    )])
    fig_pie.update_layout(
        title="Bootstrap Interval Width Distribution",
        height=350,
        paper_bgcolor="#ffffff",
        font=dict(color="#0a1628"),
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown(
        """
        <div class='card-content'>
            <h5 style='color: #0a1628; margin: 0 0 8px 0;'>What This Means</h5>
            <p style='margin: 0; font-size: 13px;'>
                <strong>Narrow intervals</strong> indicate high model certainty — prediction is stable.
                <strong>Wide intervals</strong> suggest outcome could vary substantially — veterinary officers should prepare for range.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Main application entry point."""
    inject_custom_css()
    init_session_state()
    
    try:
        df = load_data()
        models = load_models()
        stage1_shap_df, stage2_shap_df = load_shap_values()
        bootstrap_intervals_df = load_bootstrap_intervals()
    except Exception as exc:
        st.error(f"❌ Failed to load required files: {exc}")
        return
    
    render_sidebar()
    
    # Route to appropriate page
    if st.session_state.page == "Overview":
        page_overview(df, models)
    elif st.session_state.page == "Risk Prediction":
        page_risk_prediction(df, models, stage1_shap_df, stage2_shap_df, bootstrap_intervals_df)
    elif st.session_state.page == "District Forecast":
        page_district_forecast(df, models)
    elif st.session_state.page == "Model Insights":
        page_model_insights(stage1_shap_df, stage2_shap_df, bootstrap_intervals_df)


if __name__ == "__main__":
    main()
