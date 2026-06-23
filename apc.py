import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math

# ==============================================================================
# 1. PAGE SETUP & CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Forecasting Feasibility Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished interface elements
st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 600;
    }
    .status-card {
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE MANAGEMENT
# ==============================================================================
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'data_loaded': False,
        'df': None,
        'filename': None
    }

def reset_session():
    st.session_state.app_state = {
        'data_loaded': False,
        'df': None,
        'filename': None
    }
    st.toast("Application state reset successfully.", icon="🔄")

# ==============================================================================
# 3. SIDEBAR CONTROL PANEL
# ==============================================================================
with st.sidebar:
    st.title("🔮 Control Center")
    st.markdown("Configure your data input and execution environment.")
    st.markdown("---")
    
    # Input Selection Mode
    data_source = st.radio(
        "Choose Data Input Method:",
        ["Upload Local File", "Load Professional Demo Data"],
        index=0,
        help="Select 'Load Professional Demo Data' to immediately test the application features."
    )
    
    uploaded_file = None
    if data_source == "Upload Local File":
        uploaded_file = st.file_uploader(
            "Upload Excel Workbook",
            type=["xlsx", "xls"],
            help="Supported formats: .xlsx, .xls"
        )
    
    st.markdown("### Lifecycle Actions")
    col_submit, col_clear = st.columns(2)
    
    with col_submit:
        submit_clicked = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    with col_clear:
        clear_clicked = st.button("🔄 New Session", use_container_width=True)

    # Action Controller Logic
    if clear_clicked:
        reset_session()
        st.rerun()

    if submit_clicked:
        if data_source == "Load Professional Demo Data":
            # Generate a structured synthetic dataset representing actual business metrics
            np.random.seed(42)
            n_rows = 200
            demo_df = pd.DataFrame({
                "Date": pd.date_range(start="2026-01-01", periods=n_rows, freq="D"),
                "Symmetrical_Sales": np.random.normal(loc=500, scale=50, size=n_rows),  # Good for forecasting
                "Skewed_User_Signups": np.random.exponential(scale=200, size=n_rows) + 50, # Bad for forecasting
                "Inbound_Support_Calls": np.random.poisson(lam=30, size=n_rows).astype(float) # Borderline
            })
            st.session_state.app_state['df'] = demo_df
            st.session_state.app_state['filename'] = "synthetic_enterprise_metrics.xlsx"
            st.session_state.app_state['data_loaded'] = True
            st.toast("Loaded professional mock telemetry data.", icon="📈")
        
        elif data_source == "Upload Local File" and uploaded_file is not None:
            try:
                with st.spinner("Parsing binary spreadsheet matrices..."):
                    st.session_state.app_state['df'] = pd.read_excel(uploaded_file)
                    st.session_state.app_state['filename'] = uploaded_file.name
                    st.session_state.app_state['data_loaded'] = True
                st.toast("Spreadsheet successfully parsed.", icon="✅")
            except Exception as e:
                st.error(f"Execution Error parsing document: {e}")
                reset_session()
        else:
            st.error("⚠️ Active workspace empty. Please upload a file before running execution.")

# ==============================================================================
# 4. MAIN APPLICATION DASHBOARD
# ==============================================================================
st.title("📈 Forecasting Distribution & Feasibility Engine")
st.markdown("Evaluate statistical profiles of numerical features to safely validate data integrity before ingestion into target predictive forecasting architectures.")
st.markdown("---")

# Conditional Presentation UI Block
if st.session_state.app_state['data_loaded']:
    df = st.session_state.app_state['df']
    current_file = st.session_state.app_state['filename']
    
    # Metadata Overview Header Card
    st.info(f"📁 **Active Dataset Context:** `{current_file}` | Total Records: `{df.shape[0]}` Rows × `{df.shape[1]}` Columns")
    
    # Functional Module Tabs
    tab_overview, tab_analytics = st.stabs(["📋 Dataset Diagnostics & Preview", "📊 Statistical Feasibility Profiler"])
    
    # --- Tab 1: Dataset Diagnostics ---
    with tab_overview:
        st.subheader("Raw Data Matrix")
        st.markdown("Examine the head of your data frame to verify layout parsing and missing data properties.")
        st.dataframe(df.head(15), use_container_width=True)
        
        # Simple structural metadata breakdown
        st.markdown("#### Schema Summary")
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.write("**Detected Columns:**", list(df.columns))
        with col_meta2:
            st.write("**Missing Value Count per Column:**")
            st.write(df.isnull().sum().to_frame(name="Missing Elements").T)

    # --- Tab 2: Analytics & Feasibility Profiler ---
    with tab_analytics:
        # Extract purely numerical columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.error("Fatal: The uploaded workspace file contains no numerical features available for processing.")
        else:
            st.markdown("### Statistical Feature Profiles")
            st.markdown("Each numerical feature is analyzed independently. Toggle through tabs to inspect distribution alignment.")
            
            # Sub-tab design per column prevents page congestion
            feature_tabs = st.tabs([f"🔹 {c}" for c in numeric_cols])
            
            for index, col in enumerate(numeric_cols):
                with feature_tabs[index]:
                    # Isolate clean vector data
                    vector = df[col].dropna()
                    
                    if vector.empty:
                        st.warning(f"Feature processing skipped: Column `{col}` contains entirely null elements.")
                        continue
                        
                    # Calculate accurate descriptive parameters
                    mean_val = vector.mean()
                    median_val = vector.median()
                    std_val = vector.std()
                    
                    # Compute statistical variance relationship
                    base_denominator = abs(mean_val) if mean_val != 0 else 1
                    variance_percentage = ((median_val - mean_val) / base_denominator) * 100
                    
                    # Tolerance evaluation boundary condition (10% Relative Window)
                    is_feasible = math.isclose(mean_val, median_val, rel_tol=0.10, abs_tol=1e-5)
                    
                    # Dashboard KPI KPI Component Matrix
                    st.markdown("<br>", unsafe_allow_html=True)
                    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
                    
                    with kpi_1:
                        st.metric(label="Sample Mean", value=f"{mean_val:,.4f}")
                    with kpi_2:
                        st.metric(label="Sample Median", value=f"{median_val:,.4f}")
                    with kpi_3:
                        st.metric(
                            label="Central Tendency Skew", 
                            value=f"{variance_percentage:+.2f}%",
                            delta="Balanced Alignment" if is_feasible else "Significant Skew",
                            delta_color="normal" if is_feasible else "inverse"
                        )
                    with kpi_4:
                        st.metric(label="Standard Deviation (σ)", value=f"{std_val:,.2f}")
                    
                    st.markdown("---")
                    
                    # Core Verdict Execution Summary Callout Box
                    if is_feasible:
                        st.success(f"### 🎉 Feasibility Status: Ready for Forecasting")
                        st.markdown(f"The mean and median alignment variance for `{col}` settles comfortably inside the **10% symmetry threshold**. The central tendency is unskewed, demonstrating high compatibility with classical linear timeseries models (e.g., ARIMA) and standard ML regressors without forcing data transforms.")
                    else:
                        st.warning(f"### ⚠️ Feasibility Status: Skewed Distribution Warning")
                        st.markdown(f"The structural mean and median variance for `{col}` exceeds acceptable symmetric margins (**{variance_percentage:.2f}%** shift). High skewness violates standard distribution normalities. Direct forecasting models may exhibit tracking errors. **Recommendation:** Apply mathematical normalization (Log, Box-Cox, or Robust Scaling) prior to target modeling.")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Production Interactive Chart Framework
                    # Blends a comprehensive statistical distribution histogram with a top-aligned Box Plot overlay
                    fig = px.histogram(
                        df,
                        x=col,
                        marginal="box",
                        nbins=35,
                        title=f"Statistical Density & Distribution Analytics Profile: {col}",
                        template="plotly_white",
                        color_discrete_sequence=["#1f77b4"],
                        opacity=0.8
                    )
                    
                    # Add distinct baseline marker indicator lines for strategic parameter callouts
                    fig.add_vline(
                        x=mean_val, 
                        line_dash="dash", 
                        line_color="#d62728", 
                        line_width=2.5,
                        annotation_text=f"Mean: {mean_val:.2f}", 
                        annotation_position="top right"
                    )
                    fig.add_vline(
                        x=median_val, 
                        line_dash="dot", 
                        line_color="#2ca02c", 
                        line_width=2.5,
                        annotation_text=f"Median: {median_val:.2f}", 
                        annotation_position="top left"
                    )
                    
                    fig.update_layout(
                        hovermode="x unified",
                        xaxis_title=f"Measurement Interval Scale ({col})",
                        yaxis_title="Frequency Distribution Count",
                        font=dict(family="Inter, sans-serif", size=13),
                        title_font=dict(size=18, weight="bold")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

else:
    # Production Landing State Illustration Block
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.info("💡 **Welcome to the Forecasting Workspace**\n\nTo begin calculation processing, utilize the **Control Center Panel** on the left menu:\n\n1. Upload your internal enterprise spreadsheet file OR click the built-in demo data option.\n2. Click the primary colored **🚀 Run Analysis** action button to evaluate system forecasting parameters.")
