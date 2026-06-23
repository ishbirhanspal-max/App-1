import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
import warnings

# Import the forecasting engine
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Suppress math warnings for clean execution
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PREMIUM PAGE SETUP & BRANDING
# ==============================================================================
st.set_page_config(
    page_title="Forecasting Intelligence Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    h1 { font-weight: 800 !important; letter-spacing: -0.03em !important; }
    h2, h3, h4 { font-weight: 700 !important; letter-spacing: -0.01em !important; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. STATE MANAGEMENT & DATA LIFECYCLE
# ==============================================================================
if 'session_state' not in st.session_state:
    st.session_state.session_state = {
        'is_active': False,
        'dataframe': None,
        'filename': ""
    }

def terminate_session():
    st.session_state.session_state = {'is_active': False, 'dataframe': None, 'filename': ""}
    st.toast("Workspace cleared. Ready for new input.", icon="🗑️")

# ==============================================================================
# 3. CONTROL CENTER SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Workstation Controls")
    st.markdown("Manage your active dataset pipeline.")
    st.markdown("---")
    
    data_mode = st.radio(
        "Select Data Stream:",
        ["Upload Local Document", "Load Sample Enterprise Telemetry"],
        index=0
    )
    
    local_file = None
    if data_mode == "Upload Local Document":
        local_file = st.file_uploader("Inbound Spreadsheet Ingestion", type=["xlsx", "xls"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_run, col_reset = st.columns(2)
    with col_run:
        btn_run = st.button("🚀 Run Engine", use_container_width=True, type="primary")
    with col_reset:
        btn_reset = st.button("🔄 New File", use_container_width=True)
        
    btn_cancel = st.button("❌ Terminate Current Analysis", use_container_width=True)

    if btn_reset:
        terminate_session()
        st.rerun()

    if btn_cancel:
        st.session_state.session_state['is_active'] = False
        st.warning("Analysis stream paused. Session data retained.")

    if btn_run:
        if data_mode == "Load Sample Enterprise Telemetry":
            np.random.seed(101)
            timesteps = 365
            # Create synthetic data with a slight upward trend so forecasts look realistic
            base_trend = np.linspace(1000, 1500, timesteps)
            sample_df = pd.DataFrame({
                "Timestamp": pd.date_range(start="2025-01-01", periods=timesteps, freq="D"),
                "Optimal_Symmetrical_Sales": base_trend + np.random.normal(loc=0, scale=80, size=timesteps),
                "Highly_Skewed_User_Churn": np.random.lognormal(mean=2.5, sigma=0.75, size=timesteps) * 5,
            })
            st.session_state.session_state['dataframe'] = sample_df
            st.session_state.session_state['filename'] = "enterprise_telemetry_stream.xlsx"
            st.session_state.session_state['is_active'] = True
            st.toast("Enterprise telemetry successfully loaded.", icon="📈")
            
        elif data_mode == "Upload Local Document" and local_file is not None:
            try:
                with st.spinner("Decoding binary spreadsheet arrays..."):
                    st.session_state.session_state['dataframe'] = pd.read_excel(local_file)
                    st.session_state.session_state['filename'] = local_file.name
                    st.session_state.session_state['is_active'] = True
                st.toast("Spreadsheet successfully compiled.", icon="✨")
            except Exception as error:
                st.error(f"Ingestion Fault: Could not read file structure. Details: {error}")
                terminate_session()
        else:
            st.error("⚠️ Ingestion Queue Empty. Please assign an input spreadsheet first.")

# ==============================================================================
# 4. MAIN EXECUTIVE DASHBOARD INTERFACE
# ==============================================================================
st.title("⚡ Predictive Intelligence & Feasibility Engine")
st.markdown("Evaluate mathematical symmetry constraints and generate live future trend predictions using Exponential Smoothing.")
st.markdown("---")

if st.session_state.session_state['is_active'] and st.session_state.session_state['dataframe'] is not None:
    working_df = st.session_state.session_state['dataframe']
    active_file = st.session_state.session_state['filename']
    
    st.info(f"📊 **Active Worksession:** `{active_file}` | **Observations Container:** `{working_df.shape[0]}` Rows")
    
    # Updated: Added the Forecasting Engine Tab
    view_tab_preview, view_tab_analytics, view_tab_forecast = st.tabs([
        "📋 Document Matrix", 
        "🔬 Symmetry Profiler", 
        "🔮 Advanced Forecasting Engine"
    ])
    
    # --- Tab 1: Document Matrix ---
    with view_tab_preview:
        st.markdown("### 🔍 Raw Records Inspection")
        st.dataframe(working_df.head(15), use_container_width=True)

    # --- Tab 2: Symmetry Profiler ---
    with view_tab_analytics:
        target_features = working_df.select_dtypes(include=['number']).columns.tolist()
        if not target_features:
            st.error("Processing Halt: Zero numerical attributes detected.")
        else:
            st.markdown("### 📊 High-Fidelity Feature Analysis")
            feature_sub_tabs = st.tabs([f"📈 {f}" for f in target_features])
            
            for index, current_col in enumerate(target_features):
                with feature_sub_tabs[index]:
                    clean_vector = working_df[current_col].dropna()
                    if clean_vector.empty: continue
                    
                    mean_metric = clean_vector.mean()
                    median_metric = clean_vector.median()
                    variance_spread = ((median_metric - mean_metric) / (abs(mean_metric) if mean_metric != 0 else 1)) * 100
                    is_stat_feasible = math.isclose(mean_metric, median_metric, rel_tol=0.10, abs_tol=1e-5)
                    
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1: st.metric("Sample Mean", f"{mean_metric:,.4f}")
                    with m_col2: st.metric("Sample Median", f"{median_metric:,.4f}")
                    with m_col3: st.metric("Alignment Skew", f"{variance_spread:+.2f}%", delta_color="normal" if is_stat_feasible else "inverse")
                    
                    if is_stat_feasible:
                        st.success("✅ **Feasible for Direct Forecasting:** The central distribution demonstrates high data symmetry.")
                    else:
                        st.warning("⚠️ **Transform Required:** Substantial skew indicates potential volatility in standard predictive models.")
                    
                    dist_fig = px.histogram(
                        working_df, x=current_col, marginal="box", nbins=40,
                        template="plotly_white", color_discrete_sequence=["#2c3e50"]
                    )
                    dist_fig.add_vline(x=mean_metric, line_dash="dash", line_color="#e74c3c", annotation_text="Mean")
                    dist_fig.add_vline(x=median_metric, line_dash="dot", line_color="#2ecc71", annotation_text="Median")
                    st.plotly_chart(dist_fig, use_container_width=True)

    # --- Tab 3: Advanced Forecasting Engine ---
    with view_tab_forecast:
        st.markdown("### 🚀 Real-Time Exponential Smoothing (Holt-Winters)")
        st.markdown("Configure your temporal parameters to generate a forward-looking prediction path.")
        
        # Determine likely date columns (datetime types or strings that might be dates)
        all_cols = working_df.columns.tolist()
        num_cols = working_df.select_dtypes(include=['number']).columns.tolist()
        
        col_setup1, col_setup2, col_setup3 = st.columns(3)
        with col_setup1:
            x_axis = st.selectbox("Temporal X-Axis (Date/Time or Index):", all_cols, index=0)
        with col_setup2:
            y_axis = st.selectbox("Target Metric Y-Axis (Value):", num_cols, index=0)
        with col_setup3:
            periods = st.number_input("Forecast Horizon (Future Periods):", min_value=1, max_value=365, value=30)
            
        btn_forecast = st.button("🔮 Generate Predictive Forecast", type="primary")
        
        if btn_forecast:
            with st.spinner("Compiling statistical forecast..."):
                try:
                    # Isolate and clean the target data
                    df_ts = working_df[[x_axis, y_axis]].dropna().copy()
                    
                    # Ensure X-axis is sorted properly for time series
                    df_ts = df_ts.sort_values(by=x_axis).reset_index(drop=True)
                    
                    historical_x = df_ts[x_axis]
                    historical_y = df_ts[y_axis]
                    
                    # Fit Holt-Winters Exponential Smoothing Model
                    # Using additive trend. Seasonal is None by default for general flexibility
                    model = ExponentialSmoothing(
                        historical_y, 
                        trend='add', 
                        initialization_method="estimated"
                    ).fit()
                    
                    # Generate predictions
                    forecast_values = model.forecast(periods)
                    
                    # Attempt to generate future dates if X-axis is datetime
                    if pd.api.types.is_datetime64_any_dtype(historical_x):
                        last_date = historical_x.iloc[-1]
                        # Infer frequency (Daily, Monthly, etc.)
                        freq = pd.infer_freq(historical_x[-10:]) or 'D' 
                        future_x = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
                    else:
                        # Fallback to integer progression if X-axis is not a Date
                        last_val = historical_x.iloc[-1]
                        if isinstance(last_val, (int, float)):
                            step = (historical_x.iloc[-1] - historical_x.iloc[0]) / len(historical_x)
                            future_x = [last_val + (step * (i + 1)) for i in range(periods)]
                        else:
                            future_x = [f"Period {i+1}" for i in range(periods)]

                    # Build the Interactive Plot
                    fig = go.Figure()

                    # Add Historical Data Line
                    fig.add_trace(go.Scatter(
                        x=historical_x, 
                        y=historical_y,
                        mode='lines',
                        name='Historical Actuals',
                        line=dict(color='#2c3e50', width=2)
                    ))

                    # Add Forecast Data Line
                    fig.add_trace(go.Scatter(
                        x=future_x, 
                        y=forecast_values,
                        mode='lines',
                        name='Model Forecast',
                        line=dict(color='#e74c3c', width=3, dash='dash')
                    ))
                    
                    # Polish the forecast chart
                    fig.update_layout(
                        title=f"Predictive Projection: {y_axis} over Next {periods} Periods",
                        xaxis_title=f"{x_axis} Timeline",
                        yaxis_title=f"{y_axis} Metric Value",
                        hovermode="x unified",
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("Forecast generated successfully. Review the projected dashed trajectory above.")
                    
                except Exception as e:
                    st.error(f"Failed to generate forecast. Ensure your X-axis is chronological and your Y-axis is strictly numerical. Error details: {e}")

else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    l_left, l_center, l_right = st.columns([1, 2, 1])
    with l_center:
        st.info("💡 **Intelligence Workspace Inactive**\n\nPlease select an input data source on the left panel and click **🚀 Run Engine**.")
