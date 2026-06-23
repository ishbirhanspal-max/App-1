import streamlit as st
import pandas as pd
import plotly.express as px
import math

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="Forecasting Feasibility Analyzer", 
    page_icon="📈",
    layout="wide"
)

# Custom CSS injection for cards and polished metrics
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    </style>
""", unsafe_allow_html=True)

# 2. Initialize Session State Variables
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'file_data' not in st.session_state:
    st.session_state.file_data = None

# App Title & Description
st.title("📈 Forecasting Distribution & Feasibility Analyzer")
st.markdown("Upload your dataset to verify if the underlying statistical distributions match requirements for standard predictive forecasting models.")
st.markdown("---")

# 3. File Control Dashboard
st.subheader("📁 Data Source Control Panel")

# File Uploader
uploaded_file = st.file_uploader(
    "Choose an Excel file (.xlsx, .xls)", 
    type=["xlsx", "xls"],
    help="Upload your timeseries or general dataset to inspect features."
)

# Control Buttons Layout
col1, col2, col3, _ = st.columns([1, 1, 1.5, 4])

with col1:
    submit_btn = st.button("🚀 Submit File", use_container_width=True, type="primary")

with col2:
    cancel_btn = st.button("❌ Cancel", use_container_width=True)

with col3:
    clear_btn = st.button("🔄 Remove File / New File", use_container_width=True)

# 4. Button Logic & Event Handling
if clear_btn:
    st.session_state.processed = False
    st.session_state.file_data = None
    st.rerun()

if cancel_btn:
    st.session_state.processed = False
    st.warning("Analysis cancelled. Click 'Submit File' to re-run or use the control panel to add a new file.")

if submit_btn:
    if uploaded_file is not None:
        try:
            # Read and stash in session state to prevent reprocessing on every interaction
            st.session_state.file_data = pd.read_excel(uploaded_file)
            st.session_state.processed = True
            st.success("🎉 Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error parsing the Excel file. Details: {e}")
            st.session_state.processed = False
    else:
        st.error("⚠️ Please select an Excel file before clicking Submit.")

# 5. Application Execution Block
if st.session_state.processed and st.session_state.file_data is not None:
    df = st.session_state.file_data
    
    # Dataset Summary Preview Box
    st.markdown("### 📋 Dataset Overview")
    with st.expander("View Raw Dataset Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Extract numerical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        st.warning("⚠️ No numerical columns detected within the dataset columns.")
    else:
        st.markdown(f"### 📊 Distribution Analysis Summary ({len(numeric_cols)} Numerical Columns Found)")
        
        # Iterating through valid columns
        for col in numeric_cols:
            col_data = df[col].dropna()
            
            if col_data.empty:
                continue
                
            mean_val = col_data.mean()
            median_val = col_data.median()
            
            # Containerized block per variable for visual aesthetic
            with st.container():
                st.markdown(f"#### Column Metrics: `{col}`")
                
                # Visual Metric Dashboard Layout
                m_col1, m_col2, m_col3 = st.columns(3)
                
                with m_col1:
                    st.markdown(f'<div class="metric-card"><h5>Mean Value</h5><h2>{mean_val:,.4f}</h2></div>', unsafe_allow_html=True)
                with m_col2:
                    st.markdown(f'<div class="metric-card"><h5>Median Value</h5><h2>{median_val:,.4f}</h2></div>', unsafe_allow_html=True)
                
                # Forecasting Check Logic
                is_close = math.isclose(mean_val, median_val, rel_tol=0.10, abs_tol=1e-5)
                
                with m_col3:
                    if is_close:
                        st.markdown(
                            '<div class="metric-card" style="border-left: 5px solid #28a745;">'
                            '<h5>Forecasting Viability</h5><h2 style="color: #28a745;">Ready ✅</h2></div>', 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="metric-card" style="border-left: 5px solid #dc3545;">'
                            '<h5>Forecasting Viability</h5><h2 style="color: #dc3545;">Skewed ℹ️</h2></div>', 
                            unsafe_allow_html=True
                        )
                
                # Graphical Layout
                fig = px.histogram(
                    col_data, 
                    x=col, 
                    title=f"Data Distribution Histogram — {col}", 
                    nbins=35,
                    template="plotly_white",
                    labels={col: 'Values'}
                )
                
                # Smooth interactive curves inside the UI
                fig.update_layout(
                    title_font=dict(size=16, family="Arial"),
                    margin=dict(l=20, r=20, t=50, b=20),
                    bargap=0.05
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Text Context Breakdown
                if is_close:
                    st.success(f"**Verdict:** The mean and median of `{col}` match within 10% tolerance limits. Symmetrical distributions suggest the data can be directly targeted for linear, standard, or deep time-series **Forecasting**.")
                else:
                    st.info(f"**Verdict:** `{col}` shows clear statistical skewness. Standard forecasting models might experience prediction shifts. Consider applying transformations (e.g., Logarithmic scaling) prior to building downstream models.")
                
                st.markdown("<br><hr>", unsafe_allow_html=True)
