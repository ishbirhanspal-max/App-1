import streamlit as st
import pandas as pd
import plotly.express as px
import math

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Forecasting Analyzer", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Initialize Session State
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'file_data' not in st.session_state:
    st.session_state.file_data = None

# 3. Sidebar (Controls & Uploader)
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("Upload your data and manage your analysis session.")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📂 Upload Excel File", 
        type=["xlsx", "xls"]
    )
    
    st.markdown("### Actions")
    # Button Grid Layout
    col1, col2 = st.columns(2)
    with col1:
        submit_btn = st.button("🚀 Submit", use_container_width=True, type="primary")
    with col2:
        clear_btn = st.button("🔄 New File", use_container_width=True)
        
    cancel_btn = st.button("❌ Cancel Process", use_container_width=True)

    # Sidebar Logic
    if clear_btn:
        st.session_state.processed = False
        st.session_state.file_data = None
        st.rerun()

    if cancel_btn:
        st.session_state.processed = False
        st.warning("Analysis paused.")

    if submit_btn:
        if uploaded_file is not None:
            try:
                st.session_state.file_data = pd.read_excel(uploaded_file)
                st.session_state.processed = True
                st.success("File loaded!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.processed = False
        else:
            st.error("⚠️ Please select a file first.")

# 4. Main Application Area
st.title("📊 Data Distribution & Forecasting Analyzer")
st.markdown("Evaluate numerical features to determine if they meet the symmetrical distribution requirements for standard forecasting models.")

# Display content only if a file is submitted
if st.session_state.processed and st.session_state.file_data is not None:
    df = st.session_state.file_data
    
    # Clean dataset preview inside a collapsible accordion
    with st.expander("🔍 View Raw Dataset Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numerical columns detected in this dataset.")
    else:
        st.markdown(f"### 📈 Feature Analysis ({len(numeric_cols)} features)")
        st.markdown("Select a column below to view its distribution and forecasting viability.")
        
        # Create interactive tabs for each column instead of a long scrolling page
        tabs = st.tabs(numeric_cols)
        
        for index, col in enumerate(numeric_cols):
            with tabs[index]:
                col_data = df[col].dropna()
                
                if col_data.empty:
                    st.info("This column is empty or contains only missing values.")
                    continue
                
                # Statistical Calculations
                mean_val = col_data.mean()
                median_val = col_data.median()
                
                # Calculate percentage difference for the UI metric
                # Avoid division by zero
                base_val = abs(mean_val) if mean_val != 0 else 1
                diff_pct = ((median_val - mean_val) / base_val) * 100
                
                is_close = math.isclose(mean_val, median_val, rel_tol=0.10, abs_tol=1e-5)
                
                # --- KPI Metrics Row ---
                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.metric(label="Mean (Average)", value=f"{mean_val:,.4f}")
                with m2:
                    # delta_color="inverse" makes the text red if the gap is large, green if small
                    st.metric(
                        label="Median (Middle Value)", 
                        value=f"{median_val:,.4f}", 
                        delta=f"{diff_pct:,.2f}% variance",
                        delta_color="normal" if is_close else "inverse"
                    )
                with m3:
                    if is_close:
                        st.success("##### ✅ Ready for Forecasting\nSymmetrical distribution detected.")
                    else:
                        st.error("##### ⚠️ Skewed Data\nTransformation recommended.")
                
                # --- Advanced Data Visualization ---
                st.markdown("---")
                
                # Create a histogram with a marginal box plot on top
                fig = px.histogram(
                    col_data, 
                    x=col, 
                    marginal="box", # Adds a box plot above the histogram
                    nbins=40,
                    title=f"Distribution Profile: {col}",
                    color_discrete_sequence=['#4C78A8'],
                    opacity=0.85
                )
                
                # Add visual lines for Mean and Median onto the chart
                fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                              annotation_text="Mean", annotation_position="top right")
                fig.add_vline(x=median_val, line_dash="dot", line_color="green", 
                              annotation_text="Median", annotation_position="top left")
                
                # Polish the chart layout
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=60, b=20),
                    font=dict(size=14)
                )
                
                st.plotly_chart(fig, use_container_width=True)
else:
    # Empty state landing page
    st.info("👈 Please upload an Excel file and click **Submit** in the sidebar to begin analysis.")
