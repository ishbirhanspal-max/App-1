import streamlit as st
import pandas as pd
import plotly.express as px
import math

# Set up the page layout
st.set_page_config(page_title="Forecasting Data Analyzer", layout="wide")
st.title("Data Distribution & Forecasting Feasibility App")
st.write("Upload an Excel file to analyze numerical columns, view histograms, and check if the data is symmetrically distributed for forecasting.")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file here", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Read the excel file
        df = pd.read_excel(uploaded_file)
        st.write("### Dataset Preview:")
        st.dataframe(df.head())

        # Filter for only numerical columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if not numeric_cols:
            st.warning("No numerical columns found in the uploaded dataset.")
        else:
            for col in numeric_cols:
                st.markdown("---")
                st.subheader(f"Analysis for Column: **{col}**")

                # Drop empty rows for accurate calculations
                col_data = df[col].dropna()

                if col_data.empty:
                    st.write("This column is empty or contains only missing values.")
                    continue

                # Calculate Mean and Median
                mean_val = col_data.mean()
                median_val = col_data.median()

                st.write(f"**Mean:** {mean_val:.4f} &nbsp;&nbsp;|&nbsp;&nbsp; **Median:** {median_val:.4f}")

                # Plot the Histogram
                fig = px.histogram(
                    col_data, 
                    x=col, 
                    title=f"Histogram of {col}", 
                    nbins=30,
                    color_discrete_sequence=['#4C78A8']
                )
                st.plotly_chart(fig, use_container_width=True)

                # Check if mean and median are close (within 10% tolerance)
                # abs_tol is added to prevent errors if the numbers are exactly zero
                if math.isclose(mean_val, median_val, rel_tol=0.10, abs_tol=1e-5):
                    st.success("✅ The mean and median are closely aligned. The data is approximately symmetrically distributed and **can be used for Forecasting**.")
                else:
                    st.info("ℹ️ The mean and median differ significantly. The data is skewed and might need transformation before standard forecasting.")

    except Exception as e:
        st.error(f"Error reading the file. Please ensure it is a valid Excel file. Details: {e}")
