import streamlit as st
from functions.data_analysis_functions import (
    descriptive_analysis,
    input_histogram,
    pca_plot,
    plot_5x5_cr,
    plot_5x5_sr,
    load_preprocess_data,
    load_models
)

# Initialize session state for navigation if not already set.
if "data_analysis_page" not in st.session_state:
    st.session_state.data_analysis_page = "main"

def main_menu():
    st.title("Data Analysis")
    st.write("Select an analysis category:")
    if st.button("Statistical Analysis"):
        st.session_state.data_analysis_page = "statistical_analysis"
    if st.button("Contour Plots"):
        st.session_state.data_analysis_page = "contour_plots"

def statistical_analysis_menu():
    st.title("Statistical Analysis")
    st.write("Select a statistical analysis option:")
    if st.button("Descriptive Analysis"):
        st.session_state.data_analysis_page = "descriptive_analysis"
    if st.button("PCA"):
        st.session_state.data_analysis_page = "pca"
    if st.button("Input Histograms"):
        st.session_state.data_analysis_page = "input_histograms"
    if st.button("Go to Home"):
        st.session_state.data_analysis_page = "main"

def contour_plots_menu():
    st.title("Contour Plots")
    st.write("Select a contour plot option:")
    if st.button("Corrosion Rate"):
        st.session_state.data_analysis_page = "corrosion_rate"
    if st.button("Saturation Ratio"):
        st.session_state.data_analysis_page = "saturation_ratio"
    if st.button("Go to Home"):
        st.session_state.data_analysis_page = "main"

def descriptive_analysis_page():
    st.title("Descriptive Analysis")
    # Load and preprocess the data
    df, X, scaler_X = load_preprocess_data()
    # Call the descriptive analysis function
    descriptive_analysis(X)
    if st.button("Go Back"):
        st.session_state.data_analysis_page = "statistical_analysis"

def pca_page():
    st.title("Principal Component Analysis (PCA)")
    explained_variance = pca_plot()
    st.write("Explained Variance Ratios:", explained_variance)
    if st.button("Go Back"):
        st.session_state.data_analysis_page = "statistical_analysis"

def input_histograms_page():
    st.title("Input Histograms")
    input_histogram()
    if st.button("Go Back"):
        st.session_state.data_analysis_page = "statistical_analysis"

def corrosion_rate_page():
    st.title("Corrosion Rate Contour Plot")
    # Load data and models necessary for the corrosion rate plot
    df, X, scaler_X = load_preprocess_data()
    cr_model, _ = load_models()
    plot_5x5_cr(X, scaler_X, cr_model)
    if st.button("Go Back"):
        st.session_state.data_analysis_page = "contour_plots"

def saturation_ratio_page():
    st.title("Saturation Ratio Contour Plot")
    # Load data and models necessary for the saturation ratio plot
    df, X, scaler_X = load_preprocess_data()
    _, sr_model = load_models()
    plot_5x5_sr(X, scaler_X, sr_model)
    if st.button("Go Back"):
        st.session_state.data_analysis_page = "contour_plots"

# Navigation based on session state:
page = st.session_state.data_analysis_page

if page == "main":
    main_menu()
elif page == "statistical_analysis":
    statistical_analysis_menu()
elif page == "contour_plots":
    contour_plots_menu()
elif page == "descriptive_analysis":
    descriptive_analysis_page()
elif page == "pca":
    pca_page()
elif page == "input_histograms":
    input_histograms_page()
elif page == "corrosion_rate":
    corrosion_rate_page()
elif page == "saturation_ratio":
    saturation_ratio_page()
