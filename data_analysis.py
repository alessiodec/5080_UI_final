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

# Reset dataset choice whenever this page is loaded.
st.session_state["dataset_choice"] = None

# Initialize session state for navigation if not already set.
if "data_analysis_page" not in st.session_state:
    st.session_state["data_analysis_page"] = "main"

def safe_rerun():
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Rerun failed: {e}")

def main_menu():
    st.title("Data Analysis")
    st.write("Select an analysis category:")
    # Display two buttons side by side using columns.
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Statistical Analysis"):
            st.session_state["data_analysis_page"] = "statistical_analysis"
            safe_rerun()
    with col2:
        if st.button("Contour Plots"):
            st.session_state["data_analysis_page"] = "contour_plots"
            safe_rerun()

def statistical_analysis_menu():
    st.title("Statistical Analysis")

    st.write("testing")
    
    if st.button("Descriptive Analysis"):
        st.session_state["data_analysis_page"] = "descriptive_analysis"
        safe_rerun()
    if st.button("PCA"):
        st.session_state["data_analysis_page"] = "pca"
        safe_rerun()
    if st.button("Input Histograms"):
        st.session_state["data_analysis_page"] = "input_histograms"
        safe_rerun()
    if st.button("Go to Home"):
        st.session_state["data_analysis_page"] = "main"
        safe_rerun()

def contour_plots_menu():
    st.title("Contour Plots")
    if st.button("Corrosion Rate"):
        st.session_state["data_analysis_page"] = "corrosion_rate"
        safe_rerun()
    if st.button("Saturation Ratio"):
        st.session_state["data_analysis_page"] = "saturation_ratio"
        safe_rerun()
    if st.button("Go to Home"):
        st.session_state["data_analysis_page"] = "main"
        safe_rerun()

def descriptive_analysis_page():
    st.title("Descriptive Analysis")
    df, X, scaler_X = load_preprocess_data()
    descriptive_analysis(X)
    if st.button("Go Back"):
        st.session_state["data_analysis_page"] = "main"
        safe_rerun()

def pca_page():
    st.title("Principal Component Analysis (PCA)")
    explained_variance = pca_plot()
    st.write("Explained Variance Ratios:", explained_variance)
    if st.button("Go Back"):
        st.session_state["data_analysis_page"] = "main"
        safe_rerun()

def input_histograms_page():
    st.title("Input Histograms")
    input_histogram()
    if st.button("Go Back"):
        st.session_state["data_analysis_page"] = "main"
        safe_rerun()

def corrosion_rate_page():
    st.title("Corrosion Rate Contour Plot")
    df, X, scaler_X = load_preprocess_data()
    cr_model, _ = load_models()  # loads from models/CorrosionRateModel.keras
    plot_5x5_cr(X, scaler_X, cr_model)
    if st.button("Go Back"):
        st.session_state["data_analysis_page"] = "main"
        safe_rerun()

def saturation_ratio_page():
    st.title("Saturation Ratio Contour Plot")
    df, X, scaler_X = load_preprocess_data()
    _, sr_model = load_models()  # loads from models/SaturationRateModel.keras
    plot_5x5_sr(X, scaler_X, sr_model)
    if st.button("Go Back"):
        st.session_state["data_analysis_page"] = "main"
        safe_rerun()

# Wrap the navigation logic in a function
def data_analysis():
    page = st.session_state.get("data_analysis_page", "main")
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
    else:
        st.write("Unknown page state.")

if __name__ == "__main__":
    data_analysis()
