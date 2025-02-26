import streamlit as st

# --- Sub-page: Contour Plots ---
def contour_plots():
    st.title("Contour Plots")
    st.write("Generating Saturation Ratio Contour Plot...")
    # Here you would call your plotting function, e.g.:
    # plot_5x5_sr(X, scaler_X, sr_model)
    
    # Button to return to the Data Analysis home menu
    if st.button("Go to Home"):
        st.session_state.data_analysis_page = "main"

# --- Sub-page: Statistical Analysis ---
def statistical_analysis():
    st.title("Statistical Analysis")
    st.write("Generating Input Histograms...")
    # Here you would call your function to create histograms, e.g.:
    # input_histogram()
    
    # Button to return to the Data Analysis home menu
    if st.button("Go to Home"):
        st.session_state.data_analysis_page = "main"

# --- Main Data Analysis Function ---
def data_analysis():
    # Initialize the page state if it doesn't exist
    if "data_analysis_page" not in st.session_state:
        # You can set the default to "main" (Data Analysis Home)
        st.session_state.data_analysis_page = "main"
    
    # Render the appropriate sub-page based on the session state.
    if st.session_state.data_analysis_page == "main":
        st.title("Data Analysis")
        st.write("Select an option:")
        
        # Button to switch to Statistical Analysis
        if st.button("Statistical Analysis"):
            st.session_state.data_analysis_page = "statistical_analysis"
        
        # Button to switch to Contour Plots
        if st.button("Contour Plots"):
            st.session_state.data_analysis_page = "contour_plots"
    
    elif st.session_state.data_analysis_page == "statistical_analysis":
        statistical_analysis()
    
    elif st.session_state.data_analysis_page == "contour_plots":
        contour_plots()

# Allow the page to run standalone for testing.
if __name__ == "__main__":
    data_analysis()
