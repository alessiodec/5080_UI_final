import streamlit as st
from functions.physical_relationship_analysis_functions import (
    load_heatsink_data,
    run_heatsink_analysis,
    run_heatsink_evolution
)

def physical_relationship_analysis():
    st.title("Physical Relationship Analysis")
    
    # Button: Load Heatsink Data
    if st.button("Load Heatsink Data"):
        try:
            df, X, y, standardised_y, mean_y, std_y = load_heatsink_data(display_output=True)
            st.session_state["heatsink_data"] = (df, X, y, standardised_y, mean_y, std_y)
            st.success("Heatsink data loaded successfully.")
        except Exception as e:
            st.error(f"Error loading heatsink data: {e}")
    
    # Button: Run Heatsink Analysis
    if st.button("Run Heatsink Analysis"):
        if "heatsink_data" not in st.session_state:
            st.error("Please load the heatsink data first!")
        else:
            try:
                # Here you can let the user provide parameters or use defaults
                run_heatsink_analysis(pop_size=200, pop_retention=50, num_iterations=10)
                st.session_state["analysis_done"] = True
            except Exception as e:
                st.error(f"Error running heatsink analysis: {e}")
    
    # Button: Run Heatsink Evolution
    if st.button("Run Heatsink Evolution"):
        if "analysis_done" not in st.session_state:
            st.error("Please run the heatsink analysis first!")
        else:
            try:
                run_heatsink_evolution(num_iterations=10)
            except Exception as e:
                st.error(f"Error running evolution process: {e}")
    
    # Go to Home button: mimics the reference behavior
    if st.button("Go to Home"):
        st.session_state.page = "main"
        st.session_state.page = "Physical Relationship Analysis"

def run():
    physical_relationship_analysis()

if __name__ == "__main__":
    run()
