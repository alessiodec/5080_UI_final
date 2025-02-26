import streamlit as st
from functions.physical_relationship_analysis_functions import (
    load_heatsink_data,
    run_heatsink_analysis
)

# Initialize session state variables if not already set.
if "heatsink_data_loaded" not in st.session_state:
    st.session_state["heatsink_data_loaded"] = False
if "physical_relationship_page" not in st.session_state:
    st.session_state["physical_relationship_page"] = "main"

def main_menu():
    st.title("Physical Relationship Analysis")
    st.write("Select an option:")

    # Button to load heatsink data is always shown.
    if st.button("Load Heatsink Data"):
        st.session_state["physical_relationship_page"] = "load_data"
    
    # If data has been loaded, show the "Run Heatsink Analysis" button immediately.
    if st.session_state.get("heatsink_data_loaded", False):
        if st.button("Run Heatsink Analysis"):
            st.session_state["physical_relationship_page"] = "run_analysis"
    
    if st.button("Go to Home"):
        st.session_state["page"] = "main"

def load_data_page():
    st.title("Load Heatsink Data")
    try:
        df, X, y, standardised_y, mean_y, std_y = load_heatsink_data(display_output=True)
        st.session_state["heatsink_data"] = (df, X, y, standardised_y, mean_y, std_y)
        st.session_state["heatsink_data_loaded"] = True
        st.success("Heatsink data loaded successfully.")
    except Exception as e:
        st.error(f"Error loading heatsink data: {e}")
    # Provide a button to return to the main menu
    if st.button("Return to Main Menu"):
        st.session_state["physical_relationship_page"] = "main"

def run_analysis_page():
    st.title("Run Heatsink Analysis")
    st.write("Enter parameters for the analysis:")
    pop_size = st.number_input("Population Size:", min_value=10, value=200, step=10)
    pop_retention = st.number_input("Population Retention Size:", min_value=1, value=50, step=1)
    num_iterations = st.number_input("Number of Iterations:", min_value=1, value=10, step=1)
    if st.button("Confirm Parameters"):
        if not st.session_state.get("heatsink_data_loaded", False):
            st.error("Please load heatsink data first!")
        else:
            try:
                run_heatsink_analysis(pop_size, pop_retention, num_iterations)
                st.success("Heatsink analysis completed successfully.")
                # You can set a flag here if you later need it for evolution.
                st.session_state["analysis_done"] = True
            except Exception as e:
                st.error(f"Error running heatsink analysis: {e}")
    if st.button("Return to Main Menu"):
        st.session_state["physical_relationship_page"] = "main"

def main():
    page = st.session_state.get("physical_relationship_page", "main")
    if page == "main":
        main_menu()
    elif page == "load_data":
        load_data_page()
    elif page == "run_analysis":
        run_analysis_page()
    else:
        st.write("Unknown page state.")

def run():
    main()

if __name__ == "__main__":
    run()
