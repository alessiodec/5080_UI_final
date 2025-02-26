import streamlit as st
from functions.physical_relationship_analysis_functions import (
    load_heatsink_data,
    run_heatsink_analysis,
    run_heatsink_evolution
)

# Initialize navigation state for Physical Relationship Analysis if not already set.
if "physical_relationship_page" not in st.session_state:
    st.session_state["physical_relationship_page"] = "main"

def main_menu():
    st.title("Physical Relationship Analysis")
    st.write("Select an option:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Load Heatsink Data"):
            st.session_state["physical_relationship_page"] = "load_data"
    with col2:
        if st.button("Run Heatsink Analysis"):
            st.session_state["physical_relationship_page"] = "run_analysis"
    with col3:
        if st.button("Run Heatsink Evolution"):
            st.session_state["physical_relationship_page"] = "run_evolution"
    if st.button("Go to Home"):
        # Set the main app page to main (you may have a similar mechanism in app.py)
        st.session_state["page"] = "main"

def load_data_page():
    st.title("Load Heatsink Data")
    try:
        # Call the function with display_output=True so that key values are shown
        df, X, y, standardised_y, mean_y, std_y = load_heatsink_data(display_output=True)
        st.session_state["heatsink_data"] = (df, X, y, standardised_y, mean_y, std_y)
        st.success("Heatsink data loaded and stored in session state.")
    except Exception as e:
        st.error(f"Error loading heatsink data: {e}")
    if st.button("Go Back"):
        st.session_state["physical_relationship_page"] = "main"

def run_analysis_page():
    st.title("Run Heatsink Analysis")
    st.write("Enter parameters for the analysis:")
    pop_size = st.number_input("Population Size:", min_value=10, value=200, step=10)
    pop_retention = st.number_input("Population Retention Size:", min_value=1, value=50, step=1)
    num_iterations = st.number_input("Number of Iterations:", min_value=1, value=10, step=1)
    if st.button("Run Analysis"):
        try:
            run_heatsink_analysis(pop_size, pop_retention, num_iterations)
        except Exception as e:
            st.error(f"Error running heatsink analysis: {e}")
    if st.button("Go Back"):
        st.session_state["physical_relationship_page"] = "main"

def run_evolution_page():
    st.title("Run Heatsink Evolution")
    num_iterations = st.number_input("Number of Iterations:", min_value=1, value=10, step=1)
    if st.button("Run Evolution"):
        try:
            run_heatsink_evolution(num_iterations)
        except Exception as e:
            st.error(f"Error running evolution process: {e}")
    if st.button("Go Back"):
        st.session_state["physical_relationship_page"] = "main"

def run():
    page = st.session_state.get("physical_relationship_page", "main")
    if page == "main":
        main_menu()
    elif page == "load_data":
        load_data_page()
    elif page == "run_analysis":
        run_analysis_page()
    elif page == "run_evolution":
        run_evolution_page()

if __name__ == "__main__":
    run()
