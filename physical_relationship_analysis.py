import streamlit as st
from functions.physical_relationship_analysis_functions import (
    load_heatsink_data,
    run_heatsink_analysis,
    run_heatsink_evolution
)

# Initialize a session state variable to track which dataset option is selected.
if "dataset_choice" not in st.session_state:
    st.session_state["dataset_choice"] = None

def main_menu():
    st.title("Physical Relationship Analysis")
    st.write("Please choose a dataset:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Corrosion Dataset"):
            st.session_state["dataset_choice"] = "corrosion"
    with col2:
        if st.button("Heatsink Dataset"):
            st.session_state["dataset_choice"] = "heatsink"

def corrosion_page():
    st.title("Corrosion Dataset")
    st.info("Not ready yet.")

def heatsink_page():
    st.title("Heatsink Dataset Analysis")
    # Automatically load heatsink data if not already loaded.
    if "heatsink_data" not in st.session_state:
        try:
            df, X, y, standardised_y, mean_y, std_y = load_heatsink_data(display_output=True)
            st.session_state["heatsink_data"] = (df, X, y, standardised_y, mean_y, std_y)
            st.success("Heatsink data loaded successfully.")
        except Exception as e:
            st.error(f"Error loading heatsink data: {e}")
            return  # Abort further processing if data cannot be loaded.
    
    st.write("Enter parameters for analysis:")
    pop_size = st.number_input("Population Size:", min_value=10, value=200, step=10)
    pop_retention = st.number_input("Population Retention Size:", min_value=1, value=50, step=1)
    num_iterations = st.number_input("Number of Iterations:", min_value=1, value=10, step=1)
    if st.button("Confirm Parameters"):
        try:
            run_heatsink_analysis(pop_size, pop_retention, num_iterations)
            run_heatsink_evolution(num_iterations)
            st.success("Heatsink analysis and evolution completed successfully.")
        except Exception as e:
            st.error(f"Error running analysis/evolution: {e}")

def main():
    # If no dataset has been chosen, show the main menu.
    if st.session_state["dataset_choice"] is None:
        main_menu()
    # Otherwise, route based on the user’s choice.
    elif st.session_state["dataset_choice"] == "corrosion":
        corrosion_page()
    elif st.session_state["dataset_choice"] == "heatsink":
        heatsink_page()
    else:
        st.write("Invalid selection.")

def run():
    main()

if __name__ == "__main__":
    run()
