import streamlit as st
from functions.physical_relationship_analysis_functions import (
    load_heatsink_data,
    run_heatsink_analysis_and_evolution
)

# Reset the dataset choice whenever this page is loaded.
st.session_state["dataset_choice"] = None

# Helper function to safely call experimental_rerun
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Rerun failed: {e}")

# Reset dataset choice on page load
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
    if st.button("Go to Home"):
        st.session_state["page"] = "main"

def corrosion_page():
    st.title("Corrosion Dataset")
    st.info("Not ready yet.")
    if st.button("Return to Main Menu"):
        st.session_state["dataset_choice"] = None
        st.experimental_rerun()
        safe_rerun()

def heatsink_page():
    st.title("Heatsink Dataset Analysis")

    st.write("""
    - Uses the heatsink dataset to perform an evolutionary algorithm to breed well-performing candidates from the population
    - User input for population size, retention size, and the number of iterations
    - Real-time plotting of fitness and complexity throughout iterations
    - Pareto front visualisation (planned)
    - Display of the best-fit equation found (planned) \n
    ***note**: since this is hosted online, it may take a while to run expensive computations. It is recommended to use a population size of no greater then 300 to analyse the process with faster results.*
    """)

    # Automatically load heatsink data if not already loaded.
    if "heatsink_data" not in st.session_state:
        try:
            df, X, y, standardised_y, mean_y, std_y = load_heatsink_data(display_output=True)
            st.session_state["heatsink_data"] = (df, X, y, standardised_y, mean_y, std_y)
            st.success("Heatsink data loaded successfully.")
        except Exception as e:
            st.error(f"Error loading heatsink data: {e}")
            return

    st.write("Enter parameters for analysis:")
    pop_size = st.number_input("Population Size:", min_value=10, value=100, step=10)
    st.write("**Retention Size**: Number of top-performing individuals from the current generation that are preserved unchanged and carried over to the next generation, used for breeding.")
    pop_retention = st.number_input("Population Retention Size:", min_value=1, value=20, step=1)
    num_iterations = st.number_input("Number of Iterations:", min_value=1, value=10, step=1)

    if st.button("Run Analysis and Evolution"):
        try:
            run_heatsink_analysis_and_evolution(pop_size, pop_retention, num_iterations)
            st.success("Heatsink analysis and evolution completed successfully.")
        except Exception as e:
            st.error(f"Error running analysis/evolution: {e}")

    if st.button("Return to Main Menu"):
        st.session_state["dataset_choice"] = None
        st.experimental_rerun()
        safe_rerun()

def main():
    choice = st.session_state.get("dataset_choice", None)
    if choice is None:
        main_menu()
    elif choice == "corrosion":
        corrosion_page()
    elif choice == "heatsink":
        heatsink_page()
    else:
        st.write("Invalid selection.")

def run():
    main()

if __name__ == "__main__":
    run()
