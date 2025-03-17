import streamlit as st
from functions.optimisation_functions import minimise_cr

# Reset the dataset choice whenever this page is loaded.
st.session_state["dataset_choice"] = None

# Initialize session state for optimisation navigation if not already set.
if "optimisation_page" not in st.session_state:
    st.session_state["optimisation_page"] = "optimisation"

def optimisation_menu():
    st.title("Optimisation")
    
    st.write("""
    **Optimisation methods:**  \n
    - Select desired diamater and CO2 pressure, minimise CR (SR <= 1) \n
    This page allows you to minimise the corrosion rate by specifying the pipe diameter and CO₂ partial pressure. \n
    A differential evolution algorithm is used to optimise the remaining parameters while keeping some values fixed. \n
    - [more to be added] \n
    *Please wait up to one minute for each optimisation algorithm to be completed.* \n
    """)
    
    # Button to navigate to the minimisation page.
    if st.button("Minimise CR for Given d and PCO₂"):
        st.session_state["optimisation_page"] = "minimise_cr"
    # This button can be used to reset back to the Optimisation main menu
    if st.button("Go to Home"):
        st.session_state["optimisation_page"] = "optimisation"

def minimise_cr_page():
    st.title("Minimise Corrosion Rate (CR)")
    st.write("Enter the required inputs below:")
    # Input fields for pipe diameter (d) and CO₂ partial pressure (PCO₂).
    d = st.number_input("Enter pipe diameter (d, m) [0.01, 1]:", min_value=0.1, value=1.0, step=0.1)
    PCO2 = st.number_input("Enter CO₂ pressure (PCO₂, Pa) [10000, 99999]:", min_value=0.1, value=1.0, step=0.1)
    if st.button("Run Optimisation"):
        try:
            best_params, min_cr = minimise_cr(d, PCO2)
            st.success("Optimisation Completed!")
            st.write("Optimised Design Vector (pH, T, CO₂, v, d):", best_params)
            st.write("Minimum Corrosion Rate (CR):", min_cr)
        except Exception as e:
            st.error(f"Error running optimisation: {e}")
    if st.button("Go to Optimisation Menu"):
        st.session_state["optimisation_page"] = "optimisation"

def run():
    page = st.session_state.get("optimisation_page", "optimisation")
    if page == "optimisation":
        optimisation_menu()
    elif page == "minimise_cr":
        minimise_cr_page()

if __name__ == "__main__":
    run()
