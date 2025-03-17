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
    **Optimisation methods:**  
    - Select desired diameter and CO₂ pressure, then minimise CR (SR ≤ 1).  
    - This page allows you to minimise the corrosion rate by specifying the pipe diameter and CO₂ partial pressure.  
    - A differential evolution algorithm is used to optimise the remaining parameters while keeping some values fixed.  
    - [more to be added]  
    *Please wait up to one minute for each optimisation algorithm to be completed.*
    """)

    # Button to navigate to the minimisation page.
    if st.button("Minimise CR for Given d and PCO₂"):
        st.session_state["optimisation_page"] = "minimise_cr"

    # This button can be used to reset back to the Optimisation main menu.
    if st.button("Go to Home"):
        st.session_state["optimisation_page"] = "optimisation"

def minimise_cr_page():
    st.title("Minimise Corrosion Rate (CR)")
    st.write("Enter the required inputs below:")

    # Input fields for pipe diameter (d) and CO₂ partial pressure (PCO₂).
    d = st.number_input(
        "Enter pipe diameter (d, m) [0.01, 1]:",
        min_value=0.01,
        max_value=1.0,
        value=0.01,
        step=0.01
    )
    PCO2 = st.number_input(
        "Enter CO₂ pressure (PCO₂, Pa) [10000, 99999]:",
        min_value=10000.0,
        max_value=99999.0,
        value=10000.0,
        step=1000.0
    )

    if st.button("Run Optimisation"):
        try:
            # Only call the function; no extra prints or summaries here.
            minimise_cr(d, PCO2)
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
