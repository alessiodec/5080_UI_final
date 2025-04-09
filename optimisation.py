import streamlit as st
from functions.optimisation_functions import minimise_cr, plot_pareto_front_traverse

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
    - *Please wait up to one minute for each optimisation algorithm to be completed.*
    """)
    
    if st.button("Minimise CR for Given d and PCO₂"):
        st.session_state["optimisation_page"] = "minimise_cr"
        
    if st.button("Traverse Pareto Front"):
        st.session_state["optimisation_page"] = "pareto_front"
        
    if st.button("Go to Home"):
        st.session_state["optimisation_page"] = "optimisation"

def minimise_cr_page():
    st.title("Minimise Corrosion Rate (CR)")
    st.write("Enter the required inputs below:")
    
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
            minimise_cr(d, PCO2)
        except Exception as e:
            st.error(f"Error running optimisation: {e}")

    if st.button("Go to Optimisation Menu"):
        st.session_state["optimisation_page"] = "optimisation"

def traverse_pareto_page():
    st.title("Traverse Pareto Front")
    st.write("""
    This page allows you to interactively explore the Pareto front generated by the multi-objective optimisation.
    
    **Preference Slider:**  
    Use the slider below to set your preference between Sensitivity and CR.
    
    - At the left end (0.0), the weights are (1, 0), meaning full preference for Sensitivity and none for CR.  
    - At the right end (1.0), the weights are (0, 1), meaning full preference for CR and none for Sensitivity.
    - For values in between, the weight vector is (1 - slider_value, slider_value).
    
    After adjusting the slider, press the **"Plot Pareto Front"** button to confirm your selection.
    """)

    # Slider that outputs a value between 0.0 and 1.0.
    slider_value = st.slider("Preference Slider (0.0 to 1.0)", min_value=0.0, max_value=1.0,
                               value=0.5, step=0.01)
    
    # Compute the weight vector from the slider value.
    weight_vector = [1 - slider_value, slider_value]
    
    # Display the current vector.
    st.write(f"Current preference vector: {weight_vector}")

    if st.button("Plot Pareto Front"):
        try:
            fig = plot_pareto_front_traverse(weight_vector[0], weight_vector[1])
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error plotting Pareto front: {e}")

    if st.button("Go to Optimisation Menu"):
        st.session_state["optimisation_page"] = "optimisation"

def run():
    page = st.session_state.get("optimisation_page", "optimisation")
    if page == "optimisation":
        optimisation_menu()
    elif page == "minimise_cr":
        minimise_cr_page()
    elif page == "pareto_front":
        traverse_pareto_page()

if __name__ == "__main__":
    run()
