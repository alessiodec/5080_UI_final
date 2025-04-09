import streamlit as st
from functions.optimisation_functions import (
    minimise_cr, 
    plot_pareto_front_traverse, 
    get_selected_design_info
)

# Reset session state for dataset if needed.
st.session_state["dataset_choice"] = None

# Initialize session state for navigation if not already set.
if "optimisation_page" not in st.session_state:
    st.session_state["optimisation_page"] = "optimisation"

def optimisation_menu():
    st.title("Optimisation")
    st.write("""
    **Optimisation methods:**  
    - **Minimise CR for Given d and PCO₂:** Specify the pipe diameter and CO₂ pressure; a differential evolution algorithm will optimise the remaining design variables.
    - **Traverse Pareto Front:** Explore trade-offs among Pareto‐optimal designs with a slider that sets your preference between robustness (sensitivity) and low corrosion rate (CR).
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
    This page allows you to explore the Pareto front of the multi-objective optimisation.
    
    **Preference Slider:**  
    Move the slider to adjust your preference between Sensitivity (robustness) and CR (corrosion rate).
    
    - **At 0.0:** Full preference for Sensitivity (weight vector: [1, 0]).
    - **At 1.0:** Full preference for CR (weight vector: [0, 1]).
    - Values in between yield a weighted vector (1 - slider_value, slider_value).
    
    After adjusting the slider, press the **"Plot Pareto Front"** button to update the graph and display the corresponding design details.
    """)
    
    slider_value = st.slider("Preference Slider (0.0 to 1.0)", min_value=0.0, max_value=1.0,
                               value=0.5, step=0.01)
    weight_vector = [1 - slider_value, slider_value]
    st.write(f"Current preference vector: {weight_vector}")
    
    if st.button("Plot Pareto Front"):
        try:
            # plot_pareto_front_traverse returns both the figure and the index of the selected design.
            fig, robustI = plot_pareto_front_traverse(weight_vector[0], weight_vector[1])
            st.pyplot(fig)
            design_info = get_selected_design_info(robustI)
            st.write("**Selected Design Details:**")
            st.write(design_info)
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
