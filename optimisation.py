import streamlit as st
# --- Ensure all necessary functions are imported ---
from functions.optimisation_functions import (
    minimise_cr,
    plot_pareto_front_traverse,
    get_selected_design_info
)
import matplotlib.pyplot as plt # Needed for st.pyplot
# Removed traceback as it wasn't in the original user code provided for this version

# Reset the dataset choice whenever this page is loaded.
# Reset session state for dataset if needed.
# --- Keeping user's original session state handling ---
if "dataset_choice" in st.session_state: # Check if exists before trying to set
    st.session_state["dataset_choice"] = None

# Initialize session state for optimisation navigation if not already set.
# Initialize session state for navigation if not already set.
if "optimisation_page" not in st.session_state:
    st.session_state["optimisation_page"] = "optimisation"

def optimisation_menu():
    st.title("Optimisation")
    # --- Using user's original description ---
    st.write("""
    **Optimisation methods:**\n
    **Minimise CR for Given d and PCO₂:** Specify the pipe diameter and CO₂ pressure; a differential evolution algorithm will optimise the remaining design variables.
    - This page allows you to minimise the corrosion rate by specifying the pipe diameter and CO₂ partial pressure.
    - Select desired diameter and CO₂ pressure, then minimise CR (SR ≤ 1).
    - A differential evolution algorithm is used to optimise the remaining parameters while keeping some values fixed.
    - *Please wait up to one minute for each optimisation algorithm to be completed.*
    
    **Traverse Pareto Front:** Explore trade-offs among Pareto‐optimal designs with a slider that sets your preference between minimising pressure drop or thermal resistance across a heatsink.
    """)

    # --- Using user's original buttons and logic ---
    if st.button("Minimise CR for Given d and PCO₂"):
        st.session_state["optimisation_page"] = "minimise_cr"
        st.rerun() # Added rerun for immediate navigation as is typical

    if st.button("Traverse Pareto Front"):
        st.session_state["optimisation_page"] = "pareto_front"
        st.rerun() # Added rerun for immediate navigation

    if st.button("Go to Home"):
         st.session_state["optimisation_page"] = "optimisation"
         # removed rerun() to match original exactly

def minimise_cr_page():
    st.title("Minimise Corrosion Rate (CR)")
    # --- Using user's original description ---
    st.write("Enter the required inputs below:")

    # --- Using user's original number inputs ---
    d = st.number_input(
        "Enter pipe diameter (d, m) [0.01, 1]:",
        min_value=0.01,
        max_value=1.0,
        value=0.01, # Original default
        step=0.01
    )
    PCO2 = st.number_input(
        "Enter CO₂ pressure (PCO₂, Pa) [10000, 99999]:", # Original range
        min_value=10000.0, # Original min
        max_value=99999.0, # Original max
        value=10000.0, # Original default
        step=1000.0
    )

    # --- Using user's original button and logic ---
    if st.button("Run Optimisation"):
        try:
            # Assuming minimise_cr prints its own results as per optimisation_functions.py
            minimise_cr(d, PCO2)
        except Exception as e:
            st.error(f"Error running optimisation: {e}")


    # --- Using user's original button and logic ---
    if st.button("Go to Optimisation Menu"):
        st.session_state["optimisation_page"] = "optimisation"
        st.rerun() # Added rerun for immediate navigation

def traverse_pareto_page():
    st.title("Traverse Pareto Front")
    # --- Using user's original description ---
    st.write("""
    This page allows you to interactively explore the Pareto front of a multi-objective optimisation problem.

    **Preference Slider:**
    Use the slider below to set your preference between minimising Pressure Drop (left) and Thermal Resistance (right).\n
    After adjusting the slider, press the **"Plot Pareto Front"** button to confirm / update your selection and display the corresponding design.
    
    """)

    # --- Using user's original slider ---
    slider_value = st.slider("Preference Slider (0.0 to 1.0)", min_value=0.0, max_value=1.0,
                               value=0.5, step=0.01)

    # --- Using user's original weight calculation and display ---
    weight_vector = [1 - slider_value, slider_value]
    st.write(f"Current preference vector: {weight_vector}")


    # --- Using user's original button and logic, with MODIFIED print section ---
    if st.button("Plot Pareto Front"):
        try:
            # --- User's original call, captures both fig and index ---
            # plot_pareto_front_traverse returns both the figure and the index of the selected design.
            fig, robustI = plot_pareto_front_traverse(weight_vector[0], weight_vector[1])
            st.pyplot(fig)
            plt.close(fig) # Added close figure as good practice

            # --- START OF REPLACEMENT for st.write(design_info) ---
            if robustI is not None: # Check if index is valid
                st.write("**Selected Design Details:**") # Keep original header
                design_info = get_selected_design_info(robustI) # Get the dictionary

                # Extract values
                design_vars = design_info["Design Variables"].flatten()
                pred_dp = design_info["Predicted DP"].item()
                pred_tr = design_info["Predicted TR"].item()

                # Format the output strings exactly as requested
                optimal_design_str = (
                    f"**Optimal Design:** G1 = {design_vars[0]:.2f}, "
                    f"G2 = {design_vars[1]:.2f}, "
                )

                predicted_outputs_str = (
                    f"**Predicted Outputs:** PD = {pred_dp:.4f} [Pa], "
                    f"TR = {pred_tr:.4f} [°C/W]"
                )

                # Display the formatted strings using st.markdown
                st.markdown(optimal_design_str)
                st.markdown(predicted_outputs_str)
            else:
                st.warning("Could not identify a specific design for the selected weights.")
            # --- END OF REPLACEMENT ---

        except Exception as e:
            st.error(f"Error plotting Pareto front: {e}")


    # --- Using user's original button and logic ---
    if st.button("Go to Optimisation Menu"):
        st.session_state["optimisation_page"] = "optimisation"
        st.rerun() # Added rerun for immediate navigation

# --- Using user's original run function ---
def run():
    page = st.session_state.get("optimisation_page", "optimisation")
    if page == "optimisation":
        optimisation_menu()
    elif page == "minimise_cr":
        minimise_cr_page()
    elif page == "pareto_front":
        traverse_pareto_page()
    # No else block in user's original

# --- Using user's original __main__ block ---
if __name__ == "__main__":
    run()
