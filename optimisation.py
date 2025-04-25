import streamlit as st
# --- Combine imports for clarity ---
from functions.optimisation_functions import (
    minimise_cr,
    plot_pareto_front_traverse,
    get_selected_design_info
)
import matplotlib.pyplot as plt # Ensure matplotlib is imported if st.pyplot is used
import traceback # Useful for error reporting

# Reset the dataset choice whenever this page is loaded.
# Reset session state for dataset if needed.
# If "dataset_choice" is used elsewhere, keep this, otherwise it might be redundant here.
if "dataset_choice" in st.session_state:
    st.session_state["dataset_choice"] = None

# Initialize session state for optimisation navigation if not already set.
if "optimisation_page" not in st.session_state:
    st.session_state["optimisation_page"] = "optimisation" # Default to menu

def optimisation_menu():
    st.title("Optimisation")
    # --- Use consistent Markdown/write formatting ---
    st.markdown("""
    **Optimisation Methods:**

    Select one of the optimisation tasks below:

    *   **Minimise CR for Given d and PCO₂:** Specify the pipe diameter (d) and CO₂ partial pressure (PCO₂). The tool uses a differential evolution algorithm to find the optimal pH, Temperature (T), and Velocity (v) that minimise the Corrosion Rate (CR), while ensuring the Saturation Ratio (SR) is feasible (SR ≥ 1). *Please allow up to one minute for the optimisation.*

    *   **Traverse Pareto Front:** Explore the trade-off between minimising Corrosion Rate (CR) and minimising its Sensitivity ($||\nabla CR\|$) using a pre-calculated set of optimal solutions (Pareto front). Use a slider to define your preference between these two conflicting objectives.
    """)

    col1, col2, col3 = st.columns([1,1,1]) # Adjust column ratios as needed
    with col1:
        if st.button("Minimise CR for Given d and PCO₂", key="nav_min_cr"):
            st.session_state["optimisation_page"] = "minimise_cr"
            st.rerun() # Force rerun to navigate immediately
    with col2:
        if st.button("Traverse Pareto Front", key="nav_pareto"):
            st.session_state["optimisation_page"] = "pareto_front"
            st.rerun() # Force rerun to navigate immediately
    # Removed "Go to Home" button as sidebar handles main navigation

def minimise_cr_page():
    st.title("Minimise Corrosion Rate (CR)")
    st.markdown("""
    Enter the fixed values for pipe diameter (d) and CO₂ partial pressure (PCO₂).
    The optimisation will find the best **pH**, **Temperature (T)**, and **Velocity (v)** to minimise CR, subject to SR ≥ 1.
    """)

    # Use columns for input fields for better layout
    col1, col2 = st.columns(2)
    with col1:
        d = st.number_input(
            "Enter pipe diameter (d, meters) [Range: 0.01 to 1.0]",
            min_value=0.01,
            max_value=1.0,
            value=0.1, # Sensible default
            step=0.01,
            format="%.3f", # Format for meters
            key="d_input_mincr"
        )
    with col2:
        PCO2 = st.number_input(
            "Enter CO₂ pressure (PCO₂, Pascals) [Range: 100 to 1,000,000]", # Wider, more realistic range
            min_value=100.0,
            max_value=1000000.0,
            value=10000.0, # Default Pa
            step=1000.0,
            format="%.1f", # Format for Pascals
            key="pco2_input_mincr"
        )

    if st.button("Run CR Minimisation", key="run_min_cr_button"):
        with st.spinner("Running optimisation... This may take up to a minute."):
            try:
                # The minimise_cr function should handle displaying its own results (table)
                minimise_cr(d, PCO2)
                st.success("Optimisation completed.")
            except Exception as e:
                st.error(f"An error occurred during optimisation:")
                st.error(e)
                # st.code(traceback.format_exc()) # Uncomment for detailed debugging

    # Navigation back button
    if st.button("Back to Optimisation Menu", key="back_from_mincr"):
        st.session_state["optimisation_page"] = "optimisation"
        st.rerun()

def traverse_pareto_page():
    st.title("Traverse Pareto Front (Sensitivity vs. CR)")
    st.markdown("""
    Explore the trade-off between minimising Corrosion Rate (CR) and its Sensitivity ($||\nabla CR\|$).
    Sensitivity indicates how much CR changes with small input variations (lower is more robust).

    **Preference Slider:**
    Use the slider to set your preference. The weights represent importance (they sum to 1).
    - **Weight for Sensitivity:** Higher value prioritises robustness.
    - **Weight for CR:** Higher value prioritises lower corrosion.
    """)

    # Slider for CR weight (Objective 2)
    # We define the slider based on CR weight, sensitivity weight is derived
    weight_cr = st.slider(
        "Weight Importance: Minimise CR",
        min_value=0.01,  # Avoid exact 0 or 1 for numerical stability if needed
        max_value=0.99,
        value=0.5,       # Default to equal importance
        step=0.01,
        key="pareto_weight_slider",
        help="Slide right to prioritise lower CR, slide left to prioritise lower Sensitivity."
    )
    # Calculate sensitivity weight
    weight_sensitivity = 1.0 - weight_cr

    # Display the current weights clearly
    st.write(f"Current Weights: Sensitivity = {weight_sensitivity:.2f}, CR = {weight_cr:.2f}")

    if st.button("Plot Pareto Front and Show Selected Design", key="plot_pareto_button"):
        with st.spinner("Processing..."):
            try:
                # plot_pareto_front_traverse returns both the figure and the index
                fig, robustI = plot_pareto_front_traverse(weight_sensitivity, weight_cr)

                st.pyplot(fig)
                plt.close(fig) # Close the plot figure to free memory

                # --- START OF MODIFIED SECTION ---
                if robustI is not None:
                    st.write("**Selected Design Details:**")
                    design_info = get_selected_design_info(robustI) # Get the dict

                    # Extract values
                    design_vars = design_info["Design Variables"].flatten()
                    pred_cr = design_info["Predicted CR"].item()
                    pred_sr = design_info["Predicted SR"].item()

                    # Format the output strings
                    optimal_design_str = (
                        f"**Optimal Design Inputs:** pH = {design_vars[0]:.4f}, "
                        f"T = {design_vars[1]:.2f} (°C), "
                        f"PCO₂ = {design_vars[2]:.2f} (Pa), "
                        f"v = {design_vars[3]:.4f} (m/s), "
                        f"d = {design_vars[4]:.4f} (m)"
                    )

                    predicted_outputs_str = (
                        f"**Predicted Outputs:** CR = {pred_cr:.4f} (mm/year), "
                        f"SR = {pred_sr:.4f} (-)"
                    )

                    # Display the formatted strings
                    st.markdown(optimal_design_str)
                    st.markdown(predicted_outputs_str)

                    # Optional: Display objective values too for context
                    # obj_vals = design_info["Objective Values (Sensitivity, CR)"]
                    # st.write(f"**Objective Values:** Sensitivity = {obj_vals[0]:.5f}, CR = {obj_vals[1]:.5f}")

                else:
                     st.warning("Could not identify a design for the selected weights.")
                # --- END OF MODIFIED SECTION ---

            except Exception as e:
                st.error(f"An error occurred while plotting or retrieving design details:")
                st.error(e)
                # st.code(traceback.format_exc()) # Uncomment for detailed debugging

    # Navigation back button
    if st.button("Back to Optimisation Menu", key="back_from_pareto"):
        st.session_state["optimisation_page"] = "optimisation"
        st.rerun()

# --- Main run function that routes based on session state ---
def run():
    """Calls the appropriate function based on the navigation state."""
    page = st.session_state.get("optimisation_page", "optimisation") # Default to menu
    if page == "optimisation":
        optimisation_menu()
    elif page == "minimise_cr":
        minimise_cr_page()
    elif page == "pareto_front":
        traverse_pareto_page()
    else: # Fallback in case state is invalid
        st.warning("Invalid page state. Returning to Optimisation Menu.")
        st.session_state["optimisation_page"] = "optimisation"
        optimisation_menu()


# # This block is usually removed when running as part of a larger app via app.py
# if __name__ == "__main__":
#     # Set a default state if running directly for testing
#     if "optimisation_page" not in st.session_state:
#         st.session_state["optimisation_page"] = "optimisation"
#     run()
