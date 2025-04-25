import streamlit as st
import matplotlib.pyplot as plt
import traceback # Import traceback for better error reporting

# --- IMPORTANT ---
# Ensure the 'functions' directory is accessible from where you run streamlit.
# If 'optimisation.py' is in the root and 'functions' is a subdirectory,
# this import should work. If your structure is different, adjust the path.
try:
    from functions.optimisation_functions import (
        minimise_cr,
        plot_pareto_front_traverse,
        get_selected_design_info
        # Add other functions if needed by this page specifically
    )
except ImportError as e:
    st.error(f"Error importing optimisation functions: {e}")
    st.error("Please ensure 'functions/optimisation_functions.py' exists and is accessible.")
    st.stop() # Stop execution if functions can't be imported

def run():
    """
    Defines the Streamlit UI and logic for the Optimisation page.
    """
    st.title("Optimisation")

    # Use a selectbox for choosing the optimisation task
    optimisation_choice = st.selectbox(
        "Select Optimisation Task:",
        ["Select...", "Minimise Corrosion Rate (Single Objective)", "Traverse Pareto Front (Multi-Objective)"],
        key="optimisation_task_selector" # Add a key for stability
    )

    # --- Task 1: Single Objective Minimisation ---
    if optimisation_choice == "Minimise Corrosion Rate (Single Objective)":
        st.header("Minimise Corrosion Rate (CR)")
        st.markdown(
            """
            Optimise **pH**, **Temperature (T)**, and **Velocity (v)** to minimise Corrosion Rate (CR),
            subject to the constraint that the Saturation Ratio (SR) must be greater than or equal to 1 (SR ≥ 1).

            Please provide the fixed values for the other two input parameters:
            *   **Partial Pressure of CO₂ (PCO₂) [Pa]**
            *   **Pipe Diameter (d) [m]**
            """
        )

        # User inputs for fixed parameters using columns for better layout
        col1, col2 = st.columns(2)
        with col1:
            pco2_fixed = st.number_input(
                "Fixed Partial Pressure of CO₂ (PCO₂) [Pa]",
                min_value=100.0,       # Example plausible minimum
                max_value=1000000.0,   # Example plausible maximum
                value=10000.0,         # Default value
                step=100.0,            # Step size
                format="%.1f",         # Display format
                key="pco2_fixed_input" # Unique key
            )
        with col2:
            d_fixed = st.number_input(
                "Fixed Pipe Diameter (d) [m]",
                min_value=0.01,        # Example plausible minimum
                max_value=1.0,         # Example plausible maximum
                value=0.1,             # Default value
                step=0.01,             # Step size
                format="%.3f",         # Display format
                key="d_fixed_input"    # Unique key
            )

        if st.button("Run CR Minimisation", key="run_cr_min_button"):
            with st.spinner("Running optimisation... This may take up to a minute."):
                try:
                    # Call the minimise_cr function from optimisation_functions.py
                    # This function now handles printing the results table internally.
                    minimise_cr(d=d_fixed, PCO2=pco2_fixed)
                    st.success("Optimisation process completed.") # Add success message

                except Exception as e:
                    st.error(f"An error occurred during optimisation:")
                    st.error(e)
                    # Optionally show detailed traceback for debugging
                    # st.error("Traceback:")
                    # st.code(traceback.format_exc())


    # --- Task 2: Multi-Objective Pareto Front Exploration ---
    elif optimisation_choice == "Traverse Pareto Front (Multi-Objective)":
        st.header("Explore Pre-computed Pareto Front (Sensitivity vs. CR)")
        st.markdown(
            """
            This section allows you to explore a pre-calculated Pareto optimal front for the multi-objective problem
            of simultaneously minimising Corrosion Rate (CR) and its Sensitivity ($||\nabla CR\|$).
            The Sensitivity measures how much CR changes with small changes in the input variables.

            Adjust the **weights** below to indicate the relative importance you place on minimising CR versus
            minimising its Sensitivity. The tool will then identify the optimal design on the Pareto front
            that best matches your preferences according to the weighted sum method.

            *   A higher weight for **Minimise CR** prioritises lower corrosion rates.
            *   A higher weight for **Minimise Sensitivity** prioritises solutions that are more robust to input variations.
            """
        )

        # Use columns for slider and calculated weight display
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            # Slider for CR weight (Objective 2)
            weight_cr = st.slider(
                "Weight Importance: Minimise CR",
                min_value=0.01,          # Avoid exact 0 or 1 for weighted sum stability
                max_value=0.99,
                value=0.5,               # Default to equal importance
                step=0.01,
                key="weight_cr_slider",  # Unique key
                help="Higher value means lower CR is more important."
            )
        with col_w2:
            # Sensitivity weight is automatically calculated (weights must sum to 1)
            weight_sensitivity = 1.0 - weight_cr
            st.metric(
                label="Weight Importance: Minimise Sensitivity",
                value=f"{weight_sensitivity:.2f}",
                help="Higher value means lower sensitivity is more important. Calculated as (1 - Weight for CR)."
                )


        if st.button("Plot Pareto Front and Find Design", key="plot_pareto_button"):
            with st.spinner("Plotting Pareto front and identifying selected design..."):
                try:
                    # Call the plotting function which returns the figure and the index
                    fig, robustI = plot_pareto_front_traverse(weight_sensitivity, weight_cr)

                    # Display the generated plot
                    st.pyplot(fig)
                    plt.close(fig) # Good practice to close the figure explicitly

                    # --- START OF CORRECTED DISPLAY SECTION ---
                    if robustI is not None: # Check if a valid index was returned
                        st.subheader("Selected Optimal Design Details")

                        # Get the dictionary containing details for the selected index
                        design_info = get_selected_design_info(robustI)

                        # Extract design variables array and ensure it's 1D
                        design_vars = design_info["Design Variables"].flatten()

                        # Extract predicted CR and SR scalar values
                        pred_cr = design_info["Predicted CR"].item()
                        pred_sr = design_info["Predicted SR"].item()

                        # Format the optimal design string with labels and units
                        optimal_design_str = (
                            f"**Optimal Design Inputs:** \n"
                            f"- pH = {design_vars[0]:.4f} (-) \n"
                            f"- T = {design_vars[1]:.2f} (°C) \n"
                            f"- PCO₂ = {design_vars[2]:.2f} (Pa) \n"
                            f"- v = {design_vars[3]:.4f} (m/s) \n"
                            f"- d = {design_vars[4]:.4f} (m)"
                        )

                        # Format the predicted outputs string with labels and units
                        predicted_outputs_str = (
                            f"**Predicted Outputs for this Design:** \n"
                            f"- CR = {pred_cr:.4f} (mm/year) \n"
                            f"- SR = {pred_sr:.4f} (-)"
                        )

                        # Display the formatted strings using st.markdown for better text formatting
                        st.markdown(optimal_design_str)
                        st.markdown(predicted_outputs_str)

                        # Optional: Display the objective values (Sensitivity, CR) used for selection
                        obj_vals = design_info["Objective Values (Sensitivity, CR)"]
                        objective_values_str = (
                           f"**Corresponding Objective Values on Pareto Front:** \n"
                           f"- Sensitivity ($||\nabla CR\|$) = {obj_vals[0]:.5f} \n"
                           f"- CR = {obj_vals[1]:.5f} (mm/year)"
                        )
                        st.markdown(objective_values_str)

                    else:
                        st.warning("Could not determine the optimal design index for the given weights.")
                    # --- END OF CORRECTED DISPLAY SECTION ---

                except Exception as e:
                     st.error(f"An error occurred while processing the Pareto front exploration:")
                     st.error(e)
                     # Optionally show detailed traceback for debugging
                     # st.error("Traceback:")
                     # st.code(traceback.format_exc())

    # You can add more elif blocks here if you introduce other optimisation tasks later

# --- Optional: Allows running this page script directly for testing ---
# Note: This won't have the sidebar navigation from app.py when run directly.
# if __name__ == "__main__":
#    run()
