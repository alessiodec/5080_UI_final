import streamlit as st
import matplotlib.pyplot as plt
from functions.symbolic_regression_functions import run_evolution_experiment

def run():
    # Page title and explanation.
    st.title("Symbolic Regression")
    st.write("This page allows you to configure and run a symbolic regression algorithm. This page enables users to configure and execute the symbolic regression algorithm by first selecting a dataset (e.g., corrosion or heatsink) and the desired output variable, then specifying key parameters such as population size, retention size, and the number of iterations. Once executed, the code performs an evolutionary search—using genetic programming principles—to iteratively refine candidate mathematical expressions until it identifies the best-fitting model; this expression is then converted to a symbolic equation and displayed in LaTeX format, offering a clear and interpretable insight into the relationships within the data.")

    # Button to return to dataset selection if one is already selected.
    if 'dataset' in st.session_state:
        if st.button("Return to Dataset Selection"):
            st.session_state.pop('dataset', None)
            st.session_state.pop('output', None)
            st.experimental_rerun()

    # If no dataset is selected, display dataset selection buttons.
    if 'dataset' not in st.session_state:
        st.write("Please Select a Dataset:")
        col1, col2 = st.columns(2)
        if col1.button("Corrosion"):
            st.session_state['dataset'] = 'corrosion'
        if col2.button("Heatsink"):
            st.session_state['dataset'] = 'heatsink'
    # If a dataset is selected, display output variable selection.
    else:
        st.write(f"Dataset selected: {st.session_state['dataset'].capitalize()}")
        st.write("Please Select an Output Variable of Choice:")
        if st.session_state['dataset'] == 'corrosion':
            col1_out, col2_out = st.columns(2)
            if col1_out.button("Corrosion Rate"):
                st.session_state['output'] = 'corrosion_rate'
            if col2_out.button("Saturation Ratio"):
                st.session_state['output'] = 'saturation_ratio'
        elif st.session_state['dataset'] == 'heatsink':
            col1_out, col2_out = st.columns(2)
            if col1_out.button("Pressure Drop"):
                st.session_state['output'] = 'pressure_drop'
            if col2_out.button("Thermal Resistance"):
                st.session_state['output'] = 'thermal_resistance'

    # Once both dataset and output are selected, show regression input fields.
    if 'dataset' in st.session_state and 'output' in st.session_state:
        st.write("Please Select the Regression Inputs:")
        population_size = st.number_input("Population Size", value=100, step=1)
        population_retention_size = st.number_input("Population Retention Size", value=50, step=1)
        number_of_iterations = st.number_input("Number of Iterations", value=10, step=1)

        # When Run Regression is pressed, map selections and call the evolution function.
        if st.button("Run Regression"):
            # Map the selected dataset to the format expected by run_evolution_experiment.
            dataset_selected = st.session_state['dataset']
            output_selected = st.session_state['output']
            if dataset_selected.lower() == 'corrosion':
                dataset_choice = "CORROSION"
                # Map output variable: "corrosion_rate" -> "CR", "saturation_ratio" -> "SR"
                if output_selected.lower() == 'corrosion_rate':
                    output_var = "CR"
                elif output_selected.lower() == 'saturation_ratio':
                    output_var = "SR"
                else:
                    output_var = output_selected  # fallback
            elif dataset_selected.lower() == 'heatsink':
                dataset_choice = "HEATSINK"
                # Map output variable: "pressure_drop" -> "Pressure_Drop", "thermal_resistance" -> "Thermal_Resistance"
                if output_selected.lower() == 'pressure_drop':
                    output_var = "Pressure_Drop"
                elif output_selected.lower() == 'thermal_resistance':
                    output_var = "Thermal_Resistance"
                else:
                    output_var = output_selected
            else:
                st.error("Invalid dataset selection.")
                return

            # Call the evolution function with the user-provided inputs.
            run_evolution_experiment(dataset_choice, output_var, population_size, population_retention_size, number_of_iterations)

    # Always show a "Go To Home" button.
    if st.button("Go To Home"):
        st.session_state.pop('dataset', None)
        st.session_state.pop('output', None)
        st.experimental_set_query_params(page="Home")
        st.experimental_rerun()

if __name__ == "__main__":
    run()
