import streamlit as st
import matplotlib.pyplot as plt
from functions.symbolic_regression_functions import run_evolution_experiment

def run():
    # Page title and explanation.
    st.title("Symbolic Regression")
    st.write(
        "This page enables users to configure and execute the symbolic regression algorithm "
        "by first selecting a dataset (e.g., Corrosion or Heatsink) and the desired output variable, "
        "then specifying key parameters such as population size, retention size, and the number of iterations. "
        "Once executed, the code performs an evolutionary search—using genetic programming principles—to "
        "iteratively refine candidate mathematical expressions until it identifies the best-fitting model. "
        "That expression is then converted to a symbolic equation and displayed in LaTeX format."
    )

    # Button to return to dataset selection if one is already selected.
    if 'dataset' in st.session_state:
        if st.button("Return to Dataset Selection"):
            st.session_state.pop('dataset', None)
            st.session_state.pop('output', None)
            st.experimental_rerun()

    # If no dataset is selected, show dataset selection buttons.
    if 'dataset' not in st.session_state:
        st.write("Please select a dataset:")
        col1, col2 = st.columns(2)
        if col1.button("Corrosion"):
            st.session_state['dataset'] = 'corrosion'
        if col2.button("Heatsink"):
            st.session_state['dataset'] = 'heatsink'
    else:
        st.write(f"Dataset selected: {st.session_state['dataset'].capitalize()}")
        st.write("Please select an output variable:")
        if st.session_state['dataset'] == 'corrosion':
            col1_out, col2_out = st.columns(2)
            if col1_out.button("Corrosion Rate"):
                st.session_state['output'] = 'corrosion_rate'
            if col2_out.button("Saturation Ratio"):
                st.session_state['output'] = 'saturation_ratio'
        elif st.session_state['dataset'] == 'heatsink':
            col1_out, col2_out = st.columns(2)
            if col1_out.button("Pressure Drop"):
                st.session_state['output'] = 'Pressure_Drop'
            if col2_out.button("Thermal Resistance"):
                st.session_state['output'] = 'Thermal_Resistance'

    # If both dataset and output variable are selected, show evolution parameter inputs.
    if 'dataset' in st.session_state and 'output' in st.session_state:
        st.write("Please select regression parameters:")
        population_size = st.number_input("Population Size", value=100, min_value=10, step=1)
        population_retention_size = st.number_input("Population Retention Size", value=20, min_value=1, step=1)
        number_of_iterations = st.number_input("Number of Iterations", value=10, min_value=1, step=1)

        if st.button("Run Regression"):
            dataset_selected = st.session_state['dataset']
            output_selected = st.session_state['output']
            if dataset_selected.lower() == 'corrosion':
                dataset_choice = "CORROSION"
                # Map outputs: "corrosion_rate" to "CR", "saturation_ratio" to "SR"
                if output_selected.lower() == 'corrosion_rate':
                    output_var = "CR"
                elif output_selected.lower() == 'saturation_ratio':
                    output_var = "SR"
                else:
                    output_var = output_selected
            elif dataset_selected.lower() == 'heatsink':
                dataset_choice = "HEATSINK"
                # Map outputs: "Pressure_Drop" and "Thermal_Resistance" pass through as given.
                if output_selected.lower() == 'pressure_drop':
                    output_var = "Pressure_Drop"
                elif output_selected.lower() == 'thermal_resistance':
                    output_var = "Thermal_Resistance"
                else:
                    output_var = output_selected
            else:
                st.error("Invalid dataset selection.")
                return

            run_evolution_experiment(dataset_choice, output_var, population_size, population_retention_size, number_of_iterations)

    if st.button("Go To Home"):
        st.session_state.pop('dataset', None)
        st.session_state.pop('output', None)
        st.experimental_set_query_params(page="Home")
        st.experimental_rerun()

if __name__ == "__main__":
    run()
