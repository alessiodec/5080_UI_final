import streamlit as st
import matplotlib.pyplot as plt
# from functions.symbolic_regression_functions import run_symbolic_regression

def run():
    # Page title and explanation placeholder.
    st.title("Symbolic Regression")
    st.write("This page allows you to configure and run a symbolic regression algorithm. [Explanation placeholder for the page]")

    # If a dataset is selected, always show the "Return to Dataset Selection" button at the top.
    if 'dataset' in st.session_state:
        if st.button("Return to Dataset Selection"):
            st.session_state.pop('dataset', None)
            st.session_state.pop('output', None)
            st.experimental_rerun()

    # Dataset selection: if no dataset is selected, show dataset buttons.
    if 'dataset' not in st.session_state:
        st.write("Please Select a Dataset:")
        col1, col2 = st.columns(2)
        if col1.button("Corrosion"):
            st.session_state['dataset'] = 'corrosion'
        if col2.button("Heatsink"):
            st.session_state['dataset'] = 'heatsink'
    # If a dataset has been selected, show output variable selection.
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

    # Once both dataset and output variable are selected, show regression input fields.
    if 'dataset' in st.session_state and 'output' in st.session_state:
        st.write("Please Select the Regression Inputs:")
        population_size = st.number_input("Population Size", value=100, step=1)
        population_retention_size = st.number_input("Population Retention Size", value=50, step=1)
        number_of_iterations = st.number_input("Number of Iterations", value=10, step=1)

        # Display Run Regression button once all inputs are provided.
        if st.button("Run Regression"):
            st.write("Run Regression pressed. [No functionality implemented yet]")

    # Always show the "Go To Home" button at the bottom.
    if st.button("Go To Home"):
        st.session_state.pop('dataset', None)
        st.session_state.pop('output', None)
        st.experimental_set_query_params(page="Home")
        st.experimental_rerun()
