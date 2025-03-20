import streamlit as st

def run():
    # Title and explanation for the page.
    st.title("Symbolic Regression")
    st.write("This page allows you to configure and run a symbolic regression algorithm. [Explanation placeholder for the page]")

    # Prompt the user to select a dataset.
    st.write("Please Select a Dataset:")

    # Create two columns for the dataset selection buttons.
    col1, col2 = st.columns(2)
    if col1.button("Corrosion"):
        st.session_state['dataset'] = 'corrosion'
        # Reset output if dataset is changed.
        if 'output' in st.session_state:
            del st.session_state['output']
    if col2.button("Heatsink"):
        st.session_state['dataset'] = 'heatsink'
        if 'output' in st.session_state:
            del st.session_state['output']

    # If a dataset has been selected, display the output variable options.
    if 'dataset' in st.session_state:
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

    # Once both dataset and output variable are selected, display the regression inputs.
    if 'dataset' in st.session_state and 'output' in st.session_state:
        st.write("Please Select the Regression Inputs:")
        population_size = st.number_input("Population Size", value=100, step=1)
        population_retention_size = st.number_input("Population Retention Size", value=50, step=1)
        number_of_iterations = st.number_input("Number of Iterations", value=10, step=1)

    # "Go To Home" button to return to the main page.
    if st.button("Go To Home"):
        st.session_state.pop('dataset', None)
        st.session_state.pop('output', None)
        st.experimental_set_query_params(page="Home")
        st.experimental_rerun()
