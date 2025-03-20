import streamlit as st

def run():
    # Page title and explanation placeholder.
    st.title("Symbolic Regression")
    st.write("This page allows you to configure and run a symbolic regression algorithm. [Explanation placeholder for the page]")

    # If no dataset has been selected, show the dataset selection buttons.
    if 'dataset' not in st.session_state:
        st.write("Please Select a Dataset:")
        col1, col2 = st.columns(2)
        if col1.button("Corrosion"):
            st.session_state['dataset'] = 'corrosion'
        if col2.button("Heatsink"):
            st.session_state['dataset'] = 'heatsink'

    # If a dataset has been selected, show the output variable selection.
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

    # Once both the dataset and output variable are selected, show the regression input fields.
    if 'dataset' in st.session_state and 'output' in st.session_state:
        st.write("Please Select the Regression Inputs:")
        population_size = st.number_input("Population Size", value=100, step=1)
        population_retention_size = st.number_input("Population Retention Size", value=50, step=1)
        number_of_iterations = st.number_input("Number of Iterations", value=10, step=1)

    # Navigation buttons
    # If a dataset is selected, show both "Return to Dataset Selection" and "Go To Home" side by side.
    if 'dataset' in st.session_state:
        col_return, col_home = st.columns(2)
        if col_return.button("Return to Dataset Selection"):
            st.session_state.pop('dataset', None)
            st.session_state.pop('output', None)
            st.experimental_rerun()
        if col_home.button("Go To Home"):
            st.session_state.pop('dataset', None)
            st.session_state.pop('output', None)
            st.experimental_set_query_params(page="Home")
            st.experimental_rerun()
    # If no dataset is selected, only show the "Go To Home" button.
    else:
        if st.button("Go To Home"):
            st.experimental_set_query_params(page="Home")
            st.experimental_rerun()
