import streamlit as st

def run():
    # Prompt the user to select a dataset.
    st.write("Please Select a Dataset:")

    # Create two columns for the dataset selection buttons.
    col1, col2 = st.columns(2)
    if col1.button("Corrosion"):
        st.session_state['dataset'] = 'corrosion'
    if col2.button("Heatsink"):
        st.session_state['dataset'] = 'heatsink'

    # If a dataset has been selected, display the output variable options.
    if 'dataset' in st.session_state:
        st.write("Please Select an Output Variable of Choice:")
        # Depending on the selected dataset, create two columns with appropriate buttons.
        if st.session_state['dataset'] == 'corrosion':
            col1_out, col2_out = st.columns(2)
            if col1_out.button("Corrosion Rate"):
                st.write("Corrosion Rate selected. [Further implementation goes here]")
            if col2_out.button("Saturation Ratio"):
                st.write("Saturation Ratio selected. [Further implementation goes here]")
        elif st.session_state['dataset'] == 'heatsink':
            col1_out, col2_out = st.columns(2)
            if col1_out.button("Pressure Drop"):
                st.write("Pressure Drop selected. [Further implementation goes here]")
            if col2_out.button("Thermal Resistance"):
                st.write("Thermal Resistance selected. [Further implementation goes here]")

    # "Go To Home" button is always visible.
    if st.button("Go To Home"):
        # Optionally clear the dataset selection if needed.
        if 'dataset' in st.session_state:
            del st.session_state['dataset']
        st.experimental_set_query_params(page="Home")
        st.experimental_rerun()
