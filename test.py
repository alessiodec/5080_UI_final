import streamlit as st

def run():
    st.title("Test Page")
    st.write("Select one of the options below.")
    
    # Initialize session state variables if not already set
    if "dataset_choice" not in st.session_state:
        st.session_state["dataset_choice"] = None
    if "output_choice" not in st.session_state:
        st.session_state["output_choice"] = None
    
    # STEP 1: First level buttons: "C" or "H"
    col1, col2 = st.columns(2)
    with col1:
        if st.button("C"):
            st.session_state["dataset_choice"] = "CORROSION"  # Save as a string
            # Reset output choice if re-selecting dataset
            st.session_state["output_choice"] = None
    with col2:
        if st.button("H"):
            st.session_state["dataset_choice"] = "HEATSINK"    # Save as a string
            st.session_state["output_choice"] = None

    # Display the chosen dataset if one is selected
    if st.session_state["dataset_choice"]:
        st.write(f"Dataset choice selected: {st.session_state['dataset_choice']}")
        
        # STEP 2: Display second level buttons based on the first selection.
        # If "C" was pressed, show buttons "CR" and "SR".
        # If "H" was pressed, show buttons "PD" and "TR".
        if st.session_state["dataset_choice"] == "CORROSION":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("CR"):
                    st.session_state["output_choice"] = "CR"  # Save as a string
            with col2:
                if st.button("SR"):
                    st.session_state["output_choice"] = "SR"  # Save as a string
        elif st.session_state["dataset_choice"] == "HEATSINK":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("PD"):
                    st.session_state["output_choice"] = "PD"  # Save as a string
            with col2:
                if st.button("TR"):
                    st.session_state["output_choice"] = "TR"  # Save as a string

    # STEP 3: Once the second-level button is pressed, display the number input fields.
    if st.session_state.get("output_choice"):
        st.write(f"Output choice selected: {st.session_state['output_choice']}")
        
        # Number input fields for user-defined parameters.
        # The keys ensure values are stored in session_state.
        pop_size = st.number_input("Pop size", min_value=0, step=1, key="pop_size")
        pop_ret_size = st.number_input("Pop ret size", min_value=0, step=1, key="pop_ret_size")
        num_itns = st.number_input("Num itns", min_value=0, step=1, key="num_itns")
    
        # STEP 4: Confirm button that prints out the current variable names and values.
        if st.button("Confirm"):
            st.write("### Current Variable Values:")
            st.write(f"**dataset_choice:** {st.session_state['dataset_choice']}")
            st.write(f"**output_choice:** {st.session_state['output_choice']}")
            st.write(f"**Pop size:** {st.session_state['pop_size']}")
            st.write(f"**Pop ret size:** {st.session_state['pop_ret_size']}")
            st.write(f"**Num itns:** {st.session_state['num_itns']}")
