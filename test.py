# test.py
import streamlit as st
import time

# Import the evolution experiment function from functions/test_functions.py
from functions.test_functions import run_evolution_experiment

def run_ui():
    st.title("Regression Evolution Experiment")
    st.write("Please select a dataset, an output variable and the evolution parameters.")

    # --- STEP 1: Dataset selection ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button("C"):
            st.session_state["dataset_choice"] = "CORROSION"
            st.session_state["output_choice"] = None  # Reset output selection
    with col2:
        if st.button("H"):
            st.session_state["dataset_choice"] = "HEATSINK"
            st.session_state["output_choice"] = None

    if "dataset_choice" in st.session_state and st.session_state["dataset_choice"]:
        st.markdown(f"**Dataset Choice:** {st.session_state['dataset_choice']}")

        # --- STEP 2: Output variable selection ---
        if st.session_state["dataset_choice"] == "CORROSION":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("CR"):
                    st.session_state["output_choice"] = "CR"
            with col2:
                if st.button("SR"):
                    st.session_state["output_choice"] = "SR"
        elif st.session_state["dataset_choice"] == "HEATSINK":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Thermal_Resistance"):
                    st.session_state["output_choice"] = "Thermal_Resistance"
            with col2:
                if st.button("Pressure_Drop"):
                    st.session_state["output_choice"] = "Pressure_Drop"

    # --- STEP 3: Parameter inputs ---
    if "output_choice" in st.session_state and st.session_state["output_choice"]:
        st.markdown(f"**Output Choice:** {st.session_state['output_choice']}")
        pop_size = st.number_input("Population Size", min_value=1, value=1500, step=1, key="pop_size")
        pop_ret_size = st.number_input("Population Retention Size", min_value=1, value=300, step=1, key="pop_ret_size")
        num_itns = st.number_input("Number of Iterations", min_value=1, value=10, step=1, key="num_itns")

        # Create a placeholder for the real-time plot and debug log.
        plot_placeholder = st.empty()

        # --- STEP 4: Confirm and run experiment ---
        if st.button("Confirm"):
            st.write("### Confirmed Inputs:")
            st.write(f"**Dataset:** {st.session_state['dataset_choice']}")
            st.write(f"**Output Variable:** {st.session_state['output_choice']}")
            st.write(f"**Population Size:** {pop_size}")
            st.write(f"**Population Retention Size:** {pop_ret_size}")
            st.write(f"**Number of Iterations:** {num_itns}")
            st.write("Running evolution experiment… please wait.")

            # Run the evolution experiment and pass the plot placeholder (which will be updated)
            run_evolution_experiment(
                dataset_choice=st.session_state["dataset_choice"],
                output_var=st.session_state["output_choice"],
                population_size=pop_size,
                population_retention_size=pop_ret_size,
                number_of_iterations=num_itns,
                st_container=plot_placeholder
            )

if __name__ == "__main__":
    run_ui()
