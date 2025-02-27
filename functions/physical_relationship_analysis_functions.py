import os
import streamlit as st
import numpy as np
import pandas as pd
import warnings
import time
import matplotlib.pyplot as plt

from .physical_relationship_analysis_files import Engine
from .physical_relationship_analysis_files import config

def get_data_path(filename):
    """
    Returns the absolute path to the specified data file.
    """
    return os.path.join(os.path.dirname(__file__), "physical_relationship_analysis_files", filename)

def load_heatsink_data(file_path=None, display_output=False):
    """
    Loads and processes the heatsink dataset.
    
    Parameters:
        file_path (str): Path to the dataset. If None, defaults to the file in the script directory.
        display_output (bool): If True, displays mean, std, and DataFrame via st.write.
        
    Returns:
        df (DataFrame): The processed DataFrame.
        X (ndarray): Feature array from columns 'Geometric1' and 'Geometric2'.
        y (ndarray): Target variable array from column 'Pressure_Drop'.
        standardised_y (ndarray): The standardized target variable.
        mean_y (float): Mean of y.
        std_y (float): Standard deviation of y.
    """
    if file_path is None:
        file_path = get_data_path("Latin_Hypercube_Heatsink_1000_samples.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Ensure the file is correctly placed.")
    with open(file_path, "r") as f:
        text = f.read()
    data = [x.split(' ') for x in text.split('\n') if x.strip() != '']
    df = pd.DataFrame(data, columns=['Geometric1', 'Geometric2', 'Thermal_Resistance', 'Pressure_Drop'])
    df = df.apply(pd.to_numeric)
    X = df[['Geometric1', 'Geometric2']].values
    y = df['Pressure_Drop'].values.reshape(-1,)  # Using Pressure_Drop as target
    mean_y = np.mean(y)
    std_y = np.std(y)
    config.mean_y = mean_y
    config.std_y = std_y
    if display_output:
        st.write("Mean of y:", mean_y)
        st.write("Standard deviation of y:", std_y)
        st.write("DataFrame:", df)
    standardised_y = (y - mean_y) / std_y
    return df, X, y, standardised_y, mean_y, std_y

def run_heatsink_analysis(pop_size, pop_retention, num_iterations):
    """
    Runs the heatsink analysis in a stateful, iteration-by-iteration manner.
    The evolution state is stored in st.session_state so that one iteration is processed per run.
    After each iteration, st.experimental_rerun() is called to refresh the UI.
    
    Args:
        pop_size (int): The number of individuals in the population.
        pop_retention (int): The number of individuals retained after selection.
        num_iterations (int): Total number of iterations for the evolution process.
    """
    # Update configuration values
    config.POPULATION_SIZE = pop_size
    config.POPULATION_RETENTION_SIZE = pop_retention
    config.FIT_THRESHOLD = 10  # Constant threshold

    if "heatsink_data" not in st.session_state:
        st.error("❌ Heatsink data has not been loaded. Run 'Load Heatsink Data' first.")
        return

    # Unpack stored heatsink data and update config.X and config.y
    df, X, y, standardised_y, mean_y, std_y = st.session_state["heatsink_data"]
    config.X, config.y = X, standardised_y

    # Initialize evolution state if not already initialized.
    if "evolution_state" not in st.session_state:
        st.write("🚀 Initializing Population... This may take a moment.")
        with st.spinner("Generating initial population..."):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                init_population = Engine.initialize_population(verbose=1)
        st.session_state["evolution_state"] = {
            "current_iter": 0,
            "iterations": [],
            "avg_fitness": [],
            "avg_complexity": [],
            "best_fitness": [],
            "population": init_population,
            "start_time": time.time()
        }
        st.write("✅ Population initialized.")

    # Retrieve the evolution state.
    evolution_state = st.session_state["evolution_state"]
    current_iter = evolution_state["current_iter"]
    chart_placeholder = st.empty()

    if current_iter < num_iterations:
        # Run one iteration.
        pop = evolution_state["population"]
        new_population = Engine.generate_new_population(population=pop.copy(), verbose=1)
        avg_fit, avg_comp, best_fit = Engine.evaluate_population(new_population)

        # Update state.
        evolution_state["current_iter"] = current_iter + 1
        evolution_state["iterations"].append(current_iter + 1)
        evolution_state["avg_fitness"].append(avg_fit)
        evolution_state["avg_complexity"].append(avg_comp)
        evolution_state["best_fitness"].append(best_fit)
        evolution_state["population"] = new_population

        elapsed_time = time.time() - evolution_state["start_time"]
        st.write(f"Iteration {current_iter+1}: Best Fit={best_fit:.8f}, Avg Fit={avg_fit:.8f}, Elapsed Time={elapsed_time:.2f}s")

        # Create the plot based on the current evolution data.
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(evolution_state["iterations"], evolution_state["avg_fitness"], 'bo-', label="Avg Fitness")
        ax.plot(evolution_state["iterations"], evolution_state["avg_complexity"], 'ro-', label="Complexity")
        ax.plot(evolution_state["iterations"], evolution_state["best_fitness"], 'go-', label="Best Fitness")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness - 1-$R^2$")
        ax.set_yscale("log")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title("Population Metrics Over Iterations")
        chart_placeholder.pyplot(fig)

        # Wait briefly then re-run to process the next iteration.
        time.sleep(0.1)
        st.experimental_rerun()
    else:
        st.success("✅ Evolution Completed!")
        # Final plot.
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(evolution_state["iterations"], evolution_state["avg_fitness"], 'bo-', label="Avg Fitness")
        ax.plot(evolution_state["iterations"], evolution_state["avg_complexity"], 'ro-', label="Complexity")
        ax.plot(evolution_state["iterations"], evolution_state["best_fitness"], 'go-', label="Best Fitness")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness - 1-$R^2$")
        ax.set_yscale("log")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title("Population Metrics Over Iterations")
        chart_placeholder.pyplot(fig)
        # Optionally, clear the evolution state.
        st.session_state.pop("evolution_state", None)

def run_heatsink_evolution(num_iterations):
    """
    (Optional) Runs the evolution process for a user-defined number of iterations.
    This function can be adapted similarly to run_heatsink_analysis if needed.
    """
    st.warning("run_heatsink_evolution is not implemented in the stateful approach. Please use run_heatsink_analysis().")
