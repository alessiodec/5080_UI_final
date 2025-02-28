import os
import streamlit as st  # Only needed if you want to display using st.write
import numpy as np
import pandas as pd
import warnings
import time
import matplotlib.pyplot as plt

from .physical_relationship_analysis_files import Engine
from .physical_relationship_analysis_files import config

# Get absolute path to the data file
def get_data_path(filename):
    return os.path.join(os.path.dirname(__file__), "physical_relationship_analysis_files", filename)

def load_heatsink_data(file_path=None, display_output=False):
    """
       Loads and processes the heatsink dataset.
       
       Parameters:
           file_path (str): Path to the dataset. If None, defaults to file in script directory.
           display_output (bool): If True, display mean, std, and DataFrame via st.write.
           
       Returns:
           df (DataFrame): The processed DataFrame.
           X (ndarray): Feature array from columns 'Geometric1' and 'Geometric2'.
           y (ndarray): Target variable array from column 'Pressure_Drop'.
           standardised_y (ndarray): The standardized target variable.
           mean_y (float): Mean of y.
           std_y (float): Standard deviation of y.
    """
    # Default file path (inside the same folder as this script)
    if file_path is None:
        file_path = get_data_path("Latin_Hypercube_Heatsink_1000_samples.txt")

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Ensure the file is correctly placed.")

    # Read the file using a context manager
    with open(file_path, "r") as f:
        text = f.read()

    # Split the text into rows and then each row into columns
    data = [x.split(' ') for x in text.split('\n') if x.strip() != '']

    # Create DataFrame with proper column names and convert to numeric types
    df = pd.DataFrame(data, columns=['Geometric1', 'Geometric2', 'Thermal_Resistance', 'Pressure_Drop'])
    df = df.apply(pd.to_numeric)

    # Extract features and target
    X = df[['Geometric1', 'Geometric2']].values
    y = df['Pressure_Drop'].values.reshape(-1,)  # Using Pressure_Drop as target

    # Compute mean and standard deviation of y
    mean_y = np.mean(y)
    std_y = np.std(y)

    # Update config values
    config.mean_y = mean_y
    config.std_y = std_y

    # Optionally display the computed values and DataFrame in the Streamlit app
    # if display_output:
    #    st.write("Mean of y:", mean_y)
    #    st.write("Standard deviation of y:", std_y)
    #    st.write("DataFrame:", df)

    # Standardize y
    standardised_y = (y - mean_y) / std_y

    return df, X, y, standardised_y, mean_y, std_y

def run_heatsink_analysis(pop_size, pop_retention):
    """
       Runs the heatsink analysis based on user-defined population parameters and number of iterations.
    
       Args:
           pop_size (int): The number of individuals in the population.
           pop_retention (int): The number of individuals retained after selection.
           num_iterations (int): The number of iterations (generations) to run the evolution process.
    """
    # Update the configuration
    config.POPULATION_SIZE = pop_size
    config.POPULATION_RETENTION_SIZE = pop_retention
    config.FIT_THRESHOLD = 10  # Keeping the threshold constant

    # Ensure required data exists in session state
    if "heatsink_data" not in st.session_state:
        st.error("❌ Heatsink data has not been loaded. Run 'Load Heatsink Data' first.")
        return

    # Unpack stored heatsink data and update config.X and config.y
    df, X, y, standardised_y, mean_y, std_y = st.session_state["heatsink_data"]
    config.X, config.y = X, standardised_y

    st.write("🚀 Initializing Population... This may take a moment.")
    start_time = time.time()

    with st.spinner("Generating initial population..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            init_population = Engine.initialize_population(verbose=1)

    st.write(f"✅ Population initialized in {time.time() - start_time:.2f} seconds")

    # Display a few individuals for reference
    # for i, individual in enumerate(init_population[:10]):
    #    st.text(f"{i}: Fitness={individual.fitness:.4f}, Complexity={individual.complexity}, Eq={individual.individual}")

    Engine.evaluate_population(init_population)

    st.write("⚙️ Simplifying Population...")
    start_time = time.time()

    with st.spinner("Simplifying expressions..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            simplified_pop = Engine.simplify_and_clean_population(init_population)

    st.write(f"✅ Population simplified in {time.time() - start_time:.2f} seconds")

    # ---- Evolution Loop with Real-Time Graph Updates ----
    st.write("📈 Running Evolution Process...")
    chart_placeholder = st.empty()  # Placeholder for dynamic graph updates

    # Initialize tracking arrays
    avg_fitness_arr = []
    avg_complexity_arr = []
    best_fitness_arr = []
    iterations = []

    # IMPORTANT: Initialize new_population from init_population
    new_population = init_population.copy()

    evolution_start = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(num_iterations):
            # Generate new population from current new_population copy
            new_population = Engine.generate_new_population(population=new_population.copy(), verbose=1)
            avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)

            avg_fitness_arr.append(avg_fitness)
            avg_complexity_arr.append(avg_complexity)
            best_fitness_arr.append(optimal_fitness)
            iterations.append(i + 1)

            elapsed_time = time.time() - evolution_start
            # st.write(f"Iteration {i+1}: Best Fit={optimal_fitness:.8f}, Avg Fit={avg_fitness:.8f}, Elapsed Time={elapsed_time:.2f}s")

            # Clear and update the plot dynamically
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(iterations, avg_fitness_arr, 'bo-', label="Avg Fitness")
            ax.plot(iterations, avg_complexity_arr, 'ro-', label="Complexity")
            ax.plot(iterations, best_fitness_arr, 'go-', label="Best Fitness")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness - 1-$R^2$")
            ax.set_yscale("log")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_title("Population Metrics Over Iterations")
            chart_placeholder.pyplot(fig)

            time.sleep(0.1)

    st.success("✅ Heatsink Analysis Completed!")

    st.write(f"Best Fit={optimal_fitness:.8f}")

def run_heatsink_evolution(num_iterations):
    """
       Runs the evolution process for a user-defined number of iterations.
    
       Args:
           num_iterations (int): Number of iterations to run the evolution process.
    """
    if "heatsink_data" not in st.session_state:
        st.error("❌ Heatsink data not found! Please load it first.")
        return

    # Ensure X and y exist in config (needed for Engine functions)
    config.X, config.y = st.session_state.heatsink_data[1], st.session_state.heatsink_data[3]

    # Initialize population correctly (instead of using raw X values)
    new_population = Engine.initialize_population(verbose=1)

    avg_fitness_arr = []
    avg_complexity_arr = []
    best_fitness_arr = []
    iterations = list(range(num_iterations))

    start_time = time.time()

    # Streamlit placeholder to update graph dynamically
    chart_placeholder = st.empty()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        for i in iterations:
            new_population = Engine.generate_new_population(population=new_population, verbose=1)
            avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)

            avg_fitness_arr.append(avg_fitness)
            avg_complexity_arr.append(avg_complexity)
            best_fitness_arr.append(optimal_fitness)

            elapsed_time = time.time() - start_time

            st.write(f"Iter {i+1}: Best Fit={optimal_fitness:.8f}, Avg Fit={avg_fitness:.8f}, Avg Comp={avg_complexity:.5f}, Iter Time={elapsed_time:.2f}s")

            # --- Clear previous figure and update ---
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(iterations[: i+1], avg_fitness_arr, 'bo-', label="Avg Fitness")
            ax.plot(iterations[: i+1], avg_complexity_arr, 'ro-', label="Complexity")
            ax.plot(iterations[: i+1], best_fitness_arr, 'go-', label="Best Fitness")

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness - 1-$R^2$")
            ax.set_yscale("log")
            ax.legend()
            ax.set_title("Population Metrics Over Iterations")

            # Update the existing plot dynamically
            chart_placeholder.pyplot(fig)

            time.sleep(0.1)

    st.success("✅ Evolution process completed!")
