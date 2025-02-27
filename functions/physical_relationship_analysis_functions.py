import os
import streamlit as st  # Only needed if you want to display using st.write
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
        display_output (bool): If True, displays the mean, std, and DataFrame via st.write.
        
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
    Runs the heatsink analysis based on user-defined population parameters and number of iterations.
    Uses a single persistent Matplotlib figure to update the plot in real time.
    
    Args:
        pop_size (int): The number of individuals in the population.
        pop_retention (int): The number of individuals retained after selection.
        num_iterations (int): The number of iterations (generations) to run the evolution process.
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

    st.write("🚀 Initializing Population... This may take a moment.")
    start_time = time.time()
    with st.spinner("Generating initial population..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            init_population = Engine.initialize_population(verbose=1)
    st.write(f"✅ Population initialized in {time.time() - start_time:.2f} seconds")

    # Display a few individuals for reference
    for i, individual in enumerate(init_population[:10]):
        st.text(f"{i}: Fitness={individual.fitness:.4f}, Complexity={individual.complexity}, Eq={individual.individual}")

    Engine.evaluate_population(init_population)

    st.write("⚙️ Simplifying Population...")
    start_time = time.time()
    with st.spinner("Simplifying expressions..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            simplified_pop = Engine.simplify_and_clean_population(init_population)
    st.write(f"✅ Population simplified in {time.time() - start_time:.2f} seconds")

    # ---- Evolution Loop with In-Place Graph Updates ----
    st.write("📈 Running Evolution Process...")
    chart_placeholder = st.empty()   # Placeholder for the plot
    status_text = st.empty()         # Placeholder for iteration status
    progress_bar = st.progress(0)    # Progress bar for UI updates

    # Initialize tracking arrays
    avg_fitness_arr = []
    avg_complexity_arr = []
    best_fitness_arr = []
    iterations = []
    new_population = init_population.copy()
    evolution_start = time.time()

    # Create a single persistent figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    line_avg, = ax.plot([], [], 'bo-', label="Avg Fitness")
    line_complex, = ax.plot([], [], 'ro-', label="Complexity")
    line_best, = ax.plot([], [], 'go-', label="Best Fitness")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness - 1-$R^2$")
    ax.set_yscale("log")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("Population Metrics Over Iterations")
    chart_placeholder.pyplot(fig)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(num_iterations):
            # Generate new population and evaluate metrics
            new_population = Engine.generate_new_population(population=new_population.copy(), verbose=1)
            avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)

            iterations.append(i + 1)
            avg_fitness_arr.append(avg_fitness)
            avg_complexity_arr.append(avg_complexity)
            best_fitness_arr.append(optimal_fitness)

            elapsed_time = time.time() - evolution_start
            status_text.text(f"Iteration {i+1}: Best Fit={optimal_fitness:.8f}, Avg Fit={avg_fitness:.8f}, Elapsed Time={elapsed_time:.2f}s")
            progress_bar.progress((i + 1) / num_iterations)

            # Update the data of the existing line objects
            line_avg.set_data(iterations, avg_fitness_arr)
            line_complex.set_data(iterations, avg_complexity_arr)
            line_best.set_data(iterations, best_fitness_arr)

            # Adjust axes limits and update the view
            ax.relim()
            ax.autoscale_view()

            # Update the existing plot in the same placeholder
            chart_placeholder.pyplot(fig)
            time.sleep(0.1)

    st.success("✅ Heatsink Analysis Completed!")

def run_heatsink_evolution(num_iterations):
    """
    Runs the evolution process for a user-defined number of iterations.
    Similar to run_heatsink_analysis, it updates a persistent plot in real time.
    
    Args:
        num_iterations (int): Number of iterations to run the evolution process.
    """
    if "heatsink_data" not in st.session_state:
        st.error("❌ Heatsink data not found! Please load it first.")
        return

    config.X, config.y = st.session_state.heatsink_data[1], st.session_state.heatsink_data[3]
    new_population = Engine.initialize_population(verbose=1)

    avg_fitness_arr = []
    avg_complexity_arr = []
    best_fitness_arr = []
    iterations = []
    start_time = time.time()
    chart_placeholder = st.empty()
    status_text = st.empty()
    progress_bar = st.progress(0)

    # Create a persistent figure and axis for the evolution plot
    fig, ax = plt.subplots(figsize=(8, 6))
    line_avg, = ax.plot([], [], 'bo-', label="Avg Fitness")
    line_complex, = ax.plot([], [], 'ro-', label="Complexity")
    line_best, = ax.plot([], [], 'go-', label="Best Fitness")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness - 1-$R^2$")
    ax.set_yscale("log")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("Population Metrics Over Iterations")
    chart_placeholder.pyplot(fig)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(num_iterations):
            new_population = Engine.generate_new_population(population=new_population, verbose=1)
            avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)

            iterations.append(i + 1)
            avg_fitness_arr.append(avg_fitness)
            avg_complexity_arr.append(avg_complexity)
            best_fitness_arr.append(optimal_fitness)

            elapsed_time = time.time() - start_time
            status_text.text(f"Iteration {i+1}: Best Fit={optimal_fitness:.8f}, Avg Fit={avg_fitness:.8f}, Avg Comp={avg_complexity:.5f}, Iter Time={elapsed_time:.2f}s")
            progress_bar.progress((i + 1) / num_iterations)

            line_avg.set_data(iterations, avg_fitness_arr)
            line_complex.set_data(iterations, avg_complexity_arr)
            line_best.set_data(iterations, best_fitness_arr)

            ax.relim()
            ax.autoscale_view()

            chart_placeholder.pyplot(fig)
            time.sleep(0.1)

    st.success("✅ Evolution process completed!")
