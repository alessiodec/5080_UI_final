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
    return os.path.join(os.path.dirname(__file__), "physical_relationship_analysis_files", filename)

def load_heatsink_data(file_path=None, display_output=False):
    """
    Loads and processes the heatsink dataset.
    Returns:
        df, X, y, standardised_y, mean_y, std_y
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
    y = df['Pressure_Drop'].values.reshape(-1,)
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
    Initializes and simplifies the population, then runs an evolution loop that
    updates a persistent graph in real time. After the loop, only the final best
    fitness is printed alongside the final graph.
    """
    # Update configuration
    config.POPULATION_SIZE = pop_size
    config.POPULATION_RETENTION_SIZE = pop_retention
    config.FIT_THRESHOLD = 10  # Fixed threshold

    if "heatsink_data" not in st.session_state:
        st.error("❌ Heatsink data has not been loaded. Run 'Load Heatsink Data' first.")
        return

    df, X, y, standardised_y, mean_y, std_y = st.session_state["heatsink_data"]
    config.X, config.y = X, standardised_y

    st.write("🚀 Initializing Population... This may take a moment.")
    start_time = time.time()
    with st.spinner("Generating initial population..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            init_population = Engine.initialize_population(verbose=0)
    st.write(f"✅ Population initialized in {time.time() - start_time:.2f} seconds.")

    # Evaluate and simplify population
    Engine.evaluate_population(init_population)
    st.write("⚙️ Simplifying Population...")
    start_time = time.time()
    with st.spinner("Simplifying expressions..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            simplified_population = Engine.simplify_and_clean_population(init_population)
    st.write(f"✅ Population simplified in {time.time() - start_time:.2f} seconds.")

    st.write("📈 Running Evolution Process...")

    # Create a persistent figure and axis for updating in real time.
    fig, ax = plt.subplots(figsize=(8, 6))
    line_avg, = ax.plot([], [], 'bo-', label="Avg Fitness")
    line_comp, = ax.plot([], [], 'ro-', label="Complexity")
    line_best, = ax.plot([], [], 'go-', label="Best Fitness")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness - 1-$R^2$")
    ax.set_yscale("log")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Population Metrics Over Iterations")

    # Create a placeholder for real-time graph updates.
    chart_placeholder = st.empty()

    # Initialize tracking arrays.
    avg_fitness_arr = []
    avg_complexity_arr = []
    best_fitness_arr = []
    iterations = []

    new_population = simplified_population.copy()
    evolution_start = time.time()

    for i in range(num_iterations):
        new_population = Engine.generate_new_population(population=new_population.copy(), verbose=0)
        avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)
        avg_fitness_arr.append(avg_fitness)
        avg_complexity_arr.append(avg_complexity)
        best_fitness_arr.append(optimal_fitness)
        iterations.append(i + 1)

        # Update the persistent figure's data.
        line_avg.set_data(iterations, avg_fitness_arr)
        line_comp.set_data(iterations, avg_complexity_arr)
        line_best.set_data(iterations, best_fitness_arr)
        ax.relim()
        ax.autoscale_view()

        # Update the placeholder with the modified figure.
        chart_placeholder.pyplot(fig)
        time.sleep(1.0)  # Adjust delay as needed

    final_best = best_fitness_arr[-1] if best_fitness_arr else None
    st.write(f"Final Best Fitness: {final_best:.8f}" if final_best is not None else "No best fitness computed.")
    st.success("✅ Heatsink Analysis Completed!")

