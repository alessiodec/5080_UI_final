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
    Runs the heatsink analysis based on user-defined population parameters and number of iterations.
    """
    # Update the configuration.
    config.POPULATION_SIZE = pop_size
    config.POPULATION_RETENTION_SIZE = pop_retention
    config.FIT_THRESHOLD = 10  # Fixed threshold.

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
    st.write(f"✅ Population initialized in {time.time() - start_time:.2f} seconds")

    # (No printing of individual details)

    Engine.evaluate_population(init_population)

    st.write("⚙️ Simplifying Population...")
    start_time = time.time()
    with st.spinner("Simplifying expressions..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            simplified_pop = Engine.simplify_and_clean_population(init_population)
    st.write(f"✅ Population simplified in {time.time() - start_time:.2f} seconds")

    st.write("📈 Running Evolution Process...")
    chart_placeholder = st.empty()  # Placeholder for real-time updates.

    # Tracking arrays.
    avg_fitness_arr = []
    avg_complexity_arr = []
    best_fitness_arr = []
    iterations = []

    new_population = init_population.copy()
    evolution_start = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(num_iterations):
            new_population = Engine.generate_new_population(population=new_population.copy(), verbose=0)
            avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)
            avg_fitness_arr.append(avg_fitness)
            avg_complexity_arr.append(avg_complexity)
            best_fitness_arr.append(optimal_fitness)
            iterations.append(i + 1)
            # Update the real-time graph.
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

    final_best = best_fitness_arr[-1] if best_fitness_arr else None
    st.write(f"Final Best Fitness: {final_best:.8f}" if final_best is not None else "No best fitness computed.")
    st.success("✅ Heatsink Analysis Completed!")

def run_heatsink_evolution(num_iterations):
    """
    Runs the evolution process for a user-defined number of iterations.
    """
    if "heatsink_data" not in st.session_state:
        st.error("❌ Heatsink data not found! Please load it first.")
        return

    config.X, config.y = st.session_state["heatsink_data"][1], st.session_state["heatsink_data"][3]
    new_population = Engine.initialize_population(verbose=0)

    avg_fitness_arr = []
    avg_complexity_arr = []
    best_fitness_arr = []

    start_time = time.time()
    chart_placeholder = st.empty()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(num_iterations):
            new_population = Engine.generate_new_population(population=new_population, verbose=0)
            avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)
            avg_fitness_arr.append(avg_fitness)
            avg_complexity_arr.append(avg_complexity)
            best_fitness_arr.append(optimal_fitness)
            time.sleep(0.1)
            # Update the real-time graph each iteration.
            fig, ax = plt.subplots(figsize=(8, 6))
            iterations = list(range(1, i + 2))
            ax.plot(iterations, avg_fitness_arr, 'bo-', label="Avg Fitness")
            ax.plot(iterations, avg_complexity_arr, 'ro-', label="Complexity")
            ax.plot(iterations, best_fitness_arr, 'go-', label="Best Fitness")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness - 1-$R^2$")
            ax.set_yscale("log")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_title("Population Metrics Over Iterations")
            chart_placeholder.pyplot(fig)

    final_best = best_fitness_arr[-1] if best_fitness_arr else None
    st.write(f"Final Best Fitness: {final_best:.8f}" if final_best is not None else "No best fitness computed.")
    st.success("✅ Evolution process completed!")
