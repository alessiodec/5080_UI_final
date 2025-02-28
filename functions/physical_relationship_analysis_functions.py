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

    standardised_y = (y - mean_y) / std_y

    return df, X, y, standardised_y, mean_y, std_y

def run_heatsink_analysis_and_evolution(pop_size, pop_retention, num_iterations):
    config.POPULATION_SIZE = pop_size
    config.POPULATION_RETENTION_SIZE = pop_retention
    config.FIT_THRESHOLD = 10

    if "heatsink_data" not in st.session_state:
        st.error("❌ Heatsink data has not been loaded.")
        return

    df, X, y, standardised_y, mean_y, std_y = st.session_state["heatsink_data"]
    config.X, config.y = X, standardised_y

    st.write("🚀 Initializing Population...")
    with st.spinner("Generating initial population..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            population = Engine.initialize_population(verbose=1)

    Engine.evaluate_population(population)

    st.write("⚙️ Running Evolution Process...")
    chart_placeholder = st.empty()

    avg_fitness_arr = []
    avg_complexity_arr = []
    best_fitness_arr = []
    iterations = []

    start_time = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(num_iterations):
            population = Engine.generate_new_population(population=population.copy(), verbose=1)
            avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(population)

            avg_fitness_arr.append(avg_fitness)
            avg_complexity_arr.append(avg_complexity)
            best_fitness_arr.append(optimal_fitness)
            iterations.append(i + 1)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(iterations, avg_fitness_arr, 'bo-', label="Avg Fitness")
            ax.plot(iterations, avg_complexity_arr, 'ro-', label="Complexity")
            ax.plot(iterations, best_fitness_arr, 'go-', label="Best Fitness")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness - 1-$R^2$")
            ax.set_yscale("log")
            ax.legend()
            ax.set_title("Population Metrics Over Iterations")

            chart_placeholder.pyplot(fig)

            time.sleep(2) # edit this to make sure each iteration is loaded

    st.success("✅ Heatsink Analysis and Evolution Completed!")
