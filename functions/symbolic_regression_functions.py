import os
import time
import warnings
import requests
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import streamlit as st
from sklearn.metrics import r2_score
from deap import gp

# Import modules from the symbolic regression folder
from functions.symbolic_regression_files import config
from functions.symbolic_regression_files import EngineDict as Engine
from functions.symbolic_regression_files import Plotting as Plot
from functions.symbolic_regression_files import Simplification as simp


def run_evolution_experiment(dataset_choice, output_var, population_size, population_retention_size, number_of_iterations=50):
    """
    Runs the symbolic regression evolution experiment using the specified parameters.
    
    Parameters:
      dataset_choice (str): "HEATSINK" or "CORROSION"
      output_var (str): For HEATSINK, either "Thermal_Resistance" or "Pressure_Drop"; 
                        for CORROSION, either "CR" or "SR"
      population_size (int): Number of individuals in the population.
      population_retention_size (int): Number of individuals to retain.
      number_of_iterations (int): Number of evolution iterations (default is 50).
    
    Behavior:
      - Loads and preprocesses the chosen dataset.
      - Configures the evolution parameters.
      - Initializes the population and evolves it while updating a real‑time plot and progress bar in Streamlit.
      - At the end, displays the best simplified equation formatted as, for example: "CR = [expression]".
    """
    # Set the dataset in config
    config.DATASET = dataset_choice

    # --- Data Loading and Preprocessing ---
    with st.spinner("Loading and preprocessing data..."):
        if dataset_choice == 'CORROSION':
            csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
            response = requests.get(csv_url)
            df = pd.read_csv(io.StringIO(response.text))
            df.rename(columns={"Pp CO2": "PpCO2"}, inplace=True)
            df = df.replace('', np.nan)
            df = df.astype(float)

            # Create logarithmic columns
            df["LogP"] = np.log10(df["PpCO2"])
            df["LogV"] = np.log10(df["v"])
            df["LogD"] = np.log10(df["d"])

            transformation_dict = {
                "pH":    [5, 6],
                "Tc":    [0, 100],
                "LogP":  [-1, 1],
                "LogV":  [-1, 1],
                "LogD":  [-2, 0],
            }

            np.random.seed(42)
            sample_size = 2000
            sample_indices = np.random.choice(df.index, size=sample_size, replace=False)
            df_sampled = df.loc[sample_indices]

            X = df_sampled[["pH", "Tc", "LogP", "LogV", "LogD"]].values
            y = df_sampled[output_var].values.reshape(-1,)

            transformed_X = np.array([
                (X[:, i] - transformation_dict[col][0]) / (transformation_dict[col][1] - transformation_dict[col][0]) + 1
                for i, col in enumerate(["pH", "Tc", "LogP", "LogV", "LogD"])
            ]).T

            mean_y = np.mean(y)
            std_y = np.std(y)
            config.mean_y = mean_y
            config.std_y = std_y
            standardised_y = (y - mean_y) / std_y

            config.X = transformed_X
            config.y = standardised_y

        elif dataset_choice == 'HEATSINK':
            heatsink_file = os.path.join("functions", "symbolic_regression_files", "data", "Latin_Hypercube_Heatsink_1000_samples.txt")
            with open(heatsink_file, "r") as f:
                text = f.read()
            data = [x.split(' ') for x in text.split('\n') if x.strip() != '']
            df = pd.DataFrame(data, columns=['Geometric1', 'Geometric2', 'Thermal_Resistance', 'Pressure_Drop'])
            df = df.apply(pd.to_numeric)

            X = df[['Geometric1', 'Geometric2']].values
            y = df[output_var].values.reshape(-1,)
            mean_y = np.mean(y)
            std_y = np.std(y)
            config.mean_y = mean_y
            config.std_y = std_y
            standardised_y = (y - mean_y) / std_y

            config.X = X
            config.y = standardised_y
        else:
            st.error("Invalid dataset choice provided.")
            return
    st.success("Data loaded and preprocessed.")

    # --- Set Evolution Parameters ---
    config.POPULATION_SIZE = population_size
    config.POPULATION_RETENTION_SIZE = population_retention_size
    config.FIT_THRESHOLD = 1
    config.USE_SIMPLIFICATION = False
    config.DISPLAY_ERROR_MESSAGES = False
    config.VERBOSE = True

    config.SIMPLIFICATION_INDEX_INTERVAL = 20
    config.EARLY_STOPPING_THRESHOLD = 20
    config.FITNESS_REDUCTION_THRESHOLD = 5
    config.USE_SIMPLIFICATION = True
    config.FITNESS_REDUCTION_FACTOR = 0.8
    config.FIT_THRESHOLD = 10
    config.DISPLAY_ERROR_MESSAGES = False

    # --- Initialize Population ---
    with st.spinner("Generating initial population..."):
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Use verbose=1 if your Engine.initialize_population supports it.
            init_population = Engine.initialize_population(verbose=1)
            Engine.evaluate_population(init_population)
        init_time = time.time() - t0
        st.write(f"Initial population generated in {init_time:.2f} seconds.")
    st.success("Population initialized.")

    new_population = init_population.copy()
    avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)
    iterations = [0]
    avg_fitness_arr = [avg_fitness]
    avg_complexity_arr = [avg_complexity]
    best_fitness_arr = [optimal_fitness]
    changing_pareto_fronts = []
    changing_pareto_front_indecies = []
    fitness_reduction_indecies = []
    time_array = [0]

    # --- Real-Time Plotting Setup Using Streamlit ---
    fig, ax = plt.subplots()
    plot_placeholder = st.empty()  # Placeholder for updating plot

    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    start_time = time.time()
    with st.spinner("Evolving population..."):
        for j, i in enumerate(range(iterations[-1], iterations[-1] + number_of_iterations)):
            status_placeholder.text(f"Evolution iteration {j+1} of {number_of_iterations}")
            new_population = Engine.generate_new_population(population=new_population.copy())

            if config.USE_SIMPLIFICATION and i % config.SIMPLIFICATION_INDEX_INTERVAL == 0:
                _, old_avg_complexity, _ = Engine.evaluate_population(new_population)
                new_population = Engine.simplify_population(new_population)
                _, avg_complexity, _ = Engine.evaluate_population(new_population)

            avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)
            avg_fitness_arr.append(avg_fitness)
            avg_complexity_arr.append(avg_complexity)
            best_fitness_arr.append(optimal_fitness)
            iterations.append(i + 1)

            finish_time = time.time()
            elapsed_time = finish_time - start_time
            start_time = finish_time
            time_array.append(time_array[-1] + elapsed_time)

            if hasattr(config, 'PARETO_INDEX_INTERVAL') and config.PARETO_INDEX_INTERVAL is not None:
                if i % config.PARETO_INDEX_INTERVAL == 0:
                    changing_pareto_fronts.append(Engine.return_pareto_front(new_population))
                    changing_pareto_front_indecies.append(i)

            if len(avg_fitness_arr) > config.FITNESS_REDUCTION_THRESHOLD:
                if (min(avg_fitness_arr[-config.FITNESS_REDUCTION_THRESHOLD:]) == avg_fitness_arr[-config.FITNESS_REDUCTION_THRESHOLD]) and \
                   (config.FIT_THRESHOLD * config.FITNESS_REDUCTION_FACTOR > avg_fitness):
                    config.FIT_THRESHOLD *= config.FITNESS_REDUCTION_FACTOR
                    fitness_reduction_indecies.append(i + 1)

            if len(avg_fitness_arr) > config.EARLY_STOPPING_THRESHOLD:
                if min(avg_fitness_arr[-config.EARLY_STOPPING_THRESHOLD:]) == avg_fitness_arr[-config.EARLY_STOPPING_THRESHOLD]:
                    break

            ax.cla()
            ax.plot(iterations, av
