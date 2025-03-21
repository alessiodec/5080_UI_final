import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
import sympy as sp
from sklearn.preprocessing import StandardScaler
from deap import gp

# --- Import Internal Modules ---
import config
import EngineDict as Engine
import Plotting as Plot
import Simplification as simp
import requests
import io


def load_and_process_dataset(dataset_choice, output_var):
    """
    Loads and preprocesses the dataset for symbolic regression.

    Args:
        dataset_choice (str): Name of the dataset file.
        output_var (str): The target variable for regression.

    Returns:
        tuple: (X, y)
    """
    if dataset_choice == 'CORROSION':
        csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
        response = requests.get(csv_url)
        df = pd.read_csv(io.StringIO(response.text))
        df.rename(columns={"Pp CO2": "PpCO2"}, inplace=True)
        df = df.replace('', np.nan)
        df = df.astype(float)

        # Create logarithmic columns
        df["LogP"] = np.log10(df["PCO2"])
        df["LogV"] = np.log10(df["v"])
        df["LogD"] = np.log10(df["d"])

        # Normalization details
        transformation_dict = {
            "pH": [5, 6],
            "Tc": [0, 100],
            "LogP": [-1, 1],
            "LogV": [-1, 1],
            "LogD": [-2, 0],
        }

        # Sample a subset
        np.random.seed(42)
        sample_size = 2000
        sample_indices = np.random.choice(df.index, size=sample_size, replace=False)
        df_sampled = df.loc[sample_indices]

        # Extract features and target
        X = df_sampled[["pH", "Tc", "LogP", "LogV", "LogD"]].values
        y = df_sampled[output_var].values.reshape(-1,)

        # Normalize features
        transformed_X = np.array([
            (X[:, i] - transformation_dict[col][0]) / (transformation_dict[col][1] - transformation_dict[col][0]) + 1
            for i, col in enumerate(["pH", "Tc", "LogP", "LogV", "LogD"])
        ]).T

    elif dataset_choice == 'HEATSINK':
        with open("Data/Latin_Hypercube_Heatsink_1000_samples.txt", "r") as f:
            text = f.read()
        data = [x.split(' ') for x in text.split('\n') if x.strip()]
        df = pd.DataFrame(data, columns=['Geometric1', 'Geometric2', 'Thermal_Resistance', 'Pressure_Drop'])
        df = df.apply(pd.to_numeric)

        X = df[['Geometric1', 'Geometric2']].values
        y = df[output_var].values.reshape(-1,)

    else:
        raise ValueError("Invalid dataset choice.")

    # Standardize target variable
    mean_y = np.mean(y)
    std_y = np.std(y)
    config.mean_y = mean_y
    config.std_y = std_y
    y_standardized = (y - mean_y) / std_y

    return transformed_X if dataset_choice == 'CORROSION' else X, y_standardized


def run_evolution_experiment(dataset_choice, output_var, population_size, population_retention_size, num_iterations):
    """
    Runs the symbolic regression evolution process.

    Args:
        dataset_choice (str): The dataset to use.
        output_var (str): Target variable for regression.
        population_size (int): Number of individuals in the population.
        population_retention_size (int): Number of individuals retained per generation.
        num_iterations (int): Number of iterations to run.
    """
    # --- Load Dataset ---
    X, y = load_and_process_dataset(dataset_choice, output_var)
    config.X, config.y = X, y

    # --- Set Configuration Parameters ---
    config.POPULATION_SIZE = population_size
    config.POPULATION_RETENTION_SIZE = population_retention_size
    config.FIT_THRESHOLD = 1
    config.USE_SIMPLIFICATION = True
    config.VERBOSE = True
    config.SIMPLIFICATION_INDEX_INTERVAL = 20
    config.EARLY_STOPPING_THRESHOLD = 20
    config.FITNESS_REDUCTION_THRESHOLD = 5
    config.FITNESS_REDUCTION_FACTOR = 0.8
    config.FIT_THRESHOLD = 10

    # --- Initialize Population ---
    warnings.simplefilter("ignore", RuntimeWarning)
    init_population = Engine.initialize_population()
    Engine.evaluate_population(init_population)
    new_population = init_population.copy()

    # --- Track Metrics ---
    avg_fitness_arr, avg_complexity_arr, best_fitness_arr = [], [], []
    iterations = [0]
    time_array = [0]

    # --- Streamlit UI Setup ---
    progress_bar = st.progress(0)
    status_text = st.empty()
    fig, ax = plt.subplots()

    start_time = time.time()
    
    for i in range(num_iterations):
        new_population = Engine.generate_new_population(new_population.copy())

        # Simplification
        if config.USE_SIMPLIFICATION and i % config.SIMPLIFICATION_INDEX_INTERVAL == 0:
            _, _, _ = Engine.evaluate_population(new_population)
            new_population = Engine.simplify_population(new_population)
        
        # Evaluate Population
        avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)
        avg_fitness_arr.append(avg_fitness)
        avg_complexity_arr.append(avg_complexity)
        best_fitness_arr.append(optimal_fitness)
        iterations.append(i + 1)

        # Timing
        elapsed_time = time.time() - start_time
        time_array.append(time_array[-1] + elapsed_time)
        start_time = time.time()

        # Update Progress
        progress_bar.progress(int((i + 1) / num_iterations * 100))
        status_text.text(f"Iteration {i + 1}/{num_iterations} - Best Fitness: {optimal_fitness:.5f}")

        # Real-Time Plotting
        ax.cla()
        ax.plot(iterations, avg_fitness_arr, label="Average Fitness")
        ax.plot(iterations, avg_complexity_arr, label="Complexity")
        ax.plot(iterations, best_fitness_arr, label="Best Fitness")
        ax.legend()
        ax.set_yscale("log")
        ax.set_title("Symbolic Regression Evolution Progress")
        st.pyplot(fig)
    
    # --- Select Best Individual ---
    pareto_front = list(Engine.return_pareto_front(new_population))
    pareto_front.sort(key=lambda x: x['fitness'])
    best_indiv = pareto_front[0]

    # --- Convert to Symbolic Expression ---
    best_expr = simp.convert_expression_to_sympy(best_indiv['individual'])
    equation = sp.Eq(sp.Symbol(output_var), best_expr)
    
    # --- Display Final Result ---
    st.subheader("Best Discovered Equation:")
    st.latex(sp.latex(equation))
