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

# Import modules from our symbolic regression package
from functions.symbolic_regression_files import config
from functions.symbolic_regression_files import EngineDict as Engine
from functions.symbolic_regression_files import Plotting as Plot
from functions.symbolic_regression_files import Simplification as simp

def run_evolution_experiment(dataset_choice, output_var, population_size, population_retention_size, number_of_iterations=50):
    st.write("DEBUG: Starting evolution experiment")
    st.write(f"DEBUG: Dataset: {dataset_choice}, Output: {output_var}")
    
    # Set dataset in config
    config.DATASET = dataset_choice

    # Reinitialize the primitive set and toolbox (via EngineDict) based on the updated config
    Engine.initialize_primitive_set()

    # --- Data Loading and Preprocessing ---
    with st.spinner("Loading and preprocessing dataset..."):
        if dataset_choice == 'CORROSION':
            st.write("DEBUG: Loading CORROSION dataset")
            csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
            response = requests.get(csv_url)
            df = pd.read_csv(io.StringIO(response.text))
            df.rename(columns={"Pp CO2": "PpCO2"}, inplace=True)
            df = df.replace('', np.nan).astype(float)
            df["LogP"] = np.log10(df["PpCO2"])
            df["LogV"] = np.log10(df["v"])
            df["LogD"] = np.log10(df["d"])
            transformation_dict = {"pH": [5,6], "Tc": [0,100], "LogP": [-1,1], "LogV": [-1,1], "LogD": [-2,0]}
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
            st.write("DEBUG: Finished loading CORROSION dataset")
        elif dataset_choice == 'HEATSINK':
            st.write("DEBUG: Loading HEATSINK dataset")
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
            st.write("DEBUG: Finished loading HEATSINK dataset")
        else:
            st.error("Invalid dataset choice provided.")
            return
    st.write("DEBUG: Dataset loading complete")

    # --- Set Evolution Parameters ---
    st.write("DEBUG: Setting evolution parameters")
    config.FIT_THRESHOLD = 10  # as in VS code version
    config.POPULATION_SIZE = population_size
    config.POPULATION_RETENTION_SIZE = population_retention_size
    config.DISPLAY_ERROR_MESSAGES = False
    config.VERBOSE = True
    # Disable in-loop simplification by setting interval beyond iterations
    config.SIMPLIFICATION_INDEX_INTERVAL = number_of_iterations + 1
    config.EARLY_STOPPING_THRESHOLD = 20
    config.FITNESS_REDUCTION_THRESHOLD = 5
    config.USE_SIMPLIFICATION = False  # Avoid collapsing expressions to constants during evolution
    config.FITNESS_REDUCTION_FACTOR = 0.8
    st.write("DEBUG: Evolution parameters set")

    # --- Initialize Population ---
    with st.spinner("Initializing population..."):
        st.write("DEBUG: Initializing population")
        init_population = Engine.initialize_population()  # returns dictionary
        st.write("DEBUG: Evaluating initial population")
        Engine.evaluate_population(init_population)
    st.write("DEBUG: Population initialization complete")

    # --- Evolution Loop ---
    new_population = init_population.copy()
    avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)
    iterations = [0]
    avg_fitness_arr = [avg_fitness]
    avg_complexity_arr = [avg_complexity]
    best_fitness_arr = [optimal_fitness]
    evolution_progress = st.progress(0)
    chart_placeholder = st.empty()
    start_time = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(number_of_iterations):
            st.write(f"DEBUG: Starting iteration {i+1}")
            new_population = Engine.generate_new_population(population=new_population.copy())
            st.write(f"DEBUG: Population generated at iteration {i+1}")
            # Evaluate population and record metrics
            avg_fit, avg_comp, best_fit = Engine.evaluate_population(new_population)
            iterations.append(i+1)
            avg_fitness_arr.append(avg_fit)
            avg_complexity_arr.append(avg_comp)
            best_fitness_arr.append(best_fit)
            evolution_progress.progress(int((i+1)/number_of_iterations*100))
            # Update the plot
            fig, ax = plt.subplots()
            ax.plot(iterations, avg_fitness_arr, 'bo-', label="Avg Fitness")
            ax.plot(iterations, avg_complexity_arr, 'ro-', label="Avg Complexity")
            ax.plot(iterations, best_fitness_arr, 'go-', label="Best Fitness")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness (1 - R^2)")
            ax.set_yscale("log")
            ax.legend(loc='upper right')
            ax.set_title(f"Population Metrics: {dataset_choice} // {output_var}")
            chart_placeholder.pyplot(fig)
            plt.close(fig)
            time.sleep(0.5)
    st.write("DEBUG: Evolution iterations complete")

    # --- Determine Best Individual ---
    st.write("DEBUG: Determining best individual")
    pareto_front = Engine.return_pareto_front(new_population)
    pareto_list = list(pareto_front)
    pareto_list.sort(key=lambda x: x['fitness'])
    best_indiv = pareto_list[0]
    st.write("DEBUG: Best individual determined", best_indiv)
    st.write("DEBUG: Converting best individual to sympy expression")
    best_sympy_expr = simp.convert_expression_to_sympy(best_indiv['individual'])
    equation = sp.Eq(sp.Symbol(output_var), best_sympy_expr)
    st.latex(sp.latex(equation))
    st.write("DEBUG: Evolution experiment complete")
