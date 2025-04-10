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
    st.write("DEBUG: Starting run_evolution_experiment")
    st.write(f"DEBUG: Dataset choice: {dataset_choice}, Output variable: {output_var}")
    
    # Set the dataset in config
    config.DATASET = dataset_choice
    
    # Reinitialize the primitive set and toolbox based on the updated dataset
    Engine.initialize_primitive_set()
    
    # --- Data Loading and Preprocessing ---
    with st.spinner("Loading and preprocessing dataset..."):
        if dataset_choice == 'CORROSION':
            st.write("DEBUG: Loading CORROSION dataset")
            # Download and load the CORROSION dataset
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

            # Transformation ranges for normalisation
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
            st.write("DEBUG: Finished loading CORROSION dataset")

        elif dataset_choice == 'HEATSINK':
            st.write("DEBUG: Loading HEATSINK dataset")
            # Adjust the file path for the HEATSINK dataset
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
    config.FIT_THRESHOLD = 10   # Changed from 1000 to 10 (as in normal SR)
    config.POPULATION_SIZE = population_size
    config.POPULATION_RETENTION_SIZE = population_retention_size
    config.DISPLAY_ERROR_MESSAGES = False
    config.VERBOSE = True

    # Additional configuration for evolution
    # Disable simplification during evolution (or set the interval beyond the iteration count)
    config.SIMPLIFICATION_INDEX_INTERVAL = number_of_iterations + 1  
    config.EARLY_STOPPING_THRESHOLD = 20
    config.FITNESS_REDUCTION_THRESHOLD = 5
    config.USE_SIMPLIFICATION = False  
    config.FITNESS_REDUCTION_FACTOR = 0.8
    st.write("DEBUG: Evolution parameters set")

    # --- Initialize Population ---
    with st.spinner("Initializing population..."):
        st.write("DEBUG: Calling Engine.initialize_population()")
        init_population = Engine.initialize_population()  
        st.write("DEBUG: Evaluating initial population")
        Engine.evaluate_population(init_population)
    st.write("DEBUG: Population initialization complete")

    # Optionally, you can re-enable simplification after initialization if needed:
    # config.USE_SIMPLIFICATION = True

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
    st.write("DEBUG: Setting up real-time plotting")
    fig, ax = plt.subplots()
    plot_placeholder = st.empty()  # This placeholder will be updated in real time

    # Create a progress bar for evolution iterations
    evolution_progress = st.progress(0)
    st.write("DEBUG: Starting evolution iterations")

    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(iterations[-1], iterations[-1] + number_of_iterations):
            st.write(f"DEBUG: Starting iteration {i+1}")
            # Evolution step: Generate new population (pass verbose=1 for consistency)
            new_population = Engine.generate_new_population(population=new_population.copy())
            st.write(f"DEBUG: Generated new population at iteration {i+1}")

            # (Simplification has been disabled to avoid collapsing expressions to constants)
            # Evaluate population and record metrics
            avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)
            avg_fitness_arr.append(avg_fitness)
            avg_complexity_arr.append(avg_complexity)
            best_fitness_arr.append(optimal_fitness)
            iterations.append(i + 1)
            st.write(f"DEBUG: Iteration {i+1} metrics -- Avg Fitness: {avg_fitness}, Avg Complexity: {avg_complexity}")

            finish_time = time.time()
            elapsed_time = finish_time - start_time
            start_time = finish_time
            time_array.append(time_array[-1] + elapsed_time)

            if hasattr(config, 'PARETO_INDEX_INTERVAL') and config.PARETO_INDEX_INTERVAL is not None:
                if i % config.PARETO_INDEX_INTERVAL == 0:
                    changing_pareto_fronts.append(Engine.return_pareto_front(new_population))
                    changing_pareto_front_indecies.append(i)
                    st.write(f"DEBUG: Recorded pareto front at iteration {i+1}")

            if len(avg_fitness_arr) > config.FITNESS_REDUCTION_THRESHOLD:
                if (min(avg_fitness_arr[-config.FITNESS_REDUCTION_THRESHOLD:]) == avg_fitness_arr[-config.FITNESS_REDUCTION_THRESHOLD]) and \
                   (config.FIT_THRESHOLD * config.FITNESS_REDUCTION_FACTOR > avg_fitness):
                    config.FIT_THRESHOLD = config.FIT_THRESHOLD * config.FITNESS_REDUCTION_FACTOR
                    fitness_reduction_indecies.append(i + 1)
                    st.write(f"DEBUG: Adjusted FIT_THRESHOLD to {config.FIT_THRESHOLD} at iteration {i+1}")

            if len(avg_fitness_arr) > config.EARLY_STOPPING_THRESHOLD:
                if min(avg_fitness_arr[-config.EARLY_STOPPING_THRESHOLD:]) == avg_fitness_arr[-config.EARLY_STOPPING_THRESHOLD]:
                    st.write(f"DEBUG: Early stopping triggered at iteration {i+1}")
                    break

            # --- Update the Real-Time Plot ---
            ax.cla()
            ax.plot(iterations, avg_fitness_arr, 'bo-', label="Average Population Fitness")
            ax.plot(iterations, avg_complexity_arr, 'ro-', label="Complexity")
            ax.plot(iterations, best_fitness_arr, 'go-', label="Lowest Population Fitness")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_ylabel("Fitness - 1-$R^2$")
            ax.set_xlabel("Iteration")
            ax.set_yscale("log")
            ax.set_title(f"Population Metrics {dataset_choice} // {output_var}")

            plot_placeholder.pyplot(fig)
            evolution_progress.progress(int((i + 1) / number_of_iterations * 100))
            time.sleep(0.5)  # Allow time for the plot to visibly update

    st.write("DEBUG: Evolution iterations complete")

    # --- Determine the Best Individual ---
    st.write("DEBUG: Determining best individual")
    pareto_front = Engine.return_pareto_front(new_population)
    pareto_front = list(pareto_front)
    pareto_front.sort(key=lambda x: x['fitness'], reverse=False)
    best_indiv = pareto_front[0]
    st.write("DEBUG: Best individual determined", best_indiv)

    # --- Convert Best Individual to a Sympy Expression and Wrap as an Equation ---
    st.write("DEBUG: Converting best individual to sympy expression")
    best_sympy_expr = simp.convert_expression_to_sympy(best_indiv['individual'])
    equation = sp.Eq(sp.Symbol(output_var), best_sympy_expr)
    st.latex(sp.latex(equation))
    st.write("DEBUG: Evolution experiment complete")
