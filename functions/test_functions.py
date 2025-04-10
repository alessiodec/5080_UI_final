# functions/test_functions.py
import time
import warnings
import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from deap import gp

# Import modules from the test_files folder using absolute imports.
from functions.test_files import config
from functions.test_files import EngineDict as Engine
from functions.test_files import plotting    # Included if needed
from functions.test_files import simplification as simp

def run_evolution_experiment(dataset_choice, output_var, population_size, population_retention_size, number_of_iterations, st_container):
    """
    Perform the evolution experiment.
    
    Loads the dataset (CORROSION or HEATSINK), preprocesses it,
    sets evolution configuration parameters, and runs the evolutionary loop.
    Real-time metrics are plotted in st_container.
    
    Debug outputs are shown at each major step.
    """
    # Create a debug log container using st_container.subcontainer()
    debug_log = st_container.container()  # You can also use st.empty() if desired.
    debug_log.text("DEBUG: Starting evolution experiment...")

    # --- Set configuration variable ---
    config.DATASET = dataset_choice
    debug_log.text(f"DEBUG: DATASET set to {dataset_choice}")

    # --- Dataset Selection and Preprocessing ---
    if dataset_choice == 'CORROSION':
        debug_log.text("DEBUG: Loading CORROSION dataset...")
        csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
        response = requests.get(csv_url)
        df = pd.read_csv(io.StringIO(response.text))
        df.rename(columns={"Pp CO2": "PpCO2"}, inplace=True)
        df = df.replace('', np.nan)
        df = df.astype(float)
        df["LogP"] = np.log10(df["PCO2"])
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
        debug_log.text("DEBUG: CORROSION dataset loaded and preprocessed.")
        
    elif dataset_choice == 'HEATSINK':
        debug_log.text("DEBUG: Loading HEATSINK dataset...")
        heatsink_file = "functions/test_files/data/Latin_Hypercube_Heatsink_1000_samples.txt"
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
        debug_log.text("DEBUG: HEATSINK dataset loaded and preprocessed.")
    else:
        raise ValueError("Invalid dataset_choice provided.")

    # --- Set Configuration Parameters ---
    config.POPULATION_SIZE = population_size
    config.POPULATION_RETENTION_SIZE = population_retention_size
    # Note: In this new code we set FIT_THRESHOLD as 10 (as seen in your heatsink code) to speed up acceptance.
    config.FIT_THRESHOLD = 10  
    config.USE_SIMPLIFICATION = True
    config.DISPLAY_ERROR_MESSAGES = False
    config.VERBOSE = True
    config.SIMPLIFICATION_INDEX_INTERVAL = 20
    config.EARLY_STOPPING_THRESHOLD = 20
    config.FITNESS_REDUCTION_THRESHOLD = 5
    config.FITNESS_REDUCTION_FACTOR = 0.8
    debug_log.text("DEBUG: Configuration parameters set.")

    # --- Initialize Population ---
    debug_log.text("DEBUG: Initializing population...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        init_population = Engine.initialize_population()
        Engine.evaluate_population(init_population)
    new_population = init_population.copy()
    avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)
    iterations = [0]
    avg_fitness_arr = [avg_fitness]
    avg_complexity_arr = [avg_complexity]
    best_fitness_arr = [optimal_fitness]
    debug_log.text(f"DEBUG: Initial population created. Avg fitness = {avg_fitness:.4f}")

    start_time = time.time()
    fig, ax = plt.subplots()

    # --- Evolution Loop ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(number_of_iterations):
            debug_log.text(f"DEBUG: Iteration {i+1}/{number_of_iterations} starting...")
            new_population = Engine.generate_new_population(population=new_population.copy())
            if config.USE_SIMPLIFICATION and i % config.SIMPLIFICATION_INDEX_INTERVAL == 0:
                new_population = Engine.simplify_population(new_population)
                debug_log.text(f"DEBUG: Population simplified at iteration {i+1}.")
            avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)
            avg_fitness_arr.append(avg_fitness)
            avg_complexity_arr.append(avg_complexity)
            best_fitness_arr.append(optimal_fitness)
            iterations.append(iterations[-1] + 1)
            elapsed = time.time() - start_time
            start_time = time.time()

            # Update plot
            ax.cla()
            ax.plot(iterations, avg_fitness_arr, label="Average Fitness")
            ax.plot(iterations, avg_complexity_arr, label="Complexity")
            ax.plot(iterations, best_fitness_arr, label="Lowest Fitness")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_ylabel("Fitness (1-$R^2$)")
            ax.set_xlabel("Iteration")
            ax.set_yscale("log")
            ax.set_title(f"Population Metrics {dataset_choice} // {output_var}")
            st_container.pyplot(fig)
            debug_log.text(f"DEBUG: Iteration {i+1} complete. Avg fitness = {avg_fitness:.4f}, Best fitness = {optimal_fitness:.6f}, elapsed = {elapsed:.2f}s")
            plt.pause(0.1)
            time.sleep(0.1)

    # --- Select Best Individual ---
    debug_log.text("DEBUG: Selecting best individual from Pareto front...")
    pareto_front = list(Engine.return_pareto_front(new_population))
    pareto_front.sort(key=lambda x: x['fitness'])
    best_indiv = pareto_front[0]

    # --- Simplify and Construct Final Equation ---
    debug_log.text("DEBUG: Simplifying best individual to construct final equation...")
    best_sympy_expr = simp.convert_expression_to_sympy(best_indiv['individual'])
    equation = sp.Eq(sp.Symbol(output_var), best_sympy_expr)

    # Clear container and output final equation.
    st_container.empty()
    debug_log.empty()  # clear debug log container
    st_container.write("### Final Simplified Equation:")
    st_container.latex(sp.pretty(equation, use_unicode=True))
