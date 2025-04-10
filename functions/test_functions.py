import time
import warnings
import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from deap import gp

# Import modules from the test_files folder.
from functions.test_files import config
from functions.test_files import EngineDict as Engine
from functions.test_files import plotting    # Not actively used in the loop below, but included if needed.
from functions.test_files import simplification as simp

def run_evolution_experiment(dataset_choice, output_var, population_size, population_retention_size, number_of_iterations, st_container):
    """
    Perform the evolution experiment.
    
    Loads the dataset (CORROSION or HEATSINK), preprocesses the data,
    sets evolution configuration parameters, and runs the evolutionary loop.
    Real-time metrics are plotted into the provided st_container.
    Finally, the best simplified equation is rendered using Streamlit.
    """
    # --- Set configuration variable ---
    config.DATASET = dataset_choice

    # --- Dataset Selection and Preprocessing ---
    if dataset_choice == 'CORROSION':
        # Download and load the corrosion dataset
        csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
        response = requests.get(csv_url)
        df = pd.read_csv(io.StringIO(response.text))
        df.rename(columns={"Pp CO2": "PpCO2"}, inplace=True)
        df = df.replace('', np.nan)
        df = df.astype(float)
        
        # Create new logarithmic columns
        df["LogP"] = np.log10(df["PCO2"])
        df["LogV"] = np.log10(df["v"])
        df["LogD"] = np.log10(df["d"])
        
        # Transformation details
        transformation_dict = {
            "pH":    [5, 6],
            "Tc":    [0, 100],
            "LogP":  [-1, 1],
            "LogV":  [-1, 1],
            "LogD":  [-2, 0],
        }
        
        # Sample a subset
        np.random.seed(42)
        sample_size = 2000
        sample_indices = np.random.choice(df.index, size=sample_size, replace=False)
        df_sampled = df.loc[sample_indices]
        
        # Feature and target extraction
        X = df_sampled[["pH", "Tc", "LogP", "LogV", "LogD"]].values
        y = df_sampled[output_var].values.reshape(-1,)
        
        # Normalize features
        transformed_X = np.array([
            (X[:, i] - transformation_dict[col][0]) / (transformation_dict[col][1] - transformation_dict[col][0]) + 1
            for i, col in enumerate(["pH", "Tc", "LogP", "LogV", "LogD"])
        ]).T
        
        # Standardise target
        mean_y = np.mean(y)
        std_y = np.std(y)
        config.mean_y = mean_y
        config.std_y = std_y
        standardised_y = (y - mean_y) / std_y
        
        config.X = transformed_X
        config.y = standardised_y

    elif dataset_choice == 'HEATSINK':
        # Load heatsink dataset from file
        heatsink_file = "functions/test_files/data/Latin_Hypercube_Heatsink_1000_samples.txt"
        with open(heatsink_file, "r") as f:
            text = f.read()
        data = [x.split(' ') for x in text.split('\n') if x.strip() != '']
        df = pd.DataFrame(data, columns=['Geometric1', 'Geometric2', 'Thermal_Resistance', 'Pressure_Drop'])
        df = df.apply(pd.to_numeric)
        
        X = df[['Geometric1', 'Geometric2']].values
        y = df[output_var].values.reshape(-1,)
        
        # Standardise target variable
        mean_y = np.mean(y)
        std_y = np.std(y)
        config.mean_y = mean_y
        config.std_y = std_y
        standardised_y = (y - mean_y) / std_y
        
        config.X = X
        config.y = standardised_y
    else:
        raise ValueError("Invalid dataset_choice provided.")

    # --- Set Configuration Parameters ---
    config.POPULATION_SIZE = population_size
    config.POPULATION_RETENTION_SIZE = population_retention_size
    config.FIT_THRESHOLD = 1
    config.USE_SIMPLIFICATION = True
    config.DISPLAY_ERROR_MESSAGES = False
    config.VERBOSE = True
    config.SIMPLIFICATION_INDEX_INTERVAL = 20
    config.EARLY_STOPPING_THRESHOLD = 20
    config.FITNESS_REDUCTION_THRESHOLD = 5
    config.FITNESS_REDUCTION_FACTOR = 0.8
    config.FIT_THRESHOLD = 10

    # --- Initialize Population ---
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

    start_time = time.time()
    fig, ax = plt.subplots()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(number_of_iterations):
            # Generate new population
            new_population = Engine.generate_new_population(population=new_population.copy())
            
            # Simplification step at given intervals
            if config.USE_SIMPLIFICATION and i % config.SIMPLIFICATION_INDEX_INTERVAL == 0:
                new_population = Engine.simplify_population(new_population)
            
            # Evaluate population
            avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)
            avg_fitness_arr.append(avg_fitness)
            avg_complexity_arr.append(avg_complexity)
            best_fitness_arr.append(optimal_fitness)
            iterations.append(iterations[-1] + 1)

            # Timing (optional)
            elapsed = time.time() - start_time
            start_time = time.time()

            # --- Update Plot in Real Time ---
            ax.cla()  # Clear current axes
            ax.plot(iterations, avg_fitness_arr, label="Average Fitness")
            ax.plot(iterations, avg_complexity_arr, label="Complexity")
            ax.plot(iterations, best_fitness_arr, label="Lowest Fitness")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_ylabel("Fitness (1-$R^2$)")
            ax.set_xlabel("Iteration")
            ax.set_yscale("log")
            ax.set_title(f"Population Metrics {dataset_choice} // {output_var}")
            
            # Update the Streamlit container
            st_container.pyplot(fig)
            plt.pause(0.1)
            time.sleep(0.1)

    # --- Select the Best Individual from the Pareto Front ---
    pareto_front = list(Engine.return_pareto_front(new_population))
    pareto_front.sort(key=lambda x: x['fitness'])
    best_indiv = pareto_front[0]

    # --- Simplify and Construct the Final Equation ---
    best_sympy_expr = simp.convert_expression_to_sympy(best_indiv['individual'])
    equation = sp.Eq(sp.Symbol(output_var), best_sympy_expr)

    # Clear the placeholder and display the final equation.
    st_container.empty()
    st_container.write("### Final Simplified Equation:")
    st_container.latex(sp.pretty(equation, use_unicode=True))
