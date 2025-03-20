# functions/symbolic_regression_functions.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the backend files from symbolic_regression_files.
from functions.symbolic_regression_files import config, EngineDict, Plotting, Simplification

def run_symbolic_regression(dataset, output_variable, pop_size, pop_retention_size, num_iterations):
    """
    Runs the symbolic regression algorithm based on the provided parameters.
    
    Parameters:
      - dataset: str, either "CORROSION" or "HEATSINK"
      - output_variable: str, e.g., "Corrosion Rate", "Saturation Ratio", "Pressure Drop", "Thermal Resistance"
      - pop_size: int, the population size to use
      - pop_retention_size: int, the number of individuals to retain between generations
      - num_iterations: int, the number of generations to evolve
      
    Returns:
      - population: dict, the final evolved population
      - pareto_df: pandas.DataFrame, a dataframe with Pareto front scores
      - DV: str, a code for the selected dependent variable (e.g. "CR", "PD", etc.)
    """
    # Set the dataset in config (ensure uppercase for consistency)
    dataset = dataset.upper()
    config.DATASET = dataset
    
    # Set population parameters in config
    config.POPULATION_SIZE = pop_size
    config.POPULATION_RETENTION_SIZE = pop_retention_size
    
    # Map the output variable to a dependent variable code (DV)
    DV = None
    if dataset == "CORROSION":
        if output_variable.lower() == "corrosion rate":
            DV = "CR"
        elif output_variable.lower() == "saturation ratio":
            DV = "SR"
    elif dataset == "HEATSINK":
        if output_variable.lower() == "pressure drop":
            DV = "PD"
        elif output_variable.lower() == "thermal resistance":
            DV = "TR"
    
    # Load the dataset.
    # In your final implementation, you can load real data from a file in
    # functions/symbolic_regression_files/data. For demonstration, we use dummy data.
    if dataset == "HEATSINK":
        # Heatsink expects 2 input features.
        config.X = np.random.rand(100, 2)
        config.y = np.random.rand(100)
    elif dataset == "CORROSION":
        # Corrosion expects 5 input features.
        config.X = np.random.rand(100, 5)
        config.y = np.random.rand(100)
    
    # Initialize the population using the EngineDict functions.
    population = EngineDict.initialize_population()
    
    # Evolve the population for the specified number of iterations.
    for i in range(num_iterations):
        population = EngineDict.generate_new_population(population)
    
    # Get the Pareto front scores as a DataFrame.
    pareto_df = EngineDict.get_pareto_scores(population)
    
    # Plot the Pareto front.
    Plotting.plot_pareto(population)
    plt.title("Pareto Front")
    # In your Streamlit code, you can display this plot with st.pyplot(plt.gcf())
    
    return population, pareto_df, DV

if __name__ == "__main__":
    # For testing purposes
    pop, pareto, dv = run_symbolic_regression("HEATSINK", "Thermal Resistance", 1500, 300, 10)
    print("Dependent Variable Code:", dv)
    print(pareto)
    plt.show()
