# functions/symbolic_regression_functions.py

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

    # Set dataset in config (happens quickly, no spinner needed)
    config.DATASET = dataset_choice

    # Reinitialize the primitive set and toolbox (happens quickly)
    Engine.initialize_primitive_set()

    # --- Data Loading and Preprocessing ---
    with st.spinner("Loading and preprocessing dataset..."):
        if dataset_choice == 'CORROSION':
            # Load and process CORROSION data
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
            # Check for zero std deviation which prevents standardization
            if std_y < 1e-9:
                st.error(f"Critical Error: Standard deviation of target '{output_var}' is near zero. Cannot proceed.")
                return
            config.mean_y = mean_y
            config.std_y = std_y
            standardised_y = (y - mean_y) / std_y
            config.X = transformed_X
            config.y = standardised_y

        elif dataset_choice == 'HEATSINK':
            # Load and process HEATSINK data
            heatsink_file = os.path.join("functions", "symbolic_regression_files", "data", "Latin_Hypercube_Heatsink_1000_samples.txt")
            # Check if file exists
            if not os.path.exists(heatsink_file):
                 st.error(f"Heatsink data file not found at: {heatsink_file}")
                 return
            with open(heatsink_file, "r") as f:
                text = f.read()
            data = [x.split(' ') for x in text.split('\n') if x.strip() != '']
            df = pd.DataFrame(data, columns=['Geometric1', 'Geometric2', 'Thermal_Resistance', 'Pressure_Drop'])
            df = df.apply(pd.to_numeric)
            X = df[['Geometric1', 'Geometric2']].values
            y = df[output_var].values.reshape(-1,)
            mean_y = np.mean(y)
            std_y = np.std(y)
            # Check for zero std deviation
            if std_y < 1e-9:
                st.error(f"Critical Error: Standard deviation of target '{output_var}' is near zero. Cannot proceed.")
                return
            config.mean_y = mean_y
            config.std_y = std_y
            standardised_y = (y - mean_y) / std_y
            config.X = X
            config.y = standardised_y
        else:
            # Should not happen if UI validates correctly, but good practice
            st.error("Invalid dataset choice provided.")
            return
        # Short pause to ensure spinner is visible
        time.sleep(0.5)

    # --- Set Evolution Parameters (Quick step, no spinner needed) ---
    config.FIT_THRESHOLD = 1000 # Adjusted threshold
    config.POPULATION_SIZE = population_size
    config.POPULATION_RETENTION_SIZE = population_retention_size
    config.DISPLAY_ERROR_MESSAGES = False # Keep errors off unless debugging
    config.VERBOSE = False # Turn off verbose EngineDict prints for cleaner UI
    config.SIMPLIFICATION_INDEX_INTERVAL = number_of_iterations + 1 # Disable in-loop default simplification
    config.EARLY_STOPPING_THRESHOLD = 20
    config.FITNESS_REDUCTION_THRESHOLD = 5
    config.USE_SIMPLIFICATION = False # Avoid collapsing expressions during evolution
    config.FITNESS_REDUCTION_FACTOR = 0.8

    # --- Initialize Population ---
    init_population = {} # Define before spinner in case initialization fails
    try:
        with st.spinner("Initializing population..."):
            init_population = Engine.initialize_population()  # Generate initial individuals
            if not init_population: # Check if initialization yielded any individuals
                 st.error("Failed to initialize population. Check parameters or evaluation function.")
                 return
            Engine.evaluate_population(init_population) # Evaluate fitness
            time.sleep(0.5) # Ensure spinner visibility
    except Exception as e:
        st.error(f"An error occurred during population initialization: {e}")
        # Optionally display traceback if needed for debugging
        # st.exception(e)
        return


    # --- Evolution Loop ---
    st.info("Running evolutionary process...") # Indicate start of the main loop
    new_population = init_population.copy()
    try:
        avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)
    except ZeroDivisionError:
         st.error("Initial population is empty after evaluation. Cannot proceed.")
         return
    except Exception as e:
         st.error(f"Error evaluating initial population before loop: {e}")
         return

    iterations = [0]
    avg_fitness_arr = [avg_fitness]
    avg_complexity_arr = [avg_complexity]
    best_fitness_arr = [optimal_fitness]

    # Setup for progress bar and plot updating
    progress_text = "Evolution Progress: Iteration 0/{}"
    evolution_progress = st.progress(0, text=progress_text.format(number_of_iterations))
    chart_placeholder = st.empty()
    start_time = time.time() # For timing if needed later

    # Run the evolution loop
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning) # Ignore common numpy warnings
        for i in range(number_of_iterations):
            try:
                # Generate the next generation
                new_population = Engine.generate_new_population(population_dict=new_population.copy())
                if not new_population:
                    st.warning(f"Iteration {i+1}: generate_new_population returned empty. Using previous population.")
                    # Decide how to handle - reuse old pop? break? For now, reuse old.
                    new_population = init_population # Revert to initial or previous? Let's reuse previous non-empty.
                    # Need to track the last known good population
                    # This part needs refinement based on desired behavior for empty generation

                # Evaluate population and record metrics
                avg_fit, avg_comp, best_fit = Engine.evaluate_population(new_population)

                # Update tracking arrays
                iterations.append(i+1)
                avg_fitness_arr.append(avg_fit)
                avg_complexity_arr.append(avg_comp)
                best_fitness_arr.append(best_fit)

                # Update progress bar
                progress_percent = int((i+1)/number_of_iterations*100)
                evolution_progress.progress(progress_percent, text=f"Evolution Progress: Iteration {i+1}/{number_of_iterations}")

                # Update the plot dynamically
                fig, ax = plt.subplots()
                ax.plot(iterations, avg_fitness_arr, 'bo-', label="Avg Fitness")
                ax.plot(iterations, avg_complexity_arr, 'ro-', label="Avg Complexity")
                ax.plot(iterations, best_fitness_arr, 'go-', label="Best Fitness")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Fitness (1 - R^2)")
                ax.set_yscale("log") # Keep log scale for fitness
                ax.legend(loc='best') # Use 'best' location for legend
                ax.set_title(f"Population Metrics: {dataset_choice} // {output_var}")
                ax.grid(True) # Add grid for better readability
                chart_placeholder.pyplot(fig)
                plt.close(fig) # Close the figure to free memory

                # Optional small delay for smoother UI update, can be removed if too slow
                # time.sleep(0.1)

            except Exception as loop_error:
                st.error(f"Error during evolution loop at iteration {i+1}: {loop_error}")
                # Optionally display traceback
                # st.exception(loop_error)
                st.warning("Stopping evolution due to error.")
                break # Exit the loop on error

    # Clear progress bar after loop finishes or breaks
    evolution_progress.empty()

    # --- Determine Best Individual ---
    final_equation = None
    with st.spinner("Analyzing results and finding best expression..."):
        if not new_population:
            st.error("Evolution resulted in an empty population. Cannot determine best individual.")
            return

        pareto_front = Engine.return_pareto_front(new_population)
                if not pareto_front: # Correctly checks if the list is empty
             st.warning("Pareto front is empty. Selecting best overall fitness individual.")
             # Fallback: find best fitness from the whole population
             pareto_list = sorted(list(new_population.values()), key=lambda x: x['fitness'])
        else:
            # Process Pareto front if not empty
            pareto_list = list(pareto_front) # Convert numpy array elements if needed
            pareto_list.sort(key=lambda x: x['fitness']) # Sort by fitness (lowest is best)

        if not pareto_list:
             st.error("Could not find any suitable individuals after evolution.")
             return

        best_indiv = pareto_list[0] # Best fitness is the first element

        # Simplify and convert the best individual
        try:
             # Use Simplification module correctly
             best_sympy_expr = simp.convert_expression_to_sympy(best_indiv['individual'])
             # Further simplification if desired (might be slow)
             # best_sympy_expr = simp.simplify_sympy_expression(best_sympy_expr)

             # Create the equation object
             final_equation = sp.Eq(sp.Symbol(output_var), best_sympy_expr)

        except Exception as final_error:
             st.error(f"Error processing final expression: {final_error}")
             # st.exception(final_error)

        time.sleep(0.5) # Ensure spinner visibility

    # --- Display Final Result ---
    if final_equation:
        st.success("Evolution Complete! Best Equation Found:")
        st.latex(sp.latex(final_equation))
    else:
        st.warning("Evolution finished, but failed to generate the final equation display.")
