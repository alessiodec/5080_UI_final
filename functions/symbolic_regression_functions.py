import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import sympy as sp
from sklearn.metrics import r2_score
from deap import gp
import warnings

# --- Import and Setup Config ---
import config
import EngineDict as Engine
import Plotting as Plot
import Simplification as simp

def run_evolution_experiment(dataset_choice, output_var, population_size, population_retention_size, number_of_iterations):
    config.DATASET = dataset_choice  
    config.POPULATION_SIZE = population_size
    config.POPULATION_RETENTION_SIZE = population_retention_size
    config.FIT_THRESHOLD = 10
    config.USE_SIMPLIFICATION = True

    # --- Initialize Population ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        new_population = Engine.initialize_population()
        Engine.evaluate_population(new_population)

    # --- Initialize Logging Arrays ---
    avg_fitness_arr = []
    avg_complexity_arr = []
    best_fitness_arr = []
    iterations = []
    
    start_time = time.time()
    fig, ax = plt.subplots()

    # --- Evolution Loop ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(number_of_iterations):
            new_population = Engine.generate_new_population(population=new_population.copy())
            avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)

            avg_fitness_arr.append(avg_fitness)
            avg_complexity_arr.append(avg_complexity)
            best_fitness_arr.append(optimal_fitness)
            iterations.append(i + 1)

            # --- Real-Time Plotting in Streamlit ---
            ax.cla()
            ax.plot(iterations, avg_fitness_arr, label="Average Population Fitness")
            ax.plot(iterations, avg_complexity_arr, label="Complexity")
            ax.plot(iterations, best_fitness_arr, label="Lowest Population Fitness")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_ylabel("Fitness - 1-$R^2$")
            ax.set_xlabel("Iteration")
            ax.set_yscale("log")
            ax.set_title(f"Population Metrics: {dataset_choice} // {output_var}")
            
            st.pyplot(fig)
            time.sleep(0.1)  # Simulate real-time updates

    # --- Determine the Best Individual ---
    pareto_front = Engine.return_pareto_front(new_population)
    pareto_front = sorted(pareto_front, key=lambda x: x['fitness'])
    best_indiv = pareto_front[0]

    # --- Simplify and Display the Final Equation ---
    best_sympy_expr = simp.convert_expression_to_sympy(best_indiv['individual'])
    equation = sp.Eq(sp.Symbol(output_var), best_sympy_expr)

    st.subheader("Final Symbolic Regression Equation")
    st.latex(sp.latex(equation))  # Display equation using LaTeX in Streamlit
