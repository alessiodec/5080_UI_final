# functions/symbolic_regression_files/config.py

import streamlit as st

# --- THRESHOLDS ----
FIT_THRESHOLD = 100
COMPLEXITY_MAX_THRESHOLD = 400
COMPLEXITY_MIN_THRESHOLD = 1

# --- GLOBAL VARIABLES
POPULATION_SIZE = 1500
POPULATION_RETENTION_SIZE = 300
TORNEMENT_SIZE = 3

# --- SETTINGS --- 
DATASET = 'BENCHMARK'  # 'CORROSION', 'HEATSINK', or 'BENCHMARK' – used for having the same code...
USE_RMSE = False       # if False then we use 1-R^2 
USE_SIMPLIFICATION = True  # if True then we use the simplification code

DISPLAY_ERROR_MESSAGES = False         # Set to True for debugging
DISPLAY_SIMPLIFY_ERROR_MESSAGES = False  # For simplification debugging

TORN_SELECTION_METHOD = 'pareto'    # 'pareto' for Pareto-optimal or 'fitness' for fittest
MATE_MUTATE_SELECTION_METHOD = 'all'  # 'pareto', 'fitness', or 'all' – all will be carried forward

VERBOSE = 1           # Whether to display progress messages
DISPLAY_INTERVAL = 200  # How often to display progress

# --- Reading Intervals ---
PARETO_INDEX_INTERVAL = 5
SIMPLIFICATION_INDEX_INTERVAL = 10
EARLY_STOPPING_THRESHOLD = 10
FITNESS_REDUCTION_THRESHOLD = 4
FITNESS_REDUCTION_FACTOR = 0.9

# Variables for function generation in other modules (e.g., EngineDict)
TOOLBOX = None
PSET = None

# Data placeholders (set later in the notebook or application)
X = None
y = None
std_y = None
mean_y = None

if __name__ == "__main__":
    st.write("This is the config file for the symbolic regression system. It holds all global parameters and thresholds.")
