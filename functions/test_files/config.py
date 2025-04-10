# functions/test_files/config.py

# --- THRESHOLDS ---
FIT_THRESHOLD = 10             # Lowered to speed up initial population acceptance
COMPLEXITY_MAX_THRESHOLD = 400
COMPLEXITY_MIN_THRESHOLD = 1

# --- GLOBAL VARIABLES ---
POPULATION_SIZE = 1500
POPULATION_RETENTION_SIZE = 300
TORNEMENT_SIZE = 3

# --- SETTINGS ---
# DATASET will be set dynamically (CORROSION or HEATSINK)
DATASET = 'BENCHMARK'
USE_RMSE = False
USE_SIMPLIFICATION = True

DISPLAY_ERROR_MESSAGES = False
DISPLAY_SIMPLIFY_ERROR_MESSAGES = False

# For selection and mating:
SELECTION_METHOD = 'pareto'     # Options: 'pareto' or 'fitness'
# (In some of your older code, this was referenced as TORN_SELECTION_METHOD; adjust accordingly.)
MATE_MUTATE_SELECTION_METHOD = 'all'

VERBOSE = 1
DISPLAY_INTERVAL = 200

# --- Iteration Parameters ---
PARETO_INDEX_INTERVAL = 5
SIMPLIFICATION_INDEX_INTERVAL = 20
EARLY_STOPPING_THRESHOLD = 20
FITNESS_REDUCTION_THRESHOLD = 5
FITNESS_REDUCTION_FACTOR = 0.8

# Global variables for data; these will be set at runtime.
TOOLBOX = None
PSET = None
X = None
y = None
std_y = None
mean_y = None
