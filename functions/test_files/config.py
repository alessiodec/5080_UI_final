# functions/test_files/config.py

# --- THRESHOLDS ---
FIT_THRESHOLD = 1000             # Lowered to speed up initial population acceptance
COMPLEXITY_MAX_THRESHOLD = 400
COMPLEXITY_MIN_THRESHOLD = 1

# --- GLOBAL VARIABLES ---
POPULATION_SIZE = 100
POPULATION_RETENTION_SIZE = 50
TORNEMENT_SIZE = 2

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
SIMPLIFICATION_INDEX_INTERVAL = 10
EARLY_STOPPING_THRESHOLD = 10
FITNESS_REDUCTION_THRESHOLD = 4
FITNESS_REDUCTION_FACTOR = 0.9

# Global variables for data; these will be set at runtime.
TOOLBOX = None
PSET = None
X = None
y = None
std_y = None
mean_y = None
