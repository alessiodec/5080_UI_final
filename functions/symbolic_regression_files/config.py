# config.py
# Global configuration for SR Beta

# --- THRESHOLDS ---
FIT_THRESHOLD = 100
COMPLEXITY_MAX_THRESHOLD = 400
COMPLEXITY_MIN_THRESHOLD = 1

# --- GLOBAL VARIABLES ---
POPULATION_SIZE = 1500
POPULATION_RETENTION_SIZE = 300
TORNEMENT_SIZE = 3

# --- SETTINGS ---
DATASET = 'BENCHMARK'  # Options: 'CORROSION', 'HEATSINK', or 'BENCHMARK'
USE_RMSE = False       # If False, fitness = 1 - R^2
USE_SIMPLIFICATION = True

DISPLAY_ERROR_MESSAGES = False
DISPLAY_SIMPLIFY_ERROR_MESSAGES = False

TORN_SELECTION_METHOD = 'pareto'        # 'pareto' or 'fitness'
MATE_MUTATE_SELECTION_METHOD = 'all'      # 'pareto', 'fitness', or 'all'

VERBOSE = 1
DISPLAY_INTERVAL = 200

# --- Reading and update intervals ---
PARETO_INDEX_INTERVAL = 5
SIMPLIFICATION_INDEX_INTERVAL = 10
EARLY_STOPPING_THRESHOLD = 10
FITNESS_REDUCTION_THRESHOLD = 4
FITNESS_REDUCTION_FACTOR = 0.9

# Global placeholders (set in run_evolution_experiment)
TOOLBOX = None
PSET = None
X = None
y = None
std_y = None
mean_y = None
