# SETTINGS - DEFAULT VALUES
# Everything in here is a global varibale, visable and changable by both files 
FIT_THRESHOLD = 1000
COMPLEXITY_THRESHOLD = 300

POPULATION_SIZE = 200
POPULATION_RETENTION_SIZE = 50

TORNEMENT_SIZE = 2

USE_RMSE = 0 #if =0 then we use 1-R^2
USE_SIMPLIFICATION = 1 # if =1 then we using the simplification code

PARETO_INDEX_INTERVAL = 5
SIMPLIFICATION_INDEX_INTERVAL = 10

# Dont set both to 1
# If we want to use the pareto selcetion method of the parents and children of a mate mutate - otherwise its just the best 2 of the 4...
# If neither -> it will be fitness optimal torn selection and all will be carried forwards in the Mate_Mutate Stuff
SELECTION_METHOD = 'pareto' # Either 'pareto' or 'fitness' or any other string
EARLY_STOPPING_THRESHOLD = 10
FITNESS_REDUCTION_THRESHOLD = 4
FITNESS_REDUCTION_FACTOR = 0.9

# Allows for artificail function generation in the other code
TOOLBOX = None
PSET = None

# Set X and y in the .ipynb code so it can be changed later - evaluate_expression should take theese values by default
X = None
y = None
std_y = None
mean_y = None
