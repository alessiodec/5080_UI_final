# functions/symbolic_regression_files/EngineDict.py

"""
Functions for implementing the SR - 14.2.25 - HEATSINK
Currently only valid for the Thermal Resistance of the Heatsink dataset - needs to be made dynamic.
For now these are just population functions.
"""

from . import config
import random
import numpy as np
import warnings
import math
import operator
from . import Simplification as simp
import pandas as pd

from deap import gp
from sklearn.metrics import root_mean_squared_error, r2_score
from deap import base, creator, gp, tools, algorithms
from functools import partial
from datetime import datetime

import sympy as sp
from sympy import Add, Mul, Pow, Number, Symbol
from sympy import log as sympy_log, exp as sympy_exp

import streamlit as st  # For Streamlit display

# Custom exception for our operations
class CustomOperationException(Exception):
    pass

# ---------- Protected Operations ----------

def protectedDiv(left, right):
    try:
        return left / right
    except Exception:
        raise CustomOperationException('protectedDiv Error')

def protectedLog(value):
    try:
        return math.log(value)
    except Exception:
        raise CustomOperationException('protectedLog Error')
        
def protectedExp(value):
    try:
        return math.exp(value)
    except Exception:
        raise CustomOperationException('protectedExp Error')

# Generate random constants between -1 and 1 (rounded to 4dp)
def random_constant():
    return round(random.uniform(-1, 1), 4)

# ---------- Define the set of operations --------------

if config.DATASET == 'HEATSINK':
    num_inputs = 2
elif config.DATASET == 'CORROSION':
    num_inputs = 5
elif config.DATASET == 'BENCHMARK':
    num_inputs = 3
else:
    raise CustomOperationException('config.DATASET has not been set correctly - should be either "BENCHMARK", "HEATSINK", or "CORROSION"')

pset = gp.PrimitiveSet("MAIN", arity=num_inputs)
pset.addPrimitive(operator.add, 2)     
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)      
pset.addPrimitive(protectedDiv, 2) 
pset.addPrimitive(operator.pow, 2)
pset.addPrimitive(protectedExp, 1)    
pset.addPrimitive(protectedLog, 1)
pset.addEphemeralConstant("randConst", partial(random_constant))

if config.DATASET == 'HEATSINK':
    pset.renameArguments(ARG0='G1')
    pset.renameArguments(ARG1='G2')
elif config.DATASET == 'CORROSION':
    pset.renameArguments(ARG0='pH')
    pset.renameArguments(ARG1='T')
    pset.renameArguments(ARG2='LogP')
    pset.renameArguments(ARG3='LogV')
    pset.renameArguments(ARG4='LogD')
elif config.DATASET == 'BENCHMARK':
    pset.renameArguments(ARG0='X1')
    pset.renameArguments(ARG1='X2')
    pset.renameArguments(ARG2='X3')

# ---------- Define Types (with safe re-creation for Streamlit) ----------

try:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
except Exception:
    pass

try:
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
except Exception:
    pass

# ---------- Create Toolbox and Register Functions ----------

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

config.TOOLBOX = toolbox
config.PSET = pset

# ---------- Evaluation Functions ----------

def evaluate_individual(individual):
    func = gp.compile(expr=individual, pset=pset)
    complexity = len(individual)
    '''
    # Check for required terminals in HEATSINK
    if config.DATASET == 'HEATSINK' and not (('G1' in str(individual)) and ('G2' in str(individual))):
        return config.FIT_THRESHOLD + 1, complexity

    # Add a similar check for CORROSION
    if config.DATASET == 'CORROSION':
        individual_str = str(individual)
        required_terms = ["pH", "T", "LogP", "LogV", "LogD"]
        if not all(term in individual_str for term in required_terms):
            return config.FIT_THRESHOLD + 1, complexity

    '''
    try:
        if config.DATASET == 'HEATSINK':
            y_pred = [func(x[0], x[1]) for x in config.X]
        elif config.DATASET == 'CORROSION': 
            y_pred = [func(x[0], x[1], x[2], x[3], x[4]) for x in config.X]
        elif config.DATASET == 'BENCHMARK':
            y_pred = [func(x[0], x[1], x[2]) for x in config.X]
        else: 
            raise CustomOperationException('config.DATASET is not set correctly')
        
        y_pred = np.array(y_pred).reshape(-1,)
    
        if config.USE_RMSE:
            fitness = root_mean_squared_error(config.y, y_pred)
        else:
            fitness = 1 - r2_score(config.y, y_pred)
    
    except Exception as e:
        if config.DISPLAY_ERROR_MESSAGES:
            st.write(e)
        fitness = config.FIT_THRESHOLD + 1
    
    return fitness, complexity


def evaluate_population(population):
    total_fitness = 0
    total_complexity = 0
    min_fitness = config.FIT_THRESHOLD

    for individual in population.values():
        fitness = individual["fitness"]
        complexity = individual["complexity"]
        total_fitness += fitness
        total_complexity += complexity
        min_fitness = min(min_fitness, fitness)

    return total_fitness / len(population), total_complexity / len(population), min_fitness

def display_progress(population, last_printed_index):
    if len(population) == last_printed_index:
        return last_printed_index
        
    if config.VERBOSE and (len(population) % config.DISPLAY_INTERVAL == 0) and (len(population) > 0):
        avg_fit, avg_comp, best_fit = evaluate_population(population)
        st.write(f'Len:{len(population)}, Avg Fit:{avg_fit:.4f}, Avg Comp:{avg_comp:.4f}, Best Fit:{best_fit:.6f}')
        return len(population)

def convert_individual_to_key(individual):
    format_str = ''
    for node in individual:
        try:
            float(node.name)
            format_str += '_COEFF_'
        except Exception:
            if node.name == 'randConst':
                format_str += '_COEFF_'
            else:
                format_str += node.name
    return format_str  

# ---------- Population Simplification ----------

def simplify_population(population):
    if config.VERBOSE:
        st.write('\n-------------- SIMPLIFICATION --------------')
    
    simplified_population = {}   
    index_tracker = 0
    
    for individual in population.values():
        index_tracker = display_progress(population=simplified_population, last_printed_index=index_tracker)
        
        if individual['is_simplified']:
            key = convert_individual_to_key(individual['individual'])
            if key not in simplified_population or individual['fitness'] < simplified_population[key]['fitness']:
                simplified_population[key] = individual
            continue
    
        simplified_indiv = simp.simplify_individual(individual['individual'])
        
        if simplified_indiv is None:
            continue
            
        fitness, new_complexity = evaluate_individual(simplified_indiv)
    
        if new_complexity > config.COMPLEXITY_MAX_THRESHOLD or new_complexity < config.COMPLEXITY_MIN_THRESHOLD or fitness > config.FIT_THRESHOLD:
            continue
            
        elif new_complexity > individual['complexity']:
            key = convert_individual_to_key(individual['individual'])
            if config.DISPLAY_ERROR_MESSAGES:
                st.write(f"failure -> individual: {individual['individual']}")
            if key not in simplified_population or fitness < simplified_population[key]['fitness']:
                simplified_population[key] = individual
                simplified_population[key]['is_simplified'] = True
        else:
            key = convert_individual_to_key(simplified_indiv)
            if key not in simplified_population or fitness < simplified_population[key]['fitness']:
                simplified_population[key] = {
                    'complexity': new_complexity,
                    'fitness': fitness,
                    'individual': simplified_indiv,
                    'is_simplified': True,
                }
    
    # Return original population if simplification filtered out everything.
    if not simplified_population:
        return population
    
    return simplified_population


# ---------- Population Initialization ----------

def initialize_population():
    init_population = {}
    index_tracker = 0
    
    while len(init_population) < config.POPULATION_SIZE:
        individual = toolbox.individual()
        fitness, complexity = evaluate_individual(individual=individual)
        
        # Remove fitness filtering: accept the individual if complexity is within bounds.
        if complexity > config.COMPLEXITY_MAX_THRESHOLD or complexity < config.COMPLEXITY_MIN_THRESHOLD:
            continue
            
        key = convert_individual_to_key(individual)
            
        if key not in init_population or fitness < init_population[key]['fitness']:
            init_population[key] = {
                'complexity': complexity,
                'fitness': fitness,
                'individual': individual,
                'is_simplified': False,
            }
            index_tracker = display_progress(population=init_population, last_printed_index=index_tracker)
    
    if config.USE_SIMPLIFICATION:
         init_population = simplify_population(init_population)
    
    return init_population



# ---------- Pareto and Dominance Functions ----------

def return_pareto_front(population):
    results = []
    for individual in population.values():
        results.append((individual['fitness'], individual['complexity']))
    results = np.array(results)
    
    is_pareto = np.ones(results.shape[0], dtype=bool)
    for i, c in enumerate(results):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(results[is_pareto] < c, axis=1)
            is_pareto[i] = True
    
    return np.array(list(population.values()))[is_pareto]

def dominates(ind1, ind2):
    fitness_1, complexity_1 = ind1['fitness'], ind1['complexity']
    fitness_2, complexity_2 = ind2['fitness'], ind2['complexity']
    return (fitness_1 <= fitness_2 and complexity_1 <= complexity_2) and (fitness_1 < fitness_2 or complexity_1 < complexity_2)

def generate_new_generation_NSGA_2(n, population, tournament_selection=False):
    dominated_counts = [0] * len(population)

    if isinstance(population, dict):
        population = list(population.values())

    for i, ind1 in enumerate(population):
        for j, ind2 in enumerate(population[i:]):
            if dominates(ind1, ind2):
                dominated_counts[j + i] += 1
            elif dominates(ind2, ind1):
                dominated_counts[i] += 1
                
    pareto_fronts = [[] for _ in range(max(dominated_counts) + 1)]
    
    for i, ind in enumerate(population):
        pareto_fronts[dominated_counts[i]].append(ind)
    
    pareto_index = 0
    next_generation = []
    
    while len(next_generation) < n:
        if len(next_generation) + len(pareto_fronts[pareto_index]) <= n:
            next_generation.extend(pareto_fronts[pareto_index])
            pareto_index += 1
        elif tournament_selection:
            pareto_fronts[pareto_index].sort(key=lambda x: x['fitness'], reverse=False)
            for i in range(n - len(next_generation)):
                next_generation.append(pareto_fronts[pareto_index][i])
        else:
            selected_ind = random.choice(pareto_fronts[pareto_index])
            pareto_fronts[pareto_index].remove(selected_ind)
            next_generation.append(selected_ind)

    return next_generation

# ---------- Tournament Selection, Mating and Mutation ----------

def tournament_selection(parent_generation: list, n_selected=2):
    selected = []
    tournament = random.sample(parent_generation, config.TORNEMENT_SIZE)
    
    if config.TORN_SELECTION_METHOD == 'pareto':
        selected = generate_new_generation_NSGA_2(n_selected, tournament, tournament_selection=True)
    else:
        tournament.sort(key=lambda x: x['fitness'], reverse=False)
        selected = tournament[:2]

    return selected

def mate_and_mutate(parent1, parent2, cxpb=0.95, mutpb=0.5):
    offspring1 = toolbox.clone(parent1['individual'])
    offspring2 = toolbox.clone(parent2['individual'])
    
    try:
        if random.random() < cxpb:
            toolbox.mate(offspring1, offspring2)
    except Exception as e: 
        if config.DISPLAY_ERROR_MESSAGES:
            st.write(f"Failed to MATE: {e}")
    
    try:
        if random.random() < mutpb:
            toolbox.mutate(offspring1)
        if random.random() < mutpb:
            toolbox.mutate(offspring2)
    except Exception as e: 
        if config.DISPLAY_ERROR_MESSAGES:
            st.write(f"Failed to MUTATE: {e}")
    
    custom_parent_arr = [parent1, parent2]
    
    for individual in [offspring1, offspring2]:
        fitness, complexity = evaluate_individual(individual)
        offspring = {
            'complexity': complexity,
            'fitness': fitness,
            'individual': individual,
            'is_simplified': False
        }
        custom_parent_arr.append(offspring)
    
    return custom_parent_arr

def generate_new_population(population: dict):
    new_gen_parents = generate_new_generation_NSGA_2(config.POPULATION_RETENTION_SIZE, population)
    new_population = {}
    index_tracker = 0
    
    max_attempts = config.POPULATION_SIZE * 10  # Allow up to 10× as many attempts as the desired size
    attempts = 0
    
    while len(new_population) < config.POPULATION_SIZE and attempts < max_attempts:
        attempts += 1
        parent1, parent2 = tournament_selection(new_gen_parents)
        mate_mutation_results = mate_and_mutate(parent1, parent2)
    
        if config.MATE_MUTATE_SELECTION_METHOD == 'pareto':
            selected_mate_mutation_results = return_pareto_front({i: m for i, m in enumerate(mate_mutation_results)})
        elif config.MATE_MUTATE_SELECTION_METHOD == 'fitness':
            mate_mutation_results.sort(key=lambda x: x['fitness'], reverse=False)
            selected_mate_mutation_results = mate_mutation_results[:2]
        else:
            selected_mate_mutation_results = mate_mutation_results
    
        for individual in selected_mate_mutation_results:
            if (individual['complexity'] > config.COMPLEXITY_MAX_THRESHOLD or
                individual['complexity'] < config.COMPLEXITY_MIN_THRESHOLD or
                individual['fitness'] > config.FIT_THRESHOLD):
                continue
            
            key = convert_individual_to_key(individual['individual'])
            if key not in new_population or individual['fitness'] < new_population[key]['fitness']:
                new_population[key] = individual
                index_tracker = display_progress(population=new_population, last_printed_index=index_tracker)
    
    if attempts >= max_attempts:
        st.write("WARNING: Maximum attempts reached in generate_new_population; returning previous population.")
    
    # If no new individuals were generated, return the original population instead of an empty dict.
    if len(new_population) == 0:
        return population
    
    return new_population




# ---------- File I/O and Post-processing ----------

def write_population_to_file(population: list, DV: str, STD: str):
    now = datetime.now()
    timestamp = now.strftime("%H-%M-%S_%d_%m")
    
    Possible_DVs = ['CR', 'PD', 'TR', 'SR']
    if DV not in Possible_DVs:
        st.write(f'ERROR: {DV} not in {Possible_DVs} - please change')
        return

    Possible_STDs = ['STD', 'NOTSTD']
    if STD not in Possible_STDs:
        st.write(f'ERROR: {STD} not in {Possible_STDs} - please change')
        return
    
    filename = f"Prev_Generations_Log/{DV}/{STD}_{timestamp}.txt"
    
    with open(filename, "a") as f:
        for i, individual in enumerate(population):
            f.write(str(individual['individual']))
            f.write('\n')

def read_population_from_file(filename: str):
    with open(filename, "r") as file:
        content = file.read()
    
    population = {}
    for expression in content.split('\n'):
        if expression == '':
            continue

        expression = expression.replace('neg(', 'mul(-1,')
        individual = gp.PrimitiveTree.from_string(expression, pset)
        fitness, complexity = evaluate_individual(individual)
        key = convert_individual_to_key(individual)
        if key not in population or fitness < population[key]['fitness']:
            population[key] = {
                'complexity': complexity,
                'fitness': fitness,
                'individual': individual,
                'is_simplified': False,
            }
        
    return population

def unstandardise_and_simplify_population(population: dict):
    injected_population = {}
    
    for individual in population.values():
        new_expression_str = f"add(mul({individual['individual']}, {config.std_y}), {config.mean_y})"
        individual = gp.PrimitiveTree.from_string(new_expression_str, pset)
        fitness, complexity = evaluate_individual(individual)
        key = convert_individual_to_key(individual)
        if key not in injected_population or fitness < injected_population[key]['fitness']:
            injected_population[key] = {
                'complexity': complexity,
                'fitness': fitness,
                'individual': individual,
                'is_simplified': False,
            }

    if config.USE_SIMPLIFICATION:
        injected_population = simplify_population(injected_population)
    
    return injected_population

def extend_population_with_saved_expressions(filenames: list, population: dict):
    for filename in filenames:
        read_population = read_population_from_file(filename)
        for individual in read_population.values():
            key = convert_individual_to_key(individual['individual'])
            if key not in population or individual['fitness'] < population[key]['fitness']:
                population[key] = individual
                
    return population

def get_pareto_scores(population):
    pareto_front = return_pareto_front(population)
    pareto_front = list(pareto_front)
    pareto_front.sort(key=lambda x: x['fitness'], reverse=True)
    
    scores = []
    lastFit = None
    lastComplexity = 0
    
    for individual in pareto_front:
        curFit = individual["fitness"]
        curComplexity = individual["complexity"]
    
        if lastFit is None:
            cur_score = 0.0
        else:
            if curFit > 0.0:
                if config.USE_RMSE:
                    cur_score = -np.log(curFit / lastFit) / (curComplexity - lastComplexity)
                else:
                    cur_score = -(curFit - lastFit) / (curComplexity - lastComplexity)
            else:
                cur_score = np.inf
    
        scores.append(cur_score)
        lastFit = curFit
        lastComplexity = curComplexity

    scores_df = pd.DataFrame(pareto_front)
    scores_df['score'] = scores
    scores_df['individual'] = scores_df['individual'].astype(str)
    
    return scores_df

def initialize_primitive_set():
    global toolbox  # Declare at the very beginning
    from deap import base, creator, gp, tools
    from functools import partial
    import operator
    import random
    import math
    from . import config

    # Determine the number of inputs based on the dataset
    if config.DATASET == 'HEATSINK':
        num_inputs = 2
    elif config.DATASET == 'CORROSION':
        num_inputs = 5
    elif config.DATASET == 'BENCHMARK':
        num_inputs = 3
    else:
        raise ValueError("Invalid dataset in config.DATASET")

    # Create the primitive set with the correct number of inputs
    pset = gp.PrimitiveSet("MAIN", arity=num_inputs)

    # Add primitives as before
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.pow, 2)
    pset.addPrimitive(protectedExp, 1)
    pset.addPrimitive(protectedLog, 1)
    pset.addEphemeralConstant("randConst", partial(random_constant))
    
    # Rename arguments based on the dataset
    if config.DATASET == 'HEATSINK':
        pset.renameArguments(ARG0='G1')
        pset.renameArguments(ARG1='G2')
    elif config.DATASET == 'CORROSION':
        pset.renameArguments(ARG0='pH')
        pset.renameArguments(ARG1='T')
        pset.renameArguments(ARG2='LogP')
        pset.renameArguments(ARG3='LogV')
        pset.renameArguments(ARG4='LogD')
    elif config.DATASET == 'BENCHMARK':
        pset.renameArguments(ARG0='X1')
        pset.renameArguments(ARG1='X2')
        pset.renameArguments(ARG2='X3')

    # Update the global config for PSET
    config.PSET = pset

    # Create the toolbox and register functions
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    
    # Update the global config for TOOLBOX
    config.TOOLBOX = toolbox

    # Now update the module-level variable to use the new toolbox
    toolbox = config.TOOLBOX


if __name__ == "__main__":
    st.write('Should not be run as main - EngineDict.py just contains utility functions')
