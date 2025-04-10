"""
Functions for implementing the SR - 14.2.25 - HEATSINK/CORROSION
Now extended to work for both datasets. In this “beta” version the population is maintained as a list rather than a dictionary.
This file contains functions for evaluating individuals, initializing the population, performing pareto selection, tournament selection,
mating/mutation, and generating new populations.
"""

from . import config
import random
import numpy as np
import warnings
import math
import operator
from . import Simplification as simp
import pandas as pd

from deap import gp, base, creator, tools, algorithms
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

# Random constant between -1 and 1 (rounded to 4dp)
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
    # --- Terminal checks (to discourage constant individuals) ---
    if config.DATASET == 'HEATSINK':
        if not (('G1' in str(individual)) and ('G2' in str(individual))):
            return config.FIT_THRESHOLD + 1, complexity
    elif config.DATASET == 'CORROSION':
        individual_str = str(individual)
        required_terms = ["pH", "T", "LogP", "LogV", "LogD"]
        if not all(term in individual_str for term in required_terms):
            return config.FIT_THRESHOLD + 1, complexity
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
            fitness = np.sqrt(np.mean((config.y - y_pred) ** 2))
            # Alternatively, one can use a proper root_mean_squared_error function.
        else:
            fitness = 1 - r2_score(config.y, y_pred)
    except Exception as e:
        if config.DISPLAY_ERROR_MESSAGES:
            st.write(e)
        fitness = config.FIT_THRESHOLD + 1
    return fitness, complexity

def evaluate_population(population):
    # population is expected to be a list of individuals (each is a dict)
    total_fitness = 0
    total_complexity = 0
    best_fitness = population[0]['fitness']
    for individual in population:
        total_fitness += individual['fitness']
        total_complexity += individual['complexity']
        best_fitness = min(best_fitness, individual['fitness'])
    return total_fitness / len(population), total_complexity / len(population), best_fitness

# ---------- Utility Functions ----------
def display_progress(population, last_printed_index):
    if len(population) == last_printed_index:
        return last_printed_index
    if config.VERBOSE and (len(population) % config.DISPLAY_INTERVAL == 0) and (len(population) > 0):
        avg_fit, avg_comp, best_fit = evaluate_population(population)
        st.write(f'Len:{len(population)}, Avg Fit:{avg_fit:.4f}, Avg Comp:{avg_comp:.4f}, Best Fit:{best_fit:.6f}')
        return len(population)
    return last_printed_index

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
def simplify_population(population_dict):
    # population_dict is a dictionary; we will simplify each and then return the dictionary
    if config.VERBOSE:
        st.write('\n-------------- SIMPLIFICATION --------------')
    simplified_population = {}
    for key, individual in population_dict.items():
        if individual['is_simplified']:
            simplified_population[key] = individual
            continue
        simplified_indiv = simp.simplify_individual(individual['individual'])
        if simplified_indiv is None:
            continue
        fitness, new_complexity = evaluate_individual(simplified_indiv)
        if new_complexity > config.COMPLEXITY_MAX_THRESHOLD or new_complexity < config.COMPLEXITY_MIN_THRESHOLD or fitness > config.FIT_THRESHOLD:
            continue
        else:
            new_key = convert_individual_to_key(simplified_indiv)
            if new_key not in simplified_population or fitness < simplified_population[new_key]['fitness']:
                simplified_population[new_key] = {
                    'complexity': new_complexity,
                    'fitness': fitness,
                    'individual': simplified_indiv,
                    'is_simplified': True,
                }
    if not simplified_population:
        return population_dict
    return simplified_population

def initialize_population():
    # Build population in a dictionary to enforce uniqueness, then return as a list.
    init_population_dict = {}
    while len(init_population_dict) < config.POPULATION_SIZE:
        individual = toolbox.individual()
        fitness, complexity = evaluate_individual(individual=individual)
        if complexity > config.COMPLEXITY_MAX_THRESHOLD or complexity < config.COMPLEXITY_MIN_THRESHOLD:
            continue
        key = convert_individual_to_key(individual)
        if key not in init_population_dict or fitness < init_population_dict[key]['fitness']:
            init_population_dict[key] = {
                'complexity': complexity,
                'fitness': fitness,
                'individual': individual,
                'is_simplified': False,
            }
    if config.USE_SIMPLIFICATION:
         init_population_dict = simplify_population(init_population_dict)
    return list(init_population_dict.values())

# ---------- Pareto and Dominance Functions ----------
def return_pareto_front(population):
    # population is a list
    results = [(ind['fitness'], ind['complexity']) for ind in population]
    results = np.array(results)
    is_pareto = np.ones(results.shape[0], dtype=bool)
    for i, c in enumerate(results):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(results[is_pareto] < c, axis=1)
            is_pareto[i] = True
    pareto = [population[i] for i, flag in enumerate(is_pareto) if flag]
    return pareto

def dominates(ind1, ind2):
    fitness_1, complexity_1 = ind1['fitness'], ind1['complexity']
    fitness_2, complexity_2 = ind2['fitness'], ind2['complexity']
    return (fitness_1 <= fitness_2 and complexity_1 <= complexity_2) and (fitness_1 < fitness_2 or complexity_1 < complexity_2)

def generate_new_generation_NSGA_2(n, population, tournament_selection=False):
    # population is a list
    dominated_counts = [0] * len(population)
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
            pareto_fronts[pareto_index].sort(key=lambda x: x['fitness'])
            for i in range(n - len(next_generation)):
                next_generation.append(pareto_fronts[pareto_index][i])
        else:
            selected_ind = random.choice(pareto_fronts[pareto_index])
            pareto_fronts[pareto_index].remove(selected_ind)
            next_generation.append(selected_ind)
    return next_generation

# ---------- Tournament Selection, Mating and Mutation ----------
def tournament_selection(parent_generation, n_selected=2):
    tournament = random.sample(parent_generation, config.TORNEMENT_SIZE)
    if config.TORN_SELECTION_METHOD == 'pareto':
        selected = generate_new_generation_NSGA_2(n_selected, tournament, tournament_selection=True)
    else:
        tournament.sort(key=lambda x: x['fitness'])
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
    results = [parent1, parent2]
    for ind in [offspring1, offspring2]:
        fitness, complexity = evaluate_individual(ind)
        offspring = {
            'complexity': complexity,
            'fitness': fitness,
            'individual': ind,
            'is_simplified': False
        }
        results.append(offspring)
    return results

def generate_new_population(population):
    # 'population' is a list
    new_gen_parents = generate_new_generation_NSGA_2(config.POPULATION_RETENTION_SIZE, population)
    new_population_dict = {}
    max_attempts = config.POPULATION_SIZE * 10  # Prevent infinite loop
    attempts = 0
    while len(new_population_dict) < config.POPULATION_SIZE and attempts < max_attempts:
        attempts += 1
        parent1, parent2 = tournament_selection(new_gen_parents)
        mate_mutation_results = mate_and_mutate(parent1, parent2)
        if config.MATE_MUTATE_SELECTION_METHOD == 'pareto':
            selected = return_pareto_front(mate_mutation_results)
        elif config.MATE_MUTATE_SELECTION_METHOD == 'fitness':
            mate_mutation_results.sort(key=lambda x: x['fitness'])
            selected = mate_mutation_results[:2]
        else:
            selected = mate_mutation_results
        for individual in selected:
            if (individual['complexity'] > config.COMPLEXITY_MAX_THRESHOLD or
                individual['complexity'] < config.COMPLEXITY_MIN_THRESHOLD or
                individual['fitness'] > config.FIT_THRESHOLD):
                continue
            key = convert_individual_to_key(individual['individual'])
            if key not in new_population_dict or individual['fitness'] < new_population_dict[key]['fitness']:
                new_population_dict[key] = individual
    if len(new_population_dict) == 0:
        return population
    return list(new_population_dict.values())

# ---------- File I/O and Post-processing ----------
def write_population_to_file(population, DV, STD):
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
        for individual in population:
            f.write(str(individual['individual']))
            f.write('\n')

def read_population_from_file(filename):
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
    return list(population.values())

def unstandardise_and_simplify_population(population):
    # population is a list
    injected_population = []
    for individual in population:
        new_expression_str = f"add(mul({individual['individual']}, {config.std_y}), {config.mean_y})"
        individual_tree = gp.PrimitiveTree.from_string(new_expression_str, pset)
        fitness, complexity = evaluate_individual(individual_tree)
        new_indiv = {
            'complexity': complexity,
            'fitness': fitness,
            'individual': individual_tree,
            'is_simplified': False
        }
        injected_population.append(new_indiv)
    simplified_injected_pop = simplify_population({convert_individual_to_key(ind['individual']): ind for ind in injected_population})
    return list(simplified_injected_pop.values()) if isinstance(simplified_injected_pop, dict) else list(simplified_injected_pop)

def extend_population_with_saved_expressions(filenames, population):
    # population is a list; we build a dict and merge
    pop_dict = {convert_individual_to_key(ind['individual']): ind for ind in population}
    for filename in filenames:
        saved = read_population_from_file(filename)
        for individual in saved:
            key = convert_individual_to_key(individual['individual'])
            if key not in pop_dict or individual['fitness'] < pop_dict[key]['fitness']:
                pop_dict[key] = individual
    return list(pop_dict.values())

def get_pareto_scores(population):
    pareto_front = return_pareto_front(population)
    pareto_front = sorted(pareto_front, key=lambda x: x['fitness'], reverse=True)
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
    global toolbox
    from deap import base, creator, gp, tools
    from functools import partial
    import operator, random, math
    from . import config

    if config.DATASET == 'HEATSINK':
        num_inputs = 2
    elif config.DATASET == 'CORROSION':
        num_inputs = 5
    elif config.DATASET == 'BENCHMARK':
        num_inputs = 3
    else:
        raise ValueError("Invalid dataset in config.DATASET")

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
    
    config.PSET = pset
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    config.TOOLBOX = toolbox
    toolbox = config.TOOLBOX

if __name__ == "__main__":
    st.write('Should not be run as main - EngineDict.py just contains utility functions')
