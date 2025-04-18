"""
EngineDict.py
--------------
This module implements the evolutionary operations for symbolic regression.
It is designed for SR Beta and works for HEATSINK, CORROSION, or BENCHMARK datasets.
Population is maintained as a dictionary for uniqueness.
"""

from . import config
import random
import numpy as np
import math
import operator
from . import Simplification as simp
import pandas as pd

from deap import gp, base, creator, tools
from functools import partial
from datetime import datetime

import sympy as sp
from sympy import log as sympy_log, exp as sympy_exp

import streamlit as st  # For Streamlit logging/display if needed

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
    raise CustomOperationException('config.DATASET must be one of "BENCHMARK", "HEATSINK", or "CORROSION"')

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

# ---------- Define Types (with safe re-creation) ----------
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
    # Terminal checks
    if config.DATASET == 'HEATSINK':
        if not (('G1' in str(individual)) and ('G2' in str(individual))):
            return config.FIT_THRESHOLD + 1, complexity
    elif config.DATASET == 'CORROSION':
        individual_str = str(individual)
        # Require at least one of the expected terminals
        if not any(term in individual_str for term in ["pH", "T", "LogP", "LogV", "LogD"]):
            return config.FIT_THRESHOLD + 1, complexity
    try:
        if config.DATASET == 'HEATSINK':
            y_pred = [func(x[0], x[1]) for x in config.X]
        elif config.DATASET == 'CORROSION':
            y_pred = [func(x[0], x[1], x[2], x[3], x[4]) for x in config.X]
        elif config.DATASET == 'BENCHMARK':
            y_pred = [func(x[0], x[1], x[2]) for x in config.X]
        else:
            raise CustomOperationException('config.DATASET not set correctly')
        y_pred = np.array(y_pred).reshape(-1,)
        if config.USE_RMSE:
            fitness = np.sqrt(np.mean((config.y - y_pred) ** 2))
        else:
            fitness = 1 - r2_score(config.y, y_pred)
    except Exception as e:
        if config.DISPLAY_ERROR_MESSAGES:
            st.write(e)
        fitness = config.FIT_THRESHOLD + 1
    return fitness, complexity

def evaluate_population(population_dict):
    total_fitness = 0
    total_complexity = 0
    best_fitness = config.FIT_THRESHOLD
    for individual in population_dict.values():
        total_fitness += individual["fitness"]
        total_complexity += individual["complexity"]
        best_fitness = min(best_fitness, individual["fitness"])
    num = len(population_dict)
    return total_fitness / num, total_complexity / num, best_fitness

# ---------- Utility Functions ----------
def display_progress(population_dict, last_printed_index):
    if len(population_dict) == last_printed_index:
        return last_printed_index
    if config.VERBOSE and (len(population_dict) % config.DISPLAY_INTERVAL == 0) and len(population_dict) > 0:
        avg_fit, avg_comp, best_fit = evaluate_population(population_dict)
        st.write(f'Len: {len(population_dict)}, Avg Fit: {avg_fit:.4f}, Avg Comp: {avg_comp:.4f}, Best Fit: {best_fit:.6f}')
        return len(population_dict)
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
    if config.VERBOSE:
        st.write("\n-------------- SIMPLIFICATION --------------")
    simplified_population = {}
    for key, individual in population_dict.items():
        # If already simplified, just keep it
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
    init_population = {}
    while len(init_population) < config.POPULATION_SIZE:
        individual = toolbox.individual()
        fitness, complexity = evaluate_individual(individual)
        if complexity > config.COMPLEXITY_MAX_THRESHOLD or complexity < config.COMPLEXITY_MIN_THRESHOLD or fitness > config.FIT_THRESHOLD:
            continue
        key = convert_individual_to_key(individual)
        if key not in init_population or fitness < init_population[key]['fitness']:
            init_population[key] = {
                'complexity': complexity,
                'fitness': fitness,
                'individual': individual,
                'is_simplified': False,
            }
    if config.USE_SIMPLIFICATION:
         init_population = simplify_population(init_population)
    return init_population  # Return dictionary

# ---------- Pareto and Dominance Functions ----------
def return_pareto_front(population_dict):
    results = []
    for individual in population_dict.values():
        results.append((individual['fitness'], individual['complexity']))
    results = np.array(results)
    is_pareto = np.ones(results.shape[0], dtype=bool)
    for i, c in enumerate(results):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(results[is_pareto] < c, axis=1)
            is_pareto[i] = True
    pareto = [list(population_dict.values())[i] for i, flag in enumerate(is_pareto) if flag]
    return pareto

def dominates(ind1, ind2):
    f1, c1 = ind1['fitness'], ind1['complexity']
    f2, c2 = ind2['fitness'], ind2['complexity']
    return (f1 <= f2 and c1 <= c2) and (f1 < f2 or c1 < c2)

def generate_new_generation_NSGA_2(n, population_dict, tournament_selection=False):
    # Convert dictionary to list for internal processing
    population_list = list(population_dict.values())
    dominated_counts = [0] * len(population_list)
    for i, ind1 in enumerate(population_list):
        for j, ind2 in enumerate(population_list[i:]):
            if dominates(ind1, ind2):
                dominated_counts[i+j] += 1
            elif dominates(ind2, ind1):
                dominated_counts[i] += 1
    pareto_fronts = [[] for _ in range(max(dominated_counts)+1)]
    for i, ind in enumerate(population_list):
        pareto_fronts[dominated_counts[i]].append(ind)
    pareto_index = 0
    next_generation = {}
    while len(next_generation) < n:
        if len(next_generation) + len(pareto_fronts[pareto_index]) <= n:
            for ind in pareto_fronts[pareto_index]:
                key = convert_individual_to_key(ind['individual'])
                next_generation[key] = ind
            pareto_index += 1
        elif tournament_selection:
            pareto_fronts[pareto_index].sort(key=lambda x: x['fitness'])
            for i in range(n - len(next_generation)):
                key = convert_individual_to_key(pareto_fronts[pareto_index][i]['individual'])
                next_generation[key] = pareto_fronts[pareto_index][i]
        else:
            selected_ind = random.choice(pareto_fronts[pareto_index])
            pareto_fronts[pareto_index].remove(selected_ind)
            key = convert_individual_to_key(selected_ind['individual'])
            next_generation[key] = selected_ind
    return next_generation

# ---------- Tournament Selection, Mating and Mutation ----------
def tournament_selection(parent_generation_dict, n_selected=2):
    parent_list = list(parent_generation_dict.values())
    tournament = random.sample(parent_list, config.TORNEMENT_SIZE)
    if config.TORN_SELECTION_METHOD == 'pareto':
        selected = generate_new_generation_NSGA_2(n_selected, {convert_individual_to_key(ind['individual']): ind for ind in tournament}, tournament_selection=True)
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
        new_offspring = {
            'complexity': complexity,
            'fitness': fitness,
            'individual': ind,
            'is_simplified': False
        }
        results.append(new_offspring)
    return results

def generate_new_population(population_dict):
    new_gen_parents = generate_new_generation_NSGA_2(config.POPULATION_RETENTION_SIZE, population_dict)
    new_population = {}
    while len(new_population) < config.POPULATION_SIZE:
        parent1, parent2 = tournament_selection(new_gen_parents)
        mate_mutation_results = mate_and_mutate(parent1, parent2)
        if config.MATE_MUTATE_SELECTION_METHOD == 'pareto':
            selected = return_pareto_front({convert_individual_to_key(ind['individual']): ind for ind in mate_mutation_results})
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
            if key not in new_population or individual['fitness'] < new_population[key]['fitness']:
                new_population[key] = individual
    if len(new_population) == 0:
        return population_dict
    return new_population

# ---------- File I/O and Post-processing ----------
def write_population_to_file(population_dict, DV, STD):
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
        for individual in population_dict.values():
            f.write(str(individual['individual']) + "\n")

def read_population_from_file(filename):
    with open(filename, "r") as file:
        content = file.read()
    population = {}
    for expression in content.split('\n'):
        if expression == '':
            continue
        # Adjust negative representation if needed
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

def unstandardise_and_simplify_population(population_dict):
    injected_population = {}
    for individual in population_dict.values():
        new_expression_str = f"add(mul({individual['individual']}, {config.std_y}), {config.mean_y})"
        ind_tree = gp.PrimitiveTree.from_string(new_expression_str, pset)
        fitness, complexity = evaluate_individual(ind_tree)
        key = convert_individual_to_key(ind_tree)
        if key not in injected_population or fitness < injected_population[key]['fitness']:
            injected_population[key] = {
                'complexity': complexity,
                'fitness': fitness,
                'individual': ind_tree,
                'is_simplified': False,
            }
    if config.USE_SIMPLIFICATION:
        injected_population = simplify_population(injected_population)
    return injected_population

def extend_population_with_saved_expressions(filenames, population_dict):
    for filename in filenames:
        saved = read_population_from_file(filename)
        for individual in saved.values():
            key = convert_individual_to_key(individual['individual'])
            if key not in population_dict or individual['fitness'] < population_dict[key]['fitness']:
                population_dict[key] = individual
    return population_dict

def get_pareto_scores(population_dict):
    pareto_front = return_pareto_front(population_dict)
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
    import pandas as pd
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
