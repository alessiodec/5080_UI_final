# functions/test_files/EngineDict.py
from . import config
import random
import numpy as np
import warnings
import math
import operator
import pandas as pd
import time
from datetime import datetime

import sympy as sp
from deap import gp, base, creator, tools
from functools import partial
from . import simplification as simp

# --- Custom Exception ---
class CustomOperationException(Exception):
    pass

# --- Protected Operations ---
def protectedDiv(left, right):
    try:
        return left / right
    except:
        raise CustomOperationException('protectedDiv Error')

def protectedLog(value):
    try:
        return math.log(value)
    except:
        raise CustomOperationException('protectedLog Error')

def protectedExp(value):
    try:
        return math.exp(value)
    except:
        raise CustomOperationException('protectedExp Error')

def random_constant():
    return round(random.uniform(-1, 1), 4)

# --- Setup Primitive Set based on config.DATASET ---
if config.DATASET == 'HEATSINK':
    num_inputs = 2
elif config.DATASET == 'CORROSION':
    num_inputs = 5
elif config.DATASET == 'BENCHMARK':
    num_inputs = 3
else:
    raise CustomOperationException('config.DATASET must be "BENCHMARK", "HEATSINK", or "CORROSION"')

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

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

config.TOOLBOX = toolbox
config.PSET = pset

def evaluate_individual(individual):
    func = gp.compile(expr=individual, pset=pset)
    complexity = len(individual)
    try:
        if config.DATASET == 'HEATSINK':
            y_pred = [func(x[0], x[1]) for x in config.X]
        elif config.DATASET == 'CORROSION':
            y_pred = [func(x[0], x[1], x[2], x[3], x[4]) for x in config.X]
        elif config.DATASET == 'BENCHMARK':
            y_pred = [func(x[0], x[1], x[2]) for x in config.X]
        else:
            raise CustomOperationException("Invalid DATASET in config")
        y_pred = np.array(y_pred).reshape(-1,)
        from sklearn.metrics import r2_score
        fitness = 1 - r2_score(config.y, y_pred)
    except Exception as e:
        if config.DISPLAY_ERROR_MESSAGES:
            print(e)
        fitness = config.FIT_THRESHOLD + 1
    return fitness, complexity

def evaluate_population(population):
    total_fitness = 0
    total_complexity = 0
    min_fitness = config.FIT_THRESHOLD
    for individual in population:
        fitness = individual["fitness"]
        complexity = individual["complexity"]
        total_fitness += fitness
        total_complexity += complexity
        if fitness < min_fitness:
            min_fitness = fitness
    return total_fitness / len(population), total_complexity / len(population), min_fitness

def display_progress(population, last_printed_index):
    if len(population) == last_printed_index:
        return last_printed_index
    if config.VERBOSE and (len(population) % config.DISPLAY_INTERVAL == 0) and (len(population) > 0):
        avg_fit, avg_comp, best_fit = evaluate_population(population)
        print(f'Len:{len(population)}, Avg Fit:{avg_fit:.4f}, Avg Comp:{avg_comp:.4f}, Best Fit:{best_fit:.4f}')
        return len(population)

def convert_individual_to_key(individual):
    format_str = ''
    for node in individual:
        try:
            float(node.name)
            format_str += '_COEFF_'
        except:
            if node.name == 'randConst':
                format_str += '_COEFF_'
            else:
                format_str += node.name
    return format_str

# Use the simplification functions from the separate module as needed.
def simplify_population(population):
    simplified_population = {}
    index_tracker = 0
    for individual in population:
        index_tracker = display_progress(population=simplified_population, last_printed_index=index_tracker)
        if individual.get('is_simplified', False):
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
                print(f"failure -> individual: {individual['individual']}")
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
    return simplified_population

def initialize_population():
    init_population = {}
    index_tracker = 0
    while len(init_population) < config.POPULATION_SIZE:
        individual = toolbox.individual()
        fitness, complexity = evaluate_individual(individual=individual)
        # Store individual as a dict
        if config.USE_SIMPLIFICATION:
            # Optionally simplify here
            pass
        if fitness < config.FIT_THRESHOLD and complexity < config.COMPLEXITY_MAX_THRESHOLD:
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
                dominated_counts[j+i] += 1
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
        else:
            selected_ind = random.choice(pareto_fronts[pareto_index])
            pareto_fronts[pareto_index].remove(selected_ind)                           
            next_generation.append(selected_ind)
    return next_generation

def tournament_selection(parent_generation, n_selected=2):
    selected = []
    tournament = random.sample(parent_generation, config.TORNEMENT_SIZE)
    if config.SELECTION_METHOD == 'pareto':
        selected = generate_new_generation_NSGA_2(n_selected, tournament, tournament_selection=True)
    else:
        tournament.sort(key=lambda x: x['fitness'])
        selected = tournament[:2]
    return selected

def mate_and_mutate(parent1, parent2, cxpb=0.95, mutpb=0.5):
    offspring1, offspring2 = toolbox.clone(parent1['individual']), toolbox.clone(parent2['individual'])
    try:
        if random.random() < cxpb:
            toolbox.mate(offspring1, offspring2)
    except Exception as e: 
        if config.DISPLAY_ERROR_MESSAGES:
            print(f"Failed to mate: {e}")
    try:
        if random.random() < mutpb:
            toolbox.mutate(offspring1)
        if random.random() < mutpb:
            toolbox.mutate(offspring2)
    except Exception as e: 
        if config.DISPLAY_ERROR_MESSAGES:
            print(f"Failed to mutate: {e}")
    custom_parent_arr = [parent1, parent2]
    for individual in [offspring1, offspring2]:
        fitness, complexity = evaluate_individual(individual)
        custom_parent_arr.append({
            'complexity': complexity,
            'fitness': fitness,
            'individual': individual,
            'is_simplified': False
        })
    return custom_parent_arr

def generate_new_population(population, verbose=1):
    new_gen_parents = generate_new_generation_NSGA_2(config.POPULATION_RETENTION_SIZE, population)
    new_population = {}
    new_population_strings = []
    while len(new_population) < config.POPULATION_SIZE:
        parents = tournament_selection(new_gen_parents)
        mate_mutation_results = mate_and_mutate(parent1=parents[0], parent2=parents[1])
        if config.SELECTION_METHOD == 'pareto':
            selected_mate_mutation_results = return_pareto_front({i: ind for i, ind in enumerate(mate_mutation_results)})
        elif config.SELECTION_METHOD == 'fitness':
            mate_mutation_results.sort(key=lambda x: x['fitness'])
            selected_mate_mutation_results = mate_mutation_results[:2]
        else:
            selected_mate_mutation_results = mate_mutation_results
        for individual in selected_mate_mutation_results:
            if individual['fitness'] < config.FIT_THRESHOLD and individual['complexity'] < config.COMPLEXITY_MAX_THRESHOLD and (str(individual['individual']) not in new_population_strings):
                new_population[convert_individual_to_key(individual['individual'])] = individual
                new_population_strings.append(str(individual['individual']))
                if verbose and len(new_population) % 100 == 1:
                    avg_fit, avg_comp, best_fit = evaluate_population(new_population)
                    print(f'Len:{len(new_population)}, Avg Fit:{avg_fit:.4f}, Avg Comp:{avg_comp:.4f}, Best Fit:{best_fit:.4f}')
    return new_population

if __name__ == "__main__":
    print('EngineDict.py is a module and should not be run as main.')
