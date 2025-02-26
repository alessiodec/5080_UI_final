# Functions for implementing the SR - 14.2.25
# Currently only valid for the Thermal Resistance of the Heatsink dataset - needs to be made dynamic
# For now theese are just population functions - struggling with a few functions because I can't accsess the Global FITNESS_THRESHOLD

# -------- Functions missing: ----------- 
# evaluate_individual() -> requires FITNESS_THRESHOLD (changed - takes settings in....)
# simplify_deap_individual() -> requires evaluate_individual (also changed)
# simplify_and_clean_population() -> requires FITNESS_THRESHOLD

# Create a Datatype for individuals to have (Dont need to keep recalculating stuff)
from . import config
import random
import numpy as np
import warnings
import math
import operator

from deap import gp
from sklearn.metrics import root_mean_squared_error, r2_score
from deap import base, creator, gp, tools, algorithms
from functools import partial
from datetime import datetime

import sympy as sp
from sympy import Add, Mul, Pow, Number, Symbol
from sympy import log as sympy_log, exp as sympy_exp

class CustomOperationException(Exception):
    pass

# The Try, Except block in the evaluate_expr() func has made the custom operations redundent but they are useful boiler plate code
def protectedDiv(left, right):
    try:
        return left / right
    except:
        raise CustomOperationException('This is a Custom Error - Can implement stuff here')

def protectedLog(value):
    try:
        return math.log(value)
    except:
        raise CustomOperationException('This is a Custom Error - Can implement stuff here')
        
def protectedExp(value):
    try:
        return math.exp(value)
    except:
        raise CustomOperationException('This is a Custom Error - Can implement stuff here')
        
def random_constant():
    return random.uniform(-1, 1)

# ---------- Define the set of operations --------------
pset = gp.PrimitiveSet("MAIN", arity=2)  # Two inputs: Geometric1 and Geometric2
pset.addPrimitive(operator.add, 2)     
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)      
pset.addPrimitive(protectedDiv, 2) 
pset.addPrimitive(operator.pow, 2)
pset.addPrimitive(operator.neg, 1)     # * -1

pset.addPrimitive(protectedExp, 1)    
pset.addPrimitive(protectedLog, 1)     # I think the 1 means it only takes one arg

pset.addEphemeralConstant("randConst", partial(random_constant))  # Random constants between -n and n
pset.renameArguments(ARG0='G1')
pset.renameArguments(ARG1='G2')

# Define types: Minimize fitness, individuals are trees
# I am getting warning everytime I re-run because theese are already defined - does not break programme
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize fitness
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Create a toolbox and register necessary functions
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)  # Tree generation
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", gp.cxOnePoint)  #gp.cxOne.. works rather than tools.
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# Allows it to be used in the other code
config.TOOLBOX = toolbox
config.PSET = pset

# Define the indicudal class that everything is wrapped in
class CustomIndividual():
    def __init__(self, individual, fitness, complexity):
        self.individual = individual
        self.fitness = fitness
        self.complexity = complexity
    
    #Function to pass in an X and then get the preditions back
    # No error handelling
    def predict(self, X):
        func = gp.compile(expr=self.individual, pset=pset)
        y_pred = [func(x[0], x[1]) for x in X] #This is not dynamic - could try somthing like *x
        return np.array(y_pred).reshape(-1,)
        
# --------- Function to evaluate an individual ---------
# Idea - could just have this attached to the class - then call it when nessisary?
def evaluate_individual(individual):
    # Im not sure about this config implementation - ypu have to update config.X before calling it
    # Compile the individual into a callable function
    func = gp.compile(expr=individual, pset=pset)
    complexity = len(individual)
    
    # Evaluate the function and Catch ZeroDivision Errors Etc, 
    try:
        # Ensure that both Varibles are in the expression:
        if not (('G1' in str(individual)) and ('G2' in str(individual))):
            fitness = config.FIT_THRESHOLD + 1
            return fitness, complexity
            
        y_pred = [func(x[0], x[1]) for x in config.X] #This is not dynamic - could try somthing like *x
        y_pred = np.array(y_pred).reshape(-1,)
    
        # This could be changed part way through
        if config.USE_RMSE:
            fitness = root_mean_squared_error(config.y, y_pred)
        else:
            fitness = 1 - r2_score(config.y, y_pred)
            
    except Exception as e:
        # print(e)
        fitness = config.FIT_THRESHOLD + 1

    return fitness, complexity


#  ------------ CODE TO SIMPLIFY EXPRESSIONS ------------
# 1.Writing a custom function to convert between the datatypes: 
symbol_map = {
    "protectedDiv": "/",
    "add": "+",
    "sub": "-",
    "neg": "*-1",
    "protectedExp": "exp",
    "protectedLog": "log",
    "mul": "*",
    "ARG0": "x0",
    "ARG1": "x1",    
}

def sympy_simplify_expression(individual):
    ''' 
    Function takes an toolbox.individual (or CustomIndividual.individual) and returns a sympy.simplify(individual)
    '''
    
    stack = []
    for i in range(len(individual)):
        node = individual[-(i+1)]
             
        if isinstance(node, gp.Terminal):
            if node.name == 'randConst':
                stack.append(str(node.value))
            
            else:
                #Catch is the name is a number 
                try:
                    number = float(node.name)
                    stack.append(node.name)
                    
                except ValueError:
                    stack.append(symbol_map[node.name])
                
        elif isinstance(node, gp.Primitive):
            if node.name == "neg":
                stack.append(f'-1*({stack.pop()})')
                
            elif node.name == "protectedExp":
                stack.append(f'exp({stack.pop()})')
    
            elif node.name == "protectedLog":
                stack.append(f'log({stack.pop()})')

            elif node.name == "pow":
                left = stack.pop()
                right = stack.pop()
                stack.append(f'pow({left}, {right})')
                
            else:
                left = stack.pop()
                right = stack.pop()
                stack.append(f'({left}){symbol_map[node.name]}({right})')
                

    try:
        sympy_expr = sp.sympify(stack[-1])            
        simplified_expr = sp.simplify(sympy_expr)
        return simplified_expr
        
    except sp.SympifyError as e:
        raise ValueError(f"Error converting individual to sympy expression: {e}")



# 2. Recursive function that converts back from a Sympy expression to a DEAP one 
def sympy_to_deap(expr):
    """
    Converts a Sympy expression `expr` into a DEAP-style string:
      - log(x) -> protectedLog(x)
      - exp(x) -> protectedExp(x)
      - x + y   -> add(x,y)
      - x - y   -> sub(x,y)
      - x * y   -> mul(x,y)
      - x / y   -> protectedDiv(x,y)  [including multi-factor merges]
      - -x      -> neg(x)
      
    This version merges multiple inverse factors (Pow(...,-1)) into a single denominator,
    e.g. x0*(x1^-1)*((x0+x1)^-1) => protectedDiv(x0, mul(x1, (x0+x1))).
    """

    # 1) Base cases
    # ------------------------------------------------
    
    # A. Symbol (e.g., x0, x1)
    if expr.is_Symbol:
        return str(expr)
    
    # B. Number (e.g., 2, 3.14)
    if expr.is_Number:
        return str(expr.evalf()) if expr.is_Float else str(expr)

    # 2) Detect unary negation if top-level is Mul(-1, x)
    # ---------------------------------------------------
    # e.g. -x => neg(x)
    if expr.is_Mul:
        # Check if this is exactly (-1) * something
        args = expr.args
        if len(args) == 2 and args[0] == -1:
            # That means expr = -1 * x => neg(x)
            return f"neg({sympy_to_deap(args[1])})"
        # Otherwise, handle a general multiplication below.

    # 3) Known functions (log, exp)
    # ------------------------------------------------
    if expr.is_Function:
        func = expr.func
        args = expr.args
        
        # log(x) => protectedLog(x)
        if func == sympy_log:
            return f"protectedLog({sympy_to_deap(args[0])})"
        
        # exp(x) => protectedExp(x)
        if func == sympy_exp:
            return f"protectedExp({sympy_to_deap(args[0])})"
        
        # If you have other functions (sin, cos, etc.), add them similarly or raise:
        raise ValueError(f"Unsupported Sympy function: {func.__name__}")

    # 4) Handle + / -
    # ------------------------------------------------
    # Sympy represents x - y as Add(x, -y). 
    if expr.is_Add:
        args = list(expr.args)
        if len(args) == 2:
            left, right = args
            # If right = -1*y => sub(x,y)
            if right.is_Mul and len(right.args) == 2 and right.args[0] == -1:
                return f"sub({sympy_to_deap(left)},{sympy_to_deap(right.args[1])})"
            else:
                # x + y => add(x,y)
                return f"add({sympy_to_deap(left)},{sympy_to_deap(right)})"
        else:
            # More than 2 => fold them left-to-right
            from functools import reduce
            subexprs = [sympy_to_deap(a) for a in args]
            return reduce(lambda a, b: f"add({a},{b})", subexprs)

    # 5) Handle * / (including merges)
    # ------------------------------------------------
    if expr.is_Mul:
        # Gather factors
        args = expr.args
        
        # Count negative-exponent factors: x^( -1 ), x^( -2 ), etc.
        # We'll treat each x^-n as going to the denominator.
        pos_factors = []
        neg_power_factors = []
        
        for factor in args:
            # is it "f^( -1 )" or "f^(some negative integer)"?
            if factor.is_Pow and factor.args[1].is_Integer and factor.args[1] < 0:
                # e.g. factor = x1^-1 or something^(-2)
                exponent = factor.args[1]
                base = factor.args[0]
                # We'll store the base repeated 'abs(exponent)' times in negative-power list
                # Typically for exponent = -1 or -2, etc.
                neg_power_factors.extend([base]*(-exponent))
            else:
                # positive factor
                pos_factors.append(factor)

        # CASE A: if we have at least one negative-power factor, we unify them into a single denominator
        if neg_power_factors:
            # numerator = mul(...) of all pos_factors (if > 1)
            # denominator = mul(...) of all neg_power_factors
            # Then combine as protectedDiv(numerator, denominator)
            
            from functools import reduce
            
            # If no "pos_factors", that means the top-level expression was purely invert(something).
            # That effectively means 1 / (something). So let's put a "1" as the numerator.
            if not pos_factors:
                pos_factors = [sp.Integer(1)]
            
            # Build the numerator (fold if multiple)
            if len(pos_factors) == 1:
                numerator_str = sympy_to_deap(pos_factors[0])
            else:
                # fold them into mul(...)
                subexprs_num = [sympy_to_deap(p) for p in pos_factors]
                numerator_str = reduce(lambda a,b: f"mul({a},{b})", subexprs_num)
            
            # Build the denominator
            if len(neg_power_factors) == 1:
                denominator_str = sympy_to_deap(neg_power_factors[0])
            else:
                subexprs_den = [sympy_to_deap(nf) for nf in neg_power_factors]
                denominator_str = reduce(lambda a,b: f"mul({a},{b})", subexprs_den)
            
            return f"protectedDiv({numerator_str},{denominator_str})"
        
        # CASE B: no negative-power factors => normal multiplication of all pos_factors
        if len(pos_factors) == 2:
            # just x*y => mul(x,y)
            return f"mul({sympy_to_deap(pos_factors[0])},{sympy_to_deap(pos_factors[1])})"
        else:
            # more than 2 => fold them left-to-right
            from functools import reduce
            subexprs = [sympy_to_deap(a) for a in pos_factors]
            return reduce(lambda a, b: f"mul({a},{b})", subexprs)

    # 6) Handle exponentiation x^y
    # ------------------------------------------------
    if expr.is_Pow:
        base, exponent = expr.args
        # Typically x^-1 or x^-2 might be handled in the Mul logic above,
        # but if you see a normal exponent here, let's do "pow(x,y)"
        return f"pow({sympy_to_deap(base)},{sympy_to_deap(exponent)})"

    # 7) Anything unhandled => error
    # ------------------------------------------------
    raise ValueError(f"Unsupported or unexpected expression: {expr}")

# 3. code to combine both together, takes a CustomIndividual Class and will simplify its indivudual
# Default values for fitness threshold and use rmse want changing
def simplify_deap_individual(custom_individual):
    try:
        # first simply a stringified version of the original epxression
        sympy_simplifyed_expr = sympy_simplify_expression(custom_individual.individual)
    
        # convert it back to DEAP format
        deap_simplified_version = sympy_to_deap(sympy_simplifyed_expr)
    
        # make the string into a new indivudial
        new_individual = gp.PrimitiveTree.from_string(deap_simplified_version.replace('x0', 'G1').replace('x1', 'G2'), pset)
    
        # Eval fitness again - pass args in from the wrapper function (except individual=new_individual)
        fitness, complexity = evaluate_individual(individual=new_individual)
        
        # make new custom_individual
        new_custom_individual = CustomIndividual(new_individual, fitness, complexity)
    
        # return either that or the original
        if new_custom_individual.complexity < custom_individual.complexity:
            return new_custom_individual
        else:
            return custom_individual
            
    except Exception as e:
        # print(e)
        return custom_individual


def simplify_and_clean_population(population):
    simplified_population = []
    
    for individual in population:
        simplified_indiv = simplify_deap_individual(individual)
        if simplified_indiv.fitness < config.FIT_THRESHOLD and simplified_indiv.complexity < config.COMPLEXITY_THRESHOLD:
            simplified_population.append(simplify_deap_individual(individual))
    
    return simplified_population


# Evaluation metrics for a population
def evaluate_population(population):
    total_fitness = 0
    total_complexity = 0
    optimal_fitness = population[0].fitness

    for individual in population:
        total_fitness += individual.fitness
        total_complexity += individual.complexity
        if individual.fitness < optimal_fitness:
            optimal_fitness = individual.fitness
        
    return [total_fitness/len(population), total_complexity/len(population), optimal_fitness] 


# Initialise the population: Default args includede but not nessisary
def initialize_population(verbose=1):
    init_population = []
    init_population_strings = []
    
    while len(init_population) < config.POPULATION_SIZE:
        individual = toolbox.individual()
        fitness, complexity = evaluate_individual(individual=individual)    
        custom_individual = CustomIndividual(individual, fitness, complexity)

        if config.USE_SIMPLIFICATION:
            custom_individual = simplify_deap_individual(custom_individual)
            
        if custom_individual.fitness < config.FIT_THRESHOLD and custom_individual.complexity < config.COMPLEXITY_THRESHOLD:
            if str(custom_individual.individual) not in init_population_strings:
                init_population.append(custom_individual)
                init_population_strings.append(str(custom_individual.individual))

                if verbose and len(init_population) % 100 == 1:
                    avg_fit, avg_comp, best_fit = evaluate_population(init_population)
                    print(f'Len:{len(init_population)}, Avg Fit:{avg_fit:.4f}, Avg Comp:{avg_comp:.4f}, Best Fit:{best_fit:.4f}')
    
    return init_population

# ------------ PARETO, DOMINANCE AND PARENT SELECTION ------------- 

# Get the pareto front of the population: Adapted from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def return_pareto_front(population):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    
    results = []
    for individual in population:
        results.append((individual.fitness, individual.complexity))
    results = np.array(results)
    
    is_pareto = np.ones(results.shape[0], dtype = bool) #Array of 1's of length population
    for i, c in enumerate(results):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(results[is_pareto]<c, axis=1)  # Keep any point with a lower cost
            is_pareto[i] = True  # And keep self
    return np.array(population)[is_pareto]


# Pareto dominance comparison for 2 class type individuals
def dominates(ind1, ind2):
    """Returns True if ind1 dominates ind2."""
    # ind1 dominates if it is better in at least one objective and not worse in any
    fitness_1, complexity_1 = ind1.fitness, ind1.complexity
    fitness_2, complexity_2 = ind2.fitness, ind2.complexity
    return (fitness_1 <= fitness_2 and complexity_1 <= complexity_2) and (fitness_1 < fitness_2 or complexity_1 < complexity_2)

def generate_new_generation_NSGA_2(n, population):
    # Initialise array of 0's for each member of population
    dominated_counts = [0] * len(population)

    # for each member add 1 for every other member that dominates it
    # end with array of the number of other individuals that dominate each point
    for i, ind1 in enumerate(population):
        for j, ind2 in enumerate(population[i:]):
            if dominates(ind1, ind2):
                dominated_counts[j+i] += 1
            elif dominates(ind2, ind1):
                dominated_counts[i] += 1
                
    # Initialise array of []'s for each dominated count - if the higest dominated count is 3 we have 3 pareto fronts
    pareto_fronts = [[] for _ in range(max(dominated_counts) + 1)]

    # for whatever the dominated count is add the indivudal to that array
    for i, ind in enumerate(population):
        pareto_fronts[dominated_counts[i]].append(ind)
    
    pareto_index = 0
    next_generation = []

    # essentially a while loop adding items to next generation from the lowest pareto fronts first
    while len(next_generation) < n:
        
        if len(next_generation) + len(pareto_fronts[pareto_index]) <= n:
            next_generation.extend(pareto_fronts[pareto_index])
            pareto_index += 1
            
        else:
            selected_ind = random.choice(pareto_fronts[pareto_index])
            pareto_fronts[pareto_index].remove(selected_ind)                           
            next_generation.append(selected_ind)

    return next_generation


#  ------------- TORNEMENT SELECTION + MATE AND MUTATE --------
def tournament_selection(parent_generation, n_selected=2):
    """
    Perform tournament selection on a set of parents.

    Args:
        parents (list): List of parent individuals (each with fitness values).
        tournsize (int): Number of individuals in each tournament.
        n_selected (int): Number of individuals to select.
        selection_method (str): pareto = pareto optimal of the selected parents, otherwise its the lowest two fitnessess

    Returns:
        list: Selected individuals from the parent pool.

    Uses pareto dominance - if one is pareto optimal, it will be chosen, 
    if they are on the same pareto front, randomly choose 
    """
    

    selected = []
    tournament = random.sample(parent_generation, config.TORNEMENT_SIZE)

    # for i, individual in enumerate(tournament):
    #     print(f"Equation {i}: fitness = {individual.fitness:.4f}, Complexity = {individual.complexity}, Individual:{individual.individual}")

    if config.SELECTION_METHOD == 'pareto':
        # Similar setup to NSGA-2 -> but only get the top n_selected poitns
        selected = generate_new_generation_NSGA_2(n_selected, tournament)
        
    else:
        
        #If we are not using pareto optimaility to win the tornement selection -> we are using the lowest two fitnessess
        # COME BACK TO THIS 
        tournament.sort(key=lambda x: x.fitness, reverse=False)
        selected = tournament[:2]

    return selected


# MATE AND MUTATE
def mate_and_mutate(parent1, parent2, cxpb=0.9, mutpb=0.3, newpb=0.02):
    """
    Mate two parents and apply mutation.
    
    Parameters:
        toolbox: The DEAP toolbox with registered mate and mutate operators.
        parent1: The first parent (individual).
        parent2: The second parent (individual).
        cxpb: Crossover probability.
        mutpb: Mutation probability.

    Returns:
        A list containing the two parents and their offspring after crossover and mutation.
    """
    # Clone parents to prevent overwriting original individuals - randomly add new ones in ...
    # if random.random() < newpb:
    #     offspring1 = toolbox.individual()
    #     offspring2 = toolbox.clone(parent2.individual)
    # else: 
    offspring1, offspring2 = toolbox.clone(parent1.individual), toolbox.clone(parent2.individual)
    
    # Apply crossover with probability cxpb
    try:
        if random.random() < cxpb:
            toolbox.mate(offspring1, offspring2)
            # print("MATE")
    except: 
        print("Failed to MATE")
    
    # Apply mutation to both offspring with probability mutpb
    try:
        if random.random() < mutpb:
            toolbox.mutate(offspring1)

        if random.random() < mutpb:
            toolbox.mutate(offspring2)

    except:
        print("Failed to MUTATE")
    
    custom_parent_arr = [parent1, parent2]
    
    for individual in [offspring1, offspring2]:
        fitness, complexity = evaluate_individual(individual=individual)
        
        custom_individual = CustomIndividual(individual, fitness, complexity)
        custom_parent_arr.extend([custom_individual])
    
    return custom_parent_arr

# GENERATING NEW POP:
def generate_new_population(population, verbose=1):
    '''
    selection method can be fitness or pareto
    '''
    new_gen_parents = generate_new_generation_NSGA_2(config.POPULATION_RETENTION_SIZE, population)
    new_population = []
    new_population_strings = []
    
    while len(new_population) < config.POPULATION_SIZE:
        parents = tournament_selection(parent_generation=new_gen_parents)
        mate_mutation_results = mate_and_mutate(parent1=parents[0], parent2=parents[1])    

        # Either select the pareto optimal MATE MUTATE points
        if config.SELECTION_METHOD == 'pareto':
            selected_mate_mutation_results = return_pareto_front(mate_mutation_results)

        # Or the ones with the two best fitness - Look at this again, I want to replace parents if the children are better
        elif config.SELECTION_METHOD == 'fitness':
            mate_mutation_results.sort(key=lambda x: x.fitness, reverse=False)
            selected_mate_mutation_results = mate_mutation_results[:2]

        # Otherwise just keep all of them
        else:
            selected_mate_mutation_results = mate_mutation_results

        # Construct the Pop
        for individual in selected_mate_mutation_results:
            if individual.fitness < config.FIT_THRESHOLD and individual.complexity < config.COMPLEXITY_THRESHOLD and (str(individual.individual) not in new_population_strings):
                new_population.append(individual) 
                new_population_strings.append(str(individual.individual))

                # Just for looking at the growing the data more
                if verbose and len(new_population) % 100 == 1:
                    avg_fit, avg_comp, best_fit = evaluate_population(new_population)
                    print(f'Len:{len(new_population)}, Avg Fit:{avg_fit:.4f}, Avg Comp:{avg_comp:.4f}, Best Fit:{best_fit:.4f}')
    
    return new_population


# ---------- Writing too, reading from, and unstandardising functions ----------
def write_population_to_file(population, DV, STD):
    ''' 
    This also involves generation of a filename:
    format:
    
    Prev_Generations_Log/DV/STD_SECOND:MIN:HOUR_DAY_MONTH.txt

    Prev_Generations_Log -> all records are kept here
    DV -> the dependant varibles are seporated: Pressure Drop=PD, Thermal Resistance=TR, Corrosion Rate=CR, Saturation Ratio=SR
    STD -> wehter or not the dependant varible has been standardised or not - 'STD' or 'NOTSTD'
    SECOND_MIN_HOUR_DAY_MONTH -> timestamp to prevent things being overwitten
    '''
    # Generate the filename
    now = datetime.now()
    timestamp = now.strftime("%H-%M-%S_%d_%m")
    
    Possible_DVs = ['CR', 'PD', 'TR', 'SR']
    if DV not in Possible_DVs:
        print(f'ERROR: {DV} not in {Possible_DVs} - please change')
        return

    Possible_STDs = ['STD', 'NOTSTD']
    if STD not in Possible_STDs:
        print(f'ERROR: {STD} not in {Possible_STDs} - please change')
        return
    
    filename = f"Prev_Generations_Log/{DV}/{STD}_{timestamp}.txt"
    
    # Write to file
    with open(filename, "a") as f:
        for i, individual in enumerate(population):
            f.write(str(individual.individual))
            f.write('\n')


def read_population_from_file(filename):
    file = open(filename, "r")
    content = file.read()
    file.close()
    
    population = []
    for expression in content.split('\n'):
        if expression == '':
            continue
            
        individual = gp.PrimitiveTree.from_string(expression, pset)
        fitness, complexity = evaluate_individual(individual)
        custom_individual = CustomIndividual(individual, fitness, complexity)
        population.append(custom_individual)
        
    return population


def unstandardise_and_simplify_population(population):
    injected_population = []
    
    for individual in population:
        new_expression_str = f"add(mul({individual.individual}, {config.std_y}), {config.mean_y})" #This is the format for if y has been standardised - can be ammended for G1 transformations etc
        individual = gp.PrimitiveTree.from_string(new_expression_str, pset)
        fitness, complexity = evaluate_individual(individual)
        custom_individual = CustomIndividual(individual, fitness, complexity)
        injected_population.append(custom_individual)

    simplified_injected_pop = simplify_and_clean_population(injected_population)
    return simplified_injected_pop


if __name__ == "__main__":
    print('Should not be run as main - Engine.py just contains utils')





