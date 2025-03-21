# functions/symbolic_regression_files/Simplification.py

#  ------------ CODE JUST TO SIMPLIFY EXPRESSIONS ------------
from . import config
from . import EngineDict as Engine

# Both for timer
import multiprocessing
import signal

import sympy as sp
from sympy import Add, Mul, Pow, Number, Symbol
from sympy import log as sympy_log, exp as sympy_exp
from deap import base, creator, gp, tools, algorithms

import streamlit as st  # Added for Streamlit output

# 1. Map to convert between the Strings:
symbol_map = {
    "protectedDiv": "/",
    "add": "+",
    "sub": "-",
    "protectedExp": "exp",
    "protectedLog": "log",
    "mul": "*",
    "ARG0": "x0",
    "ARG1": "x1",   
    "ARG2": "x2",
    "ARG3": "x3",
    "ARG4": "x4",
}

# 2. Function to take a toolbox.individual and return a sympy expression
def convert_expression_to_sympy(individual):
    ''' 
    Function takes a toolbox.individual (or population[entry].individual) and returns a sympy expression.
    '''
    stack = []
    for i in range(len(individual)):
        node = individual[-(i+1)]
        if isinstance(node, gp.Terminal):
            if node.name == 'randConst':
                stack.append(str(node.value))
            else:
                try:
                    float(node.name)
                    stack.append(node.name)
                except ValueError:
                    stack.append(symbol_map[node.name])
        elif isinstance(node, gp.Primitive):
            if node.name == "protectedExp":
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
        return sympy_expr
    except sp.SympifyError as e:
        raise ValueError(f"Error converting individual to sympy expression: {e}")

# Recursive functions to get complexity of sympy expressions
def count_atoms(expr):
    if expr.is_Atom:
        return 1
    return sum(count_atoms(arg) for arg in expr.args)

def calc_complexity_sympy(expr):
    terminal_counts = count_atoms(expr)
    complexity = sp.count_ops(expr) + terminal_counts
    return complexity

# There is not a sympy function that does this automatically
def simplify_power_bases(expr):
    expr = expr.replace(
        lambda e: isinstance(e, sp.Pow) and e.base.is_Number and e.base == 0,
        lambda e: sp.Integer(0)
    )
    expr = expr.replace(
        lambda e: isinstance(e, sp.Pow) and e.base.is_Number and e.base == 1,
        lambda e: sp.Integer(1)
    )
    return expr

def simplify_sympy_expression(expr):
    '''
    Simplify the given sympy expression using a single pass of several common methods.
    Instead of recursively simplifying until no improvement, we take one pass and select the candidate
    with the lowest complexity.
    '''
    original_complexity = calc_complexity_sympy(expr)
    candidates = [expr]

    # Simplify power bases
    candidate = simplify_power_bases(expr)
    candidates.append(candidate)

    # Factorise
    try:
        candidate = expr.factor()
        candidates.append(candidate)
    except Exception:
        pass

    # Power simplification
    try:
        candidate = sp.powsimp(expr, force=True)
        candidates.append(candidate)
    except Exception:
        pass

    # Denest powers
    try:
        candidate = sp.powdenest(expr, force=True)
        candidates.append(candidate)
    except Exception:
        pass

    # Combine logs
    try:
        candidate = sp.logcombine(expr, force=True)
        candidates.append(candidate)
    except Exception:
        pass

    # Select the candidate with the lowest complexity (allowing a tolerance of 5%)
    best_expr = expr
    best_complexity = original_complexity
    for cand in candidates:
        try:
            c = calc_complexity_sympy(cand)
            if c < best_complexity * 1.05:  # allow a 5% tolerance
                best_expr = cand
                best_complexity = c
        except Exception:
            continue

    return best_expr

# Timeout helper classes and functions (kept as is)
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def simplify_sympy_expression_with_timeout(expr, timeout=5):
    # Timeout version commented out for now
    pass

# 3. Recursive function to convert from a sympy expression back to a DEAP-style string.
def sympy_to_deap(expr):
    if expr.is_Symbol:
        return str(expr)
    if expr.is_Number:
        return str(round(expr.evalf(), 4)) if expr.is_Float else str(round(expr, 4))
    if expr.is_Function:
        func = expr.func
        args = expr.args
        if func == sympy_log:
            return f"protectedLog({sympy_to_deap(args[0])})"
        if func == sympy_exp:
            return f"protectedExp({sympy_to_deap(args[0])})"
        raise ValueError(f"Unsupported Sympy function: {func.__name__}")
    if expr.is_Add:
        args = list(expr.args)
        if len(args) == 2:
            left, right = args
            if right.is_Mul and len(right.args) == 2 and right.args[0] == -1:
                return f"sub({sympy_to_deap(left)},{sympy_to_deap(right.args[1])})"
            else:
                return f"add({sympy_to_deap(left)},{sympy_to_deap(right)})"
        else:
            from functools import reduce
            subexprs = [sympy_to_deap(a) for a in args]
            return reduce(lambda a, b: f"add({a},{b})", subexprs)
    if expr.is_Mul:
        args = expr.args
        pos_factors = []
        neg_power_factors = []
        for factor in args:
            if factor.is_Pow and factor.args[1].is_Integer and factor.args[1] < 0:
                exponent = factor.args[1]
                base = factor.args[0]
                neg_power_factors.extend([base] * (-exponent))
            else:
                pos_factors.append(factor)
        if neg_power_factors:
            from functools import reduce
            if not pos_factors:
                pos_factors = [sp.Integer(1)]
            if len(pos_factors) == 1:
                numerator_str = sympy_to_deap(pos_factors[0])
            else:
                subexprs_num = [sympy_to_deap(p) for p in pos_factors]
                numerator_str = reduce(lambda a, b: f"mul({a},{b})", subexprs_num)
            if len(neg_power_factors) == 1:
                denominator_str = sympy_to_deap(neg_power_factors[0])
            else:
                subexprs_den = [sympy_to_deap(nf) for nf in neg_power_factors]
                denominator_str = reduce(lambda a, b: f"mul({a},{b})", subexprs_den)
            return f"protectedDiv({numerator_str},{denominator_str})"
        if len(pos_factors) == 2:
            return f"mul({sympy_to_deap(pos_factors[0])},{sympy_to_deap(pos_factors[1])})"
        else:
            from functools import reduce
            subexprs = [sympy_to_deap(a) for a in pos_factors]
            return reduce(lambda a, b: f"mul({a},{b})", subexprs)
    if expr.is_Pow:
        base, exponent = expr.args
        return f"pow({sympy_to_deap(base)},{sympy_to_deap(exponent)})"
    raise ValueError(f"Unsupported or unexpected expression: {expr}")

# 4. Combine everything: take a toolbox.individual and return the simplified one.
def simplify_individual(individual):
    try:
        sympy_expr = convert_expression_to_sympy(individual)
        simplified_sympy_expr = simplify_sympy_expression(sympy_expr)
        deap_simplified_version = sympy_to_deap(simplified_sympy_expr)
        if config.DATASET == 'HEATSINK':
            deap_simplified_version = deap_simplified_version.replace('x0', 'G1').replace('x1', 'G2')
        elif config.DATASET == 'CORROSION':
            deap_simplified_version = deap_simplified_version.replace('x0', 'pH').replace('x1', 'T') \
                                                               .replace('x2', 'LogP').replace('x3', 'LogV') \
                                                               .replace('x4', 'LogD')
        elif config.DATASET == 'BENCHMARK':
            deap_simplified_version = deap_simplified_version.replace('x0', 'X1').replace('x1', 'X2') \
                                                               .replace('x2', 'X3')
            
        new_individual = gp.PrimitiveTree.from_string(deap_simplified_version, config.PSET)
        return new_individual
    except TimeoutError as e:
        st.write(e)
        return None
    except Exception as e:
        if config.DISPLAY_ERROR_MESSAGES:
            st.write(e)
        return None 

if __name__ == "__main__":
    st.write('Should not be run as main - Simplification.py just contains simplification code')
