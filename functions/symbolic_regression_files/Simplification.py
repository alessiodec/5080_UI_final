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
                # Catch if the name is a number 
                try:
                    number = float(node.name)
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
    # If the expression is an atom (like a Symbol or a number), count it as 1.
    if expr.is_Atom:
        return 1
    # Otherwise, sum the counts from all subexpressions.
    return sum(count_atoms(arg) for arg in expr.args)

def calc_complexity_sympy(expr):
    '''
    Function to calculate the complexity of a sympy expression.
    '''
    terminal_counts = count_atoms(expr)
    complexity = sp.count_ops(expr) + terminal_counts
    return complexity

# There is not a sympy function that does this automatically
def simplify_power_bases(expr):
    """
    Simplify power expressions:
    - Replace 0**n with 0
    - Replace 1**n with 1
    Note: This rule applies regardless of the exponent, so be cautious with cases like 0**0.
    """
    # Replace 0**anything with 0
    expr = expr.replace(
        lambda e: isinstance(e, sp.Pow) and e.base.is_Number and e.base == 0,
        lambda e: sp.Integer(0)
    )
    # Replace 1**anything with 1
    expr = expr.replace(
        lambda e: isinstance(e, sp.Pow) and e.base.is_Number and e.base == 1,
        lambda e: sp.Integer(1)
    )
    return expr

def simplify_sympy_expression(expr):
    '''
    Simplify the given sympy expression using custom methods (factor, powsimp, etc.).
    '''
    has_updated = False
    lowest_complexity = calc_complexity_sympy(expr)
    
    if config.DISPLAY_SIMPLIFY_ERROR_MESSAGES:
        st.write(f'Lowest Complexity: {lowest_complexity}, Expr: {expr}')

    # Simplify bases of powers:
    new_complexity = calc_complexity_sympy(simplify_power_bases(expr))
    if new_complexity < lowest_complexity:
        lowest_complexity = new_complexity
        expr = simplify_power_bases(expr)
        has_updated = True
    
    if config.DISPLAY_SIMPLIFY_ERROR_MESSAGES:
        st.write(f'New Complexity: {new_complexity}, Power Simplified: {simplify_power_bases(expr)} Expression: {expr}')
    
    # Factorise
    new_complexity = calc_complexity_sympy(expr.factor())
    if new_complexity < lowest_complexity:
        lowest_complexity = new_complexity
        expr = expr.factor()
        has_updated = True

    if config.DISPLAY_SIMPLIFY_ERROR_MESSAGES:
        st.write(f'New Complexity: {new_complexity}, Factorised: {expr.factor()} Expression: {expr}')

    # powsimp
    new_complexity = calc_complexity_sympy(sp.powsimp(expr, force=True))
    if new_complexity < lowest_complexity:
        lowest_complexity = new_complexity
        expr = sp.powsimp(expr, force=True)
        has_updated = True

    if config.DISPLAY_SIMPLIFY_ERROR_MESSAGES:
        st.write(f'New Complexity: {new_complexity}, Power simp: {sp.powsimp(expr, force=True)} Expression: {expr}')

    # powdenest
    new_complexity = calc_complexity_sympy(sp.powdenest(expr, force=True))
    if new_complexity < lowest_complexity:
        lowest_complexity = new_complexity
        expr = sp.powdenest(expr, force=True)
        has_updated = True

    if config.DISPLAY_SIMPLIFY_ERROR_MESSAGES:
        st.write(f'New Complexity: {new_complexity}, Power denest: {sp.powdenest(expr, force=True)} Expression: {expr}')

    # logcombine
    new_complexity = calc_complexity_sympy(sp.logcombine(expr, force=True))
    if new_complexity < lowest_complexity:
        lowest_complexity = new_complexity
        expr = sp.logcombine(expr, force=True)
        has_updated = True

    if config.DISPLAY_SIMPLIFY_ERROR_MESSAGES:
        st.write(f'New Complexity: {new_complexity}, Log Combine: {sp.logcombine(expr, force=True)} Expression: {expr}')
    
    if has_updated:
        # Recursive call: keep simplifying until no further improvements
        expr = simplify_sympy_expression(expr)
    return expr

# Timeout helper classes and functions (kept as is)
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def simplify_sympy_expression_with_timeout(expr, timeout=5):
    """
    Simplify a sympy expression with a maximum allowed time (in seconds).
    If the simplification finishes before the timeout, the result is returned immediately.
    Otherwise, a TimeoutError is raised.
    Note: This approach uses signal.alarm and only works on Unix-like systems.
    """
    '''
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        result = simplify_sympy_expression(expr)
        signal.alarm(0)
        return result
    except TimeoutException:
        raise TimeoutError(f"simplify_sympy_expression of {expr} timed out after {timeout} seconds")
    finally:
        signal.signal(signal.SIGALRM, original_handler)
    '''
    # Timeout version commented out for now
    pass

# 3. Recursive function to convert from a sympy expression back to a DEAP-style string.
def sympy_to_deap(expr):
    """
    Converts a sympy expression `expr` into a DEAP-style string.
    """
    # 1) Base cases:
    if expr.is_Symbol:
        return str(expr)
    if expr.is_Number:
        return str(round(expr.evalf(), 4)) if expr.is_Float else str(round(expr, 4))
        
    # 2) Handle known functions
    if expr.is_Function:
        func = expr.func
        args = expr.args
        if func == sympy_log:
            return f"protectedLog({sympy_to_deap(args[0])})"
        if func == sympy_exp:
            return f"protectedExp({sympy_to_deap(args[0])})"
        raise ValueError(f"Unsupported Sympy function: {func.__name__}")

    # 3) Handle addition
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

    # 4) Handle multiplication and division (merging negative exponents)
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

    # 5) Handle exponentiation
    if expr.is_Pow:
        base, exponent = expr.args
        return f"pow({sympy_to_deap(base)},{sympy_to_deap(exponent)})"

    raise ValueError(f"Unsupported or unexpected expression: {expr}")

# 4. Combine everything: take a toolbox.individual and return the simplified one.
def simplify_individual(individual):
    try:
        # 1. Convert individual into a sympy expression.
        sympy_expr = convert_expression_to_sympy(individual)
        # 2. Simplify using custom method.
        simplified_sympy_expr = simplify_sympy_expression(sympy_expr)
        # 3. Convert back to DEAP-style string.
        deap_simplified_version = sympy_to_deap(simplified_sympy_expr)
        # 4. Adjust string based on dataset.
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
