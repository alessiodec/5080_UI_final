# Simplification.py
from . import config
from . import EngineDict as Engine
import sympy as sp
from sympy import log as sympy_log, exp as sympy_exp
from deap import gp

# Map for conversion between DEAP names and sympy
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

def convert_expression_to_sympy(individual):
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
        return sp.simplify(sympy_expr)
    except sp.SympifyError as e:
        raise ValueError(f"Error converting individual to sympy expression: {e}")

def count_atoms(expr):
    if expr.is_Atom:
        return 1
    return sum(count_atoms(arg) for arg in expr.args)

def calc_complexity_sympy(expr):
    terminal_counts = count_atoms(expr)
    return sp.count_ops(expr) + terminal_counts

def simplify_power_bases(expr):
    expr = expr.replace(lambda e: isinstance(e, sp.Pow) and e.base.is_Number and e.base == 0,
                          lambda e: sp.Integer(0))
    expr = expr.replace(lambda e: isinstance(e, sp.Pow) and e.base.is_Number and e.base == 1,
                          lambda e: sp.Integer(1))
    return expr

def simplify_sympy_expression(expr):
    original_complexity = calc_complexity_sympy(expr)
    candidates = [expr, simplify_power_bases(expr)]
    try:
        candidates.append(expr.factor())
    except Exception:
        pass
    try:
        candidates.append(sp.powsimp(expr, force=True))
    except Exception:
        pass
    try:
        candidates.append(sp.powdenest(expr, force=True))
    except Exception:
        pass
    try:
        candidates.append(sp.logcombine(expr, force=True))
    except Exception:
        pass
    best_expr = expr
    best_complexity = original_complexity
    for cand in candidates:
        try:
            c = calc_complexity_sympy(cand)
            if c < best_complexity * 1.05:
                best_expr = cand
                best_complexity = c
        except Exception:
            continue
    return best_expr

def sympy_to_deap(expr):
    if expr.is_Symbol:
        return str(expr)
    if expr.is_Number:
        return str(round(expr.evalf(),4)) if expr.is_Float else str(round(expr,4))
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
        if len(args)==2:
            left, right = args
            if right.is_Mul and len(right.args)==2 and right.args[0]==-1:
                return f"sub({sympy_to_deap(left)},{sympy_to_deap(right.args[1])})"
            else:
                return f"add({sympy_to_deap(left)},{sympy_to_deap(right)})"
        else:
            from functools import reduce
            subexprs = [sympy_to_deap(a) for a in args]
            return reduce(lambda a,b: f"add({a},{b})", subexprs)
    if expr.is_Mul:
        args = expr.args
        pos_factors = []
        neg_power_factors = []
        for factor in args:
            if factor.is_Pow and factor.args[1].is_Integer and factor.args[1] < 0:
                exponent = factor.args[1]
                base = factor.args[0]
                neg_power_factors.extend([base]*(-exponent))
            else:
                pos_factors.append(factor)
        if neg_power_factors:
            from functools import reduce
            if not pos_factors:
                pos_factors=[sp.Integer(1)]
            if len(pos_factors)==1:
                numerator_str = sympy_to_deap(pos_factors[0])
            else:
                numerator_str = reduce(lambda a,b: f"mul({a},{b})", [sympy_to_deap(p) for p in pos_factors])
            if len(neg_power_factors)==1:
                denominator_str = sympy_to_deap(neg_power_factors[0])
            else:
                denominator_str = reduce(lambda a,b: f"mul({a},{b})", [sympy_to_deap(nf) for nf in neg_power_factors])
            return f"protectedDiv({numerator_str},{denominator_str})"
        if len(pos_factors)==2:
            return f"mul({sympy_to_deap(pos_factors[0])},{sympy_to_deap(pos_factors[1])})"
        else:
            from functools import reduce
            return reduce(lambda a,b: f"mul({a},{b})", [sympy_to_deap(a) for a in pos_factors])
    if expr.is_Pow:
        base, exponent = expr.args
        return f"pow({sympy_to_deap(base)},{sympy_to_deap(exponent)})"
    raise ValueError(f"Unsupported or unexpected expression: {expr}")

def simplify_individual(individual):
    try:
        sym_expr = convert_expression_to_sympy(individual)
        simplified_sym_expr = simplify_sympy_expression(sym_expr)
        deap_str = sympy_to_deap(simplified_sym_expr)
        if config.DATASET=='HEATSINK':
            deap_str = deap_str.replace('x0','G1').replace('x1','G2')
        elif config.DATASET=='CORROSION':
            deap_str = deap_str.replace('x0','pH').replace('x1','T').replace('x2','LogP').replace('x3','LogV').replace('x4','LogD')
        elif config.DATASET=='BENCHMARK':
            deap_str = deap_str.replace('x0','X1').replace('x1','X2').replace('x2','X3')
        new_ind = gp.PrimitiveTree.from_string(deap_str, config.PSET)
        return new_ind
    except Exception as e:
        if config.DISPLAY_ERROR_MESSAGES:
            print(e)
        return individual

if __name__=="__main__":
    print('Should not be run as main - Simplification.py just contains simplification code')
