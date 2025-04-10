import numpy as np
import matplotlib.pyplot as plt
from . import config
from . import EngineDict as Engine

def plot_pareto(population, yscale='log', ymax=None, ymin=None):
    pareto_front = Engine.return_pareto_front(population)
    pareto_plot_data = []
    for individual in pareto_front:
        pareto_plot_data.append((individual['fitness'], individual['complexity']))
    pareto_plot_data = np.array(pareto_plot_data)
    
    population_plot_data = []
    for individual in population.values():
        population_plot_data.append((individual['fitness'], individual['complexity']))
    population_plot_data = np.array(population_plot_data)
    
    utopia_point = [min(population_plot_data[:, 1]), min(population_plot_data[:, 0])]
    if config.VERBOSE:
        print(utopia_point)
    
    plt.scatter(population_plot_data[:, 1], population_plot_data[:, 0], s=15, label="Population")
    plt.scatter(pareto_plot_data[:, 1], pareto_plot_data[:, 0], s=15, label="Pareto Front")
    plt.scatter(utopia_point[0], utopia_point[1], label='Utopia')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel("Fitness $1-R^2$")
    plt.xlabel("Complexity")
    
    if yscale == 'log':
        plt.yscale("log")
    if (ymax is not None and ymin is not None):
        plt.ylim(ymin, ymax)

def plot_next_gen_parents(population, yscale='log', ymax=None, ymin=None):
    pareto_front = Engine.return_pareto_front(population)
    top_n = Engine.generate_new_generation_NSGA_2(config.POPULATION_RETENTION_SIZE, population)
    top_n_data = []
    for i, individual in enumerate(top_n):
        top_n_data.append((individual['fitness'], individual['complexity']))
    top_n_data = np.array(top_n_data)
    pareto_plot_data = []
    for individual in pareto_front:
        pareto_plot_data.append((individual['fitness'], individual['complexity']))
    pareto_plot_data = np.array(pareto_plot_data)
    
    population_plot_data = []
    for individual in population.values():
        population_plot_data.append((individual['fitness'], individual['complexity']))
    population_plot_data = np.array(population_plot_data)
    
    utopia_point = [min(population_plot_data[:, 1]), min(population_plot_data[:, 0])]
    if config.VERBOSE:
        print(utopia_point)
    
    plt.scatter(population_plot_data[:, 1], population_plot_data[:, 0], s=15, label="Population")
    plt.scatter(top_n_data[:, 1], top_n_data[:, 0], s=15, label="Parent Candidates")
    plt.scatter(pareto_plot_data[:, 1], pareto_plot_data[:, 0], s=15, label="Pareto Front")
    plt.scatter(utopia_point[0], utopia_point[1], label='Utopia')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel("Fitness $1-R^2$")
    plt.xlabel("Complexity")
    
    if yscale == 'log':
        plt.yscale("log")
    if (ymax is not None and ymin is not None):
        plt.ylim(ymin, ymax)
