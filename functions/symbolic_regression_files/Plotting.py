import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from . import config
from . import EngineDict as Engine

def plot_pareto(population, yscale='log', ymax=None, ymin=None):
    """
    Plots the Pareto front and the entire population.
    
    Returns:
        A matplotlib figure object that can be rendered via st.pyplot().
    """
    pareto_front = Engine.return_pareto_front(population)
    pareto_plot_data = np.array([(ind['fitness'], ind['complexity']) for ind in pareto_front])
    population_plot_data = np.array([(ind['fitness'], ind['complexity']) for ind in population])
    
    utopia_point = [min(population_plot_data[:, 1]), min(population_plot_data[:, 0])]
    if config.VERBOSE:
        st.write("Utopia point:", utopia_point)
    
    fig, ax = plt.subplots()
    ax.scatter(population_plot_data[:, 1], population_plot_data[:, 0], s=15, label="Population")
    ax.scatter(pareto_plot_data[:, 1], pareto_plot_data[:, 0], s=15, label="Pareto Front")
    ax.scatter(utopia_point[0], utopia_point[1], label='Utopia')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("Fitness $1-R^2$")
    ax.set_xlabel("Complexity")
    if yscale == 'log':
        ax.set_yscale("log")
    if (ymax is not None and ymin is not None):
        ax.set_ylim(ymin, ymax)
    return fig

def plot_next_gen_parents(population, yscale='log', ymax=None, ymin=None):
    """
    Plots the population, candidate parent points, and the Pareto front for the next generation.
    
    Returns:
        A matplotlib figure object that can be rendered via st.pyplot().
    """
    pareto_front = Engine.return_pareto_front(population)
    top_n = Engine.generate_new_generation_NSGA_2(config.POPULATION_RETENTION_SIZE, population)
    
    top_n_data = np.array([(ind['fitness'], ind['complexity']) for ind in top_n])
    pareto_plot_data = np.array([(ind['fitness'], ind['complexity']) for ind in pareto_front])
    population_plot_data = np.array([(ind['fitness'], ind['complexity']) for ind in population])
    
    utopia_point = [min(population_plot_data[:, 1]), min(population_plot_data[:, 0])]
    if config.VERBOSE:
        st.write("Utopia point:", utopia_point)
    
    fig, ax = plt.subplots()
    ax.scatter(population_plot_data[:, 1], population_plot_data[:, 0], s=15, label="Population")
    ax.scatter(top_n_data[:, 1], top_n_data[:, 0], s=15, label="Parent Candidates")
    ax.scatter(pareto_plot_data[:, 1], pareto_plot_data[:, 0], s=15, label="Pareto Front")
    ax.scatter(utopia_point[0], utopia_point[1], label='Utopia')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("Fitness $1-R^2$")
    ax.set_xlabel("Complexity")
    if yscale == 'log':
        ax.set_yscale("log")
    if (ymax is not None and ymin is not None):
        ax.set_ylim(ymin, ymax)
    return fig
