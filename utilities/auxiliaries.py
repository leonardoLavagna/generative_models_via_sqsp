#------------------------------------------------------------------------------
# auxiliaries.py
#
# This module provides utility functions for computing loss metrics and 
# generating parameters, which are useful in optimization tasks involving 
# probability distributions and quantum computing.
#
# Functions included:
# - loss(samples, p_i_set): Computes the loss between predicted  and target values
# - objective_function(thetas_to_optimize, idx_thetas_to_optimize, thetas, p_i_set, shots)
#   Evaluates the objective function by updating parameterized angles, running a quantum
#   circuit, and computing the loss
# - generate_parameters(n, k=2): Generates `n` random parameters within a 
#   specified range for optimization.
#
# These functions are useful in training variational quantum circuits, 
# parameter optimization, and empirical loss computation.
#
# © Marco Casalbore & Leonardo Lavagna 2025
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------


import numpy as np
from utilities.grover_state_preparation import *
from qiskit import transpile
from qiskit.circuit import ParameterVector
from config import *


all_states = [format(i, f"0{m}b") for i in range(2**m)]


def loss(samples, p_i_set):
    """
    Compute the loss between predicted and target values.
    
    Args:
        samples (array-like): Predicted values.
        p_i_set (array-like): Target values.

    Returns:
        float: Computed loss value.
    """
    samples = np.array(samples)
    p_i_set = np.array(p_i_set)
    min_len = min(len(samples), len(p_i_set))
    y_hat = samples[:min_len]
    y = p_i_set[:min_len] 
    squared_diff = np.sum((y - y_hat) ** 2)
    euclidean_dist = np.sqrt(squared_diff)
    return euclidean_dist + np.sum(np.abs(y - y_hat))


def objective_function(thetas_to_optimize, idx_thetas_to_optimize, thetas, p_i_set, shots):
    """
    Evaluates the objective function by updating parameterized angles, 
    running a quantum circuit, and computing the loss.

    Args:
        thetas_to_optimize (list or array): Values to update in the thetas list.
        idx_thetas_to_optimize (list of int): Indices in thetas to replace.
        thetas (list or array): Full parameter list for the circuit.
        p_i_set (array-like): Target probability distribution.
        shots (int): Number of measurement shots for quantum execution.

    Returns:
        float: The computed objective value based on the loss function.
    """
    angles = thetas.copy()
    for i, index in enumerate(idx_thetas_to_optimize):
        angles[index] = thetas_to_optimize[i]
    qc = state_expansion(m, angles)
    t_qc = transpile(qc, backend=backend)
    job = backend.run(t_qc, shots=shots)
    counts = job.result().get_counts(qc)
    samples = np.array([counts.get(state, 0) for state in all_states], dtype=float)
    if samples.sum() > 0:
        samples /= samples.sum()
    else:
        samples = np.zeros_like(samples)
    objective = loss(samples, p_i_set)
    return objective



def generate_parameters(n, k=2):
    """
    Generate a list of random parameters.
    
    Args:
        n (int): Number of parameters to generate.
        k (float, optional): Scaling factor for the range (default is 2).

    Returns:
        list: A list of `n` randomly generated parameters.
    """
    return list(np.random.uniform(low=0, high=k * np.pi, size=n))
