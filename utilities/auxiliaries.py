#------------------------------------------------------------------------------
# auxiliaries.py
#
# This module provides utility functions for computing loss metrics and 
# generating parameters, which are useful in optimization tasks involving 
# probability distributions and quantum computing.
#
# Functions included:
# - compute_loss(y_hat, y, loss_type="mmd"): Computes the loss between predicted 
#   and target values using L1, L2, or MMD loss.
# - generate_parameters(n, k=2): Generates `n` random parameters within a 
#   specified range for optimization.
# - callback_fn(current_params): A callback function to monitor loss history 
#   during optimization.
# - compute_loss_partial(opt_thetas, full_thetas, opt_indices, p_target): 
#   Computes the loss for a subset of optimized parameters within a larger 
#   parameter set.
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
from config import *


all_states = [format(i, f"0{m}b") for i in range(0, 2**m)]


def compute_loss(y_hat, y, loss_type="mmd"):
    """
    Compute the loss between predicted and target values.
    
    Args:
        y_hat (array-like): Predicted values.
        y (array-like): Target values.
        loss_type (str, optional): The type of loss to compute. Options are:
            - "l1": L1 loss (sum of absolute differences).
            - "l2": L2 loss (Euclidean distance).
            - "mmd": MMD-based loss (default).

    Returns:
        float: Computed loss value.

    Raises:
        ValueError: If an unsupported loss type is provided.
    """
    y_hat = np.array(y_hat)
    y = np.array(y)
    min_len = min(len(y_hat), len(y))
    y_hat = y_hat[:min_len]
    y = y[:min_len]
    if loss_type == "l1":  
        return np.sum(np.abs(y - y_hat))
    elif loss_type == "l2":  
        return np.sqrt(np.sum((y - y_hat) ** 2))
    elif loss_type == "mmd":  
        squared_diff = np.sum((y - y_hat) ** 2)
        euclidean_dist = np.sqrt(squared_diff)
        return euclidean_dist + 1e-6 * np.sum(np.abs(y - y_hat))
    else:
        raise ValueError("Unsupported loss. Use 'l1', 'l2' o 'mmd'.")


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


def compute_loss_partial(opt_thetas, full_thetas, opt_indices, p_target):
    """
    Compute loss for a subset of optimized parameters.

    Args:
        opt_thetas (list or array): The optimized parameters.
        full_thetas (list or array): The full parameter list.
        opt_indices (list): Indices in `full_thetas` that are updated by `opt_thetas`.
        p_target (array-like): Target probability distribution.

    Returns:
        float: Computed loss value.
    """
    new_thetas = full_thetas.copy()
    for i, idx in enumerate(opt_indices):
        new_thetas[idx] = opt_thetas[i]
    qc_partial = state_expansion(m, list(new_thetas))
    t_qc_partial = transpile(qc_partial, backend=backend)
    job_partial = backend.run(t_qc_partial, shots=shots)
    counts_partial = job_partial.result().get_counts(qc_partial)
    exp_distribution = np.array([counts_partial.get(state, 0) for state in all_states], dtype=float)
    if exp_distribution.sum() > 0:
        exp_distribution /= exp_distribution.sum()
    return compute_loss(exp_distribution, p_target, loss_type="mmd")
