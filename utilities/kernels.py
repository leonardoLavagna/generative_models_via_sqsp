#------------------------------------------------------------------------------
# kernels.py
#
# This module provides functionality for computing the squared Maximum Mean 
# Discrepancy (MMD) using a mixture of Radial Basis Function (RBF) kernels. 
# The MMD is a metric that measures the distance between two probability 
# distributions, and is commonly used in machine learning for distribution 
# comparison and statistical tests.
#
# Functions included:
# - mix_rbf_kernel(x, y, sigma_list): Computes a mixture of RBF kernels between 
#   two datasets `x` and `y` using a list of sigma values for the kernels.
# - RBFMMD2: A class for computing the squared MMD between two datasets `x` 
#   and `y` using the mixture of RBF kernels defined in `sigma_list`.
#
# This module is useful for tasks involving distribution matching and 
# comparison, such as domain adaptation, generative models, and other statistical 
# learning applications that require MMD as a loss function.
#
# Refs:
# [1] https://github.com/GiggleLiu/QuantumCircuitBornMachine/tree/master
# [2] https://arxiv.org/abs/1804.04168
#
# © Leonardo Lavagna 2025
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------


import numpy as np


def mix_rbf_kernel(x, y, sigma_list):
    """
    Computes a mixture of RBF kernels between two datasets.

    Args:
        x (numpy.ndarray): Dataset x, shape (n_samples_x, n_features).
        y (numpy.ndarray): Dataset y, shape (n_samples_y, n_features).
        sigma_list (list or np.ndarray): List of sigma values for the RBF kernels.

    Returns:
        numpy.ndarray: The kernel matrix computed between x and y.
    """
    # Ensure sigma values are positive
    if any(sigma <= 0 for sigma in sigma_list):
        raise ValueError("All sigma values must be positive.")
    
    exponent = np.sum((x[:, None] - y[None, :]) ** 2, axis=-1)
    # Use standard RBF kernel formula with sigma^2
    return sum(np.exp(-exponent / (2 * sigma ** 2)) for sigma in sigma_list)


class RBFMMD2:
    """
    Computes the squared Maximum Mean Discrepancy (MMD) using an RBF kernel.

    Args:
        sigma_list (list or np.ndarray): List of sigma values for the RBF kernels.
    """
    def __init__(self, sigma_list):
        self.sigma_list = sigma_list

    def compute(self, x, y):
        """
        Computes the squared MMD between two datasets.

        Args:
            x (numpy.ndarray): Dataset x, shape (n_samples_x, n_features).
            y (numpy.ndarray): Dataset y, shape (n_samples_y, n_features).

        Returns:
            float: The squared MMD value between the two datasets.
        """
        return (
            mix_rbf_kernel(x, x, self.sigma_list) +
            mix_rbf_kernel(y, y, self.sigma_list) -
            2 * mix_rbf_kernel(x, y, self.sigma_list)
        )
