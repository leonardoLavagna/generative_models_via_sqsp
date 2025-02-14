#------------------------------------------------------------------------------
# sampling.py
#
# This module provides functions for sampling from a probability distribution 
# and estimating empirical probabilities from a dataset. The functions are useful 
# for tasks such as Monte Carlo sampling, data resampling, and probability 
# estimation in machine learning and statistics.
#
# Functions included:
# - sample_from_prob(x, pl, num_sample): Samples `num_sample` elements from the 
#   dataset `x` based on the provided probability distribution `pl`.
# - prob_from_sample(dataset, hndim): Computes the empirical probability distribution 
#   from a dataset, given the number of possible outcomes `hndim`.
#
# These utilities are valuable for resampling tasks, empirical data analysis, 
# and in algorithms requiring probabilistic sampling from data or distributions.
#
# Refs:
# [1] https://github.com/GiggleLiu/QuantumCircuitBornMachine/tree/master
# [2] https://arxiv.org/abs/1804.04168
# [3] https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
#
# © Leonardo Lavagna 2025
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------
# Import necessary library
import numpy as np

def sample_from_prob(x, pl, num_sample):
    """
    Sample x ~ pl.

    This function performs sampling of `num_sample` elements from the dataset `x` 
    based on the given probability distribution `pl`. The probabilities are normalized 
    to ensure they sum to 1 before sampling. It returns the sampled elements from `x`.

    Args:
        x (numpy.ndarray): Dataset `x` from which to sample, shape (n_samples, n_features).
        pl (numpy.ndarray): Probability distribution over the dataset `x`, shape (n_samples,).
        num_sample (int): The number of samples to draw.

    Returns:
        numpy.ndarray: The sampled elements from `x`, shape (num_sample, n_features).
    """
    # Normalize the probability distribution to ensure it sums to 1
    pl = pl / pl.sum()
    
    # Perform random sampling based on the normalized probability distribution
    indices = np.random.choice(len(x), num_sample, p=pl)
    
    # Return the sampled elements from x
    return x[indices]


def prob_from_sample(dataset, hndim):
    """
    Empirical probability from data.

    This function computes the empirical probability distribution from a dataset. 
    It counts the occurrences of each element in the dataset and normalizes the 
    counts to produce a probability distribution.

    Args:
        dataset (numpy.ndarray): The dataset to compute the probability distribution from.
        hndim (int): The number of possible distinct outcomes in the dataset.

    Returns:
        numpy.ndarray: The empirical probability distribution, shape (hndim,).
    """
    # Count the occurrences of each element in the dataset
    counts = np.bincount(dataset, minlength=hndim)
    
    # Normalize the counts to obtain probabilities
    return counts / counts.sum()
