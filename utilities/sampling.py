import numpy as np


def sample_from_prob(x, pl, num_sample):
    """
    Sample x ~ pl.
    """
    pl = pl / pl.sum()
    indices = np.random.choice(len(x), num_sample, p=pl)
    return x[indices]


def prob_from_sample(dataset, hndim):
    """
    Empirical probability from data.
    """
    counts = np.bincount(dataset, minlength=hndim)
    return counts / counts.sum()
