import numpy as np


def mix_rbf_kernel(x, y, sigma_list):
    """
    Computes a mixture of RBF kernels between two datasets.
    """
    exponent = np.sum((x[:, None] - y[None, :]) ** 2, axis=-1)
    return sum(np.exp(-1.0 / (2 * sigma) * exponent) for sigma in sigma_list)


class RBFMMD2:
    """
    Computes the squared Maximum Mean Discrepancy (MMD) using an RBF kernel.
    """
    def __init__(self, sigma_list):
        self.sigma_list = sigma_list

    def compute(self, x, y):
        """
        Computes the squared MMD between two datasets.
        """
        return (
            mix_rbf_kernel(x, x, self.sigma_list) +
            mix_rbf_kernel(y, y, self.sigma_list) -
            2 * mix_rbf_kernel(x, y, self.sigma_list)
        )
