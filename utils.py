import numpy as np


def sample_Z(m, n):
    """
    Uniform prior for G(Z)
    """
    return np.random.uniform(-1., 1., size=[m, n])
