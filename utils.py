import numpy as np
import os

def sample_Z(m, n):
    """
    Uniform prior for G(Z)
    """
    return np.random.uniform(-1., 1., size=[m, n])

def path_exists(path):
	return os.path.exists(path)

def create_dir_if_not_exists(directory):
	if not path_exists(directory):
		os.makedirs(directory)
