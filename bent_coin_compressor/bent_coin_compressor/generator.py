import numpy as np


def generate_file(n_bits, one_proba):
    return (np.random.uniform(size=n_bits) <= one_proba).astype(int)