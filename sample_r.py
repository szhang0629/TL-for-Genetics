import numpy as np
import random


def sample_r(N, seed):
    random.seed(seed)
    return random.sample(list(range(N)), N)


def sample_index(seed_index, y):
    sample = sample_r(y.shape[0], seed_index)
    sample_ = np.asarray(sample, 'int').tolist()
    point = round(len(sample_) * 0.8)
    return sample_[:point], sample_[point:]
