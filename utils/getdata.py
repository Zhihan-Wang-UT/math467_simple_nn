'''
    Adobted from Math 467 Theory and Computational Methods for Optimization
    project1 by: Dr. Chunming Wang

    Translated from MATLAB to Python by: Zhihan Wang
'''

import random
import numpy as np
import torch

from numpy.random import standard_normal as randn
from numpy.random import rand as rand


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def get_data(n_data: int, n_input: int, student_id = 1234567890):
    random_seed(student_id)
    x0 = 5 * randn((n_input,1))
    r0 = 2 * np.abs(randn((n_input,1)))

    x = 80 * (rand(n_input, n_data) - 0.5)

    dist = np.sqrt(np.sum( np.power( ((x - x0 * np.ones((1, n_data))) / (r0 * np.ones((1, n_data)))), 2 ), 0 ).reshape(1,-1))

    y=np.zeros((1,n_data))
    y = (dist<=10).astype(int)
    z=rand(1,n_data)

    ind = np.asarray(z>0.95).nonzero()
    y.put(ind, 1-y.take(ind))
    return x, y



