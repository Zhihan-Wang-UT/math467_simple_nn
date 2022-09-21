'''
    Adobted from Math 467 Theory and Computational Methods for Optimization
    project1 by: Dr. Chunming Wang

    Translated from MATLAB to Python by: Zhihan Wang
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # This is for MacOS
import numpy as np
import torch

def get_nn_weight(network):
    sd = network.state_dict()
    weights = np.array([])
    for key, val in sd.items():
        weights = np.append(weights, val.reshape(-1).numpy())
    return weights

def set_nn_weight(network, weights):
    with torch.no_grad():
        sd = network.state_dict()
        idx = 0
        for key, val in sd.items():
            sd[key] = torch.tensor(weights[idx:idx+val.reshape(-1).shape[0]]).reshape(val.shape)
            idx += val.reshape(-1).shape[0]
        network.load_state_dict(sd)
    return network