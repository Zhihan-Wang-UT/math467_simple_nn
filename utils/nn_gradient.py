'''
    Adobted from Math 467 Theory and Computational Methods for Optimization
    project1 by: Dr. Chunming Wang

    Translated from MATLAB to Python by: Zhihan Wang
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # This is for MacOS
import numpy as np
import torch

def get_network_grad(network,x_val):
    grads = []
    y_val = network(x_val)
    for i in range(y_val.shape[0]):
        batch_grad = []
        for j in range(y_val.shape[1]):
            batch_param_grad = []
            y_val[i,j].backward(retain_graph=True)
            for param in network.parameters():
                batch_param_grad.append(param.grad.view(-1).clone().detach())
                param.grad.data.zero_()
            batch_grad.append(torch.cat(batch_param_grad))

        grads.append(torch.stack(batch_grad))
    return torch.stack(grads)