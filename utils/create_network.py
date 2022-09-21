'''
    Adobted from Math 467 Theory and Computational Methods for Optimization
    project1 by: Dr. Chunming Wang

    Translated from MATLAB to Python by: Zhihan Wang
'''


from importlib.util import set_package
from typing import OrderedDict
import torch


class simple_network(torch.nn.Module):
    def __init__(self, n_input, layers, activationfunction=torch.nn.Sigmoid):
        super(simple_network, self).__init__()
        layers = [n_input] + layers
        self.layers = layers

        tmp = []
        for i in range(len(layers)-1):
            tmp.append(basic_block(layers[i], layers[i+1], activationfunction))
        self.network = torch.nn.Sequential(*tmp)

    def forward(self, x):
        return self.network(x)

class basic_block(torch.nn.Module):
    def __init__(self, n_input, n_output, activationfunction=torch.nn.Sigmoid):
        super(basic_block, self).__init__()
        self.fc = torch.nn.Linear(n_input, n_output)
        self.activation = activationfunction()
    
    def forward(self, x):
        return self.activation(self.fc(x))


def create_network(n_input, layers, activationfunction=torch.nn.Sigmoid):
    return simple_network(n_input, layers, activationfunction)

