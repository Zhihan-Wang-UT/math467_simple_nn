'''
    Adobted from Math 467 Theory and Computational Methods for Optimization
    project1 by: Dr. Chunming Wang

    Translated from MATLAB to Python by: Zhihan Wang
'''


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt
from utils.getdata import get_data
from utils import simple_network, create_network, visualize_nn
from utils import get_nn_weight, set_nn_weight

from numpy.random import standard_normal as randn
from numpy.random import rand as rand
import torch


'''
Description: This is a script the performs a test of basic neural network
routines.
Usage: Experiment_Basic_NNTools(nTrials)
'''

def experiment_basic_nn_tools(n_trails = 1000):

    # Generate data and plot distribution
    x_data,y_data = get_data(100, 2, 1234567890)
    x_data = x_data.transpose()
    ind0 = np.where(y_data==0)[1]
    ind1 = np.where(y_data==1)[1]

    fig, ax = plt.subplots( figsize=(6,4) )
    ax.bar([0,1],[len(ind0),len(ind1)])
    ax.set_xticks([0,1])
    plt.show(block=False)
    

    # draw network graph
    network = create_network(2,[2,4,1])
    visualize_nn(network)


    # Initialize network using randomly generated weights.
    weight = get_nn_weight(network)
    weight = np.squeeze(0.01*randn([*weight.shape,n_trails]))
    rms = np.zeros((n_trails,1))
    for i in range(n_trails):
        network = set_nn_weight(network, weight[:,i])
        yVal = network(torch.tensor(x_data,dtype=torch.float32))
        yVal = yVal.detach().numpy().transpose()
        rms[i] = np.sqrt(np.sum((y_data-yVal)**2)/(n_trails-1))
    fig, ax = plt.subplots( figsize=(6,4) )
    ax.hist(rms,20)
    plt.show()




if __name__ == '__main__':
    experiment_basic_nn_tools()



