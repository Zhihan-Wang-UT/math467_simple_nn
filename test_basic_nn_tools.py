import os
from re import X
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt
from utils.getdata import get_data
from utils import simple_network, create_network, visualize_nn
from utils import get_nn_weight, set_nn_weight, get_network_grad

from numpy.random import standard_normal as randn
from numpy.random import rand as rand
import torch

network = create_network(2,[4,4,2])
visualize_nn(network)

weight = get_nn_weight(network)
weight = 0.01*randn(weight.shape)
network = set_nn_weight(network,weight)

C = 51

x1 = -1+2*np.arange(C)
x2 = x1[:,np.newaxis]*np.ones((1,C))
x1 = np.ones((C,1))*x1
x_val = np.vstack((x1.reshape(1,-1),x2.reshape(1,-1)))
x_val = x_val.transpose()

y_val = network(torch.tensor(x_val,dtype=torch.float32))
y_val = y_val.detach().numpy().transpose()
y1 = y_val[0,:].reshape(C,C)
y2 = y_val[1,:].reshape(C,C)
fig, ax = plt.subplots(1,2, figsize=(12,6))
ax[0].imshow(y1, extent=[-10,10,-10,10])
ax[0].set_xlabel('x_1')
ax[0].set_ylabel('x_2')
ax[0].set_title('y_1')
ax[1].imshow(y2, extent=[-10,10,-10,10])
ax[1].set_xlabel('x_1')
ax[1].set_ylabel('x_2')
ax[1].set_title('y_2')
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")
plt.show(block=False)


from torch.autograd import grad 

network.train()
x_val = randn((100,2))
x_val = torch.tensor(x_val,dtype=torch.float32, requires_grad=True)

y_val = network(x_val)
network.zero_grad()
y_grad = get_network_grad(network,x_val)

d_weight = 0.01*randn(weight.shape)*weight
# d_weight = 0.00001*randn(weight.shape)
network = set_nn_weight(network,weight+d_weight)
y_val_new = network(x_val)

delta_y_val = y_val_new - y_val
delta_y_val = delta_y_val.detach().numpy()
ad_y_val = np.zeros(delta_y_val.shape)

for k in range(delta_y_val.shape[0]):
    ad_y_val[k,:] = np.matmul(y_grad[k,:,:],d_weight)

relative_error = (delta_y_val-ad_y_val)/delta_y_val
fig, ax = plt.subplots( figsize=(6,4) )
l = len(relative_error)
ax.plot(list(range(l)),relative_error,label='y_1')
# ax.plot(list(range(l)),relative_error[:,1],label='y_2')
ax.set_xlabel('Order of Input')
ax.set_ylabel('Relative Error in Output Variation (percent)')
plt.show(block=False)


plt.show() # Pause the program for the display window
