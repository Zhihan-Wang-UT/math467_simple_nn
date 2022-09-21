import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from getdata import get_data
import numpy as np
import matplotlib.pyplot as plt


x,y = get_data(1000,2,1234567890)
ind1 = np.where(y==1)[1]
ind0 = np.where(y==0)[1]
fix, ax = plt.subplots( figsize=(6,6) )
ax.plot(x[0,ind1],x[1,ind1],'o',markerfacecolor='r',markeredgecolor='r',markersize=5)
ax.plot(x[0,ind0],x[1,ind0],'o',markerfacecolor='b',markeredgecolor='b',markersize=5)
plt.show(block=False)
plt.show()

