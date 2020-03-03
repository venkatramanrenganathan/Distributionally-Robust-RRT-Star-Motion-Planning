# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:20:38 2020

@author: vxr131730
"""


# Import all the packages.

###############################################################################
###############################################################################

# Import all the required libraries
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import EllipseCollection
from matplotlib.offsetbox import AnnotationBbox, AuxTransformBox
from scipy.linalg import block_diag
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from IPython import display
# To avoid getting error while diving by zero in degenerate cases
np.seterr(divide = 'ignore') 


###############################################################################
############################################################################### 

# Our 2-dimensional distribution will be over variables X and Y
N = 150
X = np.linspace(-5, 5, N)
Y = np.linspace(-5, 5, N)
X, Y = np.meshgrid(X, Y)

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

fig = plt.figure()
ax = fig.gca(projection='3d')

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.1,3))
ax.view_init(25, -15) # 27,-21

def update(k):
    
    global psurf
    global cset
    
    psurf.remove()
    
    print('K = ', k)
    # Mean vector and covariance matrix
    mu    = np.array([-1, -1])+k*0.1
    Sigma = np.array([[ 0.2 , -0.1], [-0.1,  0.4]])
    
    if k%2 == 0:
        mu    = np.array([-1, -1])+k*0.1
        Sigma = np.array([[ 0.6 , -0.3], [-0.3,  0.7]])
            
    
    F = multivariate_normal(mu, Sigma)
    Z = F.pdf(pos)
    
    # Create a surface plot and projected filled contour plot under it.
    
    ax = fig.gca(projection='3d')
    psurf = ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.viridis)
    
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)
    
    return psurf,

ani = FuncAnimation(fig, update, frames=np.arange(40), blit=False, repeat = False)

ani.save('animation.gif', writer='imagemagick', fps=60)
