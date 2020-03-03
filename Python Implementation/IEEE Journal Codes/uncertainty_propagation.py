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
import matplotlib as mpl
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
from itertools import product, combinations
from IPython import display
np.seterr(divide = 'ignore') 
mpl.style.use('seaborn')
###############################################################################
###############################################################################

# Our 2-dimensional distribution will be over variables X and Y

def update(k):
    
    global psurf
    global cset
    global obsContour
    
    psurf.remove()
    
    print('K = ', k)
    # Mean vector and covariance matrix
    if k%2 == 0:
        mu = np.array([-1, pow(-1,k)])+k*0.1
        Sigma = np.array([[ 0.6 , -0.3], [-0.3,  0.7]])
    else:
        mu = np.array([-1, 1])+k*0.1
        Sigma = np.array([[ 0.5 , -0.2], [-0.2,  0.6]])
            
    
    F = multivariate_normal(mu, Sigma)
    Z = F.pdf(pos)
    
    # Create a surface plot and projected filled contour plot under it.
    
    ax = fig.gca(projection='3d')
    psurf = ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.coolwarm)
    
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.40, cmap=cm.coolwarm)
    
    # Plot a cuboid of edgelengths: a, b, c
    a = 1
    b = -4
    c = 0.1
    x,y,z = get_cube()
    
    ax.plot_surface(x*a, y*b, z*c, cmap=cm.OrRd)
    obsContour = ax.contour(x*a, y*b, z*c, zdir='z', offset=-0.40, cmap=cm.coolwarm)
    
    return psurf,


###############################################################################
###############################################################################
def get_cube():   
    phi = np.arange(1,10,2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi)*np.sin(Theta)
    y = np.sin(Phi)*np.sin(Theta)
    z = np.cos(Theta)/np.sqrt(2)
    return x,y,z

###############################################################################
###############################################################################

N = 100
X = np.linspace(-10, 10, N)
Y = np.linspace(-10, 10, N)
X, Y = np.meshgrid(X, Y)

# Pack X and Y into a single 3-dimensional array
pos          = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y


k = 0
# Mean vector and covariance matrix
mu = np.array([-1, -1])+k*0.1

if k%2 == 0:
    Sigma = np.array([[ 0.6 , -0.3], [-0.3,  0.7]])
else:
    Sigma = np.array([[ 0.5 , -0.2], [-0.2,  0.6]])
        
F = multivariate_normal(mu, Sigma)
Z = F.pdf(pos)


fig = plt.figure(figsize = [16,9])
ax  = fig.gca(projection='3d')

psurf = ax.plot_surface(X, Y, Z, 
                        rstride=3, 
                        cstride=3, 
                        linewidth=1, 
                        antialiased=True, 
                        cmap=cm.coolwarm)

cset = ax.contourf(X, Y, Z, 
                   zdir='z', 
                   offset=-0.40, 
                   cmap=cm.coolwarm)


# Plot a cuboid of edgelengths: a, b, c
a = 1
b = -4
c = 0.1
x,y,z = get_cube()

ax.plot_surface(x*a, y*b, z*c, cmap=cm.OrRd)
obsContour = ax.contour(x*a, y*b, z*c, zdir='z', offset=-0.40, cmap=cm.coolwarm)

# Adjust the limits, ticks and view angle
# Pass the custom coordinates as extra arguments
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z, P(x)')
ax.set_zlim(-0.5,0.5)
ax.set_zticks(np.linspace(-0.5,0.5,10))
ax.view_init(20, -15) # 27,-21

# Show Animation
ani = FuncAnimation(fig, update, frames=np.arange(40), blit=False)
ani.save('runningDistribution.gif', writer='imagemagick', fps=40)
###############################################################################
###############################################################################