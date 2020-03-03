# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:58:07 2020

@author: vxr131730
"""

# Import all the required libraries
import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from casadi import *
from casadi.tools import *
np.seterr(divide = 'ignore')

###############################################################################
###############################################################################
###############################################################################

def main():    
    
    # Close any existing figure
    plt.close('all')
    
    # Define steer function variables
    dt   = 0.2   # discretized time step
    N    = 10    # Prediction horizon
    rDia = 0.1   # Diameter of robot 
    
    # Define MPC input constraints
    xMin = -2    # Min Position
    xMax = 2    # Max Position
    vMax = 0.6   # Max Linear Velocity
    vMin = -vMax # Min Linear Velocity
    wMax = pi/4  # Max Angular Velocity
    wMin = -wMax # Min Angular Velocity 
    
    # Define the weighing matrices Q for states and R for inputs
    Q = np.diag([1, 5, 0.1]) 
    R = np.diag([0.5, 0.05])
    
    # Create state variables for using in Casadi    
    # Create the robot state & set up some aliases
    states    = struct_symSX(["x","y", "theta"]) # vertcat(x, y, theta)     
    x,y,theta = states[...]
    numStates = states.size    
    print('The states are:', states)    
    
    # Create input variables for using in Casadi & set up some aliases
    controls    = struct_symSX(["v", "omega"])    
    v, omega    = controls[...]
    numControls = controls.size
    print('The controls are:', controls)
    
    # Define the Nonlinear state equation (Righthand side)
    nonlinearDynamics          = struct_SX(states)
    nonlinearDynamics["x"]     = v*cos(theta)
    nonlinearDynamics["y"]     = v*sin(theta)
    nonlinearDynamics["theta"] = omega    
    print('The nonlinear dynamics is given by:', nonlinearDynamics)
    
    # Nonlinear State Update function f(x,u)
    # Given {states, controls} as inputs, returns {nonlinearDynamics} as output
    f = Function('f',[states,controls], [nonlinearDynamics])
    
    ## Create the bounds for constraints values
    
    # Bounds For Dynamics Equality Constraint (Multiple Shooting)
    lbg_values = np.zeros(numStates*(N+1))
    ubg_values = np.zeros(numStates*(N+1))
    
    # Bounds For Initialize State Constraints
    lbx_values = np.zeros((numStates*(N+1)+numControls*N,1))
    ubx_values = np.zeros((numStates*(N+1)+numControls*N,1))
    
    # Create the indixes list for states and controls
    xIndex     = np.arange(0, numStates*(N+1), numStates).tolist()
    yIndex     = np.arange(1, numStates*(N+1), numStates).tolist()
    thetaIndex = np.arange(2, numStates*(N+1), numStates).tolist()
    vIndex     = np.arange(numStates*(N+1), numStates*(N+1)+numControls*N, numControls).tolist()
    omegaIndex = np.arange(numStates*(N+1)+1, numStates*(N+1)+numControls*N, numControls).tolist()
    
    # Feed Bounds For State Constraints
    lbx_values[xIndex,:]     = xMin
    lbx_values[yIndex,:]     = xMin
    lbx_values[thetaIndex,:] = -float("inf")
    ubx_values[xIndex,:]     = xMax
    ubx_values[yIndex,:]     = xMax
    ubx_values[thetaIndex,:] = float("inf")  
    
    # Feed Bounds For Input Constraints
    lbx_values[vIndex, :]     = vMin
    lbx_values[omegaIndex, :] = wMin
    ubx_values[vIndex, :]     = vMax 
    ubx_values[omegaIndex, :] = wMax
    
    print('lbx_values = ', lbx_values)
    print('ubx_values = ', ubx_values)
    
    # Create the arguments dictionary to hold the constraint values
    argums = {'lbg': lbg_values, 
              'ubg': ubg_values,
              'lbx': lbx_values,
              'ubx': ubx_values}
    
    # Define parameters - Things that change during optimization
    # Control Inputs for N time steps, Initial states & Reference states
    # U-controls, P[0:numStates-1]-states, P[numStates:2*numStates-1]-Reference    
    # X: Vector that represents the states over the optimization problem
    U = struct_symSX([entry("U", shape = (numControls, N))]) # Decision vars (controls)
    P = struct_symSX([entry("P", shape = (2*numStates,1))])  # Decision vars (states)        
    X = struct_symSX([entry("X", shape = (numStates,N+1))])  # History of states   
    
    # U = SX.sym('U',numControls, N)
    # P = SX.sym('P',2*numStates)
    # X = SX.sym('X',numStates, N+1)
    print('U is', U)
    print('P is', P)
    print('X is', X)
       
    # Specify initial state
    st_k = X["X", :, 0]
    
    # Objective function - will be defined shortly below
    obj = 0
    # constraints vector
    g = []
    
    # Add the initial state constraint
    g.append(st_k - P["P", 0:numStates])
    
    # Form the obj_fun and update constraints
    for k in range(N):
        st_k    = X["X", :, k] 
        u_k     = U["U", :, k]
        x_error = st_k - P["P", numStates:2*numStates] 
        # Calculate objective function
        obj     = obj + x_error.T @ Q @ x_error + u_k.T @ R @ u_k 
        st_next = X["X", :, k+1] 
        f_value = f(st_k, u_k)
        st_pred = st_k + dt*f_value
        # Add the constraints
        g.append(st_next - st_pred) # compute constraints 
    
    # Make the decision variables as one column vector
    # optVariables = vertcat(X.reshape((numStates*(N+1),1)), U.reshape((numControls*N,1)))
    optVariables = vertcat(X.cat, U.cat)
    print(optVariables)
    
    # Start the simulation loop
    t0      = 0                             # Initial time for each MPC iteration 
    x0      = np.zeros((numStates,1))       # Initial states
    xRef    = np.array([[1.5], [1.5], [0]]) # Reference pose
    simTime = 20                            # Maximum simulation time
    
    # Define Data Structures to store history
    xHist = x0 # For states 
    tHist = [] # For initial times
    
    # Specify the initial conditions
    u0 = np.zeros((N, numControls))
    X0 = repmat(x0, 1, N+1)
    X0 = X0.T
    print(X0)
    
    # State Nonlinear MPC iteration Loop
    mpcIter = 0
    xMPC    = []
    uMPC    = []
    
    # Create the nlp problem structure with obj_fun, variables & constraints
    nlp = {"x":optVariables, "f":obj, "g":g, "p":P}
    
    # Feed the solver options
    opts = {"print_time": False, 
            "ipopt.print_level":0, 
            "ipopt.max_iter":2000,
            "ipopt.abstol": 1e-8}
    
    # Call the IPOPT solver with nlp object and options
    S = nlpsol("solver", "ipopt", nlp, opts)
    
    # RUN MPC below using above solver. But nlpsol is throwing me errors
    #################### HELP ##############################
    
###############################################################################

if __name__ == '__main__':
    main()
    
###############################################################################
###############################################################################
###################### END OF THE FILE ########################################
###############################################################################
###############################################################################
    
    
