# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:35:09 2020

@author: vxr131730

Steer unicycle robot from pt A to pt B using nonlinear model predictive control

"""
###############################################################################
###############################################################################

# Import all the required libraries
import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from casadi import *
from casadi.tools import *
plt.interactive(True)
np.seterr(divide = 'ignore')

###########################################################################
def shift(inputParam):
    
    # Unbox input parameters
    T  = inputParam.T
    u  = inputParam.u
    f  = inputParam.f
    t0 = inputParam.t0
    x0 = inputParam.x0
    
    state   = x0
    control = u[0,:].T
    uShape  = u.shape()
    
    # Get the nonlinear propagation w.r.t given states and controls
    fValue = f(state, control)        
    nextSt = state + T*fValue
    
    outParam['x0'] = nextSt.todense()
    outParam['t0'] = t0 + T
    outParam['u0'] = vertcat(u[1:uShape[0],:], u[uShape[0],:])
    
    return outParam
    
###############################################################################
###############################################################################
###############################################################################

def nmpc(fromPoint, toPoint):    
    
    # Close any existing figure
    plt.close('all')
    
    # Define steer function variables
    dt   = 0.2   # discretized time step
    N    = 10    # Prediction horizon
    rDia = 0.1   # Diameter of robot 
    
    # Define MPC input constraints
    xMax = 2     # Max Position
    xMin = -xMax # Min Position
    vMax = 0.6   # Max Linear Velocity
    vMin = -vMax # Min Linear Velocity
    wMax = pi/4  # Max Angular Velocity
    wMin = -wMax # Min Angular Velocity 
    
    # Define the weighing matrices Q for states and R for inputs
    Q = np.diag([1, 5, 0.1]) 
    R = np.diag([0.5, 0.05])
    
    # Create state variables for using in Casadi & set up some aliases
    states    = struct_symSX(["x","y", "theta"]) # vertcat(x, y, theta)     
    x,y,theta = states[...]
    numStates = states.size         
    
    # Create input variables for using in Casadi & set up some aliases
    controls    = struct_symSX(["v", "omega"])    
    v, omega    = controls[...]
    numControls = controls.size
    
    # Define the Nonlinear state equation (Righthand side)
    nonlinearDynamics          = struct_SX(states)
    nonlinearDynamics["x"]     = v*cos(theta)
    nonlinearDynamics["y"]     = v*sin(theta)
    nonlinearDynamics["theta"] = omega  
    
    # Nonlinear State Update function f(x,u)
    # Given {states, controls} as inputs, returns {nonlinearDynamics} as output
    f = Function('f',[states,controls], [nonlinearDynamics])
    
    ## Create the bounds for constraints values    
    # Zero Bounds For Dynamics Equality Constraint
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
    
    # Start the simulation loop
    t0       = 0         # Initial time for each MPC iteration 
    x0       = fromPoint # Initial states
    xRef     = toPoint   # Reference pose
    simTime  = 20        # Maximum simulation time
    checkTol = 1e-2      # MPC Loop tolerance
    
    # Define Data Structures to store history
    xHist = [x0] # For states 
    tHist = [] # For initial times
    
    # Specify the initial conditions
    u0 = np.zeros((N, numControls))
    X0 = repmat(x0, 1, N+1).T
    
    # State Nonlinear MPC iteration Loop
    mpcIter = 0
    xMPC    = []
    uMPC    = []
    
    # Make the decision variables as one column vector    
    optVariables = vertcat(X.cat, U.cat)
    
    # Feed the solver options
    opts = {"print_time": False, 
            "ipopt.print_level":0, 
            "ipopt.max_iter":2000, 
            "ipopt.acceptable_tol": 1e-8,
            "ipopt.acceptable_obj_change_tol": 1e-6}
    
    # Create the nlp problem structure with obj_fun, variables & constraints
    nlp = {"x":optVariables, "p":P, "f":obj, "g":vcat(g)}
    
    # Call the IPOPT solver with nlp object and options
    S = nlpsol("solver", "ipopt", nlp, opts)
    
    # Start the timer
    startTime = time.time()
    
    while LA.norm(x0 - xRef) > checkTol and mpcIter < simTime/dt: 
        
        print('MPC iteration:', mpcIter)
        
        # Set the values of the parameters vector
        argums['p'] = np.vstack((x0, xRef))
        
        # Set initial value of the optimization variables
        argums['x0'] = np.vstack(((X0.T).reshape((numStates*(N+1),1)), (u0.T).reshape((numControls*N,1))))
        
        # Solve the NMPC using IPOPT solver
        soln = S(x0  = argums['x0'], 
                 p   = argums['p'], 
                 lbg = argums['lbg'], 
                 ubg = argums['ubg'], 
                 lbx = argums['lbx'], 
                 ubx = argums['ubx'])
        
        # Retrieve the solution
        xSol = soln['x']                
        
        # Extract the minimizing control    
        u = xSol[numStates*(N+1):].full().reshape(N, numControls)
        uShape = np.shape(u)         
        
        # Store only the input at the first time step
        uMPC.append(u[0,:].T) 
        
        # Update the time history
        tHist.append(t0)       
        
        # Apply the control and shift the solution                
        # Get the nonlinear propagation w.r.t given states and controls  
        fValue = f(x0, u[0,:].T)        
        x0     = x0 + dt*fValue
        x0     = x0.full()
        t0     = t0 + dt
        u0     = np.vstack((u[1:,:], u[-1,:]))  
        
        # Update the state history
        xHist.append(x0)
        
        # Reshape and get solution TRAJECTORY                
        X0 = xSol[0:numStates*(N+1)].full().reshape(N+1, numStates)        
        
        # Store the MPC trajectory        
        xMPC.append(X0)
        
        # Shift trajectory to initialize the next step
        X0 = np.vstack((X0[1:,:], X0[-1,:]))                        
        
        # Increment the MPC iteration number
        mpcIter = mpcIter + 1
    
    # End of while loop
    
    # End the timer and fetch the time for total iteration time
    totalTime = time.time() - startTime
    print('Average MPC Iteration time = ', totalTime/(mpcIter + 1))
    
    print('x0 and xRef are: ', np.c_[x0, xRef])
    
    # How close is the solution to the reference provided
    solnError = LA.norm(x0 - xRef)
    print('|| X_{MPC}(final) - X_{ref} || = ', solnError)    s
    
    
###############################################################################

if __name__ == '__main__':
    main()
    
###############################################################################
###############################################################################
###################### END OF THE FILE ########################################
###############################################################################
###############################################################################    