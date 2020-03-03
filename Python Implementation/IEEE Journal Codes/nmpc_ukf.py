# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:50:19 2020

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
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from numpy import linalg as LA
from scipy import linalg
from casadi import *
from casadi.tools import *
plt.interactive(True)
np.seterr(divide = 'ignore')

###############################################################################
###############################################################################
################## UNSCENTED KALMAN FILTER IMPLEMENTATION #####################
###############################################################################
###############################################################################

def UKF(ukfParam):
    
    # Unbox the input parameters
    zMean  = ukfParam["x0"] 
    u0     = ukfParam["u0"]     
    zCovar = ukfParam["S0"]
    n_z    = ukfParam["n_z"]
    Q      = ukfParam["Q"] 
    R      = ukfParam["R"] 
    
    # Define the global variables
    alpha          = 0.3
    beta           = 2.0
    n              = n_z
    kappa          = 3 - n
    lambda_        = alpha**2 * (n + kappa) - n
    dT             = 0.01    
    numSigmaPoints = 2*n+1
    
    # Initialize Van der Merwe's weighting matrix
    Wc = np.zeros((numSigmaPoints, 1))
    Wm = np.zeros((numSigmaPoints, 1))    
    
    # Compute the Van der Merwe's weighting matrix values    
    for i in range(numSigmaPoints):
        if i == 0:
            Wc[i,:] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
            Wm[i,:] = lambda_ / (n + lambda_)
            continue
        Wc[i,:] = 1/(2*(n + lambda_))
        Wm[i,:] = 1/(2*(n + lambda_))
       
    # Define the direction matrix
    U = linalg.cholesky((n+lambda_)*zCovar)     
    
    # Generate the sigma points using Van der Merwe algorithm
    # First SigmaPoint is always the mean
    sigmaPoints      = np.zeros((n, numSigmaPoints))    
    sigmaPoints[:,0] = zMean[:,0].T 
    
    # Generate sigmapoints symmetrically around the mean
    for k in range(n):
        sigmaPoints[:, k+1]   = sigmaPoints[:,0] + U[:, k]
        sigmaPoints[:, k+n+1] = sigmaPoints[:,0] - U[:, k]   
    
    ###################### Apriori Update #####################################
    # Compute the apriori output
    aprioriOutput = PredictSigmaPoints(u0, dT, sigmaPoints, Wm, Wc, Q)    
    
    # Unbox the apriori output
    aprioriMean   = aprioriOutput["mean"]
    aprioriCovar  = aprioriOutput["Covar"]
    aprioriPoints = aprioriOutput["aprioriPoints"] 
    
    ###########################################################################
    ###################### Aposteriori Update #################################
        
    # Compute the aposteriori output
    aposterioriOutput = UpdateSigmaPoints(sigmaPoints, Wm, Wc, R)
    
    # Unbox the aposteriori output
    aposterioriMean   = aposterioriOutput["mean"]
    aposterioriCovar  = aposterioriOutput["Covar"]
    aposterioriPoints = aposterioriOutput["aposterioriPoints"] 
    
    # Compute the residual yStar
    z     = MeasurementModel(zMean)
    yStar = z - aposterioriMean   
     
    # Prepare dictionary to compute cross covariance matrix  
    funParam = {"input1": aprioriPoints, 
                "input2": aposterioriPoints, 
                "input1Mean": aprioriMean, 
                "input2Mean": aposterioriMean, 
                "weightMatrix": Wc}  
    
    # Compute the cross covariance matrix 
    crossCovarMatrix = ComputeCrossCovariance(funParam)
    
    # Compute Unscented Kalman Gain
    uKFGain = np.dot(crossCovarMatrix, LA.inv(aposterioriCovar))
    
    # Compute Aposteriori State Update and Covariance Update
    ekfMean  = aprioriMean + np.dot(uKFGain, yStar) 
    ekfCovar = aprioriCovar - uKFGain @ aposterioriCovar @ uKFGain.T         
    
    # Prepare Output Dictionary
    ukfOutput = {"stateMean": ekfMean, "stateCovar": ekfCovar}
    
    return ukfOutput 

###############################################################################

def PredictSigmaPoints(u0, dT, sigmaPoints, Wm, Wc, Q):        
        
    # Get the shape of sigmaPoints
    ro, co = np.shape(sigmaPoints)
    # Create the data structure to hold the transformed points
    aprioriPoints = np.zeros((ro, co))
    
    # Loop through and pass each and every sigmapoint
    for i in range(co):
        aprioriPoints[:, i] = MotionModel(sigmaPoints[:, i], u0, dT)    
    
    # Compute the mean and covariance of the transformed points
    aprioriOutput = ComputeStatistics(aprioriPoints, Wm, Wc, Q)
    
    # Add the aprioriPoints to output
    aprioriOutput["aprioriPoints"] = aprioriPoints 
    
    return aprioriOutput

###############################################################################
def MotionModel(oldState, u, dT):    
    newState = oldState + [dT*u[0]*cos(oldState[2]), 
                           dT*u[0]*sin(oldState[2]), 
                           dT*u[1]]
    
    return newState

###############################################################################

def UpdateSigmaPoints(sigmaPoints, Wm, Wc, R):
    
    aprioriPoints = sigmaPoints 
    # Get the shape of aprioriPoints
    ro, co = np.shape(aprioriPoints)
       
    # Create the data structure to hold the transformed points
    aposterioriPoints = np.zeros((ro-1, co)) #3 states, 2 outputs
    
    # Loop through and pass each and every sigmapoint
    for i in range(co):
        aposterioriPoints[:, i] = MeasurementModel(aprioriPoints[:, i])
    
    # Compute the mean and covariance of the transformed points    
    aposterioriOutput = ComputeStatistics(aposterioriPoints, Wm, Wc, R)
    
    # Add the aposterioriPoints to the output dictionary
    aposterioriOutput["aposterioriPoints"] = aposterioriPoints
    
    return aposterioriOutput

###############################################################################
    
def ComputeCrossCovariance(funParam):        
    
    # Compute the crossCovarMatrix    
    input1Shape = np.shape(funParam["input1"])
    input2Shape = np.shape(funParam["input2"])
    P           = np.zeros((input1Shape[0], input2Shape[0]))
    
    for k in range(input1Shape[1]):        
        diff1 = funParam["input1"][:,k] - funParam["input1Mean"]
        diff2 = funParam["input2"][:,k] - funParam["input2Mean"]       
        P     = P + funParam["weightMatrix"][k] * np.outer(diff1, diff2) 
    
    return P

###############################################################################
def MeasurementModel(newState):    
    output = [sqrt((newState[0])**2 + (newState[1])**2), 
              atan2(newState[1], newState[0])]
    
    return output


###############################################################################
def ComputeStatistics(inputPoints, Wm, Wc, noiseCov):
    
    # Compute the weighted mean   
    inputPointsMean  = np.dot(Wm[:,0], inputPoints.T)
    
    # Compute the weighted covariance
    inputShape = np.shape(inputPoints)
    P          = np.zeros((inputShape[0], inputShape[0]))
    
    # Find the weighted covariance
    for k in range(inputShape[1]):        
        y = inputPoints[:, k] - inputPointsMean        
        P = P + Wc[k] * np.outer(y, y) 
    P = P + noiseCov
    
    # Box the Output data
    statsOutput = {"mean": inputPointsMean, "Covar": P}
    
    return statsOutput
    
###############################################################################
###############################################################################
####################### NONLINEAR MODEL PREDICTIVE CONTROL ####################
###############################################################################
###############################################################################

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

def main(): 
    
    ###########################################################################
    ###################### Problem Definition Start ###########################
    ###########################################################################
    
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
    
    # Initialize Bounds For State Constraints
    lbx_values = np.zeros((numStates*(N+1)+numControls*N,1))
    ubx_values = np.zeros((numStates*(N+1)+numControls*N,1))
    
    # Create the indices list for states and controls
    xIndex     = np.arange(0, numStates*(N+1), numStates).tolist()
    yIndex     = np.arange(1, numStates*(N+1), numStates).tolist()
    thetaIndex = np.arange(2, numStates*(N+1), numStates).tolist()
    vIndex     = np.arange(numStates*(N+1), numStates*(N+1)+numControls*N, numControls).tolist()
    omegaIndex = np.arange(numStates*(N+1)+1, numStates*(N+1)+numControls*N, numControls).tolist()
    
    # Feed Bounds For State Constraints
    lbx_values[xIndex,:]     = -float("inf") # xMin
    lbx_values[yIndex,:]     = -float("inf") # xMin
    lbx_values[thetaIndex,:] = -float("inf")
    ubx_values[xIndex,:]     = float("inf")  # xMax
    ubx_values[yIndex,:]     = float("inf")  # xMax
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
    
    ###########################################################################
    ###################### Problem Definition Complete ########################
    ###########################################################################
    
    ###########################################################################
    ###################### Simulation Loop Start ##############################
    ###########################################################################
    # Start the simulation loop
    t0       = 0                             # Initial time for each MPC iteration     
    simTime  = 20                            # Maximum simulation time
    checkTol = 0.05                          # MPC Loop tolerance    
    xRef     = np.array([[1.5], [1.5], [0]]) # Reference pose    
    x0       = np.zeros((numStates,1))       # Initial states
    S0       = np.array([[0.01, 0,    0], 
                         [0,    0.02, 0], 
                         [0,    0,    0.001]])
    SigmaW   = 0.001*np.identity(numStates)   # States:  (x,y,theta)
    SigmaV   = 0.001*np.identity(numStates-1) # Outputs: (r,phi)
    
    # Define the dictionary to pass paramters for UKF state estimation
    ukfParam = {"n_z": numStates, "Q": SigmaW, "R": SigmaV}
    
    # Define Data Structures to store history
    xHist = [x0] # For states 
    tHist = []   # For initial times
    sHist = [S0] # For covariances
    
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
    
    while mpcIter < simTime/dt: # and LA.norm(x0 - xRef) > checkTol: 
        
        if LA.norm(x0 - xRef) < checkTol:
            break
        
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
        u      = xSol[numStates*(N+1):].full().reshape(N, numControls)
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
        
        # Update the dictionary to call the UKF state estimator
        ukfParam["x0"]  = x0
        ukfParam["u0"]  = u[0,:].T
        ukfParam["S0"]  = S0  
        
        # Call the UKF to get the state estimate
        ukfOutput = UKF(ukfParam)
        
        # Unbox the UKF state estimate & covariance
        x0 = ukfOutput["stateMean"]
        S0 = ukfOutput["stateCovar"]
        
        # Reshape x0 to comply to its dimensions
        x0 = np.reshape(x0, (numStates, 1))
        
        # Update the state history
        xHist.append(x0)
        sHist.append(S0)
        
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
    print('|| X_{MPC}(final) - X_{ref} || = ', solnError)    
    
    
    totalIter = mpcIter
    print('Total number of Iterations: ', totalIter)
    
    STEER_TIME = 10
    
    iterIndices = [(totalIter // STEER_TIME) + (1 if i < (totalIter % STEER_TIME) else 0) for i in range(STEER_TIME)]
    iterIndices.insert(0,0)
    
    reqIndices     = [sum(iterIndices[:i+1]) for i in range(len(iterIndices))]
    reqIndices[-1] = reqIndices[-1] - 1
    print('Required Indices = ', reqIndices)
    
    ## Plot the MPC trajectory
    rRad   = rDia/2
    angles = np.arange(0, 2*pi, 0.005).tolist()
    xPoint = rRad*cos(angles)
    yPoint = rRad*sin(angles)
    xPoint = 0
    yPoint = 0
    
    xHistShape = np.shape(xHist)
    
    # Plot the reference state
    plt.figure(figsize = [16,9])
    plt.plot(xRef[0], xRef[1], 'sb')
    plt.plot(xHist[0][0], xHist[0][1], 'dk')
    
    # Plot the predicted and true trajectory
    for k in range(xHistShape[0]):
        xkHist = xHist[k]
        # Plot  
        plt.plot(xkHist[0,:], xkHist[1,:], '-r')    
        if k < xHistShape[0] - 1:
            xkMPCTraj = xMPC[k]
            indices1N = np.arange(N).tolist()
            rzz, = plt.plot(xkMPCTraj[indices1N,0], xkMPCTraj[indices1N,1], '--*k')
               
        rx, = plt.plot(xkHist[0,:]+xPoint, xkHist[1,:]+yPoint, '--r')
        plt.pause(0.1001)        
        if k < xHistShape[0]-1:
            rx.remove()
            rzz.remove()
    plt.pause(1.000)
    plt.close()
    
    # Plot the required States with ellipses
    plt.figure(figsize = [16,9])
    plt.plot(xRef[0], xRef[1], 'sb')
    plt.plot(xHist[0][0], xHist[0][1], 'dk')
    
    xValues      = []
    yValues      = []
    widthValues  = []
    heightValues = []
    angleValues  = []
        
    for k in range(len(reqIndices)):
        x_kHist = xHist[reqIndices[k]]        
        s_kHist = sHist[reqIndices[k]]
        s_kHist = s_kHist[0:numStates-1, 0:numStates-1]
        
        alfa     = math.atan2(x_kHist[1], x_kHist[0])
        elcovar  = np.asarray(s_kHist)            
        elE, elV = LA.eig(elcovar)
        xValues.append(x_kHist[0])
        yValues.append(x_kHist[1])        
        widthValues.append(math.sqrt(elE[0]))
        heightValues.append(math.sqrt(elE[1]))
        angleValues.append(alfa*360)
        
    # Scatter plot all the mean points
    plt.scatter(xValues, yValues)
    plt.plot(xValues, yValues)
    
    # Plot all the covariance ellipses
    
    XY = np.column_stack((xValues, yValues))                                                 
    ec = EllipseCollection(widthValues, 
                           heightValues, 
                           angleValues, 
                           units='x', 
                           offsets=XY,
                           facecolors="#C59434",
                           transOffset=plt.axes().transData)        
    ec.set_alpha(0.6)
    plt.axes().add_collection(ec)
        
        

    
###############################################################################

if __name__ == '__main__':
    main()
    
###############################################################################
###############################################################################
###################### END OF THE FILE ########################################
###############################################################################
###############################################################################