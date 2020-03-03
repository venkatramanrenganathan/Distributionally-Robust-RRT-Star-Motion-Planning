# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 22:49:36 2020

@author: vxr131730 - Venkatraman Renganathan
This script simulates Unscented Kalman Filter algorithm. 
This script is tested in Python 3.7, Windows 10, 64-bit
(C) Venkatraman Renganathan, 2019.  Email: vrengana@utdallas.edu

This program is a free software: you can redistribute it and/or modify it
under the terms of the GNU lesser General Public License, either version 
3.7, or any later version. This program is distributed in the hope that it 
will be useful, but WITHOUT ANY WARRANTY. 

"""
import numpy as np
import scipy as sp
import math

###############################################################################

def UKF(ukfParam):
    
    # Unbox the input parameters
    zMean  = ukfParam["x0"]
    zCovar = ukfParam["S0"]
    n_z    = ukfParam["n_z"]
    Q      = ukfParam["Q"] 
    R      = ukfParam["R"] 
    
    # Define the global variables
    alpha   = 0.3
    beta    = 2.0
    n       = n_z
    kappa   = 3 - n
    lambda_ = 5
    dT      = 0.01
    
    # Generate the sigma points using Van der Merwe algorithm
    # First SigmaPoint is always the mean
    sigmaPoints      = np.zeros((2*n+1, n))    
    sigmaPoints[:,0] = zMean    
    
    # Compute the Van der Merwe's weighting matrix values
    parameters["lambda_"] = alpha**2 * (n + kappa) - n
    parameters["Wc"]      = np.full(2*n + 1, 1/(2*(n + lambda_))          # Weights for Covariances
    print(parameters["Wc"])
    parameters["Wm"]      = np.full(2*n + 1, 1/(2*(n + lambda_))          # Weights for Mean
    parameters["Wc[0]"]   = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    parameters["Wm[0]"]   = lambda_ / (n + lambda_)
    
    # Define the direction matrix
    U = scipy.linalg.cholesky((n+lamda)*zCovar)         
    
    # Generate sigmapoints symmetrically around the mean
    for k in range(n):
        sigmaPoints[:, k+1]   = zMean + U[:, k]
        sigmaPoints[:, k+n+1] = zMean - U[:, k]    
    
    ###################### Apriori Update #####################################
    # Pass the sigma points through nonlinear state update model
    predictParameters["sigmaPoints"] = sigmaPoints
    predictParameters["Wm"]          = parameters["Wm"]
    predictParameters["Wc"]          = parameters["Wc"]
    predictParameters["Q"]           = Q
    
    # Compute the apriori output
    aprioriOutput = PredictSigmaPoints(predictParameters)    
    
    # Unbox the apriori output
    aprioriMean   = aprioriOutput["mean"]
    aprioriCovar  = aprioriOutput["Covar"]
    aprioriPoints = aprioriOutput["aprioriPoints"] 
    
    ###########################################################################
    ###################### Aposteriori Update #################################
    
    # Pass the predictedOutput through nonlinear measurement model
    updateParameters["aprioriPoints"] = sigmaPoints # aprioriPoints
    updateParameters["Wc"]            = parameters["Wc"]
    updateParameters["Q"]             = Q
    
    # Compute the aposteriori output
    aposterioriOutput = UpdateSigmaPoints(updateParameters)
    
    # Unbox the aposteriori output
    aposterioriMean   = aposterioriOutput["mean"]
    aposterioriCovar  = aposterioriOutput["Covar"]
    aposterioriPoints = aposterioriOutput["aposterioriPoints"] 
    
    # Compute the residual yStar
    z     = MeasurementModel(zMean)
    yStar = z - aposterioriOutput["mean"]    
     
    # Prepare dictionary to compute cross covariance matrix  
    funParam["input1"]       = aprioriPoints
    funParam["input2"]       = aposterioriPoints
    funParam["input1Mean"]   = aprioriMean
    funParam["input2Mean"]   = aposterioriMean  
    funParam["weightMatrix"] = Wc  
    
    # Compute the cross covariance matrix 
    crossCovarMatrix = ComputeCrossCovariance(funParam)
    
    # Compute Unscented Kalman Gain
    uKFGain = np.dot(crossCovarMatrix, np.inv(aposterioriCovar))
    
    # Compute Aposteriori State Update and Covariance Update
    ukfOutput["stateMean"]  = aprioriMean + np.dot(uKFGain, ySTar)
    ukfOutput["stateCovar"] = aprioriCovar - uKFGain @ aposterioriCovar @ uKFGain.T 
    
    return ukfOutput 

###############################################################################

def PredictSigmaPoints(predictParameters): 
    
    # Unbox all input parameters
    sigmaPoints = predictParameters["sigmaPoints"]
    Wm          = predictParameters["Wm"]
    Wc          = predictParameters["Wc"]
    Q           = predictParameters["Q"]
    
    # Get the shape of sigmaPoints
    ro, co = np.shape(sigmaPoints)
    # Create the data structure to hold the transformed points
    aprioriPoints = np.zeros((ro, co))
    
    # Loop through and pass each and every sigmapoint
    for i in range(co):
        aprioriPoints[:, i] = StateUpdate(sigmaPoints[:, i])
    
    # Compute the mean and covariance of the transformed points
    aprioriOutput = ComputeStatistics(aprioriPoints, Wm, Wc, Q)
    
    # Add the aprioriPoints to output
    aprioriOutput["aprioriPoints"] = aprioriPoints 
    
    return aprioriOutput

###############################################################################

def UpdateSigmaPoints(updateParameters):
    
    # Unbox all input parameters
    aprioriPoints = updateParameters["aprioriPoints"]    
    Wc            = updateParameters["Wc"]
    Q             = updateParameters["Q"]
    
    # Get the shape of aprioriPoints
    ro, co = np.shape(aprioriPoints)
    # Create the data structure to hold the transformed points
    aposterioriPoints = np.zeros((ro-1, co)) #3 states, 2 outputs
    
    # Loop through and pass each and every sigmapoint
    for i in range(co):
        aposterioriPoints[:, i] = MeasurementModel(aprioriPoints[:, i])
        
    # Compute the mean and covariance of the transformed points    
    aposterioriOutput = ComputeStatistics(aposterioriPoints, Wm, Wc, R)
    
    aposterioriOutput["aposterioriPoints"] = aposterioriPoints
    
    return aposterioriOutput

###############################################################################
    
def ComputeCrossCovariance(funParam):        
    
    # Compute the crossCovarMatrix    
    kmax, n = funParam["input1"].shape
    P = zeros((n, n))
    for k in range(kmax):
        diff1 = funParam["input1"][k] - funParam["input1Mean"]
        diff2 = funParam["input2"][k] - funParam["input2Mean"]
        P     = P + funParam["weightMatrix"][k] * np.outer(diff1, diff2) 
    
    return P

###############################################################################
def StateUpdate(oldState, u):    
    newState = oldState + dT*[[u[0]*cos(oldState[2])], 
                              [u[0]*sin(oldState[2])], 
                              [u[1]]]
    
    return newState

###############################################################################
def MeasurementModel(newState):    
    output = [[sqrt((newState[0])**2 + (newState[1])**2)], 
              [atan2(newState[1], newState[0])]]
    
    return output


###############################################################################
def ComputeStatistics(inputPoints, Wm, Wc, noiseCov):
    
    # Compute the weighted mean
    inputPointsMean  = np.dot(Wm, inputPoints)
    
    # Compute the weighted covariance
    kmax, n = inputPoints.shape
    P       = zeros((n, n))
    
    # Find the weighted covariance
    for k in range(kmax):
        y = inputPoints[:, k] - inputPointsMean
        P = P + Wc[:, k] * np.outer(y, y) 
    P = P + noiseCov
    
    inputPointsCovar = covar(inputPoints)
    
    # Box the Output data
    statisticsOutput["mean"]  = inputPointsMean
    statisticsOutput["Covar"] = P
    
    return statisticsOutput
    
###############################################################################