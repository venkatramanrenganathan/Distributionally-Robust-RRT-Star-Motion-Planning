# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:54:36 2019
@author: vxr131730 - Venkatraman Renganathan

This script simulates Path Planning with Distributionally Robust RRT*
This script is tested in Python 3.7, Windows 10, 64-bit
(C) Venkatraman Renganathan, 2019.  Email: vrengana@utdallas.edu

This program is a free software: you can redistribute it and/or modify it
under the terms of the GNU lesser General Public License, either version 
3.7, or any later version. This program is distributed in the hope that it 
will be useful, but WITHOUT ANY WARRANTY. 

"""

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
import time

np.seterr(divide = 'ignore') 

###############################################################################
###############################################################################

# Defining Global Variables
STEER_TIME     = 5     # Maximum Steering Time Horizon
DT             = 0.1   # Time tick(discretization time)
P0             = 0.0   # Optimal Cost-To-Go Matrix - Will be updated below
ENVCONSTANT    = 20.0  # Environment Constant - Used in computing search radius

###############################################################################
###############################################################################

class trajNode():
    """
    Class Representing a steering law trajectory Node
    """ 
    
    def __init__(self):
        """
        Constructor Function
        """
        self.X      = np.zeros((4, 1)) # State Vector [x-position, y-position, x-velocity, y-velocity]
        self.Sigma  = np.zeros((4, 4)) # Covariance Marix

###############################################################################
###############################################################################
        
class DR_RRTStar_Node():
    """
    Class Representing a DR_RRT* Tree Node
    """
    
    def __init__(self):
        """
        Constructor Function
        """                  
        self.cost   = 0.0                            # Cost         
        self.parent = None                           # Index of the parent node       
        self.means  = np.zeros((STEER_TIME+1, 4, 1)) # Mean Sequence
        self.covar  = np.zeros((STEER_TIME+1, 4, 4)) # Covariance Sequence        
    
    ###########################################################################
    
    def __eq__(self,other):
        """
        Overwriting equality check function to compare two same class objects
        """
        costFlag   = self.cost == other.cost
        parentFlag = self.parent == other.parent
        meansFlag  = np.array_equal(self.means, other.means)
        covarFlag  = np.array_equal(self.covar, other.covar)        
        return costFlag and parentFlag and meansFlag and covarFlag 
        
###############################################################################
###############################################################################
class DR_RRTStar():
    """
    Class for DR_RRT* Planning
    """

    def __init__(self, start, randArea, goalArea, maxIter):
        """
        Constructor function
        Input Parameters:
        start   : Start Position [x,y]         
        randArea: Ramdom Samping Area [xmin,xmax,ymin,ymax]
        goalArea: Goal Area [xmin,xmax,ymin,ymax]
        maxIter : Maximum # of iterations to run for constructing DR-RRT* Tree
        """
        # Add the Double Integrator Data  
        self.iter           = 0
        self.controlPenalty = 0.02
        self.plotFrequency  = 50
        self.initParam      = self.GetDynamicsData()        
        self.xminrand       = randArea[0]
        self.xmaxrand       = randArea[1]               
        self.yminrand       = randArea[2]
        self.ymaxrand       = randArea[3]  
        self.xmingoal       = goalArea[0]
        self.xmaxgoal       = goalArea[1]
        self.ymingoal       = goalArea[2]
        self.ymaxgoal       = goalArea[3]             
        self.maxIter        = maxIter                
        self.obstacleList   = self.initParam[9]  
        self.alfaThreshold  = 0.05
        self.alfa           = [self.alfaThreshold]*len(self.obstacleList)
        # Prepare DR-RRT* tree node with start coordinates & add to nodeList
        self.start = DR_RRTStar_Node()                                 
        # Set the covariance sequence to the initial condition value
        for k in range(STEER_TIME+1):
            self.start.means[k,0,:] = start[0]
            self.start.means[k,1,:] = start[1]
            self.start.covar[k,:,:] = self.initParam[8]  
        self.nodeList = [self.start]             
        # Update the Global Variable Optimal Cost-To-Go Matrix 
        self.initParam.append(self.CostToGo(self.initParam))          
    ###########################################################################
    
    def GetDynamicsData(self):
        """
        Returns the Dynamics data and obstacle details as one packed parameter
        """
        # Double Integrator Data                               
        A  = np.array([[1,0,DT,0],
                       [0,1,0,DT],
                       [0,0,1,0],
                       [0,0,0,1]])                                        # System Dynamics Matrix 
        
        B  = np.array([[(DT**2)/2, 0],
                        [0, (DT**2)/2],
                        [DT, 0],
                        [0,DT]])                                          # Control/Input Matrix
        C  = np.array([[1,0,0,0],
                       [0,1,0,0]])                                        # Output Matrix
        G  = B                                                            # Disturbance Input Matrix
        Q  = np.block([[40*np.identity(2), np.zeros((2,2))],
                        [np.zeros((2,2)), 2*np.identity(2)]])             # State Stage cost Penalty
        QT = np.block([[100*np.identity(2), np.zeros((2,2))],
                        [np.zeros((2,2)), 2*np.identity(2)]])             # State Terminal Penalty    
        R  = self.controlPenalty * np.identity(2)                         # Control/Input Penalty 
        W  = np.block([[np.zeros((2,2)), np.zeros((2,2))],
                        [np.zeros((2,2)), 0.1*np.array([[2,1],[1,2]])]])  # Disturbance covariance    
        S0 = np.block([[0.1*np.identity(2), np.zeros((2,2))],
                        [np.zeros((2,2)), 0.1*np.identity(2)]])          # Initial State Covariance                           
    
        # Obstacle Location Format [ox,oy,wd,ht]: 
        # ox, oy specifies the bottom left corner of rectangle with width: wd and height: ht
        obstacleList = [(0.2, 1.2, 0.2, 0.5),
                        (-0.9, 1.2, 0.25, 0.5),
                        (0.2, 0.3, 0.2, 0.2),
                        (-1.05, 0.2, 0.3, 0.4),
                        (-0.6, 0.8, 0.6, 0.3)]
        # Pack all the data into parameter for easy access across all functions
        initParam = [A,B,C,G,Q,QT,R,W,S0,obstacleList] 
        return initParam   
 
    ###########################################################################
    
    def truncate(self, f):
        return math.floor(f * 10 ** 2) / 10 ** 2
    
    ###########################################################################
    
    def GetAncestors(self, childNode):
        """
        Returns the complete list of ancestors for a given child Node
        """
        ancestorNodeList = []
        while True:            
            if childNode.parent is None:
                # It is root node - with no parents
                ancestorNodeList.append(childNode)
                break
            elif childNode.parent is not None:                
                ancestorNodeList.append(self.nodeList[childNode.parent])
                childNode = self.nodeList[childNode.parent]
        return ancestorNodeList
    
    ###########################################################################
    
    def CostToGo(self, initParam):
        """
        Returns the Optimal Cost-To-Go Matrix
        Input Parameters:
        initParam : List containing all the dynamics data matrices
        """
        A  = initParam[0]
        B  = initParam[1]        
        Q  = initParam[4]       
        QT = initParam[5]
        R  = initParam[6]        
        # Preallocate data structures
        n,m       = np.shape(B)
        P         = np.zeros((STEER_TIME+1,n,n))
        P[-1,:,:] = QT        
        # Compute Cost-To-Go Matrix
        for t in range(STEER_TIME-1,-1,-1):            
            P[t,:,:] = Q + A.T @ P[t+1,:,:] @ A - A.T @ P[t+1,:,:] @ B @ inv(R+B.T @ P[t+1,:,:] @ B) @ B.T @ P[t+1,:,:] @ A                    
        P0 = P[0,:,:]        
        return P0
    
    ###########################################################################
    
    def ComputeCost(self, calculateNode):
        """
        Returns the node cost computed using recursion
        Input parameters:
        calculateNode : Node whose cost has to be calculated        
        """
        # If the queried node is a parent node, set the cost to zero & return
        # Else recursively compute the cost as J[N] = J[N_parent] + del*J(sigma,pi)
        if calculateNode.parent is None:            
            return 0 
        elif calculateNode.parent is not None:            
            parentCost  = self.ComputeCost(self.nodeList[calculateNode.parent])  
            parentNode  = self.nodeList[calculateNode.parent]                       
            connectCost = self.ComputeDistance(calculateNode, parentNode)
            totalCost   = parentCost + connectCost
            return totalCost
    
    ###########################################################################
    
    def ComputeDistance(self, fromNode, toNode):
        """
        Returns the distance between two nodes computed using the dynamic control-based distance metric
        Input parameters:
        fromNode   : Node representing point A
        toNode     : Node representing point B        
        """
        # Use the dynamic control-based distance metric
        diffVec = (fromNode.means[-1,:,:] - toNode.means[-1,:,:])[:,0]    
        diffVec = diffVec.T 
        P0      = self.initParam[10]
        return diffVec @ P0 @ diffVec.T    
    
    ###########################################################################
    
    def RandFreeCheck(self, randNode):
        """
        Performs Collision Check For Random Sampled Point
        Input Parameters:
        randNode : Node containing position data which has to be checked for collision 
        """
        for ox, oy, wd, ht in self.obstacleList:                        
            if (randNode.means[-1,0,:] >= ox and 
               randNode.means[-1,0,:] <= ox + wd and
               randNode.means[-1,1,:] >= oy and
               randNode.means[-1,1,:] <= oy + ht):
                return False    # collision
        return True  # safe
    
    ###########################################################################

    def GetRandomPoint(self):
        """
        Returns a randomly sampled node from the obstacle free space
        """        
        while True:            
           # Get a random point in search space and initialize that as a DR-RRT* Node object with zero velocity          
           if self.iter >= 0.985*self.maxIter:
               randNode               = DR_RRTStar_Node()   
               randNode.means[-1,0,:] = random.uniform(self.xmingoal, self.xmaxgoal)
               randNode.means[-1,1,:] = random.uniform(self.ymingoal, self.ymaxgoal)               
           else:
               randNode               = DR_RRTStar_Node()   
               randNode.means[-1,0,:] = random.uniform(self.xminrand, self.xmaxrand)
               randNode.means[-1,1,:] = random.uniform(self.yminrand, self.ymaxrand)
           if self.RandFreeCheck(randNode):
               break
        return randNode  
    
    ###########################################################################
    
    def GetNearestListIndex(self, randNode):
        """
        Returns the index of the node in the tree that is closest to the randomly sampled node
        Input Parameters:        
        randNode  : The randomly sampled node around which a nearest node in the DR-RRT* tree has to be returned        
        """
        distanceList = []
        for node in self.nodeList:                        
            distanceList.append(self.ComputeDistance(node, randNode))                
        return distanceList.index(min(distanceList))
    
    ###########################################################################
    
    def SteerUsingLQGControl(self, fromNode, toNode):        
        """
        Steers from point A to point B using linear quadratic Gaussian control
        Input Parameters:
        fromNode  : Node representing Point A
        toNode    : Node representing Point B        
        """
        # Extract the Linear system model from initParam data list
        A   = self.initParam[0]
        B   = self.initParam[1]
        C   = self.initParam[2]
        G   = self.initParam[3]
        Q   = self.initParam[4]       
        QT  = self.initParam[5]
        R   = self.initParam[6]
        W   = self.initParam[7]             
        n,m = np.shape(B)
        T   = STEER_TIME
        
        # Run dynamic programming to compute optimal controller
        K = np.zeros((T,m,n))
        k = np.zeros((T,m,1))
        P = np.zeros((T+1,n,n))
        p = np.zeros((T+1,n,1)) 
    
        # Initiliaze terminal time matrices
        P[-1,:,:] = QT
        p[-1,:,:] = -np.dot(QT,toNode.means[-1,:,:])

        # Run Backward Propagation Offline
        for t in range(T-1,-1,-1):
            P[t,:,:] = Q + A.T @ P[t+1,:,:] @ A - A.T @ P[t+1,:,:] @ B @ inv(R+B.T @ P[t+1,:,:] @ B) @ B.T @ P[t+1,:,:] @ A
            K[t,:,:] = -inv(R+B.T @ P[t+1,:,:] @ B) @ B.T @ P[t+1,:,:] @ A
            k[t,:,:] = -inv(R+B.T @ P[t+1,:,:] @ B) @ B.T @ p[t+1,:,:]
            p[t,:,:] = A.T @ p[t+1,:,:] - np.dot(Q,toNode.means[-1,:,:]) + K[t,:,:].T @ B.T @ p[t+1,:,:] + A.T @ P[t+1,:,:] @ B @ k[t,:,:] + K[t,:,:].T @ (R + B.T @ P[t+1,:,:] @ B) @ k[t,:,:] 
        
        # Preallocate matrices         
        u           = np.zeros((T,m,1))                # Control Sequence
        x           = np.zeros((T+1,n,1))              # True State
        xEst        = np.zeros((T+1,n,1))              # Estimated State
        x[0,:,:]    = fromNode.means[-1,:,:]           # Feed the initial condition to the True State        
        xEst[0,:,:] = fromNode.means[-1,:,:]           # Feed the initial condition to the Estimated State
        C           = np.identity(n)                   # Output Matrix
        H           = np.identity(n)                   # Sensor Noise Marix
        G           = np.identity(n)                   # Disturbance Matrix                
        KG          = np.zeros((T+1,n,n))              # Kalman Gain Matrix
        S           = np.zeros((T+1,n,n))              # True State Covariance Matrix
        S[0,:,:]    = fromNode.covar[-1,:,:]           # Feed the final time condition to the Covariance Estimate
        A_bar       = np.zeros((T,2*n,2*n))            # New Concatenated Joint System Matrix
        B_bar       = np.zeros((T,2*n,m))              # New Concatenated Joint Input Matrix
        G_bar       = np.zeros((T,2*n,2*n))            # New Concatenated Joint Disturbance Matrix        
        P_x0        = np.zeros((n,n))                  # Initial Covariance of True State
        P_xEst_0    = np.zeros((n,n))                  # Initial Covariance of Estimated State
        pi          = np.zeros((T+1,2*n,2*n))          # Joint Covariance of Both True State and Estimated State
        pi[0,:,:]   = block_diag(P_x0, P_xEst_0)       # Feed the initial condition to the joint covariance
        SigmaV      = 0.001*np.identity(n)             # Realized the measurement noise
        xTrajs      = [trajNode() for i in range(T+1)] # Trajectory data as trajNode object for each steer time step
        
        # Steer the robot across the finite time horizon using LQG control
        for t in range(0,T):            
            # control uses estimated state
            u[t,:,:] = K[t,:,:] @ xEst[t,:,:] + k[t,:,:] 
            # Update the true state
            x[t+1,:,:] = A @ x[t,:,:] + B @ u[t,:,:]
            # update the Kalman Gain
            KG[t,:,:] = S[t,:,:] @ C.T @ inv(C @ S[t,:,:] @ C.T + H @ SigmaV @ H.T)
            # update the estimated state
            xEst[t+1,:,:] = KG[t,:,:] @ C @ A @ x[t,:,:] + (np.identity(n) - KG[t,:,:] @ C) @ A @ xEst[t,:,:] + B @ u[t,:,:]
            # stack up the true and estimated states
            A_bar[t,:,:] = np.block([[A, B @ K[t,:,:]], [KG[t,:,:] @ C @ A, (np.identity(n)-KG[t,:,:] @ C) @ A + B @ K[t,:,:]]])
            B_bar[t,:,:] = np.block([[B],[B]])
            G_bar[t,:,:] = np.block([[G, np.zeros((n,n))], [KG[t,:,:] @ C @ G, KG[t,:,:] @ H]])
            # propagate the joint covariance
            pi[t+1,:,:] = A_bar[t,:,:] @ pi[t,:,:] @ A_bar[t,:,:].T + G_bar[t,:,:] @ block_diag(W, SigmaV) @ G_bar[t,:,:].T
            # Extract the true state covariance alone
            S[t+1,:,:] = np.block([np.identity(n), np.zeros((n,n))]) @ pi[t+1,:,:] @  np.block([np.identity(n), np.zeros((n,n))]).T 
            
        # Compute the trajectory cost as x_0'Px_0
        trajCost = x[0,:,:].T @ P[0,:,:] @ x[0,:,:] 
        # Update the trajectory object at time step t+1        
        for k, xTraj in enumerate(xTrajs):                                      
            xTraj.X     = x[k,:,:]
            xTraj.Sigma = S[k,:,:] 
        return xTrajs, trajCost   
    
    ###########################################################################
    
    def PerformCollisionCheck(self,xTrajs):
        """
        Performs point-obstacle & line-obstacle check in distributionally robust fashion.
        Input Parameters: 
        xTrajs - collection of means & sigmas of points along the steered trajectory
        """
        for k, xTraj in enumerate(xTrajs): 
            if k != 0:                
                # collisionFreeFlag = True: Safe Trajectory and False: Unsafe Trajectory
                drCollisionFreeFlag = self.DRCollisionCheck(xTraj)
                if not drCollisionFreeFlag:
                    return False
                # Check for Line Rectangle Collision only from second time step in the trajectory
                lineRectangleCollisionFreeFlag = self.LineRectangleCollisionFreeCheck(xTrajs[k-1], xTrajs[k])
                if not lineRectangleCollisionFreeFlag:
                    return False
        # If everything is fine, return True
        return True
    
    ###########################################################################
    
    def DRCollisionCheck(self, trajNode):
        """
        Performs Collision Check Using Deterministic Tightening of Distributionally Robust Chance Constraint
        Input Parameters:
        trajNode : Node containing position data which has to be checked for collision         
        """
        for alfa, (ox, oy, wd, ht) in zip(self.alfa, self.obstacleList):                        
            # left,right,bottom,top
            # Check if the node's inside the bloated obstacle
            if (trajNode.X[0] >= ox - math.sqrt((1-alfa)/alfa)*math.sqrt(np.array([1,0,0,0]).T @ trajNode.Sigma @ np.array([1,0,0,0])) and      # Left 
                trajNode.X[0] <= ox + wd + math.sqrt((1-alfa)/alfa)*math.sqrt(np.array([1,0,0,0]).T @ trajNode.Sigma @ np.array([1,0,0,0]))  and  # Right
                trajNode.X[1] <= oy - math.sqrt((1-alfa)/alfa)*math.sqrt(np.array([0,1,0,0]).T @ trajNode.Sigma @ np.array([0,1,0,0]))   and      # Bottom
                trajNode.X[1] >= oy + ht + math.sqrt((1-alfa)/alfa)*math.sqrt(np.array([0,1,0,0]).T @ trajNode.Sigma @ np.array([0,1,0,0])) ):   # Top
                return False    # collision has occured
        return True  # safe     
    
    ###########################################################################
    
    def LineRectangleCollisionFreeCheck(self, fromPoint, toPoint):
        """
        Returns true or false after checking if the line joining two points intersects any obstacle
        Input Parameters:
        fromPoint   : Point A
        toPoint     : Point B
        Source Code : http://jsfiddle.net/pjnovas/fPMG5/
        """       
        # Get the coordinates of the Trajectory line connecting two points
        x1 = fromPoint.X[0]
        y1 = fromPoint.X[1]
        x2 = toPoint.X[0]
        y2 = toPoint.X[1]
        
        for alfa, (ox, oy, wd, ht) in zip(self.alfa, self.obstacleList): 
            
            # Prepare bloated version of min and max x,y positions of obstacle
            minX = ox - math.sqrt((1-alfa)/alfa)*math.sqrt(np.array([1,0,0,0]).T @ toPoint.Sigma @ np.array([1,0,0,0]))
            minY = oy - math.sqrt((1-alfa)/alfa)*math.sqrt(np.array([0,1,0,0]).T @ toPoint.Sigma @ np.array([0,1,0,0]))
            maxX = ox + wd + math.sqrt((1-alfa)/alfa)*math.sqrt(np.array([1,0,0,0]).T @ toPoint.Sigma @ np.array([1,0,0,0]))
            maxY = oy + ht + math.sqrt((1-alfa)/alfa)*math.sqrt(np.array([0,1,0,0]).T @ toPoint.Sigma @ np.array([0,1,0,0]))
            
            # Condition for Line to be Completely outside the rectangle
            if (x1 <= minX and x2 <= minX or
                y1 <= minY and y2 <= minY or
                x1 >= maxX and x2 >= maxX or
                y1 >= maxY and y2 >= maxY):
                continue 
        
            # Calculate the slope of the line
            lineSlope = (y2 - y1) / (x2 - x1)
            
            # Connect with a line to other point and check if it lies inside
            yPoint1 = lineSlope * (minX - x1) + y1
            yPoint2 = lineSlope * (maxX - x1) + y1
            xPoint1 = (minY - y1) / lineSlope + x1
            xPoint2 = (maxY - y1) / lineSlope + x1
            
            if (yPoint1 > minY and yPoint1 < maxY or
                yPoint2 > minY and yPoint2 < maxY or
                xPoint1 > minX and xPoint1 < maxX or
                xPoint2 > minX and xPoint2 < maxX):
                return False        
            
        return True # Collision Free - No Interection
    
    ###########################################################################
    
    def PrepareMinNode(self, nearestIndex, randNode, xTrajs, trajCost):
        """
        Prepares and returns the randNode to be added to the DR-RRT* tree
        Input Parameters: 
        nearestIndex : Index of the nearestNode in the DR-RRT* tree
        randNode     : Node which has to prepared for addition to DR-RRT* Tree
        xTrajs       : Trajectory data containing the sequence of means and covariances
        trajCost     : Cost to Steer from nearestNode to randNode
        """
        # Convert trajNode to DR-RRT* Tree Node        
        minNode = DR_RRTStar_Node()             
        # Associate the DR-RRT* node with sequence of means and covariances data            
        for k, xTraj in enumerate(xTrajs):                        
            minNode.means[k,:,:] = xTraj.X                
            minNode.covar[k,:,:] = xTraj.Sigma 
        # Find mincost = Cost(x_nearest) + Line(x_nearest, x_rand)
        self.nodeList[nearestIndex].cost = self.ComputeCost(self.nodeList[nearestIndex])
        minNode.cost = self.nodeList[nearestIndex].cost + trajCost
        # Associate MinNode's parent as NearestNode
        minNode.parent = nearestIndex
        return minNode
    
    ###########################################################################
    
    def FindNearNodeIndices(self, randNode):
        """
        Returns indices of all nodes that are closer to randNode within a specified radius
        Input Parameters:
        randNode : Node around which the nearest indices have to be selected        
        """
        totalNodes   = len(self.nodeList)
        searchRadius = ENVCONSTANT * math.sqrt((math.log(totalNodes) / totalNodes)) 
        distanceList = []
        for node in self.nodeList:            
            distanceList.append(self.ComputeDistance(node, randNode))             
        nearIndices  = [distanceList.index(dist) for dist in distanceList if dist <= searchRadius ** 2]        
        return nearIndices
    
    ###########################################################################
    
    def ConnectViaMinimumCostPath(self, nearestIndex, nearIndices, randNode, minNode):
        """
        Chooses the minimum cost path by selecting the correct parent
        Input Parameters:        
        nearestIndex : Index of DR-RRT* Node that is nearest to the randomNode
        nearIndices  : Indices of the nodes that are nearest to the randNode        
        randNode     : Randomly sampled node
        minNode      : randNode with minimum cost sequence to connect as of now
        """        
        # If the queried node is a root node, return the same node
        if not nearIndices:
            return minNode         
        # Create holders for mean and covariance sequences
        meanSequences  = np.zeros((len(nearIndices), STEER_TIME+1, 4, 1))
        covarSequences = np.zeros((len(nearIndices), STEER_TIME+1, 4, 4))          
        for j, nearIndex in enumerate(nearIndices):            
            # Looping except nearestNode - Uses the overwritten equality check function            
            if self.nodeList[nearIndex] == self.nodeList[nearestIndex]:                
                continue
            # Try steering from nearNode to randNodeand get the trajectory
            xTrajs, sequenceCost = self.SteerUsingLQGControl(self.nodeList[nearIndex], randNode)             
                        
            self.nodeList[nearIndex].cost = self.ComputeCost(self.nodeList[nearIndex]) 
            connectCost = self.nodeList[nearIndex].cost + sequenceCost              
            
            # Now check for collision along the trajectory
            lineRectangleCollisionFreeFlag = True
            for k, xTraj in enumerate(xTrajs):    
                # Update the meanSequences and covarSequences
                meanSequences[j,k,:,:]  = xTraj.X
                covarSequences[j,k,:,:] = xTraj.Sigma                
                # Check for DR Feasibility             
                drCollisionFreeFlag = self.DRCollisionCheck(xTraj)                
                # Check for Line Rectangle Collision and if there is one, break
                if k != 0:
                    lineRectangleCollisionFreeFlag = self.LineRectangleCollisionFreeCheck(xTrajs[k], xTrajs[k-1])                  
                if not drCollisionFreeFlag or not lineRectangleCollisionFreeFlag:                     
                    break                                     
            # Proceed only if there is no collision
            if drCollisionFreeFlag and lineRectangleCollisionFreeFlag:                                                
                if connectCost < minNode.cost:   
                    # Associate minCost to connect Node as the parent of minNode
                    minNode.cost   = connectCost
                    minNode.means  = meanSequences[j,:,:,:]
                    minNode.covar  = covarSequences[j,:,:,:]                    
                    minNode.parent = nearIndex
        return minNode
    
    ###########################################################################
    
    def ReWire(self, nearIndices, minNode):
        """
        Rewires the DR-RRT* Tree using Minimum cost path found
        Input Parameters:        
        nearIndices : Indices of the nodes that are nearest to the randomNode
        minNode     : randNode with minimum cost sequence to connect as of now
        """               
        meanSequences  = np.zeros((len(nearIndices), STEER_TIME+1, 4, 1))
        covarSequences = np.zeros((len(nearIndices), STEER_TIME+1, 4, 4))        
        # Get all ancestors of minNode
        minNodeAncestors = self.GetAncestors(minNode)
        for j, nearIndex in enumerate(nearIndices):                                  
            # Avoid looping all ancestors of minNode            
            if np.any([self.nodeList[nearIndex] == minNodeAncestor for minNodeAncestor in minNodeAncestors]):
                continue                        
            # Steer from minNode to nearNode
            xTrajs, sequenceCost = self.SteerUsingLQGControl(minNode, self.nodeList[nearIndex]) 
            # Get the connection cost
            connectCost = minNode.cost + sequenceCost
            # Perform Collision Check
            lineRectangleCollisionFreeFlag = True                       
            for k, xTraj in enumerate(xTrajs):             
                # Update the meanSequences and covarSequences
                meanSequences[j,k,:,:]  = xTraj.X
                covarSequences[j,k,:,:] = xTraj.Sigma
                # Check for DR Feasibility                                         
                drCollisionFreeFlag = self.DRCollisionCheck(xTraj)
                # Check for Line Rectangle Collision
                if k != 0:
                    lineRectangleCollisionFreeFlag = self.LineRectangleCollisionFreeCheck(xTrajs[k], xTrajs[k-1])  
                # If there is collision, exit the loop
                if not drCollisionFreeFlag or not lineRectangleCollisionFreeFlag:                                
                    break            
            if drCollisionFreeFlag and lineRectangleCollisionFreeFlag:
                # Proceed only if J[x_min] + del*J(sigma,pi) < J[X_near]
                self.nodeList[nearIndex].cost = self.ComputeCost(self.nodeList[nearIndex])                                 
                if connectCost < self.nodeList[nearIndex].cost:                                   
                    self.nodeList[nearIndex].parent = len(self.nodeList)-1
                    self.nodeList[nearIndex].cost   = connectCost
                    self.nodeList[nearIndex].means  = meanSequences[j,:,:,:]
                    self.nodeList[nearIndex].covar  = covarSequences[j,:,:,:] 
                    # Update the children of nearNode about the change in cost
                    self.UpdateDescendantsCost(self.nodeList[nearIndex])                     
                        
    ###########################################################################
    
    def UpdateDescendantsCost(self, newNode):
        """
        Updates the cost of all children nodes of newNode
        Input Parameter:
        newNode: Node whose children's costs have to be updated
        """
        # Record the index of the newNode
        newNodeIndex = self.nodeList.index(newNode)
        # Loop through the nodeList to find the children of newNode
        for childNode in self.nodeList[newNodeIndex:]:            
            # Ignore Root node and all ancestors of newNode - Just additional check
            if childNode.parent is None or childNode.parent < newNodeIndex:
                continue    
            if childNode.parent == newNodeIndex:  
                # Update the sequence cost by steering from parent to child
                xTrajs, trajCost = self.SteerUsingLQGControl(self.nodeList[newNodeIndex], childNode)
                childNode.cost = newNode.cost + trajCost
                # update the covariance and means also
                meanSequence  = np.zeros((STEER_TIME+1, 4, 1))
                covarSequence = np.zeros((STEER_TIME+1, 4, 4))
                for k, xTraj in enumerate(xTrajs):                                      
                    meanSequence[k,:,:]  = xTraj.X
                    covarSequence[k,:,:] = xTraj.Sigma
                childNode.means = meanSequence
                childNode.covar = covarSequence
                # Get one more level deeper
                self.UpdateDescendantsCost(childNode)
    
    ###########################################################################
    def PlotObstacles(self):
        """
        Plots the obstacles and the starting position.
        """
        # Plot the Starting position        
        plt.plot(self.start.means[-1,0,:], self.start.means[-1,1,:], "Xr",markersize=20)        
        plt.axis([-1.3, 1.3, -0.3, 2.3])
        # Plot the environment boundary
        xy, w, h = (-1.2, -0.2), 2.4, 2.2
        r = Rectangle(xy, w, h, fc='none', ec='gold', lw=1)        
        offsetbox = AuxTransformBox(plt.axes().transData)
        offsetbox.add_artist(r)
        ab = AnnotationBbox(offsetbox, (xy[0]+w/2.,xy[1]+w/2.),
                            boxcoords="data", pad=0.52,fontsize=20,
                            bboxprops=dict(facecolor = "none", edgecolor='k', 
                                      lw = 20))
        plt.axes().add_artist(ab) 
        
        # Change ticklabel font size
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        # Plot the rectangle obstacles
        obstacles = [Rectangle(xy        = [ox, oy], 
                               width     = wd, 
                               height    = ht, 
                               angle     = 0, 
                               color     = "k", 
                               facecolor = "k",) for (ox, oy, wd, ht) in self.obstacleList]
        for obstacle in obstacles:
            plt.axes().add_artist(obstacle)     
            
        # Data for plotting the shaded goal region
        goalHeight = self.ymingoal
        x          = [-1.20,-1.15,-1.10,-1.05, -1.00]         
        y1         = [goalHeight]*len(x)
        
        plt.text(-1.10, 1.0, 'GOAL', rotation=0, weight="bold", color='white', size=20,horizontalalignment='center', verticalalignment='top', multialignment='center')

        
        # Shade the area between y1 and line y=2.10
        plt.fill_between(x, y1, self.ymaxgoal,
                         facecolor="#CAB8CB", # The fill color
                         color='#CAB8CB')     # The outline color
    
    ###########################################################################
    
    def DrawGraph(self, lastFlag, randNode=None):                
        """
        Updates the Plot with uncertainty ellipse and trajectory at each time step
        Input Parameters:
        lastFlag: Flag to denote if its the last iteration
        randNode: Node data representing the randomly sampled point        
        """            
        xValues      = []
        yValues      = []
        widthValues  = []
        heightValues = []
        angleValues  = []
        lineObjects  = []
        
        for ellipseNode in self.nodeList:
            if ellipseNode is not None and ellipseNode.parent is not None:                
                ellNodeShape = ellipseNode.means.shape  
                xPlotValues  = []
                yPlotValues  = []
                # Prepare the trajectory x and y vectors and plot them                
                for k in range(ellNodeShape[0]):                                    
                    xPlotValues.append(ellipseNode.means[k,0,0])
                    yPlotValues.append(ellipseNode.means[k,1,0]) 
                # Plotting the risk bounded trajectories
                lx, = plt.plot(xPlotValues, yPlotValues, color = '#636D97', linewidth=2.0)
                lineObjects.append(lx)  
                # Plot only the last ellipse in the trajectory                                             
                alfa     = math.atan2(ellipseNode.means[-1,1,0], ellipseNode.means[-1,0,0])
                elcovar  = np.asarray(ellipseNode.covar[-1,:,:])            
                elE, elV = LA.eig(elcovar[0:2,0:2])
                xValues.append(ellipseNode.means[-1,0,0])
                yValues.append(ellipseNode.means[-1,1,0])
                widthValues.append(math.sqrt(elE[0]))
                heightValues.append(math.sqrt(elE[1]))
                angleValues.append(alfa*360)                  
        
        # Plot the randomly sampled point
        rx, = plt.plot(randNode.means[-1,0,:], randNode.means[-1,1,:], "^k") 
                     
        # Plot the Safe Ellipses
        XY = np.column_stack((xValues, yValues))                                                 
        ec = EllipseCollection(widthValues, 
                               heightValues, 
                               angleValues, 
                               units='x', 
                               offsets=XY,
                               facecolors="#C59434",
                               transOffset=plt.axes().transData)
        ec.set_alpha(0.5)        
        plt.axes().add_collection(ec)
        plt.pause(0.0001)        
        if not lastFlag:            
            ec.remove() 
            rx.remove()
            for lx in lineObjects:
                lx.remove() 
    
    ###########################################################################
            
    def ExpandTree(self):        
        """
        Subroutine that grows DR-RRT* Tree 
        """                         
        # Plot the environment with the obstacles and the starting position
        self.PlotObstacles()
        
        # Iterate over the maximum allowable number of nodes
        for iter in range(self.maxIter): 
            self.iter = iter                          
                
            # Get a random feasible point in the space as a DR-RRT* Tree node
            randNode = self.GetRandomPoint()            
            
            # Get index of best DR-RRT* Tree node that is nearest to the random node                      
            nearestIndex = self.GetNearestListIndex(randNode)
            # Truncate the randNode within a desired distance
            diffVec         = (self.nodeList[nearestIndex].means[-1,:,:] - randNode.means[-1,:,:])[:,0] 
            actualDistance  = math.sqrt(diffVec[0] ** 2 + diffVec[1] ** 2)    
            desiredDistance = 0.2
            if actualDistance > desiredDistance:
                distanceRatio          = desiredDistance/actualDistance
                randNode.means[-1,0,:] = (1-distanceRatio)*self.nodeList[nearestIndex].means[-1,0,:] + distanceRatio*randNode.means[-1,0,:]
                randNode.means[-1,1,:] = (1-distanceRatio)*self.nodeList[nearestIndex].means[-1,1,:] + distanceRatio*randNode.means[-1,1,:]                        
            
            # Steer from nearestNode to the randomNode using LQG Control
            # Returns a list of node points along the trajectory and cost
            xTrajs, trajCost = self.SteerUsingLQGControl(self.nodeList[nearestIndex], randNode) 
            
            # Check for Distributionally Robust Feasibility of the whole trajectory            
            collisionFreeFlag = self.PerformCollisionCheck(xTrajs)
                                
            # Entire distribution sequence was DR Feasible              
            if collisionFreeFlag:                
                # Create minNode with trajectory data & Don't add to the tree for the time being                               
                minNode = self.PrepareMinNode(nearestIndex, randNode, xTrajs, trajCost)  
                # Get all the nodes in the DR-RRT* Tree that are closer to the randomNode within a specified search radius
                nearIndices = self.FindNearNodeIndices(randNode)                    
                # Choose the minimum cost path to connect the random node
                minNode = self.ConnectViaMinimumCostPath(nearestIndex, nearIndices,randNode, minNode)
                # Add the minNode to the DR-RRT* Tree
                self.nodeList.append(minNode)
                # Rewire the tree with newly added minNode                    
                self.ReWire(nearIndices, minNode)    
                # Plot the trajectory 
                if iter % self.plotFrequency == 0 and iter < self.maxIter-1:
                    print("Iteration no:",iter)
                    self.DrawGraph(0, minNode)  
            # Plot the last iterate
            if iter == self.maxIter-1:
                print("Iteration no:",iter)
                self.DrawGraph(1, randNode)            

###############################################################################
###############################################################################
###############################################################################

def main():    
    
    # Close any existing figure
    plt.close('all')
    
    # Create the DR_RRTStar Class Object by initizalizng the required data
    dr_rrtstar = DR_RRTStar(start = [0, 0], randArea = [-1.2, 1.2, -0.1, 2.1], goalArea = [-1.2,-1.0,0.8,1.1], maxIter=1200)
    
    # Perform DR_RRTStar Tree Expansion
    dr_rrtstar.ExpandTree()    

###############################################################################

if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print("total execution time: ", t2 - t1)
###############################################################################
###############################################################################
###################### END OF THE FILE ########################################
###############################################################################
###############################################################################