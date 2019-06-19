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
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from scipy.linalg import block_diag
from numpy.linalg import inv
from numpy import linalg as LA

###############################################################################
###############################################################################

# Defining Global Variables
show_animation  = True         # Flag to decide to show animation or not
STEER_TIME      = 10           # Maximum Steering Time Horizon
DT              = 0.1          # Time tick(discretization time)
P0              = 0.0          # Optimal Cost-To-Go Matrix - Will be updated below
CT              = 1.0          # Minimum Path Cost: CT = f(\hat{x}, P)
EnvConstant     = 50.0         # Environment Constant - Used in computing search radius
M               = 5            # Number of neighbors to be considered while trying to connect
fig             = plt.figure() # Python Figure handle

###############################################################################
###############################################################################

class trajNode():
    """
    Class Representing a steering law trajectory Node
    """ 
    
    def __init__(self):
        self.X      = np.zeros((4, 1)) # State Vector [x-position, y-position, x-velocity, y-velocity]
        self.Sigma  = np.zeros((4, 4)) # Covariance Marix

###############################################################################
###############################################################################
        
class Node():
    """
    Class Representing a DR_RRT* Node
    """
    
    def __init__(self, x, y):
        """
        Constructor Function
        """
        self.x      = x                              # x position
        self.y      = y                              # y position
        self.xd     = 0.0                            # x velocity
        self.yd     = 0.0                            # y velocity
        self.cost   = 0.0                            # Cost         
        self.parent = None                           # Index of the parent node       
        self.means  = np.zeros((STEER_TIME+1, 4, 1)) # Mean Sequence
        self.covar  = np.zeros((STEER_TIME+1, 4, 4)) # Covariance Sequence
    
    ###########################################################################
    
    def __eq__(self,other):
        """
        Overwriting equality check function to compare two same class objects
        """
        a = self.x  == other.x
        b = self.y  == other.y
        c = self.xd == other.xd
        d = self.yd == other.yd
        e = np.array_equal(self.means, other.means)
        f = np.array_equal(self.covar, other.covar)
        return a and b and c and d and e and f 
        
        
###############################################################################
###############################################################################
class DR_RRTStar():
    """
    Class for DR_RRT* Planning
    """

    def __init__(self, start, randArea, expandDis=0.5, maxIter=50):
        """
        Constructor function
        Input Parameters:
        start   :Start Position [x,y]         
        randArea:Ramdom Samping Area [min,max]
        """
        # Add the Double Integrator Data    
        self.controlPenalty = 0.02
        self.initParam      = self.GetDynamicsData()
        self.start          = Node(start[0], start[1]) # Start Node Coordinates 
        self.minrand        = randArea[0]
        self.maxrand        = randArea[1]
        self.expandDis      = expandDis        
        self.maxIter        = maxIter                
        self.obstacleList   = self.initParam[9]        
        self.alfa           = [0.01 + (0.05-0.01)*random.random() 
                               for i in range(len(self.obstacleList))] 
        self.rewireNodeList = [self.start]    # Node List to hold the values of main nodes for rewiring
        # Set the covariance sequence to the initial condition value
        for k in range(STEER_TIME):
            self.start.covar[k,:,:] = self.initParam[8]               
        
    ###########################################################################
    
    def GetDynamicsData(self):
        """
        Returns the Dynamics data and obstacle details as one packed parameter
        """
        # Double Integrator Data                               
        A  = np.array([[1,0,DT,0],
                       [0, 1, 0, DT],
                       [0,0,1,0],
                       [0,0,0,1]])                                        # System Dynamics Matrix 
        
        B  = np.array([[(DT**2)/2, 0],
                        [0, (DT**2)/2],
                        [DT, 0],
                        [0,DT]])                                          # Input Matrix
        C  = np.array([[1,0,0,0],
                       [0,1,0,0]])                                        # Output Matrix
        G  = B                                                            # Disturbance Input Matrix
        Q  = np.block([[40*np.identity(2), np.zeros((2,2))],
                        [np.zeros((2,2)), 2*np.identity(2)]])             # State Stage cost Penalty
        QT = np.block([[100*np.identity(2), np.zeros((2,2))],
                        [np.zeros((2,2)), 2*np.identity(2)]])             # State Terminal Penalty    
        R  = self.controlPenalty * np.identity(2)                           # Control/Input Penalty 
        W  = np.block([[np.zeros((2,2)), np.zeros((2,2))],
                        [np.zeros((2,2)), 0.1*np.array([[2,1],[1,2]])]])  # Disturbance covariance    
        S0 = np.block([[0.01*np.identity(2), np.zeros((2,2))],
                        [np.zeros((2,2)), 0.01*np.identity(2)]])          # Initial State Covariance                           
    
        # Obstacle Location Format [ox,oy,wd,ht]: ox, oy specifies the bottom left corner of rectangle with width: wd and height: ht
        obstacleList = [(0.6, 0.6, 0.1, 0.1),
                        (0.2, 0.5, 0.2, 0.1),
                        (0.3, 0.7, 0.1, 0.2),
                        (0.1, 0.9, 0.2, 0.1),
                        (0.7, 0.8, 0.1, 0.2),
                        (0.9, 0.2, 0.1, 0.1)] 
        # Pack all the data into parameter for easy access across all functions
        initParam = [A,B,C,G,Q,QT,R,W,S0,obstacleList] 
        return initParam
    
    ###########################################################################
    
    def AddPathNode(self, pathNode):
        """
        Adds the pathNode to the DR-RRT* Tree after preparing it.
        Input Parameters:
        pathNode: Node to be added to the DR-RRT* Tree
        """
        # Create the DR-RRT* Node object and feed the data
        drPathNode    = Node(pathNode.X[0], pathNode.X[1])
        drPathNode.xd = pathNode.X[2]
        drPathNode.yd = pathNode.X[3]  
        covarShape    = drPathNode.covar.shape        
        for k in range(covarShape[0]):
            drPathNode.covar[k,:,:] = pathNode.Sigma
            drPathNode.means[k,:,:] = np.array([pathNode.X[0], pathNode.X[1], pathNode.X[2], pathNode.X[3]])
        drPathNode.parent = len(self.nodeList) - 1
        drPathNode.cost   = self.ComputeCost(drPathNode)
        # Add the node to the DR-RRT* Tree
        self.nodeList.append(drPathNode)   
    
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
                ancestorNodeList.append(childNode.parent)
                childNode = childNode.parent
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
            parentCost       = self.ComputeCost(self.nodeList[calculateNode.parent])            
            drTrajNode       = self.GetLastSequenceNode(calculateNode)
            drTrajParentNode = self.GetLastSequenceNode(self.nodeList[calculateNode.parent])
            addCost          = self.ComputeDistance(drTrajNode, drTrajParentNode,euclidFlag=1)
            calculatedCost   = parentCost + addCost
            return calculatedCost
    
    ###########################################################################
    
    def ComputeDistance(self, fromNode, toNode, euclidFlag=None):
        """
        Returns the distance between two nodes computed using the dynamic control-based distance metric
        Input parameters:
        fromNode   : Node representing point A
        toNode     : Node representing point B
        euclidFlag : Flag determining whether to use simple euclidean distance metric or a dynamic control based metric
        """
        diffVec = fromNode.X - toNode.X
        # If initialFlag is set, use the dynamic control-based distance metric
        if euclidFlag:
            diffVec = diffVec.T 
            P0      = self.initParam[10]
            return diffVec @ P0 @ diffVec.T
        # Else return the simple euclidean distance
        return math.sqrt(diffVec[0] ** 2 + diffVec[1] ** 2)
    
    ###########################################################################
    
    def RandFreeCheck(self, randNode):
        """
        Performs Collision Check For Random Sampled Point
        Input Parameters:
        randNode : Node containing position data which has to be checked for collision 
        """
        for ox, oy, wd, ht in self.obstacleList:            
            relax = 0.02 + max(self.alfa) # Conservative estimate used here - Can also use DR CHECK - But not needed
            if randNode.X[0] >= ox - relax and randNode.X[0] <= ox + wd + relax and randNode.X[1] >= oy - relax and randNode.X[1] <= oy + ht + relax:
                return False    # collision
        return True  # safe
    
    ###########################################################################

    def GetRandomPoint(self):
        """
        Returns a randomly sampled node from the obstacle free space
        """        
        while True:            
           # Get a random point in search space and initialize that as a trajNode object with zero velocity          
           randNode      = trajNode()
           randNode.X[0] = random.uniform(self.minrand, self.maxrand)
           randNode.X[1] = random.uniform(self.minrand, self.maxrand)
           randNode.X[2] = 0.0
           randNode.X[3] = 0.0
           if self.RandFreeCheck(randNode):
               break
        return randNode  
    
    ###########################################################################
    
    def GetNearestListIndices(self, randNode, M):
        """
        Returns the indices of top M nodes in the tree that are closest to the randomly sampled node
        Input Parameters:        
        randNode  : The randomly sampled node around which a nearest node in the DR-RRT* tree has to be returned
        M         : Total number of indices to be returned    
        """
        distanceList = [self.ComputeDistance(self.GetLastSequenceNode(node),randNode) for node in self.nodeList]            
        return np.argsort(distanceList)[:M]
    
    ###########################################################################
    
    def GetLastSequenceNode(self, drTreeNode):
        """
        Returns the last member of (means,covar) data sequence as trajectory node
        Input Parameters:
        drTreeNode : DR-RRT* Tree node which has to be queried to get the trajectory node representing last sequence data
        """
        # Create a plain Trajectory Node
        lastSequenceNode = trajNode()
        # Get the mean and covariance sequences
        meanSequence  = drTreeNode.means
        covarSequence = drTreeNode.covar
        # Load the last sequence data
        lastSequenceNode.X     = meanSequence[-1,:,:]
        lastSequenceNode.Sigma = covarSequence[-1,:,:]        
        return lastSequenceNode
    
    ###########################################################################
    
    def SteerUsingLQGControl(self, from_node, to_node):        
        """
        Steers from point A to point B using linear quadratic Gaussian control
        Input Parameters:
        from_node  : Node representing Point A
        to_node    : Node representing Point B        
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
        
        # Extract state data from from_node and to_node
        fromNode = from_node.X
        toNode   = to_node.X       
        
        # Run dynamic programming to compute optimal controller
        K = np.zeros((T,m,n))
        k = np.zeros((T,m,1))
        P = np.zeros((T+1,n,n))
        p = np.zeros((T+1,n,1)) 
    
        # Initiliaze terminal time matrices
        P[-1,:,:] = QT
        p[-1,:,:] = -np.dot(QT,toNode)

        # Run Backward Propagation Offline
        for t in range(T-1,0,-1):
            P[t,:,:] = Q + A.T @ P[t+1,:,:] @ A - A.T @ P[t+1,:,:] @ B @ inv(R+B.T @ P[t+1,:,:] @ B) @ B.T @ P[t+1,:,:] @ A
            K[t,:,:] = -inv(R+B.T @ P[t+1,:,:] @ B) @ B.T @ P[t+1,:,:] @ A
            k[t,:,:] = -inv(R+B.T @ P[t+1,:,:] @ B) @ B.T @ p[t+1,:,:]
            p[t,:,:] = A.T @ p[t+1,:,:] - np.dot(Q,toNode) + K[t,:,:].T @ B.T @ p[t+1,:,:] + A.T @ P[t+1,:,:] @ B @ k[t,:,:] + K[t,:,:].T @ (R + B.T @ P[t+1,:,:] @ B) @ k[t,:,:] 
        
        # Preallocate matrices        
        u            = np.zeros((T,m,1));               # Control Sequence
        x            = np.zeros((T+1,n,1));             # True State
        x[0,:,:]     = fromNode;                        # Feed the initial condition to the True State
        x_est        = np.zeros((T+1,n,1))              # Estimated State
        x_est[0,:,:] = fromNode                         # Feed the initial condition to the Estimated State
        C            = np.identity(n)                   # Output Matrix
        H            = np.identity(n)                   # Sensor Noise Marix
        G            = np.identity(n)                   # Disturbance Matrix                
        KG           = np.zeros((T+1,n,n))              # Kalman Gain Matrix
        S            = np.zeros((T+1,n,n))              # True State Covariance Matrix
        S[0,:,:]     = from_node.Sigma                  # Feed the final time condition to the Covariance Estimate
        A_bar        = np.zeros((T,2*n,2*n))            # New Concatenated Joint System Matrix
        B_bar        = np.zeros((T,2*n,m))              # New Concatenated Joint Input Matrix
        G_bar        = np.zeros((T,2*n,2*n))            # New Concatenated Joint Disturbance Matrix
        pi           = np.zeros((T+1,2*n,2*n))          # Joint Covariance of Both True State and Estimated State
        P_x0         = np.zeros((n,n))                  # Initial Covariance of True State
        P_x_est_0    = np.zeros((n,n))                  # Initial Covariance of Estimated State
        pi_0         = block_diag(P_x0, P_x_est_0)      # Joint Initial Covariance of true and estimated states    
        pi[0,:,:]    = pi_0                             # Feed the initial condition to the joint covariance
        Sigma_v      = 0.001*np.identity(n)             # Realized the measurement noise
        x_trajs      = [trajNode() for i in range(T+1)] # Trajectory data as trajNode object for each steer time step
        
        # Steer the robot across the finite time horizon using LQG control
        for t in range(T):            
            # control uses estimated state
            u[t,:,:] = K[t,:,:] @ x_est[t,:,:] + k[t,:,:] 
            # Update the true state
            x[t+1,:,:] = A @ x[t,:,:] + B @ u[t,:,:]
            # update the Kalman Gain
            KG[t,:,:] = S[t,:,:] @ C.T @ inv(C @ S[t,:,:] @ C.T + H @ Sigma_v @ H.T)
            # update the estimated state
            x_est[t+1,:,:] = KG[t,:,:] @ C @ A @ x[t,:,:] + (np.identity(n) - KG[t,:,:] @ C) @ A @ x_est[t,:,:] + B @ u[t,:,:]
            # stack up the true and estimated states
            A_bar[t,:,:] = np.block([[A, B @ K[t,:,:]], [KG[t,:,:] @ C @ A, (np.identity(n)-KG[t,:,:] @ C) @ A + B @ K[t,:,:]]])
            B_bar[t,:,:] = np.block([[B],[B]])
            G_bar[t,:,:] = np.block([[G, np.zeros((n,n))], [KG[t,:,:] @ C @ G, KG[t,:,:] @ H]])
            # propagate the joint covariance
            pi[t+1,:,:] = A_bar[t,:,:] @ pi[t,:,:] @ A_bar[t,:,:].T + G_bar[t,:,:] @ block_diag(W, Sigma_v) @ G_bar[t,:,:].T
            # Extract the true state covariance alone
            S[t+1,:,:] = np.block([np.identity(n), np.zeros((n,n))]) @ pi[t+1,:,:] @  np.block([np.identity(n), np.zeros((n,n))]).T 
            
        # Update the trajectory object at time step t+1        
        for k, x_traj in enumerate(x_trajs):                                      
            x_traj.X     = x[k,:,:]
            x_traj.Sigma = S[k,:,:]
        return x_trajs   
    
    ###########################################################################
       
    def DRCollisionCheck(self, trajNode):
        """
        Performs Collision Check Using Deterministic Tightening of Distributionally Robust Chance Constraint
        Input Parameters:
        trajNode : Node containing position data which has to be checked for collision         
        """
        for alfa, (ox, oy, wd, ht) in zip(self.alfa, self.obstacleList):                        
            xrelax  = math.sqrt((1-alfa)/alfa)*LA.norm(trajNode.Sigma @ np.array([ox,0,0,0]).T)
            yrelax  = math.sqrt((1-alfa)/alfa)*LA.norm(trajNode.Sigma @ np.array([0,oy,0,0]).T)
            xdrelax = math.sqrt((1-alfa)/alfa)*LA.norm(trajNode.Sigma @ np.array([ox+wd,0,0,0]).T)
            ydrelax = math.sqrt((1-alfa)/alfa)*LA.norm(trajNode.Sigma @ np.array([0,oy+ht,0,0]).T)
            if (trajNode.X[0] >= ox - xrelax and 
               trajNode.X[0] <= ox + wd + xdrelax and 
               trajNode.X[1] >= oy - yrelax and 
               trajNode.X[1] <= oy + ht + ydrelax):
                return False    # collision has occured
        return True  # safe     
    
    ###########################################################################
    
    def LineRectangleCollision(self, fromPoint, toPoint):
        """
        Returns true or false after checking if the line joining two points intersects any obstacle
        Input Parameters:
        fromPoint : Point A
        toPoint   : Point B
        """
        fromPoint = fromPoint.X
        toPoint   = toPoint.X
        
        x1 = fromPoint[0]
        y1 = fromPoint[1]
        x2 = toPoint[0]
        y2 = toPoint[1]
        
        for ox, oy, wd, ht in self.obstacleList: 
            # Left Check
            x3 = ox
            y3 = oy
            x4 = ox
            y4 = oy + ht
            aLeft = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            bLeft = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            # Right Check
            x3 = ox + wd
            y3 = oy
            x4 = ox + wd
            y4 = oy + ht
            aRight = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            bRight = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            # Bottom Check
            x3 = ox
            y3 = oy
            x4 = ox + wd
            y4 = oy
            aBottom = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            bBottom = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            # Top Check
            x3 = ox
            y3 = oy + ht
            x4 = ox + wd
            y4 = oy + ht
            aTop = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            bTop = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            
            leftCdtn   = aLeft >= 0   and aLeft <= 1   and bLeft >= 0   and bLeft <= 1
            rightCdtn  = aRight >= 0  and aRight <= 1  and bRight >= 0  and bRight <= 1
            topCdtn    = aTop >= 0    and aTop <= 1    and bTop >= 0    and bTop <= 1 
            bottomCdtn = aBottom >= 0 and aBottom <= 1 and bBottom >= 0 and bBottom <= 1
            
            if leftCdtn or rightCdtn or topCdtn or bottomCdtn:
                return True # Line Intersects the rectangle obstacle
            
        return False # No Interection
    
    ###########################################################################
    
    def PrepareNode(self, randNode, x_trajs, t):
        """
        Prepares and returns the randNode to be added to the DR-RRT* tree
        Input Parameters:
        randNode : Node which has to prepared for addition to DR-RRT* Tree
        x_trajs  : Trajectory data containing the sequence of means and covariances
        t        : Steer Step where the collision occurred
        """
        # Convert trajNode to DR-RRT* Tree Node
        preparedNode    = Node(randNode.X[0], randNode.X[1])
        preparedNode.xd = randNode.X[2]
        preparedNode.yd = randNode.X[3]
        # Create the DR-RRT* node with sequence of means and covariances data
        for k, x_traj in enumerate(x_trajs):
            # Get the Collision Point Data
            if k == t:
                clashpoint = x_traj.X                
                clashSigma = x_traj.Sigma
            # If trajectory has not collided, use the actual trajectory data
            if k < t:                
                preparedNode.means[k,:,:] = x_traj.X                
                preparedNode.covar[k,:,:] = x_traj.Sigma
            # If trajectory has collided, use the collision data from there till the end
            elif k >= t:
                preparedNode.means[k,:,:] = clashpoint                
                preparedNode.covar[k,:,:] = clashSigma
        return preparedNode
    
    ###########################################################################
    
    def FindNearNodeIndices(self, randNode):
        """
        Returns indices of all nodes that are closer to randNode within a specified radius
        Input Parameters:
        randNode : Node around which the nearest indices have to be selected        
        """
        totalNodes   = len(self.nodeList)
        searchRadius = EnvConstant * math.sqrt((math.log(totalNodes) / totalNodes))    
        distanceList = [self.ComputeDistance(self.GetLastSequenceNode(node), randNode, euclidFlag=1) for node in self.rewireNodeList]        
        nearinds     = [distanceList.index(i) for i in distanceList if i <= searchRadius ** 2]        
        return nearinds
    
    ###########################################################################
    
    def ChooseParent(self, nearIndices, nearestNode, randNode, minNode):
        """
        Chooses the minimum cost path by selecting the correct parent
        Input Parameters:        
        nearIndices : Indices of the nodes that are nearest to the randNode
        nearestNode : DR-RRT* Node that is nearest to the sampled random node
        randNode    : Randomly sampled node
        minNode     : Node with minimum cost as of now
        """
        # If the queried node is a root node, return the same node
        if not nearIndices:
            return minNode
        costList       = []
        meanSequences  = np.zeros((len(nearIndices), STEER_TIME+1, 4, 1))
        covarSequences = np.zeros((len(nearIndices), STEER_TIME+1, 4, 4))
        for j, nearIndex in enumerate(nearIndices):            
            nearNode = self.nodeList[nearIndex]            
            # Looping except nearestNode - Uses the overwritten equality check function
            if nearNode == nearestNode:
                continue
            # Get the last sequence data as trajNode object
            nearNode = self.GetLastSequenceNode(nearNode)
            
            # Try steering from nearNode to randNodeand get the trajectory
            x_trajs = self.SteerUsingLQGControl(nearNode, randNode) 
            
            # Now check for collision along the trajectory
            for k, x_traj in enumerate(x_trajs):  
                # Check only from second time step in the steered trajectory
                if k == 0:
                    continue
                # Update the meanSequences and covarSequences
                meanSequences[j,k,:,:]  = x_traj.X
                covarSequences[j,k,:,:] = x_traj.Sigma
                # Check for DR Feasibility             
                collisionFreeFlag = self.DRCollisionCheck(x_traj)
                if not collisionFreeFlag:
                    # There is a collision, so set the distance as infinity and break                    
                    costList.append(float("inf"))
                    break             
            if collisionFreeFlag:
                # If no collision, then consider adding the cost of the minNode directly
                # Compute the recursive cost                
                self.nodeList[nearIndex].cost = self.ComputeCost(self.nodeList[nearIndex])
                # Proceed only if J[nearNode] + del*J(sigma,Pi) < J[minNode]
                d = self.ComputeDistance(self.GetLastSequenceNode(self.nodeList[nearIndex]), self.GetLastSequenceNode(minNode))
                if self.nodeList[nearIndex].cost + d < minNode.cost:
                    costList.append(self.nodeList[nearIndex].cost + d)                     
#                if self.nodeList[nearIndex].cost + DT*(STEER_TIME+1)*CT < minNode.cost:                    
#                    costList.append(self.nodeList[nearIndex].cost + DT*(STEER_TIME+1)*CT)                     
        # Update the minNode Cost and parent data                
        if costList:
            if min(costList) == float("inf"):            
                return minNode
            minIndex = costList.index(min(costList))
            minNode.cost   = min(costList)
            minNode.parent = nearIndices[minIndex]
            # Populate minNode with the new mean,covar sequence data
            minNode.means  = meanSequences[minIndex,:,:,:]
            minNode.covar  = covarSequences[minIndex,:,:,:]
        return minNode    
    
    ###########################################################################
    
    def ReWire(self, nearIndices, minNode):
        """
        Rewires the DR-RRT* Tree using Minimum cost path found
        Input Parameters:        
        nearIndices : Indices of the nodes that are nearest to the randomNode
        minNode  : Node with minimum cost paths
        """               
        meanSequences  = np.zeros((len(nearIndices), STEER_TIME+1, 4, 1))
        covarSequences = np.zeros((len(nearIndices), STEER_TIME+1, 4, 4))
        # Get all ancestors of minNode
        minNodeAncestors = self.GetAncestors(minNode)
        for j, nearIndex in enumerate(nearIndices):                       
            # Avoid looping all ancestors of minNode            
            if np.any([self.nodeList[nearIndex] == minNodeAncestor for minNodeAncestor in minNodeAncestors]):
                continue            
            # Get the last sequence data as trajNode object
            nearNode    = self.GetLastSequenceNode(self.nodeList[nearIndex])
            minLastNode = self.GetLastSequenceNode(minNode)
            # Steer from minLastNode to nearNode
            x_trajs = self.SteerUsingLQGControl(minLastNode, nearNode)                        
            for k, x_traj in enumerate(x_trajs): 
                # Check only from second time step in the steered trajectory
                if k == 0:
                    continue   
                # Update the meanSequences and covarSequences
                meanSequences[j,k,:,:]  = x_traj.X
                covarSequences[j,k,:,:] = x_traj.Sigma
                # Check for DR Feasibility                                         
                collisionFreeFlag = self.DRCollisionCheck(x_traj)
                # If there is collision, exit the loop
                if not collisionFreeFlag:  
                    break            
            if collisionFreeFlag:
                self.nodeList[nearIndex].cost = self.ComputeCost(self.nodeList[nearIndex]) 
                d = self.ComputeDistance(self.GetLastSequenceNode(self.nodeList[nearIndex]), self.GetLastSequenceNode(minNode))
                if self.nodeList[nearIndex].cost > d + minNode.cost:                    
                    # Vanilla RRT* Rewiring main code
                    self.nodeList[nearIndex].cost   = minNode.cost + d
                    self.nodeList[nearIndex].parent = self.nodeList.index(minNode) - STEER_TIME
#                    # Previous Implementations
#                    # Prepare newNode with means,covar sequences  
#                    minNodeIndex   = self.nodeList.index(minNode)                             
#                    newNode        = self.nodeList[nearIndex]
#                    newNode.means  = meanSequences[j,:,:,:]
#                    newNode.covar  = covarSequences[j,:,:,:]
#                    newNode.parent = minNodeIndex # totalNodes - 1 # should be index of minNode
#                    newNode.cost   = minNode.cost + DT*(STEER_TIME+1)*CT
#                    # Delete nearNode from DR-RRT* Tree
#                    self.nodeList.pop(nearIndex)
#                    # Add newNode with means,covar sequences to the DR-RRT* tree
#                    self.nodeList.insert(nearIndex, newNode)
    
    ###########################################################################
    
    def DrawGraph(self, randNode=None, ellipseNode=None,freeFlag=None,initialFlag=None):                
        """
        Updates the Plot with uncertainty ellipse and trajectory at each time step
        Input Parameters:
        randNode    : Node data representing the randomly sampled point 
        ellipseNode : Node representing the data to plot the covariance ellipse        
        freeFlag    : This is set for plotting the safe/collided trajectory
        initialFlag : This is set for plotting the initial position and obstacle locations
        """        
        # Get the figure object
        fig = plt.figure()
        # Plot the start position and rectangle obstacles
        if initialFlag == 1:
            # Plot the Starting position
            plt.plot(self.start.x, self.start.y, "xr")        
            plt.axis([0, 1, 0, 1])
            plt.grid(True)  
            # Plot the rectangle obstacles
            obstacles = [Rectangle(xy        = [ox, oy], 
                                   width     = wd, 
                                   height    = ht, 
                                   angle     = 0, 
                                   color     = "k", 
                                   facecolor = "k",) for (ox, oy, wd, ht) in self.obstacleList]
            for obstacle in obstacles:
                plt.axes().add_artist(obstacle)  
            
        if initialFlag is None and randNode is not None:
            # Plot the randomly sampled point 
            rx, = plt.plot(randNode.X[0], randNode.X[1], "^k")  
            # Plotting the risk bounded trajectories
            ellNodeShape = ellipseNode.means.shape
            # Prepare the trajectory x and y vectors
            xPlotValues  = []
            yPlotValues  = []
            for k in range(ellNodeShape[0]):              
                if ellipseNode is not None:  
                    xPlotValues.append(ellipseNode.means[k,0,0])
                    yPlotValues.append(ellipseNode.means[k,1,0])
            # Plot the trajectory x,y vectors 
            plt.plot(xPlotValues, yPlotValues, "-g", marker='o', markersize=2, alpha=0.8)
            # Plot only the last ellipse in the trajectory             
            k == ellNodeShape[0]-1
            # Prepare the Ellipse Object                    
            alfa     = math.atan2(ellipseNode.means[k,1,0],
                                  ellipseNode.means[k,0,0])
            elcovar  = np.asarray(ellipseNode.covar[k,:,:])            
            elE, elV = np.linalg.eig(elcovar[0:2,0:2])
            ellObj   = Ellipse(xy     = [ellipseNode.means[k,0,0], ellipseNode.means[k,1,0]], 
                               width  = math.sqrt(elE[0]), 
                               height = math.sqrt(elE[1]), 
                               angle  = alfa * 360)
            plt.axes().add_artist(ellObj)
            ellObj.set_clip_box(plt.axes().bbox)
            ellObj.set_alpha(0.5)                        
            if freeFlag == 0:
                # Red Danger Ellipse                                       
                ellObj.set_facecolor('r')
            elif freeFlag == 1:
                # Green Safe Ellipse    
                ellObj.set_facecolor('g')
#            fig.canvas.draw()
            # Creating the Animation object
            line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
                                   interval=50, blit=False)
            plt.pause(0.00001)
            rx.remove()
    
    ###########################################################################
            
    def ExpandTree(self, animation=True):        
        """
        Subroutine that grows DR-RRT* Tree 
        """ 
        
        # Plot the initial configuration
#        self.DrawGraph(initialFlag=1)                      
        
        # Update the Global Variable Optimal Cost-To-Go Matrix 
        P0 = self.CostToGo(self.initParam)  
        self.initParam.append(P0)
        
        # Add the start node to the nodeList
        self.nodeList   = [self.start]                
        
        # Iterate over the maximum allowable number of nodes
        for iter in range(self.maxIter): 
            print("Iteration no:",iter)              
                
            # Get a random feasible point in the space as a trajNode object
            randNode = self.GetRandomPoint()            
            
            # Get indices of M best DR-RRT* Tree nodes that are nearest to the random node          
            nearestIndices = self.GetNearestListIndices(randNode, M)  
            
            pathFoundFlag = False
            # Loop through all the M=5 nearestIndices
            for nearestIndex in nearestIndices:
                # Try connecting to the randomNode untill a path is found
                if pathFoundFlag:
                    break
                # Set the nearestIndex as the nearestNode and try to connect               
                self.nearestDRNode  = self.nodeList[nearestIndex] 
                
                # Get the last sequence data of nearestNode DR-RRT* Tree Node as a trajNode object
                nearestNode = self.GetLastSequenceNode(self.nearestDRNode)
                
                # Try steering from nearestNode to the random sample using steer function
                # Steer function returns a list of node points along the trajectory 
                x_trajs = self.SteerUsingLQGControl(nearestNode, randNode) 
                
                # For each point in the trajectory, check for collision with all the obstacles                      
                for k, x_traj in enumerate(x_trajs):
                    # Check only from the second time step in the trajectory
                    if k == 0:
                        continue                                
                    # Check for Distributionally Robust Feasibility of the whole trajectory
                    # collisionFreeFlag = True : Safe Trajectory, False: Unsafe Trajectory
                    collisionFreeFlag = self.DRCollisionCheck(x_traj)
                    # Add all trajectory points except the last point in the trajectory - that will be added below
                    if collisionFreeFlag and k != len(x_trajs) and not self.LineRectangleCollision(x_trajs[k], x_trajs[k-1]):
                        self.AddPathNode(x_traj)                         
                    if not collisionFreeFlag:                    
                        # Collision with obtacle happens - Add the node to the tree, update the figure and break  
                        # Create a Node with trajectory sequence data upto the collision instant                    
                        print("Planned Trajectory Collides with obstacle")                        
                        clashNode = self.PrepareNode(randNode, x_trajs, k-1)                         
                        # Third Argument 1 means it is a collision-free trajectory, 0 means it is a collision trajectory                                                 
#                        self.DrawGraph(randNode, clashNode, freeFlag=collisionFreeFlag)  
                        # self.nodeList.append(clashNode)                    
                        break                 
                if collisionFreeFlag:
                    # Entire distribution sequence was DR Feasible - So set the pathFoundFlag to 1
                    pathFoundFlag = True
                    # Record the DR-RRT* Tree node from where successfull connection was made
                    self.rewireNodeList.append(self.nearestDRNode)
                    # Distributionally Robust (Probabilistically) Safe Trajectory with no collision
                    # Create a Node with trajectory sequence data but don't add to the tree for the time being
                    # k+STEER_TIME is used for informing that no collision has occurred while creating the DR-RRT* tree node               
                    minNode  = self.PrepareNode(randNode, x_trajs, k+STEER_TIME)  
                    # Get all the nodes in the Dr-RRT* Tree that are closer to the randomNode within a specified search radius
                    nearInds = self.FindNearNodeIndices(randNode)                    
                    # Choose the minimum cost path to connect the random node
                    minNode  = self.ChooseParent(nearInds, self.nearestDRNode, randNode, minNode)
                    # Add the minNode to the DR-RRT* Tree
                    self.nodeList.append(minNode)
                    # Rewire the tree with newly added minNode                    
                    self.ReWire(nearInds, minNode)    
                    # Third Argument 1 means it is a collision-free trajectory, 0 means it is a collision trajectory                     
#                    self.DrawGraph(randNode, minNode, freeFlag=collisionFreeFlag) 
                    # Since we found a path to connect the random sample with a DR Trajectory, break.
                    break

###############################################################################
###############################################################################
###############################################################################

def main():    
    
    # Close any existing figure
    plt.close('all')
    
    # Create the DR_RRTStar Class Object by initizalizng the required data
    dr_rrtstar = DR_RRTStar(start=[0, 0], randArea=[0, 1])
    
    # Perform DR_RRTStar Tree Expansion
    dr_rrtstar.ExpandTree(animation=show_animation)    

###############################################################################

if __name__ == '__main__':
    main()
    
###############################################################################
###############################################################################
###################### END OF THE FILE ########################################
###############################################################################
###############################################################################