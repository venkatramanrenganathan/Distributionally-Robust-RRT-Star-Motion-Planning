# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:54:36 2019
@author: vxr131730 - Venkatraman Renganathan

This script simulates Path Planning with Distributionally Robust RRT*
This script is tested in Python 3.0, Windows 10, 64-bit
(C) Venkatraman Renganathan, 2019.  Email: vrengana@utdallas.edu

This program is a free software: you can redistribute it and/or modify it
under the terms of the GNU lesser General Public License, either version 
3, or any later version. This program is distributed in the hope that it 
will be useful, but WITHOUT ANY WARRANTY. 

"""

###############################################################################
###############################################################################

# Import all the required libraries
import random
import math
import copy
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
show_animation = True  # Flag to decide to show animation or not
STEER_TIME     = 10    # Maximum Steering Time Horizon
DT             = 0.1   # Time tick(discretization time)
P0             = 0.0   # Optimal Cost-To-Go Matrix - Will be updated below
CT             = 1.0   # Minimum Path Cost - CT = f(\hat{x}, P)
EnvConstant    = 50.0  # Environment Constant - Used in computing search radius

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
        self.cost   = 0.0                            # cost 
        self.parent = None                           # index of the parent node       
        self.means  = np.zeros((STEER_TIME+1, 4, 1)) # sequence of means
        self.covar  = np.zeros((STEER_TIME+1, 4, 4)) # sequence of covariances
    
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

    def __init__(self, start, randArea, expandDis=0.5, maxIter=300):
        """
        Input Parameters:
        start   :Start Position [x,y]
        goal    :Goal Position [x,y]        
        randArea:Ramdom Samping Area [min,max]
        """
        # Add the Double Integrator Data    
        self.controlPenalty                 = 0.02
        self.initParam                     = self.getDynamicsData()
        self.start                          = Node(start[0], start[1]) # Start Node Coordinates        
        self.minrand                        = randArea[0]
        self.maxrand                        = randArea[1]
        self.expandDis                      = expandDis        
        self.maxIter                        = maxIter                
        self.obstacleList                   = self.initParam[9]        
        self.alfa                           = [0.01 + (0.05-0.01)*random.random() for i in range(len(self.obstacleList))]        
        self.start.covar[STEER_TIME-1,:,:]  = self.initParam[8]               
        
    ###########################################################################
    
    def getDynamicsData(self):
        """
        Returns the Dynamics data and obstacle details as one packed parameter
        """
        # Double Integrator Data                               
        A  = np.array([[1,0,DT,0],
                       [0, 1, 0, DT],
                       [0,0,1,0],
                       [0,0,0,1]])                                          # System Dynamics Matrix 
        
        B  = np.array([[(DT**2)/2, 0],
                        [0, (DT**2)/2],
                        [DT, 0],
                        [0,DT]])                                            # Input Matrix
        C  = np.array([[1,0,0,0],
                       [0,1,0,0]])                                          # Output Matrix
        G  = B                                                              # Disturbance Input Matrix
        Q  = np.block([[4*np.identity(2), np.zeros((2,2))],
                        [np.zeros((2,2)), 0.1*np.identity(2)]])             # State Stage cost Penalty
        QT = np.block([[100*np.identity(2), np.zeros((2,2))],
                        [np.zeros((2,2)), 0.1*np.identity(2)]])             # State Terminal Penalty    
        R  = self.controlPenalty * np.identity(2)                           # Control/Input Penalty 
        W  = np.block([[np.zeros((2,2)), np.zeros((2,2))],
                        [np.zeros((2,2)), 0.001*np.array([[2,1],[1,2]])]])  # Disturbance covariance    
        S0 = np.block([[0.001*np.identity(2), np.zeros((2,2))],
                        [np.zeros((2,2)), np.zeros((2,2))]])                # Initial State Covariance                           
    
        # Obstacle Location Format [ox,oy,wd,ht]: ox, oy specifies the bottom left corner of rectangle with width: wd and height: ht
        obstacleList = [(6, 6, 1, 1),
                        (2, 5, 2, 1),
                        (3, 7, 1, 2),
                        (1, 9, 2, 1),
                        (7, 8, 1, 2),
                        (9, 2, 2, 1)] 
        # Pack all the data into parameter for easy access across all functions
        initParam = [A,B,C,G,Q,QT,R,W,S0,obstacleList] 
        return initParam
        
    ###########################################################################

    def ExpandTree(self, animation=True):        
        """
        Subroutine that grows DR-RRT* Tree 
        """ 
        
        # Plot the initial configuration
        self.DrawGraph(initialFlag=1)                      
        
        # Update the Global Variable Optimal Cost-To-Go Matrix 
        P0 = self.CostToGo(self.initParam)  
        self.initParam.append(P0)
        
        # Add the start node to the nodeList
        self.nodeList = [self.start]
        
        for iter in range(self.maxIter): 
            print("Iteration no:",iter)              
                
            # Get a random feasible point in the space as a trajNode object
            randNode = self.GetRandomPoint()
            
            # Get the DR-RRT* Tree node that is nearest to the random node          
            nearestIndex = self.GetNearestListIndex(randNode)
            nearestDRNode  = self.nodeList[nearestIndex] 
            
            # Get the last sequence data of nearestNode DR-RRT* Tree Node as a trajNode object
            nearestNode = self.GetLastSequenceNode(nearestDRNode)
            
            # Try steering from nearestNode to the random sample using steer function
            # Steer function returns a list of node points along the trajectory 
            x_trajs = self.SteerUsingLQGControl(nearestNode, randNode) 
            
            # For each point in the trajectory, check for collision with all the obstacles                      
            for k, x_traj in enumerate(x_trajs):
                # Check only from the second time step in the trajectory
                if k == 0:
                    continue                
                # Check for Distributionally Robust Feasibility of the whole trajectory
                # collisionFreeFlag = True : Safe Trajectory
                # collisionFreeFlag = False: Unsafe Trajectory
                collisionFreeFlag = self.DRCollisionCheck(x_traj)
                if not collisionFreeFlag:
                    # Collision with obtacle happens - Add the node to the tree, update the figure and break  
                    # Create a Node with trajectory sequence data upto the collision instant
                    clashNode = self.PrepareNode(randNode, x_trajs, k-1) 
                    # Third Argument 1 means it is a collision-free trajectory, 0 means it is a collision trajectory                                                 
                    self.DrawGraph(randNode, clashNode, freeFlag=collisionFreeFlag)  
                    self.nodeList.append(clashNode)                    
                    break                
            if collisionFreeFlag:
                # Distributionally Robust (Probabilistically) Safe Trajectory with no collision
                # Create a Node with trajectory sequence data but don't add to the tree for the time being
                # k+STEER_TIME will make sure that only trajectory data is used for creating the DR-RRT* tree node               
                minNode  = self.PrepareNode(randNode, x_trajs, k+STEER_TIME)  
                # Get all the nodes in the Dr-RRT* Tree that are closer to the randomNode within a specified search radius
                nearinds = self.FindNearNodes(randNode)                    
                # Choose the minimum cost path to connect the random node
                minNode  = self.ChooseParent(nearinds, nearestDRNode, randNode, minNode)
                # Add the minNode to the DR-RRT* Tree
                self.nodeList.append(minNode)
                # Rewire the tree with newly added minNode
                self.ReWire(nearinds, minNode)    
                # Third Argument 1 means it is a collision-free trajectory, 0 means it is a collision trajectory                                                 
                self.DrawGraph(randNode, minNode, freeFlag=collisionFreeFlag)        
    
    ###########################################################################
    
    def DRCollisionCheck(self, trajNode):
        """
        Performs Collision Check Using Deterministic Tightening of Distributionally Robust Chance Constraint
        Input Parameters:
        node         : Node containing position data which has to be checked for collision         
        """
        for alfa, (ox, oy, wd, ht) in zip(self.alfa, self.obstacleList):            
            relax   = 0.5
            xrelax  = math.sqrt((1-alfa)/alfa)*LA.norm(trajNode.Sigma @ np.array([1,0,0,0]).T) + relax
            yrelax  = math.sqrt((1-alfa)/alfa)*LA.norm(trajNode.Sigma @ np.array([0,1,0,0]).T) + relax
            xdrelax = math.sqrt((1-alfa)/alfa)*LA.norm(trajNode.Sigma @ np.array([0,0,1,0]).T) + relax
            ydrelax = math.sqrt((1-alfa)/alfa)*LA.norm(trajNode.Sigma @ np.array([0,0,0,1]).T) + relax
            if trajNode.X[0] >= ox - xrelax and trajNode.X[0] <= ox + wd + xdrelax and trajNode.X[1] >= oy - yrelax and trajNode.X[1] <= oy + ht + ydrelax:
                return False    # collision has occured
        return True  # safe 
        
    ###########################################################################
    
    def CheckCollisionExtend(self, nearNode, theta, d):
        """
        Subroutine that extends DRCollisionCheck module
        Input Parameters:
        nearNode : Node to be checked for Collison
        theta    : Slope of the line connecting two points
        d        : Length of the line connecting two points
        """
        # Function returns TRUE if there is NO collision and FALSE if there is collision
        tmpNode = copy.deepcopy(nearNode)
        for i in range(int(d / self.expandDis)):
            tmpNode.X[0] += self.expandDis * math.cos(theta)
            tmpNode.X[1] += self.expandDis * math.sin(theta)
            if not self.DRCollisionCheck(tmpNode):
                return False # Collision
        return True # Safe
    
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
    
    def ReWire(self, nearIndices, minNode):
        """
        Rewires the DR-RRT* Tree using Minimum cost path found
        Input Parameters:        
        nearIndices : Indices of the nodes that are nearest to the randomNode
        minNode  : Node with minimum cost paths
        """
        totalNodes = len(self.nodeList)
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
                # Check for DR collision                         
                theta    = math.atan2(x_traj.X[1] - nearNode.X[1], x_traj.X[0] - nearNode.X[0])
                distance = self.ComputeDistance(nearNode, x_traj)
                collisionFreeFlag = self.CheckCollisionExtend(nearNode, theta, distance)
                if not collisionFreeFlag: # If there is collision, exit the loop 
                    break            
            if collisionFreeFlag:
                self.nodeList[nearIndex] = self.ComputeCost(self.nodeList[nearIndex])
                #minNode                  = self.ComputeCost(minNode)
                if self.nodeList[nearIndex].cost > minNode.cost + DT*(STEER_TIME+1)*CT:                                                                                
                    # Prepare newNode with means,covar sequences
                    minNodeIndex   = self.nodeList.index(minNode)                  
                    newNode        = self.nodeList[nearIndex]
                    newNode.means  = meanSequences[j,:,:,:]
                    newNode.covar  = covarSequences[j,:,:,:]
                    newNode.parent = totalNodes - 1 # should be index of minNode
                    newNode.cost   = minNode.cost + DT*(STEER_TIME+1)*CT
                    # Delete nearNode from DR-RRT* Tree
                    self.nodeList.pop(nearIndex)
                    # Add newNode with means,covar sequences to the DR-RRT* tree
                    self.nodeList.append(newNode)

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
                # Check for DR collision
                theta    = math.atan2(x_traj.X[1] - nearNode.X[1], x_traj.X[0] - nearNode.X[0])
                distance = self.ComputeDistance(nearNode, x_traj)
                collisionFreeFlag = self.CheckCollisionExtend(nearNode, theta, distance)
                if not collisionFreeFlag:
                    # There is a collision, so set the distance as infinity and break                    
                    costList.append(float("inf"))
                    break             
            if collisionFreeFlag:
                # If no collision, that is safe then consider adding the cost of the minNode directly
                # Compute the recursive cost
                #minNode                  = self.ComputeCost(minNode)
                self.nodeList[nearIndex] = self.ComputeCost(self.nodeList[nearIndex])
                # Proceed only if J[nearNode] + del*J(sigma,Pi) < J[minNode]
                if self.nodeList[nearIndex].cost + DT*(STEER_TIME+1)*CT < minNode.cost:                    
                    costList.append(self.nodeList[nearIndex].cost + DT*(STEER_TIME+1)*CT)                     
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
    
    def RandFreeCheck(self, randNode):
        """
        Performs Collision Check For Random Sampled Point
        Input Parameters:
        randNode : Node containing position data which has to be checked for collision 
        """
        for ox, oy, wd, ht in self.obstacleList:            
            relax = 0.5 + max(self.alfa) # Conservative estimate used here - Can also use DR CHECK - But not needed
            if randNode.X[0] >= ox - relax and randNode.X[0] <= ox + wd + relax and randNode.X[1] >= oy - relax and randNode.X[1] <= oy + ht + relax:
                return False    # collision
        return True  # safe

    ###########################################################################
    
    def FindNearNodes(self, randNode):
        """
        Returns indices of all nodes that are closer to randNode within a specified radius
        Input Parameters:
        randNode : Node around which the nearest indices have to be selected
        """
        totalNodes   = len(self.nodeList)
        searchRadius = EnvConstant * math.sqrt((math.log(totalNodes) / totalNodes))    
        distanceList = [self.ComputeDistance(self.GetLastSequenceNode(node), randNode) for node in self.nodeList]        
        nearinds     = [distanceList.index(i) for i in distanceList if i <= searchRadius ** 2]
        return nearinds
    
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

        # Run Backward Propagation
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
        for t in range(1,T):            
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
            if k == 0:
                continue                          
            x_traj.X  = x[k,:,:]
            x_traj.Sigma = S[k,:,:]
        return x_trajs   
    

    ###########################################################################
    
    def GetNearestListIndex(self, randNode):
        """
        Returns the index of the node in the tree that is closest to the randomly sampled node
        Input Parameters:        
        randNode  : The randomly sampled node around which a nearest node in the DR-RRT* tree has to be returned
        """
        dlist = [self.ComputeDistance(self.GetLastSequenceNode(node),randNode) for node in self.nodeList]        
        minind = dlist.index(min(dlist))
        return minind  
    
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
        P         = np.zeros((n,n,STEER_TIME+1))
        P[:,:,-1] = QT        
        # Compute Cost-To-Go Matrix
        for t in range(STEER_TIME-1,0,-1):
            P[:,:,t] = Q + A.T @ P[:,:,t+1]@ A - A.T @ P[:,:,t+1] @ B @ inv(R+B.T @ P[:,:,t+1] @ B) @ B.T @ P[:,:,t+1] @ A        
        P0 = P[:,:,1]        
        return P0
    
    ###########################################################################
    
    def ComputeCost(self, costNode):
        """
        Returns the node after associating the cost into it using recursion
        Input parameters:
        costNode : Node whose cost has to be calculated        
        """
        # If the queried node is a parent node, set the cost to zero & return
        # Else recursively compute the cost as J[N] = J[N_parent] + del*J(sigma,pi)
        if costNode.parent is None:
            costNode.cost = 0
            return costNode
        costNode.cost = self.ComputeCost(costNode.parent) + DT*(STEER_TIME+1)*CT
        return costNode
    
    ###########################################################################
    
    def ComputeDistance(self, fromNode, toNode):
        """
        Returns the distance between two nodes computed using the dynamic control-based distance metric
        Input parameters:
        fromNode : Node representing point A
        toNode   : Node representing point B
        """
        diffVec = fromNode.X - toNode.X
        P0       = self.initParam[10]
        distance = diffVec.T @ P0 @ diffVec        
        return distance
    
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
        # Plot the start position and rectangle obstacles
        if initialFlag == 1:
            # Plot the Starting position
            plt.plot(self.start.x, self.start.y, "xr")        
            plt.axis([-5, 20, -5, 20])
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
            for k in range(ellNodeShape[0]):
                if k == 0:
                    continue
                # Prepare the Ellipse Object
                if ellipseNode is not None:                     
                    alfa     = math.atan2(ellipseNode.means[k,1,0],ellipseNode.means[k,0,0])
                    elcovar  = np.asarray(ellipseNode.covar[k,:,:])            
                    elE, elV = np.linalg.eig(elcovar[0:2,0:2])
                    ellObj   = Ellipse(xy = [ellipseNode.means[k,0,0], ellipseNode.means[k,1,0]], 
                                       width  = math.sqrt(elE[0]), 
                                       height = math.sqrt(elE[1]), 
                                       angle  = alfa * 360)
                    plt.axes().add_artist(ellObj)
                    ellObj.set_clip_box(plt.axes().bbox)
                    ellObj.set_alpha(0.9)
                    # Plot trajectory and the intersecting Ellipse at time step k
                    if freeFlag == 0:
                        # Unsafe Trajectory 
                        plt.plot([ellipseNode.means[k,0,0], ellipseNode.means[k-1,0,0]], 
                                 [ellipseNode.means[k,1,0], ellipseNode.means[k-1,1,0]], "-r", alpha=0.8) 
                        # Collision - Red Danger Trajectory Ellipse               
                        ellObj.set_facecolor('r')
                    elif freeFlag == 1:
                        # Safe Trajectory
                        plt.plot([ellipseNode.means[k,0,0], ellipseNode.means[k-1,0,0]], 
                                 [ellipseNode.means[k,1,0], ellipseNode.means[k-1,1,0]], "-g", alpha=0.8) 
                        # No Collision - Green Safe Trajectory Ellipse    
                        ellObj.set_facecolor('g')
                plt.pause(0.001)
            rx.remove()

###############################################################################
###############################################################################
###############################################################################

def main():    
    
    # Create the DR_RRTStar Class Object by initizalizng the required data
    dr_rrtstar = DR_RRTStar(start=[0, 0], randArea=[-5, 20])
    
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