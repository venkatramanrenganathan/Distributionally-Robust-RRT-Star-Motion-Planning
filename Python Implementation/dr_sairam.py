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


# Global Variables
show_animation = True  # Flag to decide to show animation or not
STEER_TIME     = 10    # Maximum Steering Time Horizon
DT             = 0.1   # Time tick(discretization time)
P0             = 0.0   # Optimal Cost-To-Go Matrix - Will be updated below
CT             = 1.0   # Minimum Path Cost - CT = f(\hat{x}, P)


class trajNode():
    """
    Class Representing a steering law trajectory Node
    """ 
    
    def __init__(self):
        self.X      = np.zeros((4, 1)) # State Vector [x position, y position, x-velocity, y-velocity]
        self.Sigma  = np.zeros((4, 4)) # sequence of covariances
        
class Node():
    """
    Class Representing a DR_RRT* Node
    """
    
    def __init__(self, x, y):
        self.x      = x                              # x position
        self.y      = y                              # y position
        self.xd     = 0.0                            # x velocity
        self.yd     = 0.0                            # y velocity
        self.cost   = 0.0                            # cost 
        self.parent = None                           # index of the parent node       
        self.means  = np.zeros((STEER_TIME+1, 2))    # sequence of means
        self.covar  = np.zeros((STEER_TIME+1, 4, 4)) # sequence of covariances
        

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
        self.init_param                     = self.getDynamicsData()
        self.start                          = Node(start[0], start[1]) # Start Node Coordinates        
        self.minrand                        = randArea[0]
        self.maxrand                        = randArea[1]
        self.expandDis                      = expandDis        
        self.maxIter                        = maxIter                
        self.obstacleList                   = self.init_param[9]        
        self.alfa                           = [0.01 + (0.05-0.01)*random.random() for i in range(len(self.obstacleList))]        
        self.start.covar[STEER_TIME-1,:,:]  = self.init_param[8]               
        
        
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
    
        # Obstacle Location Format [ox,oy,wd,ht]- ox, oy specifies the bottom left corner of rectangle with width: wd and height: ht
        obstacleList = [(5, 5, 1, 1),
                        (3, 6, 2, 1),
                        (3, 8, 1, 2),
                        (3, 10, 2, 1),
                        (7, 5, 1, 2),
                        (9, 5, 2, 1)] 
        # Pack all the data into parameter for easy access across all functions
        init_param = [A,B,C,G,Q,QT,R,W,S0,obstacleList] 
        return init_param
        

    def expandTree(self, animation=True):        
        """
        Subroutine that grows DR-RRT* Tree 
        """ 
        
        # Plot the initial configuration
        self.DrawGraph(initialFlag=1)                      
        
        # Update the Global Variable Optimal Cost-To-Go Matrix 
        P0 = self.CostToGo(self.init_param)  
        self.init_param.append(P0)
        
        # Add the start node to the nodeList
        self.nodeList = [self.start]
        
        for iter in range(self.maxIter): 
            print(iter)              
                
            # Get a random feasible point in the space as a node object
            rnd = self.get_random_point()
            
            # Get the node that is nearest to the random sample obtained             
            nind        = self.GetNearestListIndex(self.nodeList, rnd)
            nearestNode = self.nodeList[nind]   
            
            # Try steering from nearestNode to the random sample using steer function
            # Steer function returns a list of node points along the trajectory 
            x_trajs     = self.steer_LQG(nearestNode, rnd, self.init_param) 
            
            # For each point in the trajectory, check for collision with all the obstacles                      
            for x_traj in x_trajs:  
                obstacleClashFlag = self.DRCollisionCheck(x_traj, self.obstacleList)
                if not obstacleClashFlag:
                    # Collision with obtacle happens - Add the node to the tree, update the figure and break                    
                    self.nodeList.append(x_traj) # Newly added
                    # 3rd Argument 1 means it is a collision trajectory
                    self.DrawGraph(rnd, x_traj, clashFlag=1) 
                    break                
            if obstacleClashFlag:
                # Safe Trajectory with no collision  
                minNode  = self.prepareNode(rnd, x_trajs)                
                nearinds = self.find_near_nodes(x_traj)                    
                x_traj   = self.choose_parent(x_traj, nearinds)
                self.nodeList.append(x_traj)
                self.rewire(x_traj, nearinds)    
                # Third Argument 0 means it is a collision-free trajectory                                                 
                self.DrawGraph(rnd, x_traj, clashFlag=0)        
    
    def DRCollisionCheck(self, trajNode, obstacleList):
        """
        Performs Collision Check Using Deterministic Tightening of Distributionally Robust Chance Constraint
        Input Parameters:
        node         : Node containing position data which has to be checked for collision 
        obstacleList : Location information of obstacles - Format [ox,oy,wd,ht]. ox, oy - bottom left corner of rectangle with width wd and height ht
        """
        for alfa, (ox, oy, wd, ht) in zip(self.alfa, obstacleList):            
            relax   = 0.05
            xrelax  = math.sqrt((1-alfa)/alfa)*LA.norm(trajNode.Sigma @ np.array([1,0,0,0]).T) + relax
            yrelax  = math.sqrt((1-alfa)/alfa)*LA.norm(trajNode.Sigma @ np.array([0,1,0,0]).T) + relax
            xdrelax = math.sqrt((1-alfa)/alfa)*LA.norm(trajNode.Sigma @ np.array([0,0,1,0]).T) + relax
            ydrelax = math.sqrt((1-alfa)/alfa)*LA.norm(trajNode.Sigma @ np.array([0,0,0,1]).T) + relax
            if trajNode.X[0] >= ox - xrelax and 
               trajNode.X[0] <= ox + wd + xdrelax and 
               trajNode.X[1] >= oy - yrelax and 
               trajNode.X[1] <= oy + ht + ydrelax:
                return False    # collision
        return True  # safe
        
    
    def check_collision_extend(self, nearNode, theta, d):
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
            if not self.DRCollisionCheck(tmpNode, self.obstacleList):
                return False # Collision
        return True # Safe
    
    def prepareNode(self, randNode, x_trajs):
        """
        Prepares and returns the randNode to be added to the DR-RRT* tree
        Input Parameters:
        randNode : Node which has to prepared for addition to DR-RRT* Tree
        x_trajs  : Trajectory data containing the sequence of means and covariances
        """
        # Create the 
        for k, x_traj in enumerate(x_trajs):
            randNode.means[k,:]   = np.array([x_traj.X[0], x_traj.X[1]])
            randNode.covar[k,:,:] = x_traj.Sigma
        return randNode
    
    def rewire(self, newNode, nearinds):
        """
        Rewires the DR-RRT* Tree using Minimum cost path found
        Input Parameters:
        newNode  : Node whose parents have to be rewired
        nearinds : Indices of the nodes that are nearest to the newNode
        """
        nnode = len(self.nodeList)
        for i in nearinds:                       
            # Steer from newNode to self.nodeList[i]
            x_trajs_2 = self.steer_LQG(newNode, self.nodeList[i], self.init_param)            
            collision_flag = 0
            for x_traj_2 in x_trajs_2:                                            
                theta = math.atan2(x_traj_2.y-self.nodeList[i].y, x_traj_2.x-self.nodeList[i].x)
                d     = self.ComputeDistance(self.nodeList[i], x_traj_2)
                if not self.check_collision_extend(self.nodeList[i], theta, d):
                    # If there is collision, increment the collision flag and exit the loop
                    collision_flag += 1
                    break
            self.nodeList[i] = self.computeCost(self.nodeList[i])
            newNode          = self.computeCost(newNode)
            if collision_flag == 0 and self.nodeList[i].cost > newNode.cost + DT*(STEER_TIME+1)*CT:                                        
                # Proceed only if collision free and if the cost is improved
                self.nodeList[i].parent = nnode - 1
                self.nodeList[i].cost   = newNode.cost + DT*(STEER_TIME+1)*CT
    

    def choose_parent(self, newNode, nearinds):
        """
        Chooses the minimum cost path by selecting the correct parent
        Input Parameters:
        newNode  : Node whose parents have to be chosen connecting via a minimum cost path
        nearinds : Indices of the nodes that are nearest to the newNode        
        """
        # If the queried node is a root node, return the same node
        if not nearinds:
            return newNode
        dlist = []
        for i in nearinds:            
            # Try steering from newNode to self.nodeList[i]
            x_trajs_1 = self.steer_LQG(self.nodeList[i], newNode, self.init_param)
            collision_check_flag = 0
            # Now check for collision along the trajectory
            for x_traj_1 in x_trajs_1:
                theta = math.atan2(x_traj_1.y-self.nodeList[i].y, x_traj_1.x-self.nodeList[i].x)
                d     = self.ComputeDistance(self.nodeList[i], x_traj_1)
                if not self.check_collision_extend(self.nodeList[i], theta, d):
                    # If there is a collision, increment the collision check flag and exit
                    collision_check_flag += 1 
                    dlist.append(float("inf"))
                    break                
            # If no collision, that is safe then consider adding the cost of the newNode directly
            if not collision_check_flag:
                dlist.append(self.nodeList[i].cost + self.ComputeDistance(self.nodeList[i], newNode))                
        mincost = min(dlist)
        minind = nearinds[dlist.index(mincost)]
        if mincost == float("inf"):            
            return newNode
        newNode.cost   = mincost
        newNode.parent = minind
        return newNode    
    

    def get_random_point(self):
        """
        Returns a randomly sampled node from the obstacle free space
        """        
        while True:            
           # Get a random point in search space and initialize that as a Node object
           rnd = Node(random.uniform(self.minrand, self.maxrand),random.uniform(self.minrand, self.maxrand)) 
           if self.DRCollisionCheck(rnd, self.obstacleList):
               break
        return rnd         

    def find_near_nodes(self, newNode):
        """
        Returns all nodes that are closer to a given node within some specified radius
        Input Parameters:
        newNode  : Node around which the nearest indices have to be selected
        """
        nnode = len(self.nodeList)
        r = 50.0 * math.sqrt((math.log(nnode) / nnode))
        #  r = self.expandDis * 5.0
        dlist = [self.ComputeDistance(node, newNode) for node in self.nodeList]        
        nearinds = [dlist.index(i) for i in dlist if i <= r ** 2]
        return nearinds
    
    def steer_LQG(self, from_node, to_node, init_param):        
        """
        Steers from point A to point B using linear quadratic Gaussian control
        Input Parameters:
        from_node  : Node  representing Point A
        to_node    : Node  representing Point B
        init_param : List containing all the required dynamics matrices data
        """
        # Linear system model
        A   = init_param[0]
        B   = init_param[1]
        C   = init_param[2]
        G   = init_param[3]
        Q   = init_param[4]       
        QT  = init_param[5]
        R   = init_param[6]
        W   = init_param[7]             
        n,m = np.shape(B)
        T   = STEER_TIME
        fromnode = np.array([from_node.x, from_node.y, from_node.xd, from_node.yd])
        tonode   = np.array([to_node.x, to_node.y,to_node.xd, to_node.yd]) 
        x = np.array([from_node.x - to_node.x, from_node.y - to_node.y, 0 , 0])
        x = x.reshape(n, 1)  # State vector
        
        # Run dynamic programming to compute optimal controller
        K = np.zeros((m,n,T))
        k = np.zeros((m,T))
        P = np.zeros((n,n,T+1))
        p = np.zeros((n,T+1)) 
    
        # Initiliaze terminal time matrices
        P[:,:,-1] = QT
        p[:,-1]   = -np.dot(QT,tonode)

        for t in range(T-1,0,-1):
            P[:,:,t] = Q + A.T @ P[:,:,t+1]@ A - A.T @ P[:,:,t+1] @ B @ inv(R+B.T @ P[:,:,t+1] @ B) @ B.T @ P[:,:,t+1] @ A
            K[:,:,t] = -inv(R+B.T @ P[:,:,t+1]@B) @ B.T @ P[:,:,t+1]@A
            k[:,t]   = -inv(R+B.T@P[:,:,t+1]@B)@ B.T @ p[:,t+1]
            p[:,t]   = A.T @ p[:,t+1] - np.dot(Q,tonode) + K[:,:,t].T @ B.T @ p[:,t+1] + A.T @ P[:,:,t+1] @ B @ k[:,t] + K[:,:,t].T @ (R+B.T@P[:,:,t+1]@B) @ k[:,t] 
        
        # Preallocate matrices
        x          = np.zeros((n,T+1));
        u          = np.zeros((m,T));
        x[:,1]     = fromnode;        
        C          = np.identity(n)
        H          = np.identity(n)
        G          = np.identity(n)
        P_x0       = np.zeros((n,n))
        P_x_est_0  = np.zeros((n,n))
        pi_0       = block_diag(P_x0, P_x_est_0) # Joint Covariance of true and estimated states    
        S          = np.zeros((n,n,T+1))
        x_est      = np.zeros((n,T+1))
        x_est[:,1] = fromnode                      # Estimated State
        KG         = np.zeros((n,n,T+1))           # Kalman Gain
        S[:,:,1]   = from_node.covar[STEER_TIME-1, :,:] 
        A_bar      = np.zeros((2*n,2*n,T))
        B_bar      = np.zeros((2*n,m,T))
        G_bar      = np.zeros((2*n,2*n,T))
        pi         = np.zeros((2*n,2*n,T+1))
        pi[:,:,1]  = pi_0
        x_trajs    = [trajNode() for i in range(T+1)]
        
        # Steer the robot across the finite time horizon using LQG control
        for t in range(1,T):
            # Realize the measurement noise
            Sigma_v      = 0.001*np.identity(n)
            # control uses estimated state
            u[:,t]       = K[:,:,t] @ x_est[:,t] + k[:,t] 
            # Update the true state
            x[:,t+1]     = A @ x[:,t] + B @ u[:,t]
            # update the kalmann gain
            KG[:,:,t]    = S[:,:,t] @ C.T @ inv(C @ S[:,:,t] @ C.T + H @ Sigma_v @ H.T)
            # update the estimated state
            x_est[:,t+1] = KG[:,:,t] @ C @ A @ x[:,t] + (np.identity(n) - KG[:,:,t]@C) @ A @ x_est[:,t] + B @ u[:,t]
            # stack up the true and estimated states
            A_bar[:,:,t] = np.block([[A, B @ K[:,:,t]], [KG[:,:,t] @ C @ A, (np.identity(n)-KG[:,:,t] @ C) @ A + B @ K[:,:,t]]])
            B_bar[:,:,t] = np.block([[B],[B]])
            G_bar[:,:,t] = np.block([[G, np.zeros((n,n))], [KG[:,:,t] @ C @ G, KG[:,:,t] @ H]])
            # propagate the joint covariance
            pi[:,:,t+1]  = A_bar[:,:,t] @ pi[:,:,t] @ A_bar[:,:,t].T + G_bar[:,:,t] @ block_diag(W, Sigma_v) @ G_bar[:,:,t].T
            # Extract the true state covariance alone
            S[:,:,t+1]   = np.block([np.identity(n), np.zeros((n,n))]) @ pi[:,:,t+1] @  np.block([np.identity(n), np.zeros((n,n))]).T 
            
        # Update the trajectory object at time step t+1        
        for k, x_traj in enumerate(x_trajs): 
            if k == 0:
                continue                          
            x_traj.X  = x[:,k]
            x_traj.Sigma = S[:,:,k]
        return x_trajs    

    def GetNearestListIndex(self, nodeList, rnd):
        """
        Returns the index of the node in the tree that is closest to the randomly sampled node
        Input Parameters:
        nodeList  : List of all nodes in the DR-RRT* Tree
        rnd       : The randomly sampled node around which a nearest node in the tree has to be returned
        """
        dlist = [self.ComputeDistance(node,rnd) for node in nodeList]        
        minind = dlist.index(min(dlist))
        return minind   
          
    
    def CostToGo(self, init_param):
        """
        Returns the Optimal Cost-To-Go Matrix
        Input Parameters:
        init_param : List containing all the dynamics data matrices
        """
        A  = init_param[0]
        B  = init_param[1]        
        Q  = init_param[4]       
        QT = init_param[5]
        R  = init_param[6]        
        # Preallocate data structures
        n,m       = np.shape(B)
        P         = np.zeros((n,n,STEER_TIME+1))
        P[:,:,-1] = QT        
        # Compute Cost-To-Go Matrix
        for t in range(STEER_TIME-1,0,-1):
            P[:,:,t] = Q + A.T @ P[:,:,t+1]@ A - A.T @ P[:,:,t+1] @ B @ inv(R+B.T @ P[:,:,t+1] @ B) @ B.T @ P[:,:,t+1] @ A        
        P0 = P[:,:,1]        
        return P0
    
    def computeCost(self, costNode):
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
        costNode.cost = self.computeCost(costNode.parent) + DT*(STEER_TIME+1)*CT
        return costNode
    
    def ComputeDistance(self, from_node, to_node):
        """
        Returns the distance between two nodes computed using the dynamic control-based distance metric
        Input parameters:
        from_node : Node representing point A
        to_node   : Node representing point B
        """
        diff_vec = np.array([from_node.x - to_node.x, from_node.y - to_node.y, from_node.xd - to_node.xd, from_node.yd - to_node.yd])
        P0       = self.init_param[10]
        distance = diff_vec @ P0 @ diff_vec.T        
        return distance
    
    def DrawGraph(self, rnd=None, ellNode=None,clashFlag=None,initialFlag=None):        
        """        
        Updates the Plot with final course and trajectory at each time step
        Input Parameters:
        rnd        : Node data representing the randomly sampled point 
        ellNode    : Node representing the data to plot the covariance ellipse        
        clash_Flag : This is set for plotting the trajectory that has collision
        initialFlag: This is set for plotting the initial position and obstacle locations
        """
        
        # Plot the start position and rectangle obstacles
        if initialFlag == 1:
            plt.plot(self.start.x, self.start.y, "xr")        
            plt.axis([-5, 20, -5, 20])
            plt.grid(True)  
            rects = [Rectangle(xy=[ox, oy], width=wd, height=ht, angle=0, color="k", facecolor="k",) for (ox, oy, wd, ht) in self.obstacleList]
            for rect in rects:
                plt.axes().add_artist(rect)            
            
        if rnd is not None:
            # Plot the randomly sampled point 
            rx, = plt.plot(rnd.x, rnd.y, "^k")
        
            if clashFlag == 0: # Safe Trajectory
                for node in self.nodeList:
                    if node.parent is not None:   
                        plt.plot([node.x, self.nodeList[node.parent].x], 
                                 [node.y, self.nodeList[node.parent].y], "-g", alpha=0.8)                                
            elif clashFlag == 1: # Danger Trajectory
                for node in self.nodeList:
                    if node.parent is not None:   # Safe Trajectory
                        plt.plot([node.x, self.nodeList[node.parent].x], 
                                 [node.y, self.nodeList[node.parent].y], "-r", alpha=0.8)
            
            # Plot the intersecting Ellipse        
            if ellNode is not None:            
                alfa     = math.atan2(ellNode.X[1],ellNode.X[0])
                elcovar  = np.asarray(ellNode.Sigma)            
                elE, elV = np.linalg.eig(elcovar[0:2,0:2])
                ellObj   = Ellipse(xy = [ellNode.X[0], ellNode.X[1]], 
                                   width  = math.sqrt(elE[0]), 
                                   height = math.sqrt(elE[1]), 
                                   angle  = alfa * 360)
                plt.axes().add_artist(ellObj)
                ellObj.set_clip_box(plt.axes().bbox)
                ellObj.set_alpha(0.9)
                if clashFlag == 0:   # No Collision - Green Safe Trajectory Ellipses                
                    ellObj.set_facecolor('g')
                elif clashFlag == 1: # Collision - Red Danger Trajectory Ellipses
                    ellObj.set_facecolor('r')
                  
            plt.pause(0.01)
            rx.remove()


def main():    
    print("Start " + __file__)
    
    # Create the DR_RRTStar Class Object by initizalizng the required data
    dr_rrtstar = DR_RRTStar(start=[0, 0], randArea=[-5, 20])
    
    # Perform DR_RRTStar Tree Expansion
    dr_rrtstar.expandTree(animation=show_animation)
    

if __name__ == '__main__':
    main()