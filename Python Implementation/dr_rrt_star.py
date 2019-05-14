"""
Path Planning Sample Code with RRT*
author: Venkatraman Renganathan
"""

import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from scipy.linalg import block_diag
from numpy.linalg import inv


# Global Variables
show_animation = True  # Flag to decide to show animation or not
STEER_TIME     = 10.0  # Maximum Steering Time Horizon
DT             = 0.1   # Time tick(discretization time)


class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, init_param, randArea,
                 expandDis=0.5, goalSampleRate=20, maxIter=300):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]
        """
        self.start          = Node(start[0], start[1])  # Start Node Coordinates
        self.end            = Node(goal[0], goal[1])      # Goal Node Coordinates
        self.minrand        = randArea[0]
        self.maxrand        = randArea[1]
        self.expandDis      = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter        = maxIter
        self.obstacleList   = obstacleList 
        self.start.covar    = init_param[8]
        
        # Double Integrator Data    
        self.init_param   = init_param       
        

    def Planning(self, animation=True):
        """
        Pathplanning
        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        
        for i in range(self.maxIter):            
            rnd = self.get_random_point()
            nind = self.GetNearestListIndex(self.nodeList, rnd)

            newNode = self.steer(rnd, nind, self.init_param)
            #  print(newNode.cost)

            if self.__CollisionCheck(newNode, self.obstacleList):
                nearinds = self.find_near_nodes(newNode)
                newNode = self.choose_parent(newNode, nearinds)
                self.nodeList.append(newNode)
                self.rewire(newNode, nearinds)

            if animation and i % 10 == 0:
                print ("Iteration No.", round(i/10))
                self.DrawGraph(rnd)

        # generate course
        lastIndex = self.get_best_last_index()
        if lastIndex is None:
            return None
        path = self.gen_final_course(lastIndex)
        return path

    def choose_parent(self, newNode, nearinds):
        if not nearinds:
            return newNode

        dlist = []
        for i in nearinds:
            dx = newNode.x - self.nodeList[i].x
            dy = newNode.y - self.nodeList[i].y
            d = math.sqrt(dx ** 2 + dy ** 2)
            theta = math.atan2(dy, dx)
            if self.check_collision_extend(self.nodeList[i], theta, d):
                dlist.append(self.nodeList[i].cost + d)
            else:
                dlist.append(float("inf"))

        mincost = min(dlist)
        minind = nearinds[dlist.index(mincost)]

        if mincost == float("inf"):
            print("mincost is inf")
            return newNode

        newNode.cost   = mincost
        newNode.parent = minind

        return newNode
    
    def LQGplanning(self, from_node, to_node, init_param):        
    
        # Linear system model
        A = init_param[0]
        B = init_param[1]
        C = init_param[2]
        Q = init_param[3]       
        R = init_param[4]
        W = init_param[5]
        G = init_param[6]
        QT = init_param[7]
        
        n,m = np.shape(B)
        T   = STEER_TIME
        rx, ry = [from_node.x], [from_node.y]
        tonode = np.array([to_node.x, to_node.y,0,0])
        x = np.array([from_node.x - to_node.x, from_node.y - to_node.y, 0 , 0]).reshape(n, 1)  # State vector
        
        # Run dynamic programming to compute optimal controller
        K = np.zeros((m,n,T))
        k = np.zeros((m,T))
        P = np.zeros((n,n,T+1))
        p = np.zeros((n,T+1)) 
    
        P[:,:,T+1] = QT
        p[:,T+1]   = -np.dot(QT,tonode)

        for t in range(T,0,-1):
            P[:,:,t] = Q + A.T @ P[:,:,t+1]@ A - A.T @ P[:,:,t+1] @ B @ inv(R+B.T @ P[:,:,t+1] @ B) @ B.T @ P[:,:,t+1] @ A
            K[:,:,t] = -inv(R+B.T @ P[:,:,t+1]@B) @ B.T @ P[:,:,t+1]@A
            k[:,t]   = -inv(R+B.T@P[:,:,t+1]@B)@ B.T @ p[:,t+1]
            p[:,t]   = A.T@p[:,t+1]- np.dot(Q,tonode) + K[:,:,t].T @ B.T @ p[:,t+1] + transpose(A) @ P[:,:,t+1] @ B @ k[:,t] + K[:,:,t].T @ (R+B.T@P[:,:,t+1]@B) @ k[:,t] 
        
        # Preallocate matrices
        x          = np.zeros((n,T+1));
        u          = np.zeros((m,T));
        x[:,1]     = tonode;
        V          = np.identity(n)
        C          = np.identity(n)
        H          = np.identity(n)
        G          = np.identity(n)
        P_x0       = np.zeros((n))
        P_x_est_0  = np.zeros((n))
        pi_0       = block_diag(P_x0, P_x_est_0) # Joint Covariance of true and estimated states
        Sigma_V    = np.zeros((n,n,T))
        S          = np.zeros((n,n,T+1))
        x_est      = np.zeros((n,T+1))
        x_est[:,1] = tonode                      # Estimated State
        KG         = np.zeros(n,n,T+1)           # Kalman Gain
        S[:,:,1]   = from_node.covar 
        A_bar      = np.zeros((2*n,2*n,T))
        pi         = np.zeros((2*n,2*n,T+1))
        pi[:,:,1]  = pi_0
        
        # Steer the robot across the finite time horizon using LQG control
        for t in range(1,T+1):
            # Realize the measurement noise
            Sigma_v      = 0.001*np.identity(n)
            # control uses estimated state
            u[:,t]       = K[:,:,t] @ x_est[:,t] + k[:,t] 
            # Update the true state
            x[:,t+1]     = A @ x[:,t] + B @ u[:,t]
            # update the kalmann gain
            KG[:,:,t]    = S[:,:,t] @ C.T @ inv(C @ S[:,:,t] @ C.T + H @ Sigma_V @ H.T)
            # update the estimated state
            x_est[:,t+1] = KG[:,:,t] @ C @ A @ x[:,t] + (np.identity(n) - KG[:,:,t]@C) @ A @ x_est[:,t] + B @ u[:,t]
            # stack up the true and estimated states
            A_bar[:,:,t] = np.array([[A, B @ K[:,:,t]], [KG[:,:,t]@C@A, (np.identity(n)-KG[:,:,t]@C)@A+B@K[:,:,t]]])
            B_bar[:,:,t] = np.array([[B],[B]])
            G_bar[:,:,t] = np.array([[G, np.zeros((n,n))], [KG[:,:,t]@C@G, KG[:,:,t]@H]])
            # propagate the joint covariance
            pi[:,:,t+1]  = A_bar[:,:,t]@pi[:,:,t]@A_bar[:,:,t].T + G_bar[:,:,t] @ block_diag(W, Sigma_V) @ G_bar[:,:,t].T
            # Extract the true state covariance alone
            S[:,:,t+1]   = np.array([np.identity(n), np.zeros((n,n))]) @ pi[:,:,t+1] @  np.array([np.identity(n), np.zeros((n,n))]).T 
            
        return x, S

    def steer(self, rnd, nind, init_param):
        
        # expand tree
        nearestNode = self.nodeList[nind]                
        x, S = self.LQGplanning(nearestNode, rnd, init_param)
        
        return x, S

    def get_random_point(self):

        if random.randint(0, 100) > self.goalSampleRate:
            rnd = [random.uniform(self.minrand, self.maxrand),
                   random.uniform(self.minrand, self.maxrand)]
        else:  # goal point sampling
            rnd = [self.end.x, self.end.y]

        return rnd

    def get_best_last_index(self):

        disglist = [self.calc_dist_to_goal(
            node.x, node.y) for node in self.nodeList]
        goalinds = [disglist.index(i) for i in disglist if i <= self.expandDis]

        if not goalinds:
            return None

        mincost = min([self.nodeList[i].cost for i in goalinds])
        for i in goalinds:
            if self.nodeList[i].cost == mincost:
                return i

        return None

    def gen_final_course(self, goalind):
        path = [[self.end.x, self.end.y]]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            path.append([node.x, node.y])
            goalind = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.end.x, y - self.end.y])

    def find_near_nodes(self, newNode):
        nnode = len(self.nodeList)
        r = 50.0 * math.sqrt((math.log(nnode) / nnode))
        #  r = self.expandDis * 5.0
        dlist = [(node.x - newNode.x) ** 2 +
                 (node.y - newNode.y) ** 2 for node in self.nodeList]
        nearinds = [dlist.index(i) for i in dlist if i <= r ** 2]
        return nearinds

    def rewire(self, newNode, nearinds):
        nnode = len(self.nodeList)
        for i in nearinds:
            nearNode = self.nodeList[i]

            dx = newNode.x - nearNode.x
            dy = newNode.y - nearNode.y
            d  = math.sqrt(dx ** 2 + dy ** 2)

            scost = newNode.cost + d

            if nearNode.cost > scost:
                theta = math.atan2(dy, dx)
                if self.check_collision_extend(nearNode, theta, d):
                    nearNode.parent = nnode - 1
                    nearNode.cost = scost

    def check_collision_extend(self, nearNode, theta, d):

        tmpNode = copy.deepcopy(nearNode)

        for i in range(int(d / self.expandDis)):
            tmpNode.x += self.expandDis * math.cos(theta)
            tmpNode.y += self.expandDis * math.sin(theta)
            if not self.__CollisionCheck(tmpNode, self.obstacleList):
                return False

        return True

    def DrawGraph(self, rnd=None):
        """
        Draw Graph
        """
        plt.clf()
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [
                         node.y, self.nodeList[node.parent].y], "-g", alpha=0.2)

        rects = [Rectangle(xy=[ox, oy], width=wd, height=ht, angle=0, color="k", facecolor="k",) for (ox, oy, wd, ht) in self.obstacleList]
        for rect in rects:
                plt.axes().add_artist(rect)
        
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    def GetNearestListIndex(self, nodeList, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1])
                 ** 2 for node in nodeList]
        minind = dlist.index(min(dlist))

        return minind

    def __CollisionCheck(self, node, obstacleList):
        relax_param = 0.5 # need design this in a distributionally robust way
        for (ox, oy, wd, ht) in obstacleList: # ox,oy,wd,ht - specifies the bottom left corner of rectangle with width: wd and height: ht.            
            if node.x >= ox - relax_param and node.x <= ox + wd + relax_param and node.y >= oy - relax_param and node.y <= oy + ht + relax_param:
                return False    # collision            
            
        return True  # safe


class Node():
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.covar  = np.zeros((4,4))
        self.cost   = 0.0
        self.parent = None
        


def main():
    print("Start " + __file__)
    
    # Double Integrator Data                               
    A  = np.array([[1,0,DT,0],[0, 1, 0, DT],[0,0,1,0],[0,0,0,1]])  # Dynamics Matrix
    B  = np.array([[(DT**2)/2, 0],[0, (DT**2)/2],[DT, 0],[0,DT]])  # Input Matrix
    C  = np.array([[1,0,0,0],[0,1,0,0]])                           # Output Matrix
    G  = B                                                         # Disturbance Input Matrix
    Q  = np.array([[4*np.zeros((2,2)), np.zeros((2,2))],[np.zeros((2,2)), 0.1*np.identity(2)]])           # State Stage cost Penalty
    QT = np.array([[100*np.zeros((2,2)), np.zeros((2,2))],[np.zeros((2,2)), 0.1*np.identity(2)]])         # State Terminal Penalty
    R  = 0.02*np.identity(2)                                                                              # Input Penalty 
    W  = np.array([[np.zeros((2,2)), np.zeros((2,2))],[np.zeros((2,2)), 0.001*np.array([[2,1],[1,2]])]])  # Disturbance covariance    
    S0 = np.array([[0.001*np.zeros((2,2)), np.zeros((2,2))],[np.zeros((2,2)), np.zeros((2,2))]])          # Initial State Covariance
    init_param = [A,B,C,G,Q,QT,R,W,S0]

    # ====Search Path with RRT====
    obstacleList = [
        (5, 5, 1, 1),
        (3, 6, 2, 1),
        (3, 8, 1, 2),
        (3, 10, 2, 1),
        (7, 5, 1, 2),
        (9, 5, 2, 1)
    ]  # Obstacle Location Format [ox,oy,wd,ht]- ox, oy specifies the bottom left corner of rectangle with width: wd and height: ht.]

    # Set Initial parameters
    rrt  = RRT(start=[0, 0], goal=[7, 9], randArea=[-2, 15], obstacleList=obstacleList, init_param=init_param)
    path = rrt.Planning(animation=show_animation)

    if path is None:
        print("Cannot find path!!")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt.DrawGraph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-b',linewidth=3.0)
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            ells = [Ellipse(xy=[x, y],
                width=np.random.rand(), height=np.random.rand(),
                angle=np.random.rand() * 360) for (x, y) in path]                              
            for e in ells:
                plt.axes().add_artist(e)
                e.set_clip_box(plt.axes().bbox)
                e.set_alpha(0.9)
                e.set_facecolor('r')                
            plt.axes().set_xlim(-2, 15)
            plt.axes().set_ylim(-2, 15)
            plt.show()


if __name__ == '__main__':
    main()