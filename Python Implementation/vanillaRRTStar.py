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
from matplotlib.offsetbox import AnnotationBbox, AuxTransformBox

show_animation = True


class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea,
                 expandDis=0.5, goalSampleRate=20, maxIter=600):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]
        """
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.xminrand = randArea[0]
        self.xmaxrand = randArea[1]
        self.yminrand = randArea[2]
        self.ymaxrand = randArea[3]
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList

    def Planning(self, animation=True):
        """
        Pathplanning
        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        for i in range(self.maxIter):            
            rnd = self.get_random_point()
            nind = self.GetNearestListIndex(self.nodeList, rnd)

            newNode = self.steer(rnd, nind)
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

        newNode.cost = mincost
        newNode.parent = minind

        return newNode

    def steer(self, rnd, nind):

        # expand tree
        nearestNode = self.nodeList[nind]
        theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
        newNode = Node(rnd[0], rnd[1])
        currentDistance = math.sqrt(
            (rnd[1] - nearestNode.y) ** 2 + (rnd[0] - nearestNode.x) ** 2)
        # Find a point within expandDis of nind, and closest to rnd
        if currentDistance <= self.expandDis:
            pass
        else:
            newNode.x = nearestNode.x + self.expandDis * math.cos(theta)
            newNode.y = nearestNode.y + self.expandDis * math.sin(theta)
        newNode.cost = float("inf")
        newNode.parent = None
        return newNode

    def get_random_point(self):

        if random.randint(0, 100) > self.goalSampleRate:
            rnd = [random.uniform(self.xminrand, self.xmaxrand),
                   random.uniform(self.yminrand, self.ymaxrand)]
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
        r = 10.0 * math.sqrt((math.log(nnode) / nnode))
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
            d = math.sqrt(dx ** 2 + dy ** 2)

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
        # Plot the rectangle obstacles
        obstacles = [Rectangle(xy        = [ox, oy], 
                               width     = wd, 
                               height    = ht, 
                               angle     = 0, 
                               color     = "k", 
                               facecolor = "k",) for (ox, oy, wd, ht) in self.obstacleList]
        for obstacle in obstacles:
            plt.axes().add_artist(obstacle) 

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([-1.3, 1.3, -0.3, 2.3])
        plt.grid(True)
        plt.pause(0.01)

    def GetNearestListIndex(self, nodeList, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1])
                 ** 2 for node in nodeList]
        minind = dlist.index(min(dlist))

        return minind

    def __CollisionCheck(self, node, obstacleList):
        for ox, oy, wd, ht in obstacleList:                        
            if (node.x >= ox and node.x <= ox + wd and node.y >= oy and node.y <= oy + ht):                
                return False    # collision
        return True  # safe


class Node():
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None


def main():
    print("Start " + __file__)

    # ====Search Path with RRT====
    # Obstacle Location Format [ox,oy,wd,ht]: 
    # ox, oy specifies the bottom left corner of rectangle with width: wd and height: ht
    obstacleList = [(0.3, 1.0, 0.2, 0.5),
                    (-0.5, 1.0, 0.2, 0.5),
                    (0.1, 0.5, 0.2, 0.2),
                    (-0.3, 0.5, 0.2, 0.2)] 

    # Set Initial parameters
    rrt = RRT(start=[0, 0], goal=[1.5, 1.5],
              randArea=[-1.2, 1.2, -0.1, 2.1], obstacleList=obstacleList)
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
            ax = plt.axes()                      
            for e in ells:
                ax.add_artist(e)
                e.set_clip_box(plt.axes().bbox)
                e.set_alpha(0.9)
                e.set_facecolor('r')                
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            plt.show()


if __name__ == '__main__':
    main()