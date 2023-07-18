# model_obstacle.py
"""
Set up the static and dynamic obstacle
Created on 2022/12/4
@author: Pin-Yun Hung
"""
import numpy as np
import matplotlib.pyplot as plt

class obstacleModel:

    def obstacle_V_Represent(self, obstacles):
        # obstacles are defined by vertices, which are assumed to be enumerated in clock-wise direction
        #     	[ 	[[obst1_x1;obst1_y1],[obst1_x2;obst1_y2],[obst1_x3;obst1_y4],...,[obst1_x1;obst1_y1]]..]

        nOb = np.size(obstacles, 0)
        vOb = np.ones(nOb) * 5
        lOb = []

        obstacle_x = obstacles[:, 2]  # horizon length
        obstacle_y = obstacles[:, 3]  # vertical length
        obstacle_loc = obstacles[:, 0:2]

        for i in range(nOb):
            obstacle_loc_x = obstacle_loc[i, 1] - 0.5
            obstacle_loc_y = obstacle_loc[i, 0] - 0.5
            obs_vertex = [[obstacle_loc_x, obstacle_loc_y],
                          [obstacle_loc_x + obstacle_x[i], obstacle_loc_y],
                          [obstacle_loc_x + obstacle_x[i], obstacle_loc_y + obstacle_y[i]],
                          [obstacle_loc_x, obstacle_loc_y + obstacle_y[i]],
                          [obstacle_loc_x, obstacle_loc_y]]

            lOb.append(obs_vertex)

        return lOb

    def obstacle_H_Represent(self, nOb, vOb, obstacle_vertex):

        # nOb = np.size(obstacle_vertex, 0)
        # vOb = np.ones(nOb, dtype=int) * 5
        lOb = obstacle_vertex

        # these matrices contain the H-rep
        A_all = np.zeros((sum(vOb) - nOb, 2))
        b_all = np.zeros((sum(vOb) - nOb, 1))

        # counter for lazy people
        lazyCounter = 0;

        # building H-rep
        for i in range(0, nOb):
            A_i = np.zeros((vOb[i] - 1, 2)) # n * 2
            b_i = np.zeros((vOb[i] - 1, 1)) # n * 1

            # take two subsequent vertices, and compute hyperplane
            for j in range(0, vOb[i] - 1):

                # extract two vertices
                v1 = lOb[i][j]  # vertex 1
                v2 = lOb[i][j + 1]  # vertex 2

                # find hyperplane passing through v1 and v2
                if v1[0] == v2[0]: # perpendicular hyperplane, not captured by general formula
                    if v2[1] < v1[1]:
                        A_tmp = [1, 0]
                        b_tmp = v1[0]
                    else:
                        A_tmp = [-1, 0]
                        b_tmp = -v1[0]
                elif v1[1] == v2[1]: # horizontal hyperplane, captured by general formula but included for numerical stability
                    if v1[0] < v2[0]:
                        A_tmp = [0, 1]
                        b_tmp = v1[1]
                    else:
                        A_tmp = [0, -1]
                        b_tmp = -v1[1]
                else: # general formula for non-horizontal and non-vertical hyperplanes
                    # ab = [[v1[0], 1], [v2[0], 1]] / [v1[1], v2[1]]
                    # a = ab[0]
                    # b = ab[1]
                    a = (v2[1] - v1[1]) / (v2[0] - v1[0])
                    b = v1[1] - a * v1[0]

                    if v1[0] < v2[0]: # v1 --> v2 (line moves right)
                        A_tmp = [-a, 1]
                        b_tmp = b
                    else: # v2 <-- v1 (line moves left)
                        A_tmp = [a, -1]
                        b_tmp = -b

                # store vertices
                A_i[j, :] = A_tmp
                b_i[j] = b_tmp

            # store everything
            A_all[lazyCounter: lazyCounter + vOb[i] - 1, :] = A_i
            b_all[lazyCounter: lazyCounter + vOb[i] - 1] = b_i

            # update counter
            lazyCounter = lazyCounter + vOb[i] - 1

        return A_all, b_all

    def plot(self, lOb):

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        lOb = np.asarray(lOb)
        for i in range(np.size(lOb, 0)) :
            ax.plot(lOb[i, :, 0], lOb[i, :, 1], '-ob')

        plt.show()

    def check(self, l1, l2, sq):
        feas = False
        # step 1 check if end point is in the square
        if (l1[0] >= sq[0] and l1[1] >= sq[1] and l1[0] <= sq[2] and l1[1] <= sq[3]) or \
                (l2[0] >= s1[0] and l2[1] >= s1[1] and l2[0] <= sq[2] and l2[1] <= sq[3]):
            return feas
        else:
            # step 2 check if diagonal cross the segment
            p1 = [sq[0], sq[1]]
            p2 = [sq[2], sq[3]]
            p3 = [sq[2], sq[1]]
            p4 = [sq[0], sq[3]]
            if segment(l1, l2, p1, p2) or segment(l1, l2, p3, p4):
                return feas
            else:
                feas = True
                return feas