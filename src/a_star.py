# a_star.py
"""
Generate a pre-plan path without consider the dynamic obstacle
Created on 2022/12/2
@author: Pin-Yun Hung & Yan

2022/12/6 Add a_star function which is written by Yan
"""
import heapq
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
from scipy import interpolate

class a_star:

    def __init__(self, array, start, goal):

        self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        self.close_set = set()

        self.came_from = {}

        self.gscore = {start: 0}

        self.fscore = {start: self.heuristic(start, goal)}

        self.oheap = []

        heapq.heappush(self.oheap, (self.fscore[start], start))


    def heuristic(self, a, b):

        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def solve(self, array, start, goal):
        """
        Generate path
        :param array: the grid map (type:np.array with only 0 & 1, 0 = free region, 1 = forbidden region)
        :param start: start position (type: integer)
        :param goal: target position (type: integer)
        :return:
        """

        while self.oheap:

            current = heapq.heappop(self.oheap)[1]

            if current == goal:

                data = []

                while current in self.came_from:
                    data.append(current)

                    current = self.came_from[current]

                return data

            self.close_set.add(current)

            for i, j in self.neighbors:

                neighbor = current[0] + i, current[1] + j

                tentative_g_score = self.gscore[current] + self.heuristic(current, neighbor)

                if 0 <= neighbor[0] < array.shape[0]:

                    if 0 <= neighbor[1] < array.shape[1]:

                        if array[neighbor[0]][neighbor[1]] == 1:
                            continue

                    else:

                        # array bound y walls

                        continue

                else:

                    # array bound x walls

                    continue

                if neighbor in self.close_set and tentative_g_score >= self.gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < self.gscore.get(neighbor, 0) or neighbor not in [i[1] for i in self.oheap]:
                    self.came_from[neighbor] = current

                    self.gscore[neighbor] = tentative_g_score

                    self.fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)

                    heapq.heappush(self.oheap, (self.fscore[neighbor], neighbor))

        return False

    def plot(self, route, grid, start, goal):

        route = route + [start]

        route = route[::-1]

        # plot map and path

        x_coords = []

        y_coords = []

        for i in (range(0, len(route))):
            x = route[i][0]

            y = route[i][1]

            x_coords.append(x)

            y_coords.append(y)

        fig, ax = plt.subplots(figsize=(12, 12))

        ax.imshow(grid, cmap=plt.cm.Dark2)

        ax.scatter(start[1], start[0], marker="*", color="yellow", s=200)

        ax.scatter(goal[1], goal[0], marker="*", color="red", s=200)

        ax.plot(y_coords, x_coords, color="black")

        plt.show()

    def rebuild_path(self, route):

        # since a star path is backward and x, y inverse
        route = np.asarray(route)

        path = []
        path_num = np.size(route, 0)
        for i in range(0, path_num):
            path.append([route[path_num - 1 - i][1], route[path_num - 1 - i][0]])

        return path

    def interpolate(self, path, step_size):
        xData = []
        yData = []
        new_path = []

        path_num = np.size(path, 0)
        xData = np.asarray(path)[:, 0]
        yData = np.asarray(path)[:, 1]

        for i in range(path_num - 1):
            if xData[i + 1] == xData[i]:
                y = []
                if yData[i] > yData[i + 1]:
                    y = np.arange(start=yData[i + 1], stop=np.max(yData[i]), step=step_size)
                    y = np.flip(y)
                else:
                    y = np.arange(start=yData[i], stop=np.max(yData[i + 1]), step=step_size)

                for j in range(0, np.size(y, 0)):
                    new_path.append([xData[i], y[j]])
            # elif yData[i + 1] == yData[i]:
            #     x = np.arange(start=xData[i], stop=np.max(xData[i + 1]), step=step_size)
            #     for j in range(0, np.size(y, 0)):
            #         new_path.append([x[j], yData[i]])
            else:
                xx = np.array([xData[i], xData[i + 1]])
                yy = np.array([yData[i], yData[i + 1]])
                x = np.arange(start=xData[i], stop=np.max(xData[i + 1]), step=step_size)
                f_interp = interpolate.interp1d(xx, yy, 'linear')
                y = f_interp(x)
                for j in range(0, np.size(x, 0)):
                    new_path.append([x[j], y[j]])

        path = new_path
        # a_star_path = np.asarray(path)
        # plt.plot(a_star_path[:, 0], a_star_path[:, 1], '.-b')
        # plt.show()

        return path

    def create_reference_path(self, path):

        ref_trajectory = []
        path_num = np.size(path, 0)

        for i in range(path_num - 1):
            yaw = np.arctan2(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0])
            ref_trajectory.append([path[i][0], path[i][1], yaw])

        ref_trajectory.append([path[path_num - 1][0], path[path_num - 1][1], ref_trajectory[-1][2]])

        return ref_trajectory

    def demo_data(self):
        # grid map
        grid_map = np.array([

            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

            [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],

            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        # start point and goal
        start = (0, 0)
        goal = (0, 19)

        return grid_map, start, goal




