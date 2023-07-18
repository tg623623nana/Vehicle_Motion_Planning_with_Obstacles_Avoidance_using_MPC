# model_map.py
"""
Define the demo map
Created on 2022/12/4
@author: Pin-Yun Hung
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors
from skimage.morphology import erosion, dilation, disk


class mapModel:

    def __init__(self ,map_size, resolution):
        self.grid_map = np.zeros((int((map_size[1] - 1) / resolution) + 1, int((map_size[0] - 1) / resolution) + 1))

        self.resolution = resolution

    def shape2grid(self, org_gridMap, obstacle_location):
        """
        Trun shape represent map to grid map
        :param obstacle_location: is rectangle
                                  = [ [left_buttom_vertex, left_Top_vertex, right_top_vertex, right_buttom_vertex], []... ]
                                  base on heading angle
        :param resolution:
        :return: gird_map
        """

        obs_loc = obstacle_location
        if org_gridMap == []:
            gird_map = self.grid_map
        else:
            gird_map = org_gridMap

        obstacle_num = np.size(obs_loc, 0)

        # obstacle reorder to base on world and map coordinate
        obs_loc = self.reOrderVertex(obs_loc)
        obs_loc = self.world2gridmap(obs_loc)

        # add obstacle
        for n in range(0, obstacle_num):
            x_length = int((obs_loc[n][2][0] - obs_loc[n][0][0])) + 1
            y_length = int((obs_loc[n][2][1] - obs_loc[n][0][1])) + 1
            x = int(obs_loc[n][0][0])
            y = int(obs_loc[n][0][1])

            for i in range(x_length):
                for j in range(y_length):
                    obs_x = x + i
                    obs_y = y + j
                    gird_map[obs_y, obs_x] = 1

        return gird_map

    def world2gridmap(self, obstacle):

        obs = []
        for i in range(np.size(obstacle, 0)):
            x_min = (obstacle[i][0][0] / self.resolution)
            y_min = (obstacle[i][0][1] / self.resolution)

            x_max = (obstacle[i][2][0] / self.resolution)
            y_max = (obstacle[i][2][1] / self.resolution)

            obs_vertex = [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]
            obs.append(obs_vertex)

        return obs

    def w2m(self, point):

        obs = []
        for i in range(np.size(obstacle, 0)):
            x_min = (obstacle[i][0][0] / self.resolution)
            y_min = (obstacle[i][0][1] / self.resolution)

            x_max = (obstacle[i][2][0] / self.resolution)
            y_max = (obstacle[i][2][1] / self.resolution)

            obs_vertex = [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]
            obs.append(obs_vertex)

        return obs

    def reOrderVertex(self, obstacles):

        obs = []
        for i in range(np.size(obstacles, 0)):
            x_min = min([row[0] for row in obstacles[i]])
            x_max = max([row[0] for row in obstacles[i]])

            y_min = min([row[1] for row in obstacles[i]])
            y_max = max([row[1] for row in obstacles[i]])

            obs_vertex = [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]
            obs.append(obs_vertex)

        return obs

    def dilate_map(self, grid_map, dilation_level):
        kernal = disk(dilation_level)
        map_dilate =dilation(grid_map, kernal)

        return map_dilate

    def erode_map(self, grid_map, erosion_level):
        kernal = disk(erosion_level)
        map_eroded = erosion(grid_map, kernal)

        return map_eroded

    def plot(self, grid_map):

        # setting
        colormap = colors.ListedColormap(["white", "black"])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(111)
        ax1.imshow(grid_map, cmap=colormap)
        plt.show()




