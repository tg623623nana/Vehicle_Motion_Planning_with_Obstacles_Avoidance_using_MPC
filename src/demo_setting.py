# demo_setting.py
"""
Setting poblem (map and obstacles) for demo
Created on 2022/12/13
@author: Pin-Yun Hung
"""
from model_map import mapModel
import numpy as np
from obca import obca

class problemSetting:

    def __init__(self, demo_name):

        # demo_name
        self.demo_name = demo_name

        # demo_type
        self.demo_type1 = np.array(['demo1', 'demo5'])
        self.demo_type2 = np.array(['demo2', 'demo3', 'demo4'])
        self.demo_type3 = np.array(['demo6', 'demo7'])
        self.demo_type4 = np.array(['demo8'])
        self.demo_type = ''
        self.update_demoType(demo_name)

        # map info
        self.map_size = [0, 0] # horizon, vertical, = array v * h
        self.xL = [0, 0] # the smallest x, y of map
        self.xU = [0, 0] # the largest x, y of map

        # start & goal position: [x, y, theta]
        self.startPose = [0, 0, 0]
        self.goalPose = [0, 0, 0]

        # obstacle
        self.nObs = 0
        self.vObs = 0
        self.lObs = 0
        self.obs_info = [] # obstacle informaiton:
                           ## start center & angle,
                           ## length, width (moving direction is length)
                           ## const_velocity
                           ## end center & angle
                           ## start_time, end time

        # static obstacle info
        self.static_nObs = 0 # obstacle number
        self.static_vObs = 0 # vertex number of each obstacle
        self.static_lObs = [] # vertex location (x, y)
        self.static_gridlObs = []

        # dynamic obstacle info
        self.dyn_nObs = 0
        self.dyn_vObs = 0
        self.dyn_lObs = []
        self.dyn_obs_info = []

        # Mpc
        self.obca = obca
        self.terminal_set = []

        # problem initialize
        self.set_problem(demo_name)

        # grid map
        self.resolution = 1  # unit: meter
        self.mapClass = mapModel(self.map_size, self.resolution)
        self.org_gridMap = self.mapClass.shape2grid([], self.static_gridlObs)
        self.grid_map = self.org_gridMap
        self.senseDis = 10

    def set_problem(self, demo_name):
        """
        Initialize problem setting
        :param demo_name: the name of demo
        :return:
        """

        ############################
        # Problem Definition
        ############################
        if demo_name == 'demo1':
            # map info
            self.xL = [0, 0]
            self.xU = [39, 10]
            self.map_size = [(self.xU[0] - self.xL[0]) + 1, (self.xU[1] - self.xL[1]) + 1]

            # start & goal
            self.startPose = [3, 4, 0]
            self.goalPose = [38, 4, 0]

            # static obstacle (known obstacle, i.e. wall)
            self.static_lObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1]],
                                [[10, 1], [10, 5], [15, 5], [15, 1], [10, 1]],
                                [[0, 1], [self.xU[0], 1]]]

            self.static_gridlObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1], [0, self.xU[1]], [self.xU[0], self.xU[1]]],
                                   [[10, 1], [10, 5], [15, 5], [15, 1], [10, 1]],
                                   [[0, 1], [self.xU[0], 1], [self.xU[0], 0], [0, 0]]]

            # dynamic obstacle
            self.dyn_obs_info = [[22.5, 0, np.pi / 2, 3, 3, 0.2, 22.5, 9, np.pi / 2, 0, 55]]

            # terminal set
            self.terminal_set = np.array([[25, 39], [1, 9]])

        elif demo_name == 'demo2':
            # map info
            self.xL = [0, 0]
            self.xU = [39, 10]
            self.map_size = [(self.xU[0] - self.xL[0]) + 1, (self.xU[1] - self.xL[1]) + 1]

            # start & goal
            self.startPose = [3, 4, 0]
            self.goalPose = [38, 4, 0]

            # static obstacle (known obstacle, i.e. wall)
            self.static_lObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1]],
                                [[25, 8], [25, 3], [20, 3], [20, 8], [25, 8]],
                                [[0, 1], [self.xU[0], 1]]]

            self.static_gridlObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1], [0, self.xU[1]], [self.xU[0], self.xU[1]]],
                                    [[25, 8], [25, 3], [20, 3], [20, 8], [25, 8]],
                                    [[0, 1], [self.xU[0], 1], [self.xU[0], 0], [0, 0]]]
            # dynamic obstacle
            self.dyn_obs_info = [[18.5, 0, np.pi / 2, 3, 3, 0.2, 18.5, 9, np.pi / 2, 0, 55]]

            # terminal set
            self.terminal_set = np.array([[25, 39], [1, 9]])

        elif demo_name == 'demo3':
            # map info
            self.xL = [0, 0]
            self.xU = [39, 10]
            self.map_size = [(self.xU[0] - self.xL[0]) + 1, (self.xU[1] - self.xL[1]) + 1]

            # start & goal
            self.startPose = [3, 4, 0]
            self.goalPose = [38, 4, 0]

            # static obstacle (known obstacle, i.e. wall)
            self.static_lObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1]],
                                [[25, 8], [25, 3], [20, 3], [20, 8], [25, 8]],
                                [[0, 1], [self.xU[0], 1]]]

            self.static_gridlObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1], [0, self.xU[1]], [self.xU[0], self.xU[1]]],
                                    [[25, 8], [25, 3], [20, 3], [20, 8], [25, 8]],
                                    [[0, 1], [self.xU[0], 1], [self.xU[0], 0], [0, 0]]]
            # dynamic obstacle
            self.dyn_obs_info = [[18.5, 0, np.pi / 2, 3, 3, 0.15, 18.5, 9, np.pi / 2, 0, 55]]

            # terminal set
            self.terminal_set = np.array([[25, 39], [1, 9]])

        elif demo_name == 'demo4':
            # map info
            self.xL = [0, 0]
            self.xU = [39, 10]
            self.map_size = [(self.xU[0] - self.xL[0]) + 1, (self.xU[1] - self.xL[1]) + 1]

            # start & goal
            self.startPose = [3, 4, 0]
            self.goalPose = [38, 4, 0]

            # static obstacle (known obstacle, i.e. wall)
            self.static_lObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1]],
                                [[25, 8], [25, 3], [20, 3], [20, 8], [25, 8]],
                                [[0, 1], [self.xU[0], 1]]]

            self.static_gridlObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1], [0, self.xU[1]], [self.xU[0], self.xU[1]]],
                                    [[25, 8], [25, 3], [20, 3], [20, 8], [25, 8]],
                                    [[0, 1], [self.xU[0], 1], [self.xU[0], 0], [0, 0]]]
            # dynamic obstacle
            self.dyn_obs_info = [[18.5, 0, np.pi / 2, 3, 3, 0.1, 18.5, 9, np.pi / 2, 0, 55]]

            # terminal set
            self.terminal_set = np.array([[25, 39], [1, 9]])

        elif demo_name == 'demo5':
            # map info
            self.xL = [0, 0]
            self.xU = [39, 10]
            self.map_size = [(self.xU[0] - self.xL[0]) + 1, (self.xU[1] - self.xL[1]) + 1]

            # start & goal
            self.startPose = [3, 4, 0]
            self.goalPose = [38, 4, 0]

            # static obstacle (known obstacle, i.e. wall)
            self.static_lObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1]],
                                [[10, 1], [10, 5], [15, 5], [15, 1], [10, 1]],
                                [[0, 1], [self.xU[0], 1]]]

            self.static_gridlObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1], [0, self.xU[1]], [self.xU[0], self.xU[1]]],
                                   [[10, 1], [10, 5], [15, 5], [15, 1], [10, 1]],
                                   [[0, 1], [self.xU[0], 1], [self.xU[0], 0], [0, 0]]]

            # dynamic obstacle
            self.dyn_obs_info = [[22.5, 0, np.pi / 2, 3, 3, 0.1, 22.5, 9, np.pi / 2, 0, 55]]

            # terminal set
            self.terminal_set = np.array([[25, 39], [1, 9]])

        elif demo_name == 'demo6':
            # map info
            self.xL = [0, 0]
            self.xU = [39, 10]
            self.map_size = [(self.xU[0] - self.xL[0]) + 1, (self.xU[1] - self.xL[1]) + 1]

            # start & goal
            self.startPose = [3, 4, 0]
            self.goalPose = [38, 4, 0]

            # static obstacle (known obstacle, i.e. wall)
            self.static_lObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1]], [[0, 1], [self.xU[0], 1]]]

            self.static_gridlObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1], [0, self.xU[1]], [self.xU[0], self.xU[1]]],
                                   [[0, 1], [self.xU[0], 1], [self.xU[0], 0], [0, 0]]]

            # dynamic obstacle
            self.dyn_obs_info = [[13.5, 0, np.pi / 2, 3, 3, 0.2, 13.5, 9, np.pi / 2, 0, 100], [22.5, 0, np.pi / 2, 3, 3, 0.1, 22.5, 9, np.pi / 2, 0, 200]]

            # terminal set
            self.terminal_set = np.array([[25, 39], [1, 9]])

        elif demo_name == 'demo7':
            # map info
            self.xL = [0, 0]
            self.xU = [39, 10]
            self.map_size = [(self.xU[0] - self.xL[0]) + 1, (self.xU[1] - self.xL[1]) + 1]

            # start & goal
            self.startPose = [3, 4, 0]
            self.goalPose = [38, 4, 0]

            # static obstacle (known obstacle, i.e. wall)
            self.static_lObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1]], [[0, 1], [self.xU[0], 1]]]

            self.static_gridlObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1], [0, self.xU[1]], [self.xU[0], self.xU[1]]],
                                   [[0, 1], [self.xU[0], 1], [self.xU[0], 0], [0, 0]]]

            # dynamic obstacle
            self.dyn_obs_info = [[13.5, 0, np.pi / 2, 3, 3, 0.1, 13.5, 9, np.pi / 2, 0, 100], [22.5, 0, np.pi / 2, 3, 3, 0.05, 22.5, 9, np.pi / 2, 0, 200]]

            # terminal set
            self.terminal_set = np.array([[28, 39], [1, 9]])

        elif demo_name == 'demo11':
            # map info
            self.xL = [0, 0]
            self.xU = [80, 10]
            self.map_size = [(self.xU[0] - self.xL[0]) + 1, (self.xU[1] - self.xL[1]) + 1]

            # start & goal
            self.startPose = [3, 4, 0]
            self.goalPose = [77, 4, 0]

            # static obstacle (known obstacle, i.e. wall)
            self.static_lObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1]], [[0, 1], [self.xU[0], 1]]]

            self.static_gridlObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1], [0, self.xU[1]], [self.xU[0], self.xU[1]]],
                                   [[0, 1], [self.xU[0], 1], [self.xU[0], 0], [0, 0]]]

            # dynamic obstacle
            self.dyn_obs_info = [[30.5, 0, np.pi / 2, 3, 3, 0.1, 30.5, 9, np.pi / 2, 0, 100], [39.5, 9, -np.pi / 2, 3, 3, 0.1, 39.5, 0, -np.pi / 2, 0, 200]]

            # terminal set
            self.terminal_set = np.array([[25, 39], [2, 6]])

        elif demo_name == 'demo9':
            # map info
            self.xL = [0, 0]
            self.xU = [40, 60]
            self.map_size = [(self.xU[0] - self.xL[0]) + 1, (self.xU[1] - self.xL[1]) + 1]

            # start & goal
            self.startPose = [1, 5, 0] # 3, 5, np.pi/4
            self.goalPose = [37, 58, np.pi/2] # 37, 58, np.pi/2

            # static obstacle (known obstacle, i.e. wall)
            self.static_lObs = [[[8, 0], [8, 6], [40, 6]],
                                [[12, 30], [34, 30], [34, 14], [12, 14], [12, 30]],
                                [[13, 49], [34, 49], [34, 34], [13, 34], [13, 49]],
                                [[4, 60], [4, 10], [0, 10]],
                                [[33, 60], [33, 55], [4, 55]]]

            self.static_gridlObs = [[[8, 6], [40, 6], [40, 0], [8, 0]],
                                    [[12, 30], [34, 30], [34, 14], [12, 14]],
                                    [[12, 50], [34, 50], [34, 34], [12, 34]],
                                    [[0, 60], [4, 60], [4, 10], [0, 10]],
                                    [[4, 60], [34, 60], [34, 54], [4, 54]]]

            # dynamic obstacle
            self.dyn_obs_info = [[8, 50, -np.pi / 2, 2, 2, 0.5, 8, 10, -np.pi / 2, 0, 100]]

            # terminal set
            self.terminal_set = np.array([[34, 40], [54, 60]])

        elif demo_name == 'demo10':
            # map info
            self.xL = [0, 0]
            self.xU = [99, 10]
            self.map_size = [(self.xU[0] - self.xL[0]) + 1, (self.xU[1] - self.xL[1]) + 1]

            # start & goal
            self.startPose = [3, 4, 0]
            self.goalPose = [98, 4, 0]

            # static obstacle (known obstacle, i.e. wall)
            self.static_lObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1]], [[0, 1], [self.xU[0], 1]]]

            self.static_gridlObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1], [0, self.xU[1]], [self.xU[0], self.xU[1]]],
                                   [[0, 1], [self.xU[0], 1], [self.xU[0], 0], [0, 0]]]

            # dynamic obstacle
            self.dyn_obs_info = [[99, 5, -np.pi, 3, 3, 0.5, 0, 5, -np.pi, 0, 100]]

            # terminal set
            self.terminal_set = np.array([[60, 99], [1, 9]])

        elif demo_name == 'demo8':
            # map info
            self.xL = [0, 0]
            self.xU = [39, 10]
            self.map_size = [(self.xU[0] - self.xL[0]) + 1, (self.xU[1] - self.xL[1]) + 1]

            # start & goal
            self.startPose = [3, 4, 0]
            self.goalPose = [38, 4, 0]

            # static obstacle (known obstacle, i.e. wall)
            self.static_lObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1]], [[0, 1], [self.xU[0], 1]]]

            self.static_gridlObs = [[[self.xU[0], self.xU[1] - 1], [0, self.xU[1] - 1], [0, self.xU[1]], [self.xU[0], self.xU[1]]],
                                   [[0, 1], [self.xU[0], 1], [self.xU[0], 0], [0, 0]]]

            # dynamic obstacle
            self.dyn_obs_info = [[13.5, 0, np.pi / 2, 3, 3, 0.1, 13.5, 9, np.pi / 2, 0, 100], [22.5, 9, -np.pi / 2, 3, 3, 0.1, 22.5, 0, -np.pi / 2, 0, 200]]

            # terminal set
            self.terminal_set = np.array([[25, 39], [2, 6]])

        else:
            self.customize_setting([])
            # lObs = [[[39, 9], [0, 9]], [[0, 0], [39, 0]]]  # can pass under the object
            # lObs = [[[39, 9], [0, 9]], [[25, 8], [25, 3], [20, 3], [20, 8], [25, 8]], [[0, 0], [39, 0]]] # can pass under the object
            # lObs = [[[39, 9], [0, 9], [39, 10]], [[25, 8], [25, 3.5], [20, 3.5], [20, 8]], [[0, 0], [39, 0]]] # pass under the object # based on start point
            # lObs = [[[39, 9], [0, 9], [39, 10]], [[25, 5], [25, 0], [20, 0], [20, 5], [25, 5]], [[0,0], [39, 0]]] # can't pass upper the object
            # lObs = [[[39, 9], [0, 9], [39, 10]], [[25, 7], [25, 2], [20, 2], [20, 7], [25, 7]], [[0,0], [39, 0]]] # test both
            # lObs = [[[39, 9], [0, 9], [39, 10]], [[25, 8], [25, 3], [20, 3], [20, 8], [25, 8]],
            #         [[13, 5], [13, 0], [10, 0], [10, 5], [13, 5]], [[0, 0], [39, 0]]] # strict
            # lObs = [[[39, 9], [0, 9], [39, 10]], [[25, 8], [25, 4], [20, 4], [20, 8], [25, 8]],
            #         [[13, 5], [13, 0], [10, 0], [10, 5], [13, 5]], [[0, 0], [39, 0]]] # soft
            # lObs = [[[39, 9], [0, 9], [39, 10]], [[25, 8], [25, 3], [23, 3], [20, 5], [20, 8], [25, 8]], [[0, 0], [39, 0]]] # test not horzon or vertocal obstacle
            # lObs = [[[39, 9], [1, 9]], [[21, 1.5], [24, 1.5], [24, -1.5], [21, -1.5], [21, 1.5]], [[0, 1], [39, 1]]]

        ############################
        # Problem Setting
        ############################

        # get static obstacle number & each vertex number
        self.static_nObs = np.size(self.static_lObs, 0)

        self.static_vObs = np.ones(self.static_nObs, dtype=int)
        for i in range(self.static_nObs):
            self.static_vObs[i] = np.size(self.static_lObs[i], 0)

        # get dynamic obstacle number, each vertex number & location
        self.add_dynamic_obstacle(self.dyn_obs_info)

    def customize_setting(self, map_size, start, goal, static_lObs, dyn_lObs, dyn_obs_info):
        return

    def add_dynamic_obstacle(self, dyn_obs_info):
        """
        Generate the initial vertex of dynamic obstacle & update dynamic obstacle parameter.
        - obstacle represent as rectangle
        - length define at moving direction
        :param dyn_obs_info: [start_center_x, start_center_y, start_theta,
                              length, width,
                              constant_velocity, constant_angular_velocity,
                              end_center_x, end_center_y, end_theta,
                              start_time, end_time]
        :return:
        """

        dyn_lObs = []
        for i in range(np.size(dyn_obs_info, 0)):
            cx = dyn_obs_info[i][0]
            cy = dyn_obs_info[i][1]
            theta = dyn_obs_info[i][2]
            l = dyn_obs_info[i][3]
            w = dyn_obs_info[i][4]

            obs_initial_vertex = self.get_obstacle(cx, cy, theta, l, w)
            # print(obs_initial_vertex)
            dyn_lObs.append(obs_initial_vertex)

        # update dynamic obstacle parameter
        self.dyn_nObs = np.size(dyn_lObs, 0)
        self.dyn_vObs = np.ones(np.size(dyn_lObs, 0), dtype=int) * 5
        self.dyn_lObs = dyn_lObs
        self.dyn_obs_info = dyn_obs_info

    def get_obstacle(self, center_x, center_y, theta, length, width):
        """
        Generate vertex of rectangle obstacle in clockwise
        :param center_x: x value at the center of the obstacle
        :param center_y: y value at the center of the obstacle
        :param theta: heading angle of the obstacle, unit: rad
        :param length: obstacle's length, define at the moving direction
        :param width: obstacle's width
        :return: obs_vertex = obstacle vertex [left_buttom, left_top, right_top, right_buttom]
        """
        cx = center_x
        cy = center_y
        l = length / 2
        w = width / 2

        # vertex: left_buttom, left_top, right_top, right_buttom
        vertex_1 = [cx - l * np.cos(theta) - w * np.sin(theta), cy - l * np.sin(theta) + w * np.cos(theta)]
        vertex_2 = [cx + l * np.cos(theta) - w * np.sin(theta), cy + l * np.sin(theta) + w * np.cos(theta)]
        vertex_3 = [cx + l * np.cos(theta) + w * np.sin(theta), cy + l * np.sin(theta) - w * np.cos(theta)]
        vertex_4 = [cx - l * np.cos(theta) + w * np.sin(theta), cy - l * np.sin(theta) - w * np.cos(theta)]

        # combine obs_vertex
        obs_vertex = [vertex_1, vertex_2, vertex_3, vertex_4, vertex_1]

        return obs_vertex

    def combine_obstacle(self, dynObs_exist):
        nObs = self.static_nObs
        vObs = []
        lObs = []
        obs_info = []
        # always add static obstacle first
        for i in range(np.size(self.static_vObs, 0)):
            vObs.append(self.static_vObs[i])
        for i in range(np.size(self.static_lObs, 0)):
            lObs.append(self.static_lObs[i])
        for i in range(self.static_nObs):
            obs_info.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # then add dynamic obstacle
        if dynObs_exist == 1:
            for i in range(np.size(self.dyn_vObs, 0)):
                vObs.append(self.dyn_vObs[i])
            for i in range(np.size(self.dyn_lObs, 0)):
                lObs.append(self.dyn_lObs[i])
            for i in range(self.dyn_nObs):
                obs_info.append(self.dyn_obs_info[i])
                nObs = self.static_nObs + self.dyn_nObs

        # rebuild obstacle to N+1 times
        return nObs, vObs, lObs, obs_info

    def rebuild_lObs(self, N, Ts, dynObs_exist):
        nObs, vObs, lObs, obs_info = self.combine_obstacle(dynObs_exist)

        new_lObs = []
        for k in range(N + 1):
            for i in range(nObs):
                lObs_ith = []
                for j in range(vObs[i]):
                    lObs_x = lObs[i][j][0] + Ts * obs_info[i][5] * np.cos(obs_info[i][2]) * k
                    lObs_y = lObs[i][j][1] + Ts * obs_info[i][5] * np.sin(obs_info[i][2]) * k
                    lObs_ith.append([lObs_x, lObs_y])
                new_lObs.append(lObs_ith)

        self.nObs = nObs
        self.vObs = vObs
        self.lObs = new_lObs
        self.obs_info = obs_info

    def update_demoType(self, demo_name):
        if demo_name in self.demo_type1:
            self.demo_type = 'demoType1'
        elif demo_name in self.demo_type2:
            self.demo_type = 'demoType2'
        elif demo_name in self.demo_type3:
            self.demo_type = 'demoType3'
        elif demo_name in self.demo_type4:
            self.demo_type = 'demoType4'





