# simulation.py
"""

Created on 2022/12/4
@author: Pin-Yun Hung
"""
import numpy as np
from demo_setting import problemSetting
from draw import plotClass
from matplotlib.patches import Rectangle
from model_obstacle import obstacleModel
from closed_loop import closedLoop
import matplotlib.pyplot as plt
import time
from ttictoc import tic,toc
from a_star import a_star

class simulation:

    def run(self, demo_name):
        """
        Run the Open Loop MPC
        Recommand Setting:
            Q_free = 0.01, R_free = [0.01, 0.1], P_free = Q_free, N_free = 10
            Q_fix = 0.0001, R_fix = [0.01, 1], P_fix = Q_fix, N_fix = 20
        :param demo_name:
        :return:
        """

        demo = problemSetting(demo_name)
        draw = plotClass(demo, [], [])
        # draw.plot_map(dynObs_exist=0) # view the map

        #####################
        # MPC
        #####################
        mpc = closedLoop(demo)

        # free time
        mpc.mpc_openLoop_freeTime()
        if mpc.feas == True:
            sim_title = 'FullDimension with Topt & no dynObs avoid'
            file_name =  'FullDim_BigTopt_' + demo.demo_name
            draw.fullDimension_animate(mpc, mpc.N_free, sim_title, file_name)
            print('MPC1 -- Success')
            print(mpc.Ts_opt)
        else:
            print('MPC1 -- Failed')
            print(mpc.Ts_opt)
            mpc.Ts_opt = [] # if fail, will not be able to do the second MPC

        # fix time
        mpc.mpc_openLoop_fixTime()
        if mpc.feas == True:
            sim_title = 'FullDimension with fix_T & dynObs avoid'
            file_name = 'FullDim_fixT_dynObsAvoid_' + demo.demo_name + '_NTopt%i' % mpc.N_free + '_predN%i' % mpc.N_fix
            draw.fullDimension_animate(mpc, mpc.N_fix, sim_title, file_name)
            print('MPC3 -- Success')
            print(mpc.Ts_opt)
        else:
            print('MPC3 -- Failed')
            print(mpc.Ts_opt)

    def run_closedLoop(self, demo_name):
        """
        Run the Closed Loop MPC
        Recommand Setting:
        "demo 9":
            Q_free = 0.5, R_free = [0.01, 0.1], P_free = Q_free, N_free = 5
            Q_fix = 0.001, R_fix = [0.01, 1], P_fix = Q_fix, N_fix = 5
            self.senseDis = 8
            self.terminal_set = np.array([[5, 30], [self.x0[1] + 4, 60]])
            self.dyn_obs_info = [[8, 50, -np.pi / 2, 2, 2, 0.5, 8, 10, -np.pi / 2, 0, 100]]
            reference  -5

        "demo 10":
            Q_free = 0.1, R_free = [0.01, 0.1], P_free = Q_free, N_free = 15
            Q_fix = 0.001, R_fix = [0.01, 1], P_fix = Q_fix, N_fix = 15
            self.senseDis = 12
            self.terminal_set = np.array([[self.x0[0] + 6, 99], [1, 9]])
            self.dyn_obs_info = [[99, 5, -np.pi, 3, 3, 0.5, 0, 5, -np.pi, 0, 100]]

            reference  -5

        "demo 8":
            Q_free = 0.1, R_free = [0.01, 0.1], P_free = Q_free, N_free = 15
            Q_fix = 0.001, R_fix = [0.01, 1], P_fix = Q_fix, N_fix = 15
            self.senseDis = 12
            self.terminal_set = np.array([[self.x0[0] + 6, 99], [1, 9]])
            self.dyn_obs_info = [[13.5, 0, np.pi / 2, 3, 3, 0.1, 13.5, 9, np.pi / 2, 0, 100], [22.5, 9, -np.pi / 2, 3, 3, 0.1, 22.5, 0, -np.pi / 2, 0, 200]]
            reference  -3

        "demo 1":
            Q_free = 0.1, R_free = [0.01, 0.1], P_free = Q_free, N_free = 6
            Q_fix = 0.001, R_fix = [0.01, 1], P_fix = Q_fix, N_fix = 6
            self.senseDis = 10
            self.terminal_set = np.array([[self.x0[0] + 5, 99], [1, 9]])
            self.dyn_obs_info = [[22.5, 0, np.pi / 2, 3, 3, 0.2, 22.5, 9, np.pi / 2, 0, 55]]
            reference  -5

        :param demo_name:
        :return:
        """

        demo = problemSetting(demo_name)
        # closed_mpc = mpcModel('freeTime', demo, mpc_type='diff', ref_N=None, ref_Ts=None, ref_xOpt=None)
        # closed_mpc.closed_loop_mpc('big')
        # closed_mpc.closed_loop_mpc2('big')

        closed_mpc = closedLoop(demo)
        # closed_mpc.closed_loop_mpc3()
        closed_mpc.closed_loop_mpc4()

    def run_aStar(self, demo_name):

        demo = problemSetting(demo_name)
        draw = plotClass(demo, [], [])
        mpc = closedLoop(demo)
        a_star = mpc.update_path(0, mpc.x0, mpc.xF, 0, 'A_star')
        mpc.N_free = 50
        mpc.mpc_openLoop_freeTime()
        mpc.xref = a_star
        draw.plot_fullDimension(mpc, 0)

    def show_performance(self, demo_name):

        demo = problemSetting(demo_name)
        draw = plotClass(demo, [], [])
        mpc = closedLoop(demo)

        a_star = mpc.update_path(0, mpc.x0, mpc.xF, 0, 'A_star')
        x_aStar = a_star[0]
        y_aStar = a_star[1]
        theta_aStar = a_star[2]

        mpc.N_free = 50
        mpc.mpc_openLoop_freeTime()
        openLoop_xOpt = mpc.xOpt
        openLoop_uOpt = mpc.uOpt

        mpc.N_free = 5
        x_openLoop, x_opt, u_opt, T_opt = mpc.closed_loop_mpc4()
        x_opt = np.asarray(x_opt).T
        u_opt = np.asarray(u_opt).T

        # a star
        fig = plt.figure(figsize=[6, 8])
        ax1 = plt.subplot(311)
        ax1.plot(range(np.size(a_star, 1)), x_aStar, 'k')
        plt.ylabel('x (m)')
        plt.title('A Star Planning State in N Step')

        plt.subplot(312)
        plt.plot(range(np.size(a_star, 1)), y_aStar, 'k')
        plt.ylabel('y (m)')

        plt.subplot(313)
        plt.plot(range(np.size(a_star, 1)), theta_aStar, 'k')
        plt.xlabel('step (N)')
        plt.ylabel('theta (rad)')

        # openloop OBCA
        fig = plt.figure(figsize=[6, 8])
        ax1 = plt.subplot(311)
        ax1.plot(range(np.size(openLoop_xOpt, 1)), openLoop_xOpt[0], 'k')
        plt.ylabel('x (m)')
        plt.title('OpenLoop OBCA State  (horizon N = 50)')

        plt.subplot(312)
        plt.plot(range(np.size(openLoop_xOpt, 1)), openLoop_xOpt[1], 'k')
        plt.ylabel('y (m)')

        plt.subplot(313)
        plt.plot(range(np.size(openLoop_xOpt, 1)), openLoop_xOpt[2], 'k')
        plt.xlabel('step (N)')
        plt.ylabel('theta (rad)')

        # closedLoop OBCA
        fig = plt.figure(figsize=[6, 8])
        ax1 = plt.subplot(311)
        ax1.plot(range(np.size(x_opt, 1)), x_opt[0], 'k')
        plt.ylabel('x (m)')
        plt.title('ClosedLoop OBCA State  (horizon N = 5)')

        plt.subplot(312)
        plt.plot(range(np.size(x_opt, 1)), x_opt[1], 'k')
        plt.ylabel('y (m)')


        plt.subplot(313)
        plt.plot(range(np.size(x_opt, 1)), x_opt[2], 'k')
        plt.xlabel('step (N)')
        plt.ylabel('theta (rad)')

        # closedLoop OBCA control input
        fig = plt.figure(figsize=[6, 8])
        ax1 = plt.subplot(211)
        ax1.plot(range(np.size(u_opt, 1)), u_opt[0], 'k')
        plt.ylabel('velocity (m/sec)')
        plt.title('ClosedLoop OBCA control input (horizon N = 5)')

        plt.subplot(212)
        plt.plot(range(np.size(u_opt, 1)), u_opt[1], 'k')
        plt.xlabel('step (N)')
        plt.ylabel('angular velocity (rad/sec)')


        plt.show()

    def calc_time(self, demo_name):
        demo = problemSetting(demo_name)
        draw = plotClass(demo, [], [])
        mpc = closedLoop(demo)

        start = (demo.startPose[1], demo.startPose[0])
        goal = (demo.goalPose[1], demo.goalPose[0])
        path_solver = a_star(demo.org_gridMap, start, goal)

        tic()
        a_start_path = path_solver.solve(demo.org_gridMap, start, goal)
        t_aStar = toc()
        print(t_aStar)
        # 0.024004899999999996 sec

        mpc.N_free = 10 # np.size(a_start_path, 0)
        tic()
        mpc.mpc_openLoop_freeTime()
        t_openLoopOBCA = toc()
        print(t_openLoopOBCA)
        # 136.69213979999998 sec for N = 74
        # 3.6943962000000004 sec for N = 10

