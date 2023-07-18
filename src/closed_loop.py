# closed_loop.py
"""

Created on 2022/12/15
@author: Pin-Yun Hung
"""

from casadi import *
import matplotlib.pyplot as plt
from scipy import signal
from model_obstacle import obstacleModel
from obca import obca
from a_star import a_star
from draw import plotClass

class closedLoop:

    def __init__(self, problem_setting):

        self.setting = problem_setting
        self.obs_model = obstacleModel()
        self.obca_solver = obca()
        self.path_solver = a_star(self.setting.org_gridMap, (self.setting.startPose[1], self.setting.startPose[0]),
                                  (self.setting.goalPose[1], self.setting.goalPose[0]))
        self.draw = plotClass(self.setting, [], [])

        ##################################
        # Both
        ##################################

        # sampling time
        self.Ts = 0.1

        # state & input numbers
        self.nx = 3
        self.nu = 2

        # state & input constraint
        self.xL = [self.setting.xL[0], self.setting.xL[1], -np.pi]
        self.xU = [self.setting.xU[0], self.setting.xU[1], np.pi]
        self.uL = [-0.6, -np.pi / 6]
        self.uU = [0.6, np.pi / 6]

        # state & input
        self.x0 = self.setting.startPose
        self.xF = self.setting.goalPose
        self.u0 = [0, 0]

        # type
        self.fixtime = 0

        # obstacle constraints
        self.nObs = 0
        self.vObs = np.ones(self.nObs, dtype=int)
        self.AObs = []
        self.bObs = []

        # reference
        self.xref = []
        self.uref = []

        # car
        self.ego = [1.7, 0.75, 1.7, 0.75]
        self.dmin = 0.05

        # solution
        self.xOpt = []
        self.uOpt = []
        self.feas = False
        self.Ts_opt = self.Ts

        ##################################
        # free time
        ##################################

        # cost fxn parameter
        self.Q_free = 0.1 * np.eye(self.nx) # 0.01 -> 0.1
        R1 = 0.01 * np.eye(self.nu)
        R2 = 0.1 * np.eye(self.nu)
        self.R_free = [R1, R2]
        self.P_free = self.Q_free

        # horizon
        self.N_free = 6 # bound by acc & car dynamic

        # state & input
        # self.xref = self.update_path(self.N, self.x0, self.xF, 0, 'startGoal_only')

        ##################################
        # fix time
        ##################################

        # cost fxn parameter
        self.Q_fix = 0.001 * np.eye(self.nx) # 0.0001 & R2 = 1 for opened loop
        R1 = 0.01 * np.eye(self.nu)
        R2 = 1 * np.eye(self.nu)
        self.R_fix = [R1, R2]
        self.P_fix = self.Q_fix

        # horizon
        self.N_fix = 6  # bound by acc & car dynamic

        # terminal set
        self.terminal_set = [] # self.setting.terminal_set

        ##################################
        # dynamic obstacle
        ##################################
        self.dyn_orignal_info = self.setting.dyn_obs_info
        self.dyn_fulltime_info = []
        self.dyn_loc = []

    def mpc_openLoop_freeTime(self):
        self.update_obstacle_constraint(self.N_free, self.Ts, 0)
        self.xref = self.update_path(self.N_free, self.x0, self.xF, allAviable=0, type='startGoal_only')

        self.xOpt, self.uOpt, self.feas, self.Ts_opt \
            = self.obca_solver.obca_mpc4(self.Ts, self.P_free, self.Q_free, self.R_free, self.N_free,
                                         self.x0, self.xL, self.xU, self.uL, self.uU,
                                         self.xref, self.nObs, self.vObs, self.AObs, self.bObs, self.dmin, self.ego, self.u0)

    def mpc_openLoop_fixTime(self):
        # fix time
        self.xref = self.xOpt
        self.xref = self.update_path(0, 0, 0, allAviable=1, type='')
        self.update_obstacle_constraint(self.N_fix, self.Ts_opt, 1)
        self.terminal_set = self.setting.terminal_set
        self.fixtime = 1

        self.xOpt, self.uOpt, self.feas, self.Ts_opt \
            = self.obca_solver.obca_mpc6(self.Ts, self.P_fix, self.Q_fix, self.R_fix, self.N_fix,
                                         self.x0, self.xL, self.xU, self.uL, self.uU,
                                         self.xref, self.nObs, self.vObs, self.AObs, self.bObs, self.dmin, self.ego,
                                         self.u0, self.uOpt, self.terminal_set)
        if self.feas == False:
            self.xOpt, self.uOpt, self.feas, self.Ts_opt \
                = self.obca_solver.obca_mpc8(self.Ts, self.P_fix, self.Q_fix, self.R_fix, self.N_fix,
                                             self.x0, self.xL, self.xU, self.uL, self.uU,
                                             self.xref, self.nObs, self.vObs, self.AObs, self.bObs, self.dmin,
                                             self.ego, self.u0, self.uOpt)

    def closed_loop_mpc(self):

        # count time
        k = 0

        # reference
        a_star = self.update_path(0, self.x0, self.xF, 0, 'A_star')

        # store list
        x_opt = []
        T_opt = []
        x_openLoop = []
        x_opt.append(self.x0)

        while ((self.x0[0] - self.setting.goalPose[0]) ** 2
              +(self.x0[1] - self.setting.goalPose[1]) ** 2 >= 0.1):

            # update dynamic obstacle
            self.update_obstacle(k, self.Ts_opt)

            # update constraint
            self.update_obstacle_constraint(self.N_free, self.Ts, 0)

            # update short reference
            self.xref = self.update_reference_trajectory(self.N_free, a_star, self.x0)

            # mpc
            self.xOpt, self.uOpt, self.feas, self.Ts_opt = \
                self.obca_solver.obca2(self.Ts, self.P_free, self.Q_free, self.R_free, self.N_free,
                                       self.x0, self.u0, self.xL, self.xU, self.uL, self.uU,
                                       self.xref, self.uref, self.nObs, self.vObs, self.AObs, self.bObs,
                                       self.dmin, self.ego, self.fixtime, '', [])
            # check feasibility
            if self.feas == True:
                print('MPC1 -- Success')
            else:
                print('MPC1 -- Failed')
                print(self.Ts_opt)

                # draw
                self.xOpt = np.asarray(x_opt).T
                self.xref = a_star
                self.Ts_opt = T_opt
                # self.draw.fullDimension_animate(self, k, 'sim_title', 'file_name')
                self.draw.fullDimension_closedLoop_animate(self, k, x_openLoop, 'sim_title', 'file_name')
                break

            # get control input
            self.u0 = self.uOpt[:, 0].T

            # update current state
            self.x0 = self.xOpt[:, 1].T
            print(self.x0)

            # store state
            x_opt.append(self.x0)
            T_opt.append(self.Ts_opt)
            x_openLoop.append(self.xOpt.T)

            # update time counting
            k += 1

        # draw
        self.xOpt = np.asarray(x_opt).T
        self.xref = a_star
        self.Ts_opt = T_opt
        # self.draw.fullDimension_animate(self, k, 'sim_title', 'file_name')
        self.draw.fullDimension_closedLoop_animate(self, k, x_openLoop, self.dyn_loc, 'sim_title', 'file_name')

    def closed_loop_mpc3(self):

        # count time
        k = 0

        # reference
        a_star = self.update_path(0, self.x0, self.xF, 0, 'A_star')

        # new_path = []
        # cut_num = 3
        # num  = np.size(a_star, 1) - np.size(a_star, 1) % cut_num
        # for i in range(num // cut_num):
        #     new_path.append(a_star[:, i * cut_num])
        # a_star = np.asarray(new_path).T

        # store list
        x_opt = []
        T_opt = []
        x_openLoop = []
        x_opt.append(self.x0)

        while ((self.x0[0] - self.setting.goalPose[0]) ** 2
              +(self.x0[1] - self.setting.goalPose[1]) ** 2 >= 0.1):

            # update dynamic obstacle
            self.update_obstacle(k, self.Ts_opt)
            self.sensor()

            # update constraint & reference
            if self.fixtime == 0:
                # update constraint
                self.update_obstacle_constraint(self.N_free, self.Ts, 0)

                # update short reference
                self.xref = self.update_reference_trajectory(self.N_free, a_star, self.x0)

            elif self.fixtime == 1:
                # self.xref = self.xOpt
                self.xref = self.update_reference_trajectory(self.N_fix, a_star, self.x0)
                self.xref = self.update_path(0, 0, 0, allAviable=1, type='')
                self.terminal_set = self.setting.terminal_set

                # update constraint
                self.update_obstacle_constraint(self.N_fix, self.Ts_opt, 1)
                print(self.setting.lObs)

                # # update short reference
                # self.xref = self.update_reference_trajectory(self.N_fix, a_star, self.x0)

            # mpc
            if self.fixtime == 0:
                self.xOpt, self.uOpt, self.feas, self.Ts_opt = \
                    self.obca_solver.obca2(self.Ts, self.P_free, self.Q_free, self.R_free, self.N_free,
                                           self.x0, self.u0, self.xL, self.xU, self.uL, self.uU,
                                           self.xref, self.uref, self.nObs, self.vObs, self.AObs, self.bObs,
                                           self.dmin, self.ego, self.fixtime, '', [])
            elif self.fixtime == 1:
                self.xOpt, self.uOpt, self.feas, self.Ts_opt \
                    = self.obca_solver.obca_mpc6(self.Ts, self.P_fix, self.Q_fix, self.R_fix, self.N_fix,
                                                 self.x0, self.xL, self.xU, self.uL, self.uU,
                                                 self.xref, self.nObs, self.vObs, self.AObs, self.bObs, self.dmin,
                                                 self.ego, self.u0, self.uOpt, self.terminal_set)
                if self.feas == False:
                    self.xOpt, self.uOpt, self.feas, self.Ts_opt \
                        = self.obca_solver.obca_mpc8(self.Ts, self.P_fix, self.Q_fix, self.R_fix, self.N_fix,
                                                     self.x0, self.xL, self.xU, self.uL, self.uU,
                                                     self.xref, self.nObs, self.vObs, self.AObs, self.bObs, self.dmin,
                                                     self.ego, self.u0, self.uOpt)

            # check feasibility
            if self.feas == True:
                print('MPC -- Success, fixtime = %i' % self.fixtime)
                # if self.fixtime == 1:
                #     # store state
                #     x_opt.append(self.x0)
                #     T_opt.append(self.Ts_opt)
                #     x_openLoop.append(self.xOpt.T)
                #     break
            else:
                print('MPC -- Failed, fixtime = %i' % self.fixtime)
                print(self.Ts_opt)

                # draw
                self.xOpt = np.asarray(x_opt).T
                self.xref = a_star
                self.Ts_opt = T_opt
                # self.draw.fullDimension_animate(self, k, 'sim_title', 'file_name')
                self.draw.fullDimension_closedLoop_animate(self, k - 1, x_openLoop, self.dyn_loc, 'sim_title', 'file_name')
                break

            # get control input
            self.u0 = self.uOpt[:, 0].T

            # update current state
            self.x0 = self.xOpt[:, 1].T
            print(self.x0)

            # store state
            x_opt.append(self.x0)
            T_opt.append(self.Ts_opt)
            x_openLoop.append(self.xOpt.T)

            # update time counting
            k += 1

        # draw
        self.xOpt = np.asarray(x_opt).T
        self.xref = a_star
        self.Ts_opt = T_opt
        # self.draw.fullDimension_animate(self, k, 'sim_title', 'file_name')
        self.draw.fullDimension_closedLoop_animate(self, k - 1, x_openLoop, self.dyn_loc, 'sim_title', 'file_name')

    def closed_loop_mpc4(self):

        # count time
        k = 0

        # reference
        a_star = self.update_path(0, self.x0, self.xF, 0, 'A_star')

        # new_path = []
        # cut_num = 3
        # num  = np.size(a_star, 1) - np.size(a_star, 1) % cut_num
        # for i in range(num // cut_num):
        #     new_path.append(a_star[:, i * cut_num])
        # a_star = np.asarray(new_path).T

        # store list
        x_opt = []
        u_opt = []
        T_opt = []
        x_openLoop = []
        x_opt.append(self.x0)

        while ((self.x0[0] - self.setting.goalPose[0]) ** 2
              +(self.x0[1] - self.setting.goalPose[1]) ** 2 >= 0.1):

            # update dynamic obstacle
            self.update_obstacle(k, self.Ts_opt)
            self.sensor()

            # update constraint & reference
            if k == 0 or self.fixtime == 0:
                # update constraint
                self.update_obstacle_constraint(self.N_free, self.Ts, 0)

                # update short reference
                self.xref = self.update_reference_trajectory(self.N_free, a_star, self.x0)

            elif self.fixtime == 1:
                # self.xOpt = self.update_path(0, 0, 0, allAviable=1, type='')
                self.xref = self.update_reference_trajectory(self.N_fix, a_star, self.x0)
                for i in range(self.N_fix -5):
                    self.xref[:, i] = self.xOpt[:, i + 1]

                self.xref = self.update_path(0, 0, 0, allAviable=1, type='')
                # self.terminal_set = self.setting.terminal_set

                # update terminal set
                # self.terminal_set = np.array([[5, 30], [self.x0[1] + 4, 60]])
                self.terminal_set = np.array([[self.x0[0] + 5, 99], [1, 9]])

                # update constraint
                self.update_obstacle_constraint(self.N_fix, self.Ts_opt, 1)

                # # update short reference
                # self.xref = self.update_reference_trajectory(self.N_fix, a_star, self.x0)

            # mpc
            if k == 0 or self.fixtime == 0:
                self.xOpt, self.uOpt, self.feas, self.Ts_opt \
                    = self.obca_solver.obca_mpc4(self.Ts, self.P_free, self.Q_free, self.R_free, self.N_free,
                                                 self.x0, self.xL, self.xU, self.uL, self.uU,
                                                 self.xref, self.nObs, self.vObs, self.AObs, self.bObs, self.dmin,
                                                 self.ego, self.u0)

            elif self.fixtime == 1:
                self.xOpt, self.uOpt, self.feas, self.Ts_opt \
                    = self.obca_solver.obca_mpc6(self.Ts, self.P_fix, self.Q_fix, self.R_fix, self.N_fix,
                                                 self.x0, self.xL, self.xU, self.uL, self.uU,
                                                 self.xref, self.nObs, self.vObs, self.AObs, self.bObs, self.dmin,
                                                 self.ego, self.u0, self.uOpt, self.terminal_set)
                if self.feas == False:
                    self.xOpt, self.uOpt, self.feas, self.Ts_opt \
                        = self.obca_solver.obca_mpc8(self.Ts, self.P_fix, self.Q_fix, self.R_fix, self.N_fix,
                                                     self.x0, self.xL, self.xU, self.uL, self.uU,
                                                     self.xref, self.nObs, self.vObs, self.AObs, self.bObs, self.dmin,
                                                     self.ego, self.u0, self.uOpt)

            # check feasibility
            if self.feas == True:
                print('MPC -- Success, fixtime = %i' % self.fixtime)
            else:
                print('MPC -- Failed, fixtime = %i' % self.fixtime)
                print(self.Ts_opt)

                # draw
                self.xOpt = np.asarray(x_opt).T
                self.xref = a_star
                self.Ts_opt = T_opt
                # self.draw.fullDimension_animate(self, k, 'sim_title', 'file_name')
                # self.draw.fullDimension_closedLoop_animate(self, k - 1, x_openLoop, self.dyn_loc, 'sim_title', 'file_name')
                break

            # get control input
            self.u0 = self.uOpt[:, 0].T

            # update current state
            self.x0 = self.xOpt[:, 1].T
            print(self.x0)

            # store state
            x_opt.append(self.x0)
            u_opt.append(self.u0)
            T_opt.append(self.Ts_opt)
            x_openLoop.append(self.xOpt.T)

            # update time counting
            k += 1

            if k == 30:
                break;

        # draw
        self.xOpt = np.asarray(x_opt).T
        self.xref = a_star
        self.Ts_opt = T_opt
        sim_title = 'Dynamic Avoidance With OBCA'
        file_name = 'FullDim_dynObsAvoid_' + self.setting.demo_name + '_N%i' % self.N_free + '_SensorDis%i' % self.setting.senseDis + '_terminalDis = 4'
        # self.draw.fullDimension_animate(self, k, 'sim_title', 'file_name')
        self.draw.fullDimension_closedLoop_animate(self, k - 1, x_openLoop, self.dyn_loc, sim_title, file_name)

        # return x_openLoop, x_opt, u_opt, T_opt

    def update_obstacle(self, k, Ts_opt):

        dyn_nObs = np.size(self.dyn_orignal_info, 0)

        dyn_lObs = []

        self.setting.dyn_obs_info = []
        for i in range(dyn_nObs):
            if k == self.dyn_orignal_info[i][9]:
                self.setting.dyn_obs_info.append(self.dyn_orignal_info[i])

                # calc vertex
                cx = self.setting.dyn_obs_info[-1][0]
                cy = self.setting.dyn_obs_info[-1][1]
                theta = self.setting.dyn_obs_info[-1][2]
                l = self.setting.dyn_obs_info[-1][3]
                w = self.setting.dyn_obs_info[-1][4]

                obs_initial_vertex = self.setting.get_obstacle(cx, cy, theta, l, w)

                dyn_lObs.append(obs_initial_vertex)

            elif k > self.dyn_orignal_info[i][9]:
                self.dyn_orignal_info[i][0] += Ts_opt * self.dyn_orignal_info[i][5] * np.cos(
                    self.dyn_orignal_info[i][2])
                self.dyn_orignal_info[i][1] += Ts_opt * self.dyn_orignal_info[i][5] * np.sin(
                    self.dyn_orignal_info[i][2])

                self.setting.dyn_obs_info.append(self.dyn_orignal_info[i])

                # calc vertex
                cx = self.setting.dyn_obs_info[-1][0]
                cy = self.setting.dyn_obs_info[-1][1]
                theta = self.setting.dyn_obs_info[-1][2]
                l = self.setting.dyn_obs_info[-1][3]
                w = self.setting.dyn_obs_info[-1][4]

                obs_initial_vertex = self.setting.get_obstacle(cx, cy, theta, l, w)
                dyn_lObs.append(obs_initial_vertex)

        self.dyn_loc.append(dyn_lObs)
        self.setting.add_dynamic_obstacle(self.setting.dyn_obs_info)

    def update_obstacle_constraint(self, N, Ts, dynobs_exist):

        self.setting.rebuild_lObs(N, Ts, dynObs_exist=dynobs_exist)
        self.lObs = self.setting.lObs
        self.nObs = self.setting.nObs
        self.vObs = self.setting.vObs

        fulltime_nObs = np.size(self.lObs, 0)
        fulltime_vObs = np.ones(fulltime_nObs, dtype=int)
        for i in range(fulltime_nObs):
            fulltime_vObs[i] = np.size(self.lObs[i], 0)

        self.AObs, self.bObs = self.obs_model.obstacle_H_Represent(fulltime_nObs, fulltime_vObs, self.lObs)

    def update_reference_trajectory(self, N, ref_trajectory, current_state):

        nx = np.size(current_state, 0)
        # ref_trajectory = np.asarray(ref_trajectory)
        path_num = np.size(ref_trajectory, 1)

        # find closest referecnce trajectory point
        min_dis = 100000
        start_point_index = 0;
        for i in range(0, path_num):
            dis = (current_state[0] - ref_trajectory[0, i]) ** 2 + (current_state[1] - ref_trajectory[1, i]) ** 2

            if dis < min_dis:
                min_dis = dis
                start_point_index = i
        # print(start_point_index)
        # print(path_num)
        # update path
        x_ref = np.zeros((nx, N + 1))
        for i in range(0, N + 1):
            for j in range(nx):
                if (i + start_point_index) >= path_num - 1:
                    x_ref[j, i] = ref_trajectory[j][path_num - 1]
                else:
                    x_ref[j, i] = ref_trajectory[j][i + start_point_index]

        return x_ref

    def update_path(self, N, x0, xF, allAviable, type):

        ref_x = np.zeros((self.nx, N + 1))

        if allAviable == 0:
            if type == 'startGoal_only':
                ref_x[0, 0] = x0[0]
                ref_x[1, 0] = x0[1]
                ref_x[2, 0] = x0[2]
                ref_x[0, 1] = xF[0]
                ref_x[1, 1] = xF[1]
                ref_x[2, 1] = xF[2]

                for k in range(2, N + 1):
                    ref_x[:, k] = ref_x[:, k - 1]
            elif type == 'startGoal_smooth':
                for k in range(0, N + 1):
                    ref_x[0, k] = ((xF[0] - x0[0]) / N) * k + x0[0]
                    ref_x[1, k] = ((xF[1] - x0[1]) / N) * k + x0[1]

                    if k >= 1:
                        ref_x[2, k - 1] = np.arctan2(ref_x[1, k] - ref_x[1, k - 1], ref_x[0, k] - ref_x[0, k - 1])

                ref_x[2, N] = ref_x[2, N - 1]

            elif type == 'A_star':
                start = (self.setting.startPose[1], self.setting.startPose[0])
                goal = (self.setting.goalPose[1], self.setting.goalPose[0])
                a_start_path = self.path_solver.solve(self.setting.org_gridMap, start, goal)

                a_start_path = self.path_solver.rebuild_path(a_start_path)
                a_start_path = self.path_solver.create_reference_path(a_start_path)
                a_start_path = np.asarray(a_start_path).T
                ref_x = a_start_path
                # for k in range(0, N + 1):
                #     ref_x[0, k] = a_start_path[0, k]
                #     ref_x[1, k] = a_start_path[1, k]
                #     ref_x[2, k] = a_start_path[2, k]


        elif allAviable == 1:
            ''' find closest point and update reference'''
            ref_xx = []
            for i in range(self.N_free):
                xx = np.linspace(self.xref[0][i], self.xref[0][i + 1], num=int(self.N_fix / self.N_free),
                                 endpoint=False)
                yy = np.linspace(self.xref[1][i], self.xref[1][i + 1], num=int(self.N_fix / self.N_free),
                                 endpoint=False)
                for j in range(int(self.N_fix / self.N_free)):
                    ref_xx.append([xx[j], yy[j]])
            ref_xx.append([self.xref[0][-1], self.xref[1][-1]])

            ref_x = ref_xx
            ref_x = self.path_solver.create_reference_path(ref_x)
            self.N_fix = np.size(ref_x, 0) - 1
            ref_x = np.asarray(ref_x).T
            self.Ts_opt = (self.N_free * self.Ts_opt) / self.N_fix
            self.Ts = self.Ts_opt

        return ref_x

    def sensor(self):

        dyn_exist = []

        cx = self.x0[0]
        cy = self.x0[1]
        theta = self.x0[2]
        l = self.ego[0]
        w = self.ego[1]

        vertex_2 = [cx + l * np.cos(theta) - w * np.sin(theta), cy + l * np.sin(theta) + w * np.cos(theta)]
        vertex_3 = [cx + l * np.cos(theta) + w * np.sin(theta), cy + l * np.sin(theta) - w * np.cos(theta)]
        carFront = [(vertex_2[0] + vertex_3[0]) / 2, (vertex_2[1] + vertex_3[1]) / 2]

        dyn_exist = []
        self.fixtime = 0
        print(self.dyn_loc[-1])
        print(np.size(self.dyn_loc[-1], 0))
        print(self.setting.dyn_nObs)
        for i in range(np.size(self.dyn_loc[-1], 0)):
            dyn_obs = self.dyn_loc[-1][i]
            isExist = 0
            # self.fixtime = 0
            for j in range(4):
                dis = np.sqrt((carFront[0] - dyn_obs[j][0]) ** 2 + (carFront[1] - dyn_obs[j][1]) ** 2)
                if dis <= self.setting.senseDis:
                    isExist = 1
                    self.fixtime = 1
                    # constuct dyn_loc
                    dyn_exist.append(self.setting.dyn_obs_info[i])
                    break

            if isExist == 0:
                self.dyn_loc[-1][i].append(0)
            else:
                self.dyn_loc[-1][i].append(1)
            # print(self.dyn_loc[-1][i])

        self.setting.dyn_obs_info = dyn_exist
        self.setting.dyn_nObs = np.size(self.setting.dyn_obs_info, 0)