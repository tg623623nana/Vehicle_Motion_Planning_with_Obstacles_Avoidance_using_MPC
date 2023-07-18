# draw.py
"""
Plot the result
Created on 2022/12/13
@author: Pin-Yun Hung
"""
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.animation import FuncAnimation

class plotClass:

    def __init__(self, map_model, path_model, mpc_model):
        self.map = map_model
        self.ref_traj = path_model
        self.mpc = mpc_model

    def static_map_model(self, ax1):

        colormap = colors.ListedColormap(["white", "black"])

        ax1.set_xlim([0, self.map.xU[0]])
        ax1.set_ylim([0, self.map.xU[1]])
        # ax1.imshow(self.map.org_gridMap, cmap=colormap, alpha=0.3, extent=[0, self.map.xU[0], 0, self.map.xU[1]], origin='lower')

        # plot static obstacle
        for i in range(np.size(self.map.static_lObs, 0)):
            for j in range(0, self.map.static_vObs[i] - 1):
                v1 = self.map.static_lObs[i][j]  # vertex 1
                v2 = self.map.static_lObs[i][j + 1]  # vertex 2
                ax1.plot([v1[0], v2[0]], [v1[1], v2[1]], '-k')
            if self.map.static_vObs[i] >= 4:
                ax1.plot([self.map.static_lObs[i][-1][0], self.map.static_lObs[i][0][0]],
                         [self.map.static_lObs[i][-1][1], self.map.static_lObs[i][0][1]], '-k')

        # plot start & goal
        ax1.plot([self.map.startPose[0], self.map.goalPose[0]], [self.map.startPose[1], self.map.goalPose[1]], 'ob', markersize='3')

    def plot_map(self, dynObs_exist):
        # setting

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal', adjustable='box')

        # plot map with only static obstacles
        self.static_map_model(ax1)

        # animation plotting for dynamic obstacles
        corner_list = np.zeros((self.map.dyn_nObs, 4, 2))
        def animate_map(k):
            ax1.clear()
            # plot
            plt.title('simulate time = %i' % k)

            # plot static obstacle
            self.static_map_model(ax1)

            # plot start & end
            plt.plot([self.map.startPose[0], self.map.goalPose[0]], [self.map.startPose[1], self.map.goalPose[1]], 'ob', markersize='3')

            # plot dynamic obstacle
            if (self.map.dyn_lObs == []) == False:
                for i in range(self.map.dyn_nObs):
                    if self.map.dyn_obs_info[i][9] == k:
                        for j in range(4):
                            corner_list[i][j][0] = self.map.dyn_lObs[i][j][0]
                            corner_list[i][j][1] = self.map.dyn_lObs[i][j][1]
                    if self.map.dyn_obs_info[i][10] == k:
                        corner_list[i] = np.zeros((4, 2))

                for i in range(self.map.dyn_nObs):
                    if np.all(corner_list[i] == 0) == False:
                        for j in range(4):
                            corner_list[i][j][0] = \
                                self.map.dyn_lObs[i][j][0] + self.map.dyn_obs_info[i][5] * np.cos(
                                    self.map.dyn_obs_info[i][2]) * (
                                        k - self.map.dyn_obs_info[i][9])
                            corner_list[i][j][1] = \
                                self.map.dyn_lObs[i][j][1] + self.map.dyn_obs_info[i][5] * np.sin(
                                    self.map.dyn_obs_info[i][2]) * (
                                        k - self.map.dyn_obs_info[i][9])

                        plt.plot([corner_list[i][0][0], corner_list[i][1][0], corner_list[i][2][0],
                                  corner_list[i][3][0], corner_list[i][0][0]],
                                 [corner_list[i][0][1], corner_list[i][1][1], corner_list[i][2][1],
                                  corner_list[i][3][1], corner_list[i][0][1]], '-k')

            print("time = %i" %k)

        # plot animation if has dynamic obstacles
        if (dynObs_exist == 1) and (self.map.dyn_nObs > 0):
            ani = FuncAnimation(fig, animate_map, frames=max([row[10] for row in self.map.dyn_obs_info]) + 1, interval=20, repeat=False)

        plt.show()

    def plot_fullDimension(self, mpc, dynObs_exist):
        # setting

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal', adjustable='box')

        plt.title('Trajectory: A star v.s. OpenLoop OBCA', fontweight="bold")  # y=0.75
        plt.xlabel(' x (m) ')
        plt.ylabel(' y (m) ')

        # plot map with only static obstacles
        self.static_map_model(ax1)

        # plot reference trajectory
        ax1.plot(mpc.xref[0], mpc.xref[1], '-o', color='royalblue', markersize='3', markerfacecolor='none', label='A star')

        # plot mpc state
        ax1.plot(mpc.xOpt[0, :], mpc.xOpt[1, :], 'om', markersize='3', label='OpenLoop OBCA')
        ax1.plot(mpc.x0[0], mpc.x0[1], 'og', markersize='5', label='Start')
        ax1.plot(mpc.xF[0], mpc.xF[1], 'or', markersize='5', label='Goal')

        # plot car
        ego = mpc.ego
        xOpt = mpc.xOpt.T

        # plot
        W_ev = ego[1] + ego[3]
        L_ev = ego[0] + ego[2]
        offset = L_ev / 2 - ego[2]

        # initial state
        x0_s = xOpt[0, :]
        Rot0 = np.array([[np.cos(x0_s[2]), - np.sin(x0_s[2])], [np.sin(x0_s[2]), np.cos(x0_s[2])]])
        x0 = x0_s[0:2].reshape(2, 1)
        centerCar0 = x0 + Rot0 @ np.array([offset, 0]).reshape(2, 1)

        # end state
        xF_s = xOpt[-1, :]
        RotF = np.array([[np.cos(xF_s[2]), - np.sin(xF_s[2])], [np.sin(xF_s[2]), np.cos(xF_s[2])]])
        xF = xF_s[0:2].reshape(2, 1)
        centerCarF = xF + RotF @ np.array([offset, 0]).reshape(2, 1)

        # plot start position
        self.carBox(centerCar0, x0_s[2], W_ev / 2, L_ev / 2)

        # plot end position
        self.carBox_dashed(centerCarF, xF_s[2], W_ev / 2, L_ev / 2)

        # plot animation if has dynamic obstacles
        if (dynObs_exist == 1) and (self.map.dyn_nObs > 0):
            return

        plt.legend()
        plt.show()

    # def fullDimension_animate(self, dynObs_exist):
    #     # setting
    #
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(111)
    #     ax1.set_aspect('equal', adjustable='box')
    #
    #     # plot map with only static obstacles
    #     self.static_map_model(ax1)
    #
    #     # animation plotting for dynamic obstacles
    #     corner_list = np.zeros((self.map.dyn_nObs, 4, 2))
    #     def animate_map(k):
    #         ax1.clear()
    #         # plot
    #         plt.title('simulate time = %i' % k)
    #
    #         # plot static obstacle
    #         self.static_map_model(ax1)
    #
    #         # plot start & end
    #         plt.plot([self.map.startPose[0], self.map.goalPose[0]], [self.map.startPose[1], self.map.goalPose[1]], 'ob', markersize='3')
    #
    #         # plot dynamic obstacle
    #         if (self.map.dyn_lObs == []) == False:
    #             for i in range(self.map.dyn_nObs):
    #                 if self.map.dyn_obs_info[i][9] == k:
    #                     for j in range(4):
    #                         corner_list[i][j][0] = self.map.dyn_lObs[i][j][0]
    #                         corner_list[i][j][1] = self.map.dyn_lObs[i][j][1]
    #                 if self.map.dyn_obs_info[i][10] == k:
    #                     corner_list[i] = np.zeros((4, 2))
    #
    #             for i in range(self.map.dyn_nObs):
    #                 if np.all(corner_list[i] == 0) == False:
    #                     for j in range(4):
    #                         corner_list[i][j][0] = \
    #                             self.map.dyn_lObs[i][j][0] + self.map.dyn_obs_info[i][5] * np.cos(
    #                                 self.map.dyn_obs_info[i][2]) * (
    #                                     k - self.map.dyn_obs_info[i][9])
    #                         corner_list[i][j][1] = \
    #                             self.map.dyn_lObs[i][j][1] + self.map.dyn_obs_info[i][5] * np.sin(
    #                                 self.map.dyn_obs_info[i][2]) * (
    #                                     k - self.map.dyn_obs_info[i][9])
    #                     plt.plot([corner_list[i][0][0], corner_list[i][1][0], corner_list[i][2][0],
    #                               corner_list[i][3][0], corner_list[i][0][0]],
    #                              [corner_list[i][0][1], corner_list[i][1][1], corner_list[i][2][1],
    #                               corner_list[i][3][1], corner_list[i][0][1]], '-k')
    #
    #         print("time = %i" %k)
    #
    #     # plot animation if has dynamic obstacles
    #     if (dynObs_exist == 1) and (self.map.dyn_nObs > 0):
    #         ani = FuncAnimation(fig, animate_map, frames=max([row[10] for row in self.map.dyn_obs_info]) + 1, interval=20, repeat=False)
    #
    #     plt.show()

    def fullDimension_animate(self, mpc, N, sim_title, file_name):
        xOpt = mpc.xOpt.T
        uOpt = mpc.uOpt.T
        uL = mpc.uL
        uU = mpc.uU
        Ts = mpc.Ts
        Ts_opt = mpc.Ts_opt
        ref_x = mpc.xref
        ego = mpc.ego

        # plot
        W_ev = ego[1] + ego[3]
        L_ev = ego[0] + ego[2]

        uOpt = np.vstack((uOpt, np.zeros((1, 2))))  # final position no input

        w = W_ev / 2;
        offset = L_ev / 2 - ego[2]

        # initial state
        x0_s = xOpt[0, :]
        Rot0 = np.array([[np.cos(x0_s[2]), - np.sin(x0_s[2])], [np.sin(x0_s[2]), np.cos(x0_s[2])]])
        x0 = x0_s[0:2].reshape(2, 1)
        centerCar0 = x0 + Rot0 @ np.array([offset, 0]).reshape(2, 1)

        # end state
        xF_s = xOpt[-1, :]
        RotF = np.array([[np.cos(xF_s[2]), - np.sin(xF_s[2])], [np.sin(xF_s[2]), np.cos(xF_s[2])]])
        xF = xF_s[0:2].reshape(2, 1)
        centerCarF = xF + RotF @ np.array([offset, 0]).reshape(2, 1)


        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal', adjustable='box')

        # plot map with only static obstacles
        self.static_map_model(ax1)

        # animation plotting for dynamic obstacles
        corner_list = np.zeros((self.map.dyn_nObs, 4, 2))

        def animate_obca(k):
            ax1.clear()
            # print("time = %i" % k)

            plt.suptitle(sim_title, y=0.75, fontweight="bold")
            plt.title('N = %i' % N + '  ulimit = %.2f m/sec' % uU[0] + '  %.2f rad/sec' % uU[
                1] + ' Ts_opt = %.2f sec' % Ts_opt)

            # plot static obstacle
            self.static_map_model(ax1)

            # plot reference trajectory
            plt.plot(ref_x[0], ref_x[1], '-ob', markersize='3')

            # plot dynamic obstacle
            if (self.map.dyn_lObs == []) == False:
                for i in range(self.map.dyn_nObs):
                    if self.map.dyn_obs_info[i][9] == k:
                        for j in range(4):
                            corner_list[i][j][0] = self.map.dyn_lObs[i][j][0]
                            corner_list[i][j][1] = self.map.dyn_lObs[i][j][1]
                    if self.map.dyn_obs_info[i][10] == k:
                        corner_list[i] = np.zeros((4, 2))

                for i in range(self.map.dyn_nObs):
                    if np.all(corner_list[i] == 0) == False:
                        if k > self.map.dyn_obs_info[i][9]:
                            for j in range(4):
                                corner_list[i][j][0] += Ts_opt * self.map.dyn_obs_info[i][5] * np.cos(
                                        self.map.dyn_obs_info[i][2])
                                corner_list[i][j][1] += Ts_opt * self.map.dyn_obs_info[i][5] * np.sin(
                                        self.map.dyn_obs_info[i][2])
                        plt.plot([corner_list[i][0][0], corner_list[i][1][0], corner_list[i][2][0],
                                  corner_list[i][3][0], corner_list[i][0][0]],
                                 [corner_list[i][0][1], corner_list[i][1][1], corner_list[i][2][1],
                                  corner_list[i][3][1], corner_list[i][0][1]], '-k')

            # obca
            ax1.plot(xOpt[0:k + 1, 0], xOpt[0:k + 1, 1], '.g')  # plot trajectory so far
            # x_list = np.asarray(x_cur_list)
            # ax1.plot(x_list[:,0], x_list[:,1], '.g')

            Rot = np.array([[np.cos(xOpt[k, 2]), - np.sin(xOpt[k, 2])], [np.sin(xOpt[k, 2]), np.cos(xOpt[k, 2])]])
            x_cur = xOpt[k, 0:2].reshape(2, 1)
            # Rot = np.array([[np.cos(xOpt[np.mod(k, int(Ts_opt / Ts)), 2]), - np.sin(xOpt[np.mod(k, int(Ts_opt / Ts)), 2])], [np.sin(xOpt[np.mod(k, int(Ts_opt / Ts)), 2]), np.cos(xOpt[np.mod(k, int(Ts_opt / Ts)), 2])]])
            # self.x_cur = self.x_cur + np.array([Ts * uOpt[np.mod(k, int(Ts_opt / Ts)), 0] * np.cos(xOpt[np.mod(k, int(Ts_opt / Ts)), 2]),
            #                            Ts * uOpt[np.mod(k, int(Ts_opt / Ts)), 0] * np.sin(xOpt[np.mod(k, int(Ts_opt / Ts)), 2])]).reshape(2,1)

            centerCar = x_cur + Rot @ np.array([offset, 0]).reshape(2, 1)
            # print(x_cur)

            # sim.carBox(centerCar, xOpt[np.mod(k, int(Ts_opt / Ts)), 2], W_ev / 2, L_ev / 2)
            self.carBox(centerCar, xOpt[k, 2], W_ev / 2, L_ev / 2)

            # plot start position
            plt.plot(x0[0], x0[1], "ob")
            self.carBox(centerCar0, x0_s[2], W_ev / 2, L_ev / 2)

            # plot end position
            if k == N:
                plt.plot(xF[0], xF[1], "or")
            self.carBox_dashed(centerCarF, xF_s[2], W_ev / 2, L_ev / 2)


            # plt.xlim([self.xL[0], self.xU[0]])
            # plt.ylim([self.xL[1] - 0.5, self.xU[1] - 1])
            # plt.xticks(np.arange(self.xL[0], self.xU[0] - 1, step=1))
            # plt.yticks(np.arange(self.xL[1], self.xU[1] - 1, step=1))


        if not xOpt == []:
            ani = FuncAnimation(fig, animate_obca, frames=N + 1, interval=200, repeat=False)
            save_name = file_name + '_N_%i' % N + '_ulimit_%.2f' % uU[0] + '_%.2f' % uU[1] + '.gif'
            # ani.save(save_name, writer='pillow', fps=60)
            plt.show()
        else:
            self.plot_map(dynObs_exist=0)
            print('valid')
            plt.show()

    def fullDimension_closedLoop_animate(self, mpc, N, x_openLoop, dyn_loc, sim_title, file_name):
        xOpt = mpc.xOpt.T
        uOpt = mpc.uOpt.T
        uL = mpc.uL
        uU = mpc.uU
        Ts = mpc.Ts
        Ts_opt = mpc.Ts_opt
        ref_x = mpc.xref
        ego = mpc.ego

        # plot
        W_ev = ego[1] + ego[3]
        L_ev = ego[0] + ego[2]

        uOpt = np.vstack((uOpt, np.zeros((1, 2))))  # final position no input

        w = W_ev / 2;
        offset = L_ev / 2 - ego[2]

        # initial state
        x0_s = xOpt[0, :]
        Rot0 = np.array([[np.cos(x0_s[2]), - np.sin(x0_s[2])], [np.sin(x0_s[2]), np.cos(x0_s[2])]])
        x0 = x0_s[0:2].reshape(2, 1)
        centerCar0 = x0 + Rot0 @ np.array([offset, 0]).reshape(2, 1)

        # end state
        xF_s = xOpt[-1, :]
        RotF = np.array([[np.cos(xF_s[2]), - np.sin(xF_s[2])], [np.sin(xF_s[2]), np.cos(xF_s[2])]])
        xF = xF_s[0:2].reshape(2, 1)
        centerCarF = xF + RotF @ np.array([offset, 0]).reshape(2, 1)


        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal', adjustable='box')

        # plot map with only static obstacles
        self.static_map_model(ax1)

        # animation plotting for dynamic obstacles
        corner_list = np.zeros((self.map.dyn_nObs, 4, 2))

        def animate_obca(k):
            ax1.clear()
            # print("time = %i" % k)

            plt.suptitle(sim_title, fontweight="bold") #  y=0.75
            plt.title('Spend Time = %.2f (sec)' % sum(Ts_opt[:k]))  # + '  ulimit = %.2f m/sec' % uU[0] + '  %.2f rad/sec' % uU[1])
            plt.xlabel(' x (m) ')
            plt.ylabel(' y (m) ')

            # plot static obstacle
            self.static_map_model(ax1)

            # plot reference trajectory
            plt.plot(ref_x[0], ref_x[1], '-o', color='royalblue', markersize='3')

            # if (self.map.dyn_lObs == []) == False:
            #     for i in range(self.map.dyn_nObs):
            #         if self.map.dyn_obs_info[i][9] == k:
            #             for j in range(4):
            #                 corner_list[i][j][0] = self.map.dyn_lObs[i][j][0]
            #                 corner_list[i][j][1] = self.map.dyn_lObs[i][j][1]
            #         if self.map.dyn_obs_info[i][10] == k:
            #             corner_list[i] = np.zeros((4, 2))
            #
            #     for i in range(self.map.dyn_nObs):
            #         if np.all(corner_list[i] == 0) == False:
            #             if k > self.map.dyn_obs_info[i][9]:
            #                 for j in range(4):
            #                     corner_list[i][j][0] += Ts_opt[k] * self.map.dyn_obs_info[i][5] * np.cos(
            #                             self.map.dyn_obs_info[i][2])
            #                     corner_list[i][j][1] += Ts_opt[k] * self.map.dyn_obs_info[i][5] * np.sin(
            #                             self.map.dyn_obs_info[i][2])
            #             plt.plot([corner_list[i][0][0], corner_list[i][1][0], corner_list[i][2][0],
            #                       corner_list[i][3][0], corner_list[i][0][0]],
            #                      [corner_list[i][0][1], corner_list[i][1][1], corner_list[i][2][1],
            #                       corner_list[i][3][1], corner_list[i][0][1]], '-k')

            # obca
            ax1.plot(xOpt[0:k + 1, 0], xOpt[0:k + 1, 1], 'o', color='orange', markersize='3')  # plot trajectory so far
            ax1.plot(x_openLoop[k][:, 0], x_openLoop[k][:, 1], '-om', markersize='3')

            Rot = np.array([[np.cos(xOpt[k, 2]), - np.sin(xOpt[k, 2])], [np.sin(xOpt[k, 2]), np.cos(xOpt[k, 2])]])
            x_cur = xOpt[k, 0:2].reshape(2, 1)

            centerCar = x_cur + Rot @ np.array([offset, 0]).reshape(2, 1)

            # plot dynamic obstacle
            if not dyn_loc == []:
                for i in range(np.size(dyn_loc[k], 0)):
                    dyn_obs = dyn_loc[k][i]

                    plt.plot([dyn_obs[0][0], dyn_obs[1][0], dyn_obs[2][0],
                              dyn_obs[3][0], dyn_obs[0][0]],
                             [dyn_obs[0][1], dyn_obs[1][1], dyn_obs[2][1],
                              dyn_obs[3][1], dyn_obs[0][1]], '-k')
                    if dyn_obs[5] == 1:
                        self.sensorCircle(ax1, centerCar, xOpt[k, 2], W_ev / 2, L_ev / 2, 'r')
                    else:
                        self.sensorCircle(ax1, centerCar, xOpt[k, 2], W_ev / 2, L_ev / 2, 'g')
            else:
                self.sensorCircle(ax1, centerCar, xOpt[k, 2], W_ev / 2, L_ev / 2, 'g')
            # sim.carBox(centerCar, xOpt[np.mod(k, int(Ts_opt / Ts)), 2], W_ev / 2, L_ev / 2)
            self.carBox(centerCar, xOpt[k, 2], W_ev / 2, L_ev / 2)

            # plot start position
            plt.plot(x0[0], x0[1], "ob")
            self.carBox(centerCar0, x0_s[2], W_ev / 2, L_ev / 2)

            # plot end position
            if k == N:
                plt.plot(xF[0], xF[1], "or")
            self.carBox_dashed(centerCarF, xF_s[2], W_ev / 2, L_ev / 2)

        if not xOpt == []:
            ani = FuncAnimation(fig, animate_obca, frames=N + 1, interval=200, repeat=False)
            save_name = file_name + '_N_%i' % N + '_ulimit_%.2f' % uU[0] + '_%.2f' % uU[1] + '.gif'
            ani.save(save_name, writer='pillow', fps=60)
            plt.show()
        else:
            self.plot_map(dynObs_exist=0)
            print('valid')
            plt.show()

    def sensorCircle(self, ax1, center, theta, w, l, color):
        cx = center[0]
        cy = center[1]
        vertex_2 = [cx + l * np.cos(theta) - w * np.sin(theta), cy + l * np.sin(theta) + w * np.cos(theta)]
        vertex_3 = [cx + l * np.cos(theta) + w * np.sin(theta), cy + l * np.sin(theta) - w * np.cos(theta)]

        carFront = [(vertex_2[0] + vertex_3[0])/2 , (vertex_2[1] + vertex_3[1])/2]

        sensor_circle = plt.Circle(carFront, self.map.senseDis, color=color, alpha=0.3)
        ax1.add_patch(sensor_circle)

    def carBox(self, x0, psi, w, l):
        car1 = x0[0:2] + np.array([np.cos(psi) * l + np.sin(psi) * w, np.sin(psi) * l - np.cos(psi) * w]).reshape(2,1)
        car2 = x0[0:2] + np.array([np.cos(psi) * l - np.sin(psi) * w, np.sin(psi) * l + np.cos(psi) * w]).reshape(2, 1)
        car3 = x0[0:2] + np.array([-np.cos(psi) * l + np.sin(psi) * w, -np.sin(psi) * l - np.cos(psi) * w]).reshape(2, 1)
        car4 = x0[0:2] + np.array([-np.cos(psi) * l - np.sin(psi) * w, -np.sin(psi) * l + np.cos(psi) * w]).reshape(2, 1)
        plt.plot([car1[0], car2[0], car4[0], car3[0], car1[0]], [car1[1], car2[1], car4[1], car3[1], car1[1]], "k")

    def carBox_dashed(self, x0, psi, w, l):
        car1 = x0[0:2] + np.array([np.cos(psi) * l + np.sin(psi) * w, np.sin(psi) * l - np.cos(psi) * w]).reshape(2,1)
        car2 = x0[0:2] + np.array([np.cos(psi) * l - np.sin(psi) * w, np.sin(psi) * l + np.cos(psi) * w]).reshape(2, 1)
        car3 = x0[0:2] + np.array([-np.cos(psi) * l + np.sin(psi) * w, -np.sin(psi) * l - np.cos(psi) * w]).reshape(2, 1)
        car4 = x0[0:2] + np.array([-np.cos(psi) * l - np.sin(psi) * w, -np.sin(psi) * l + np.cos(psi) * w]).reshape(2, 1)
        plt.plot([car1[0], car2[0], car4[0], car3[0], car1[0]], [car1[1], car2[1], car4[1], car3[1], car1[1]], "--k")







