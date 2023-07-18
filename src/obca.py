# obca.py
"""
OBCA -- Optimizatied Obstacle Collision Avoidance
Created on 2022/12/9
@author: Pin-Yun Hung
"""
import casadi
import numpy as np

class obca:

    def obca(self, Ts, P, Q, R, N, x0, u0, xL, xU, uL, uU, xref, uref, nObs, vObs, AObs, bObs, dmin, ego, fixtime, timeScale_size):
        """

        :param Ts:
        :param P:
        :param Q:
        :param R:
        :param N:
        :param x0:
        :param u0:
        :param xL:
        :param xU:
        :param uL:
        :param uU:
        :param xref:
        :param uref:
        :param nObs:
        :param vObs:
        :param AObs:
        :param bObs:
        :param dmin:
        :param ego:
        :param fixtime: 1 == fix, 0 = free
        :param timeScale_size:  'small' or 'big'
        :return:
        """

        # set state & input number
        nx = 3  # state x = [x, y, theta]
        nu = 2  # control input u = [v, w]

        # set R
        R1 = R[0]
        R2 = R[1]

        # optimization model
        opti = casadi.Opti()

        # state variable
        x = opti.variable(nx, N + 1)

        # control input variable
        u = opti.variable(nu, N)

        # dual variable
        l = opti.variable(np.size(AObs, 0), N + 1)  # l = lambda
        mu = opti.variable(nObs * 4, N + 1)

        # timescale variable
        if fixtime == 0:
            Topt = opti.variable(N + 1)
            opti.set_initial(Topt, 1)


        #############################
        # objective fxn
        #############################
        def obj_fixtime_rule(x, u, xref):
            cost_x = 0
            cost_u = 0
            cost_acc = 0  # acc = acceleration
            cost_terminal = 0.0

            # state cost
            for t in range(N):
                for i in range(nx):
                    for j in range(nx):
                        if t < N:
                            cost_x += (x[i, t] - xref[i, t]) * Q[i, j] * (x[j, t] - xref[j, t])

            # input cost
            for t in range(N):
                for i in range(nu):
                    for j in range(nu):
                        if t < N:
                            if uref == []:
                                cost_u += (u[i, t]) * R1[i, j] * (u[j, t])
                            else:
                                cost_u += (u[i, t] - uref[i, t]) * R1[i, j] * (u[j, t] - uref[j, t])

                        if t < (N - 1):
                            cost_acc += ((u[i, t + 1] - u[i, t]) / Ts) * R2[i, j] * (
                                        (u[j, t + 1] - u[j, t]) / Ts)
                        if t == 0:
                            cost_acc += ((u[i, 0] - u[i]) / Ts) * R2[i, j] * (
                                        (u[j, 0] - u[j]) / Ts)

            # terminal state cost
            for i in range(nx):
                for j in range(nx):
                    cost_terminal += (x[i, N] - xref[i, N]) * P[i, j] * (x[j, N] - xref[j, N])

            return cost_x + cost_u + cost_acc + cost_terminal

        def obj_freetime_rule(x, u, xref, Topt):
            cost_x = 0
            cost_u = 0
            cost_acc = 0  # acc = acceleration
            cost_terminal = 0.0
            cost_t = 0.0

            # state cost
            for t in range(N):
                for i in range(nx):
                    for j in range(nx):
                        if t < N:
                            cost_x += (x[i, t] - xref[i, t]) * Q[i, j] * (x[j, t] - xref[j, t])

            # input cost
            for t in range(N):
                for i in range(nu):
                    for j in range(nu):
                        if t < N:
                            if uref == []:
                                cost_u += (u[i, t]) * R1[i, j] * (u[j, t])
                            else:
                                cost_u += (u[i, t] - uref[i, t]) * R1[i, j] * (u[j, t] - uref[j, t])

                        if t < (N - 1):
                            cost_acc += ((u[i, t + 1] - u[i, t]) / (Topt[t] * Ts)) * R2[i, j] * (
                                        (u[j, t + 1] - u[j, t]) / (Topt[t] * Ts))
                        if t == 0:
                            cost_acc += ((u[i, 0] - u[i]) / (Topt[0] * Ts)) * R2[i, j] * (
                                        (u[j, 0] - u[j]) / (Topt[0] * Ts))

            # terminal state cost
            for i in range(nx):
                for j in range(nx):
                    cost_terminal += (x[i, N] - xref[i, N]) * P[i, j] * (x[j, N] - xref[j, N])

            # sampling time cost
            for t in range(N + 1):
                cost_t += 10 * Topt[t] + 1 * Topt[t] ** 2

            return cost_x + cost_u + cost_acc + cost_terminal + cost_t

        if fixtime == 0:
            opti.minimize(obj_freetime_rule(x, u, xref, Topt))
        else:
            opti.minimize(obj_fixtime_rule(x, u, xref))

        #############################
        # dynamic fxn
        #############################
        for k in range(N):
            if fixtime == 0:
                opti.subject_to(x[0, k + 1] == x[0, k] + (Topt[k] * Ts) * (u[0, k] * casadi.cos(x[2, k])))
                opti.subject_to(x[1, k + 1] == x[1, k] + (Topt[k] * Ts) * (u[0, k] * casadi.sin(x[2, k])))
                opti.subject_to(x[2, k + 1] == x[2, k] + (Topt[k] * Ts) * u[1, k])

                opti.subject_to(Topt[k] == Topt[k + 1])

            else:
                opti.subject_to(x[0, k + 1] == x[0, k] + Ts * (u[0, k] * casadi.cos(x[2, k])))
                opti.subject_to(x[1, k + 1] == x[1, k] + Ts * (u[0, k] * casadi.sin(x[2, k])))
                opti.subject_to(x[2, k + 1] == x[2, k] + Ts * u[1, k])

        #############################
        # state constraint
        #############################
        for i in range(nx - 1):
            opti.subject_to(opti.bounded(xL[i], x[i, :], xU[i]))

        #############################
        # input constraint
        #############################
        for i in range(nu):
            opti.subject_to(opti.bounded(uL[i], u[i, :], uU[i]))

        #############################
        # acceleration constraint
        #############################

        if fixtime == 0:
            for k in range(N):
                if k == 0:
                    acc = (u0[0] - u[0, k]) / (Topt[k] * Ts)
                    angle_acc = (u0[1] - u[1, k]) / (Topt[k] * Ts)
                    opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                    opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

                else:
                    acc = (u[0, k - 1] - u[0, k]) / (Topt[k] * Ts)
                    angle_acc = (u[1, k - 1] - u[1, k]) / (Topt[k] * Ts)
                    opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                    opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))
        else:
            for k in range(N):
                if k == 0:
                    acc = (u0[0] - u[0, k]) / Ts
                    angle_acc = (u0[1] - u[1, k]) / Ts
                    opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                    opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

                else:
                    acc = (u[0, k - 1] - u[0, k]) / Ts
                    angle_acc = (u[1, k - 1] - u[1, k]) / Ts
                    opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                    opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

        #############################
        # initial state constraint
        #############################
        opti.subject_to(x[:, 0] == x0)

        #############################
        # terminal state constraint
        #############################
        if fixtime == 0:
            opti.subject_to(x[:, N] == xref[:, N])
        else:
            opti.subject_to(x[0, N] == xref[0, N])
            opti.subject_to(x[1, N] == xref[1, N])
            opti.subject_to(opti.bounded(xref[2, N] - np.pi / 4, x[2, N], xref[2, N] + np.pi / 4))

        #############################
        # positivity constraints on dual multipliers
        #############################
        for k in range(N + 1):
            opti.subject_to(l[:, k] >= 0)
            opti.subject_to(mu[:, k] >= 0)

            if fixtime == 0 and timeScale_size == 'big':
                dis = np.sqrt((xref[0, N] - x0[0]) ** 2 + (xref[1, N] - x0[1]))
                max_Topt = dis / (N * uU[0] * Ts) + 1
                opti.subject_to(opti.bounded(0.0001, Topt[k], max_Topt))

            elif fixtime == 0 and timeScale_size == 'small':
                opti.subject_to(opti.bounded(0.8, Topt[k], 1.2))

        #############################
        # obstacle avoidance constraint
        #############################
        n = 0
        for k in range(0, N + 1):
            for i in range(nObs):
                Aobs_ith = casadi.MX(vObs[i] - 1, 2)
                bobs_ith = casadi.MX(vObs[i] - 1, 1)
                l_ith = l[n: n + vObs[i] - 1, :]
                mu_ith = mu[i * 4: (i + 1) * 4, :]

                for j in range(vObs[i] - 1):
                    Aobs_ith[j, 0] = AObs[n, 0]
                    Aobs_ith[j, 1] = AObs[n, 1]
                    bobs_ith[j, 0] = bObs[n, 0]
                    n += 1

                # norm(Aobs_ith.T * lambda) <= 1
                cost1 = 0.0
                cost2 = 0.0
                for j in range(vObs[i] - 1):
                    cost1 += Aobs_ith[j, 0] * l_ith[j, k]
                    cost2 += Aobs_ith[j, 1] * l_ith[j, k]
                opti.subject_to(cost1 ** 2 + cost2 ** 2 <= 1)

                # G.T * mu + R.T * Aobs_ith.T * lambda == 0
                cost_G1 = 0.0
                cost_G1 = 0.0
                cost_G1 = (mu_ith[0, k] - mu_ith[2, k]) + np.cos(x[2, k]) * cost1 + np.sin(x[2, k]) * cost2
                cost_G2 = (mu_ith[1, k] - mu_ith[3, k]) - np.sin(x[2, k]) * cost1 + np.cos(x[2, k]) * cost2
                opti.subject_to(cost_G1 == 0)
                opti.subject_to(cost_G2 == 0)

                # -g' * mu + (AObs_ith * t - bObs_ith) * lambda > 0
                g = casadi.MX(1, 4)
                L = ego[0] + ego[2]
                W = ego[1] + ego[3]
                g[0, 0] = L / 2
                g[0, 1] = W / 2
                g[0, 2] = L / 2
                g[0, 3] = W / 2

                offset = (ego[0] + ego[2]) / 2 - ego[2]

                t = casadi.MX(2, 1)
                t[0, 0] = x[0, k] + casadi.cos(x[2, k]) * offset
                t[1, 0] = x[1, k] + casadi.sin(x[2, k]) * offset

                cost_g = 0.0
                for j in range(4):
                    cost_g += g[j] * mu_ith[j, k]
                cost_b = 0.0
                for j in range(vObs[i] - 1):
                    cost_b += bobs_ith[j] * l_ith[j, k]

                cost_dis = -cost_g + (x[0, k] + np.cos(x[2, k]) * offset) * cost1 + \
                           (x[1, k] + np.sin(x[2, k]) * offset) * cost2 - cost_b
                opti.subject_to(cost_dis >= dmin)

        if fixtime == 0:
            opti_setting = { 'print_time': 0}
        else:
            opti_setting = {'ipopt.max_iter': 1000, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                            'ipopt.acceptable_obj_change_tol': 1e-6}  # 'ipopt.print_level': 0

        # initialize
        x_Opt = []
        u_Opt = []
        Ts_opt = 0
        feas = False
        try:
            opti.solver('ipopt', opti_setting)

            print("full dimension -- Optimization solution Find")
            sol = opti.solve()
            x_Opt = np.asarray(sol.value(x))
            u_Opt = np.asarray(sol.value(u))
            feas = True
            Ts_opt = Ts

            if fixtime == 0:
                Ts_opt = sol.value(Topt[0] * Ts)
                print(sol.value(Topt * Ts))

        except:
            print("full dimension -- maybe infeasible or Fail")
            x_Opt = np.asarray(opti.debug.value(x))
            u_Opt = np.asarray(opti.debug.value(u))
            Ts_opt =Ts

            if fixtime == 0:
                Ts_opt = opti.debug.value(Topt[0]) * Ts
                print(opti.debug.value(Topt) * Ts)

        return x_Opt, u_Opt, feas, Ts_opt

    def obca2(self, Ts, P, Q, R, N, x0, u0, xL, xU, uL, uU, xref, uref, nObs, vObs, AObs, bObs, dmin, ego, fixtime, timeScale_size, terminal_set):

        # set state & input number
        nx = 3  # state x = [x, y, theta]
        nu = 2  # control input u = [v, w]

        # set R
        R1 = R[0]
        R2 = R[1]

        # optimization model
        opti = casadi.Opti()

        # state variable
        x = opti.variable(nx, N + 1)

        # control input variable
        u = opti.variable(nu, N)

        # dual variable
        l = opti.variable(np.size(AObs, 0), N + 1)  # l = lambda
        mu = opti.variable(nObs * 4, N + 1)

        # timescale variable
        if fixtime == 0:
            Topt = opti.variable(N + 1)
            opti.set_initial(Topt, 1)

        #############################
        # objective fxn
        #############################
        def obj_fixtime_rule(x, u, xref):
            cost_x = 0
            cost_u = 0
            cost_acc = 0  # acc = acceleration
            cost_terminal = 0.0

            # state cost
            for t in range(N):
                for i in range(nx):
                    for j in range(nx):
                        if t < N:
                            cost_x += (x[i, t] - xref[i, t]) * Q[i, j] * (x[j, t] - xref[j, t])

            # input cost
            for t in range(N):
                for i in range(nu):
                    for j in range(nu):
                        if t < N:
                            cost_u += (u[i, t]) * R1[i, j] * (u[j, t])
                        if t < (N - 1):
                            cost_acc += ((u[i, t + 1] - u[i, t]) / Ts) * R2[i, j] * (
                                        (u[j, t + 1] - u[j, t]) / Ts)
                        if t == 0:
                            cost_acc += ((u[i, 0] - u[i]) / Ts) * R2[i, j] * (
                                        (u[j, 0] - u[j]) / Ts)

            # terminal state cost
            for i in range(nx):
                for j in range(nx):
                    cost_terminal += (x[i, N] - xref[i, N]) * P[i, j] * (x[j, N] - xref[j, N])

            return cost_x + cost_u + cost_acc + cost_terminal

        def obj_freetime_rule(x, u, xref, Topt):
            cost_x = 0
            cost_u = 0
            cost_acc = 0  # acc = acceleration
            cost_terminal = 0.0
            cost_t = 0.0

            # state cost
            for t in range(N):
                for i in range(nx):
                    for j in range(nx):
                        if t < N:
                            cost_x += (x[i, t] - xref[i, t]) * Q[i, j] * (x[j, t] - xref[j, t])

            # input cost
            for t in range(N):
                for i in range(nu):
                    for j in range(nu):
                        if t < N:
                            if uref == []:
                                cost_u += (u[i, t]) * R1[i, j] * (u[j, t])
                            else:
                                cost_u += (u[i, t] - uref[i, t]) * R1[i, j] * (u[j, t] - uref[j, t])

                        if t < (N - 1):
                            cost_acc += ((u[i, t + 1] - u[i, t]) / (Topt[t] * Ts)) * R2[i, j] * (
                                        (u[j, t + 1] - u[j, t]) / (Topt[t] * Ts))
                        if t == 0:
                            cost_acc += ((u[i, 0] - u[i]) / (Topt[0] * Ts)) * R2[i, j] * (
                                        (u[j, 0] - u[j]) / (Topt[0] * Ts))

            # terminal state cost
            for i in range(nx):
                for j in range(nx):
                    cost_terminal += (x[i, N] - xref[i, N]) * P[i, j] * (x[j, N] - xref[j, N])

            # sampling time cost
            for t in range(N + 1):
                cost_t += 10 * Topt[t] + 1 * Topt[t] ** 2

            return cost_x + cost_u + cost_acc + cost_terminal + cost_t

        if fixtime == 0:
            opti.minimize(obj_freetime_rule(x, u, xref, Topt))
        else:
            opti.minimize(obj_fixtime_rule(x, u, xref))

        #############################
        # dynamic fxn
        #############################
        for k in range(N):
            if fixtime == 0:
                opti.subject_to(x[0, k + 1] == x[0, k] + (Topt[k] * Ts) * (u[0, k] * casadi.cos(x[2, k])))
                opti.subject_to(x[1, k + 1] == x[1, k] + (Topt[k] * Ts) * (u[0, k] * casadi.sin(x[2, k])))
                opti.subject_to(x[2, k + 1] == x[2, k] + (Topt[k] * Ts) * u[1, k])

                opti.subject_to(Topt[k] == Topt[k + 1])

            else:
                opti.subject_to(x[0, k + 1] == x[0, k] + Ts * (u[0, k] * casadi.cos(x[2, k])))
                opti.subject_to(x[1, k + 1] == x[1, k] + Ts * (u[0, k] * casadi.sin(x[2, k])))
                opti.subject_to(x[2, k + 1] == x[2, k] + Ts * u[1, k])

        #############################
        # state constraint
        #############################
        for i in range(nx - 1):
            opti.subject_to(opti.bounded(xL[i], x[i, :], xU[i]))

        #############################
        # input constraint
        #############################
        for i in range(nu):
            opti.subject_to(opti.bounded(uL[i], u[i, :], uU[i]))

        #############################
        # acceleration constraint
        #############################

        if fixtime == 0:
            for k in range(N):
                if k == 0:
                    acc = (u0[0] - u[0, k]) / (Topt[k] * Ts)
                    angle_acc = (u0[1] - u[1, k]) / (Topt[k] * Ts)
                    opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                    opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

                else:
                    acc = (u[0, k - 1] - u[0, k]) / (Topt[k] * Ts)
                    angle_acc = (u[1, k - 1] - u[1, k]) / (Topt[k] * Ts)
                    opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                    opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))
        else:
            for k in range(N):
                if k == 0:
                    acc = (u0[0] - u[0, k]) / Ts
                    angle_acc = (u0[1] - u[1, k]) / Ts
                    opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                    opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

                else:
                    acc = (u[0, k - 1] - u[0, k]) / Ts
                    angle_acc = (u[1, k - 1] - u[1, k]) / Ts
                    opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                    opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

        #############################
        # initial state constraint
        #############################
        opti.subject_to(x[:, 0] == x0)

        #############################
        # terminal state constraint
        #############################
        if fixtime == 0:
            opti.subject_to(x[:, N] == xref[:, N])
        else:
            if not terminal_set == []:
                opti.subject_to(x[0, N] >= terminal_set[0, 0])
                opti.subject_to(opti.bounded(terminal_set[1, 0], x[1, N], terminal_set[1, 1]))

        #############################
        # positivity constraints on dual multipliers
        #############################
        for k in range(N + 1):
            opti.subject_to(l[:, k] >= 0)
            opti.subject_to(mu[:, k] >= 0)

            if fixtime == 0:
                dis = (xref[0, N] - x0[0]) + (xref[1, N] - x0[1])
                max_Topt = dis / (N * uU[0] * Ts) + 1
                opti.subject_to(opti.bounded(0.0001, Topt[k], max_Topt))

        #############################
        # obstacle avoidance constraint
        #############################
        n = 0
        for k in range(0, N + 1):
            for i in range(nObs):
                Aobs_ith = casadi.MX(vObs[i] - 1, 2)
                bobs_ith = casadi.MX(vObs[i] - 1, 1)
                l_ith = l[n: n + vObs[i] - 1, :]
                mu_ith = mu[i * 4: (i + 1) * 4, :]

                for j in range(vObs[i] - 1):
                    Aobs_ith[j, 0] = AObs[n, 0]
                    Aobs_ith[j, 1] = AObs[n, 1]
                    bobs_ith[j, 0] = bObs[n, 0]
                    n += 1

                # norm(Aobs_ith.T * lambda) <= 1
                cost1 = 0.0
                cost2 = 0.0
                for j in range(vObs[i] - 1):
                    cost1 += Aobs_ith[j, 0] * l_ith[j, k]
                    cost2 += Aobs_ith[j, 1] * l_ith[j, k]
                opti.subject_to(cost1 ** 2 + cost2 ** 2 <= 1)

                # G.T * mu + R.T * Aobs_ith.T * lambda == 0
                cost_G1 = 0.0
                cost_G2 = 0.0
                cost_G1 = (mu_ith[0, k] - mu_ith[2, k]) + np.cos(x[2, k]) * cost1 + np.sin(x[2, k]) * cost2
                cost_G2 = (mu_ith[1, k] - mu_ith[3, k]) - np.sin(x[2, k]) * cost1 + np.cos(x[2, k]) * cost2
                opti.subject_to(cost_G1 == 0)
                opti.subject_to(cost_G2 == 0)

                # -g' * mu + (AObs_ith * t - bObs_ith) * lambda > 0
                g = casadi.MX(1, 4)
                L = ego[0] + ego[2]
                W = ego[1] + ego[3]
                g[0, 0] = L / 2
                g[0, 1] = W / 2
                g[0, 2] = L / 2
                g[0, 3] = W / 2

                offset = (ego[0] + ego[2]) / 2 - ego[2]

                t = casadi.MX(2, 1)
                t[0, 0] = x[0, k] + casadi.cos(x[2, k]) * offset
                t[1, 0] = x[1, k] + casadi.sin(x[2, k]) * offset

                cost_g = 0.0
                for j in range(4):
                    cost_g += g[j] * mu_ith[j, k]
                cost_b = 0.0
                for j in range(vObs[i] - 1):
                    cost_b += bobs_ith[j] * l_ith[j, k]

                cost_dis = -cost_g + (x[0, k] + np.cos(x[2, k]) * offset) * cost1 + \
                           (x[1, k] + np.sin(x[2, k]) * offset) * cost2 - cost_b
                opti.subject_to(cost_dis >= dmin)

        if fixtime == 0:
            opti_setting = { 'print_time': 0}
        else:
            opti_setting = {'ipopt.max_iter': 1000, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                            'ipopt.acceptable_obj_change_tol': 1e-6}  # 'ipopt.print_level': 0

        # initialize
        x_Opt = []
        u_Opt = []
        Ts_opt = 0
        feas = False
        try:
            opti.solver('ipopt', opti_setting)

            sol = opti.solve()
            print("full dimension -- Optimization solution Find")
            x_Opt = np.asarray(sol.value(x))
            u_Opt = np.asarray(sol.value(u))
            feas = True
            Ts_opt = Ts

            if fixtime == 0:
                Ts_opt = sol.value(Topt[0] * Ts)
                # print(sol.value(Topt * Ts))

        except:
            print("full dimension -- Fail")
            x_Opt = np.asarray(opti.debug.value(x))
            u_Opt = np.asarray(opti.debug.value(u))
            Ts_opt =Ts

            if fixtime == 0:
                Ts_opt = opti.debug.value(Topt[0]) * Ts
                # print(opti.debug.value(Topt) * Ts)

        return x_Opt, u_Opt, feas, Ts_opt

    def obca_mpc3(self, Ts, P, Q, R, N, x0, xL, xU, uL, uU, xref, nObs, vObs, AObs, bObs, dmin, L, W, u0):

        # set nx, nu
        nx = 3  # state z = [x, y, theta]
        nu = 2  # control input u = [v, w]

        # set R
        R1 = R[0]
        R2 = R[1]

        # optimization model
        opti = casadi.Opti()

        # state variable
        x = opti.variable(nx, N + 1)

        # control input variable
        u = opti.variable(nu, N)

        # dual variable
        l = opti.variable(np.size(AObs, 0), N + 1)  # l = lambda

        # timescale variable
        Topt = opti.variable(N + 1)
        opti.set_initial(Topt, 1)

        # objective fxn
        def obj_rule(x, u, xref, Topt):
            cost_x = 0
            cost_u = 0
            cost_acc = 0  # acc = acceleration
            cost_terminal = 0.0
            cost_t = 0.0

            for t in range(N):
                for i in range(nx):
                    for j in range(nx):
                        if t < N:
                            cost_x += (x[i, t] - xref[i, t]) * Q[i, j] * (x[j, t] - xref[j, t])

            for t in range(N):
                for i in range(nu):
                    for j in range(nu):
                        if t < N:
                            cost_u += (u[i, t]) * R1[i, j] * (u[j, t])
                        if t < (N - 1):
                            cost_acc += ((u[i, t + 1] - u[i, t]) / (Topt[t] * Ts)) * R2[i, j] * ((u[j, t + 1] - u[j, t]) / (Topt[t] * Ts))
                        if t == 0:
                            cost_acc += ((u[i, 0] - u[i]) / (Topt[0] * Ts)) * R2[i, j] * ((u[j, 0] - u[j]) / (Topt[0] * Ts))

            for i in range(nx):
                for j in range(nx):
                    cost_terminal += (x[i, N] - xref[i, N]) * P[i, j] * (x[j, N] - xref[j, N])

            for t in range(N + 1):
                cost_t += 0.5 * Topt[t] + 1 * Topt[t] ** 2


            return cost_x + cost_u + cost_acc + cost_terminal + cost_t

        opti.minimize(obj_rule(x, u, xref, Topt))

        #############################
        # dynamic fxn
        #############################
        for k in range(N):
            opti.subject_to(x[0, k + 1] == x[0, k] + (Topt[k] * Ts) * (u[0, k] * casadi.cos(x[2, k])))
            opti.subject_to(x[1, k + 1] == x[1, k] + (Topt[k] * Ts) * (u[0, k] * casadi.sin(x[2, k])))
            opti.subject_to(x[2, k + 1] == x[2, k] + (Topt[k] * Ts) * u[1, k])
            # v = x[3, k] + Ts / 2 * u[1, k]
            # opti.subject_to(x[0, k + 1] == x[0, k] + Ts * (v * casadi.cos(x[2, k])))
            # opti.subject_to(x[1, k + 1] == x[1, k] + Ts * (v * casadi.sin(x[2, k])))
            # opti.subject_to(x[2, k + 1] == x[2, k] + Ts * (v * casadi.tan(u[0, k]) / L))
            # opti.subject_to(x[3, k + 1] == x[3, k] + Ts * u[1, k])
            opti.subject_to(Topt[k] == Topt[k + 1])

        #############################
        # state constraint
        #############################
        for i in range(nx):
            opti.subject_to(opti.bounded(xL[i], x[i, :], xU[i]))

        #############################
        # input constraint
        #############################
        for i in range(nu):
            opti.subject_to(opti.bounded(uL[i], u[i, :], uU[i]))

        #############################
        # acceleration constraint
        #############################
        for k in range(N):
            if k == 0:
                acc = (u0[0] - u[0, k]) / (Topt[k] * Ts)
                angle_acc = (u0[1] - u[1, k]) / (Topt[k] * Ts)
                opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

            else:
                acc = (u[0, k - 1] - u[0, k]) / (Topt[k] * Ts)
                angle_acc = (u[1, k - 1] - u[1, k]) / (Topt[k] * Ts)
                opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

        #############################
        # initial state constraint
        #############################
        opti.subject_to(x[:, 0] == x0)
        # opti.subject_to(u[0, 0] == 0.6)
        # opti.subject_to(u[1, 0] == casadi.pi/6)

        #############################
        # terminal state constraint
        #############################
        opti.subject_to(x[:, N] == xref[:, N])

        #############################
        # positivity constraints on dual multipliers
        #############################
        for k in range(N + 1):
            opti.subject_to(l[:, k] >= 0)
            opti.subject_to(Topt[k] > 0)
            # opti.subject_to(opti.bounded(0.8, Topt[k], 1.2))
            # opti.subject_to(opti.bounded(0.05, Topt[k] * Ts, 5))

        #############################
        # obstacle avoidance constraint
        #############################
        for k in range(0, N + 1):
            n = 0
            for i in range(nObs):
                Aobs_ith = casadi.MX(vObs[i] - 1, 2)
                bobs_ith = casadi.MX(vObs[i] - 1, 1)

                m = n
                for j in range(vObs[i] - 1):
                    Aobs_ith[j, 0] = AObs[n, 0]
                    Aobs_ith[j, 1] = AObs[n, 1]
                    bobs_ith[j, 0] = bObs[n, 0]
                    n += 1

                # norm(Aobs_ith.T * lambda) <= 1
                # Aobs_ith_T = Aobs_ith.T
                # l = lambda = l[i:vObs[i], k].T
                # (Aobs_ith_T[0] * l) ** 2 + (Aobs_ith_T[1] * l) ** 2
                Aobs_ith_T = Aobs_ith.T
                opti.subject_to((Aobs_ith_T[0] @ l[m: m + vObs[i] - 1, k].T) ** 2 + (
                        Aobs_ith_T[1] @ l[m: m + vObs[i] - 1, k].T) ** 2 <= 1)

                # -g' * mu + (AObs_ith * t - bObs_ith) * lambda > 0
                # g' = [L/2, W/2, L/2, W/2]
                # mu_ith = mu[i * 4: (i+1) * 4].T
                # t = [[v * cos(theta)], [v * sin(theta)]] = [[x[0, k+1] - x[0, k]], [x[1, k+1] - x[1, k]]]
                # l = lambda = l[i:vObs[i], k].T
                g = casadi.MX(1, 4)
                g[0, 0] = L / 2
                g[0, 1] = W / 2
                g[0, 2] = L / 2
                g[0, 3] = W / 2

                t = casadi.MX(2, 1)
                t[0, 0] = x[0, k]  # + casadi.cos(x[2, k]) * (L / 2)  # x[0, k + 1] - x[0, k]
                t[1, 0] = x[1, k]  # + casadi.sin(x[2, k]) * (L / 2)  # x[1, k + 1] - x[1, k]

                # v = x[3, k] + Ts / 2 * u[1, k]
                # t[0, 0] = L / 2 + x[0, k] + Ts * (v * casadi.cos(x[2, k]))  # x[0, k + 1] - x[0, k]
                # t[1, 0] = L / 2 + x[1, k] + Ts * (v * casadi.cos(x[2, k])) # x[1, k + 1] - x[1, k]
                # opti.subject_to(-g @ mu[i * 4: (i+1) * 4] + (Aobs_ith @ t - bobs_ith).T @ l[m: m + vObs[i] - 1, k] >= dmin)
                opti.subject_to((Aobs_ith @ t - bobs_ith).T @ l[m: m + vObs[i] - 1, k] >= dmin)

        opti_setting = {'ipopt.max_iter': 2000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        x_Opt = []
        u_Opt = []
        feas = False
        try:
            opti.solver('ipopt', opti_setting)
            sol = opti.solve()
            x_Opt = np.asarray(sol.value(x))
            u_Opt = np.asarray(sol.value(u))
            print("pointmass")
            print(sol.value(Topt * Ts))
            print(u_Opt)
            feas = True
        except:
            print(feas)
            x_Opt = np.asarray(opti.debug.value(x))
            u_Opt = np.asarray(opti.debug.value(u))
            # print(x_Opt)
            # print(u_Opt)
            print(opti.debug.value(Topt))
            x_Opt = []
            u_Opt = []

        return x_Opt, u_Opt, feas

    def obca_mpc4(self, Ts, P, Q, R, N, x0, xL, xU, uL, uU, xref, nObs, vObs, AObs, bObs, dmin, ego, u0):

        # set nx, nu
        nx = 3  # state z = [x, y, theta]
        nu = 2  # control input u = [v, w]

        # set R
        R1 = R[0]
        R2 = R[1]

        # optimization model
        opti = casadi.Opti()

        # state variable
        x = opti.variable(nx, N + 1)

        # control input variable
        u = opti.variable(nu, N)

        # dual variable
        l = opti.variable(np.size(AObs, 0), N + 1)  # l = lambda
        mu = opti.variable(nObs * 4, N + 1)

        # slack variable
        # s = opti.variable(nObs, N + 1)

        # timescale variable
        Topt = opti.variable(N + 1)
        opti.set_initial(Topt, 1)

        # objective fxn
        def obj_rule(x, u, xref, Topt):
            cost_x = 0
            cost_u = 0
            cost_acc = 0  # acc = acceleration
            cost_terminal = 0.0
            cost_t = 0.0
            cost_s = 0.0

            for t in range(N):
                for i in range(nx):
                    for j in range(nx):
                        if t < N:
                            cost_x += (x[i, t] - xref[i, t]) * Q[i, j] * (x[j, t] - xref[j, t])

            for t in range(N):
                for i in range(nu):
                    for j in range(nu):
                        if t < N:
                            cost_u += (u[i, t]) * R1[i, j] * (u[j, t])
                        if t < (N - 1):
                            cost_acc += ((u[i, t + 1] - u[i, t]) / (Topt[t] * Ts)) * R2[i, j] * ((u[j, t + 1] - u[j, t]) / (Topt[t] * Ts))
                        if t == 0:
                            cost_acc += ((u[i, 0] - u[i]) / (Topt[0] * Ts)) * R2[i, j] * ((u[j, 0] - u[j]) / (Topt[0] * Ts))

            for i in range(nx):
                for j in range(nx):
                    cost_terminal += (x[i, N] - xref[i, N]) * P[i, j] * (x[j, N] - xref[j, N])

            for t in range(N + 1):
                cost_t += 10 * Topt[t] + 1 * Topt[t] ** 2

            # for t in range(N + 1):
            #     for i in range(nObs):
            #         cost_s += s[i, t]
            #     cost_s = 0.5 * cost_s

            return cost_x + cost_u + cost_acc + cost_terminal + cost_t  # + cost_s +

        opti.minimize(obj_rule(x, u, xref, Topt))

        #############################
        # dynamic fxn
        #############################
        for k in range(N):
            opti.subject_to(x[0, k + 1] == x[0, k] + (Topt[k] * Ts) * (u[0, k] * casadi.cos(x[2, k])))
            opti.subject_to(x[1, k + 1] == x[1, k] + (Topt[k] * Ts) * (u[0, k] * casadi.sin(x[2, k])))
            opti.subject_to(x[2, k + 1] == x[2, k] + (Topt[k] * Ts) * u[1, k])
            # v = x[3, k] + Ts / 2 * u[1, k]
            # opti.subject_to(x[0, k + 1] == x[0, k] + Ts * (v * casadi.cos(x[2, k])))
            # opti.subject_to(x[1, k + 1] == x[1, k] + Ts * (v * casadi.sin(x[2, k])))
            # opti.subject_to(x[2, k + 1] == x[2, k] + Ts * (v * casadi.tan(u[0, k]) / L))
            # opti.subject_to(x[3, k + 1] == x[3, k] + Ts * u[1, k])
            opti.subject_to(Topt[k] == Topt[k + 1])

        #############################
        # state constraint
        #############################
        for i in range(nx - 1):
            opti.subject_to(opti.bounded(xL[i], x[i, :], xU[i]))

        #############################
        # input constraint
        #############################
        for i in range(nu):
            opti.subject_to(opti.bounded(uL[i], u[i, :], uU[i]))

        #############################
        # acceleration constraint
        #############################
        for k in range(N):
            if k == 0:
                acc = (u0[0] - u[0, k]) / (Topt[k] * Ts)
                angle_acc = (u0[1] - u[1, k]) / (Topt[k] * Ts)
                opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

            else:
                acc = (u[0, k - 1] - u[0, k]) / (Topt[k] * Ts)
                angle_acc = (u[1, k - 1] - u[1, k]) / (Topt[k] * Ts)
                opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

        #############################
        # initial state constraint
        #############################
        opti.subject_to(x[:, 0] == x0)
        # opti.subject_to(u[0, 0] == 0.6)
        # opti.subject_to(u[1, 0] == casadi.pi/6)

        #############################
        # terminal state constraint
        #############################
        opti.subject_to(x[:, N] == xref[:, N])

        #############################
        # positivity constraints on dual multipliers
        #############################
        for k in range(N + 1):
            opti.subject_to(l[:, k] >= 0)
            opti.subject_to(mu[:, k] >= 0)
            opti.subject_to(Topt[k] > 0)

            dis = (xref[0, N] - x0[0]) + (xref[1, N] - x0[1])
            max_Topt = dis / (N * uU[0] * Ts) + 1
            opti.subject_to(opti.bounded(0.0001, Topt[k], max_Topt))

        #############################
        # obstacle avoidance constraint
        #############################
        for k in range(0, N + 1):
            n = 0
            for i in range(nObs):
                Aobs_ith = casadi.MX(vObs[i] - 1, 2)
                bobs_ith = casadi.MX(vObs[i] - 1, 1)
                l_ith = l[n: n + vObs[i] - 1, :]
                mu_ith = mu[i * 4: (i+1) * 4, :]

                m = n
                for j in range(vObs[i] - 1):
                    Aobs_ith[j, 0] = AObs[n, 0]
                    Aobs_ith[j, 1] = AObs[n, 1]
                    bobs_ith[j, 0] = bObs[n, 0]
                    n += 1


                # norm(Aobs_ith.T * lambda) <= 1
                # Aobs_ith_T = Aobs_ith.T
                # l = lambda = l[i:vObs[i], k].T
                # (Aobs_ith_T[0] * l) ** 2 + (Aobs_ith_T[1] * l) ** 2
                # Aobs_ith_T = Aobs_ith.T
                # opti.subject_to((Aobs_ith_T[0] @ l[m: m + vObs[i] - 1, k].T) ** 2 + (
                #         Aobs_ith_T[1] @ l[m: m + vObs[i] - 1, k].T) ** 2 <= 1)

                cost1 = 0.0
                cost2 = 0.0
                for j in range(vObs[i] - 1):
                    cost1 += Aobs_ith[j, 0] * l_ith[j, k]
                    cost2 += Aobs_ith[j, 1] * l_ith[j, k]
                opti.subject_to(cost1 ** 2 + cost2 ** 2 <= 1)

                # G.T * mu + R.T * Aobs_ith.T * lambda == 0
                # G = [[1, 0], [0, 1], [-1, 0], [0, -1]]
                # mu_ith = mu[i * 4: (i+1) * 4].T
                # R = [ [cos(x[2, k]), -sin(x[2, k])], [sin(x[2, k]), cos(x[2, k])]]
                # Aobs_ith_T = Aobs_ith.T
                # l = lambda = l[i:vObs[i], k].T

                cost_G1 = 0.0
                cost_G1 = 0.0
                cost_G1 = (mu_ith[0, k] - mu_ith[2, k]) + np.cos(x[2, k]) * cost1 + np.sin(x[2, k]) * cost2
                cost_G2 = (mu_ith[1, k] - mu_ith[3, k]) - np.sin(x[2, k]) * cost1 + np.cos(x[2, k]) * cost2
                opti.subject_to(cost_G1 == 0)
                opti.subject_to(cost_G2 == 0)

                # -g' * mu + (AObs_ith * t - bObs_ith) * lambda > 0
                # g' = [L/2, W/2, L/2, W/2]
                # mu_ith = mu[i * 4: (i+1) * 4].T
                # t = [[v * cos(theta)], [v * sin(theta)]] = [[x[0, k+1] - x[0, k]], [x[1, k+1] - x[1, k]]]
                # l = lambda = l[i:vObs[i], k].T
                g = casadi.MX(1, 4)
                L = ego[0] + ego[2]
                W = ego[1] + ego[3]
                g[0, 0] = L / 2
                g[0, 1] = W / 2
                g[0, 2] = L / 2
                g[0, 3] = W / 2

                offset = (ego[0] + ego[2]) / 2 - ego[2]

                t = casadi.MX(2, 1)

                t[0, 0] = x[0, k]  + casadi.cos(x[2, k]) * offset
                t[1, 0] = x[1, k]  + casadi.sin(x[2, k]) * offset

                cost_g = 0.0
                for j in range(4):
                    cost_g += g[j] * mu_ith[j, k]
                cost_b = 0.0
                for j in range(vObs[i] - 1):
                    cost_b += bobs_ith[j] * l_ith[j, k]

                cost_dis = -cost_g + (x[0, k] + np.cos(x[2, k]) * offset) * cost1 + \
                           (x[1, k] + np.sin(x[2, k]) * offset) * cost2 - cost_b
                opti.subject_to(cost_dis >= dmin)

        opti_setting = {'ipopt.print_level': 0}

        # opti_setting = {'ipopt.max_iter': 5000}

        x_Opt = []
        u_Opt = []
        Ts_opt = 0
        feas = False
        try:
            print("========= Runing: full dimension =========")
            opti.solver('ipopt', opti_setting)
            # opti.solver('ipopt')
            sol = opti.solve()
            x_Opt = np.asarray(sol.value(x))
            u_Opt = np.asarray(sol.value(u))
            Ts_opt = sol.value(Topt[0] * Ts)

            feas = True
        except:
            x_Opt = np.asarray(opti.debug.value(x))
            u_Opt = np.asarray(opti.debug.value(u))
            Ts_opt = np.asarray(opti.debug.value(Topt[0]) * Ts)
            # print(x_Opt)
            # print(u_Opt)
            # print(feas)
            # print(opti.debug.value(Topt[0]))

        return x_Opt, u_Opt, feas, Ts_opt

    def obca_mpc7(self, Ts, P, Q, R, N, x0, xL, xU, uL, uU, xref, nObs, vObs, AObs, bObs, dmin, ego, u0):

        # set nx, nu
        nx = 3  # state z = [x, y, theta]
        nu = 2  # control input u = [v, w]

        # set R
        R1 = R[0]
        R2 = R[1]

        # optimization model
        opti = casadi.Opti()

        # state variable
        x = opti.variable(nx, N + 1)

        # control input variable
        u = opti.variable(nu, N)

        # dual variable
        l = opti.variable(np.size(AObs, 0), N + 1)  # l = lambda
        mu = opti.variable(nObs * 4, N + 1)

        # slack variable
        # s = opti.variable(nObs, N + 1)

        # timescale variable
        Topt = opti.variable(N + 1)
        opti.set_initial(Topt, 1)

        # objective fxn
        def obj_rule(x, u, xref, Topt):
            cost_x = 0
            cost_u = 0
            cost_acc = 0  # acc = acceleration
            cost_terminal = 0.0
            cost_t = 0.0
            cost_s = 0.0

            for t in range(N):
                for i in range(nx):
                    for j in range(nx):
                        if t < N:
                            cost_x += (x[i, t] - xref[i, t]) * Q[i, j] * (x[j, t] - xref[j, t])

            for t in range(N):
                for i in range(nu):
                    for j in range(nu):
                        if t < N:
                            cost_u += (u[i, t]) * R1[i, j] * (u[j, t])
                        if t < (N - 1):
                            cost_acc += ((u[i, t + 1] - u[i, t]) / (Topt[t] * Ts)) * R2[i, j] * (
                                        (u[j, t + 1] - u[j, t]) / (Topt[t] * Ts))
                        if t == 0:
                            cost_acc += ((u[i, 0] - u[i]) / (Topt[0] * Ts)) * R2[i, j] * (
                                        (u[j, 0] - u[j]) / (Topt[0] * Ts))

            for i in range(nx):
                for j in range(nx):
                    cost_terminal += (x[i, N] - xref[i, N]) * P[i, j] * (x[j, N] - xref[j, N])

            for t in range(N + 1):
                cost_t += 0.5 * Topt[t] + 1 * Topt[t] ** 2

            # for t in range(N + 1):
            #     for i in range(nObs):
            #         cost_s += s[i, t]
            #     cost_s = 0.5 * cost_s

            return cost_x + cost_u + cost_acc + cost_terminal + cost_t  # + cost_s

        opti.minimize(obj_rule(x, u, xref, Topt))

        #############################
        # dynamic fxn
        #############################
        for k in range(N):
            opti.subject_to(x[0, k + 1] == x[0, k] + (Topt[k] * Ts) * (u[0, k] * casadi.cos(x[2, k])))
            opti.subject_to(x[1, k + 1] == x[1, k] + (Topt[k] * Ts) * (u[0, k] * casadi.sin(x[2, k])))
            opti.subject_to(x[2, k + 1] == x[2, k] + (Topt[k] * Ts) * u[1, k])
            # v = x[3, k] + Ts / 2 * u[1, k]
            # opti.subject_to(x[0, k + 1] == x[0, k] + Ts * (v * casadi.cos(x[2, k])))
            # opti.subject_to(x[1, k + 1] == x[1, k] + Ts * (v * casadi.sin(x[2, k])))
            # opti.subject_to(x[2, k + 1] == x[2, k] + Ts * (v * casadi.tan(u[0, k]) / L))
            # opti.subject_to(x[3, k + 1] == x[3, k] + Ts * u[1, k])
            opti.subject_to(Topt[k] == Topt[k + 1])

        #############################
        # state constraint
        #############################
        for i in range(nx - 1):
            opti.subject_to(opti.bounded(xL[i], x[i, :], xU[i]))

        #############################
        # input constraint
        #############################
        for i in range(nu):
            opti.subject_to(opti.bounded(uL[i], u[i, :], uU[i]))

        #############################
        # acceleration constraint
        #############################
        for k in range(N):
            if k == 0:
                acc = (u0[0] - u[0, k]) / (Topt[k] * Ts)
                angle_acc = (u0[1] - u[1, k]) / (Topt[k] * Ts)
                opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

            else:
                acc = (u[0, k - 1] - u[0, k]) / (Topt[k] * Ts)
                angle_acc = (u[1, k - 1] - u[1, k]) / (Topt[k] * Ts)
                opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

        #############################
        # initial state constraint
        #############################
        opti.subject_to(x[:, 0] == x0)
        # opti.subject_to(u[0, 0] == 0.6)
        # opti.subject_to(u[1, 0] == casadi.pi/6)

        #############################
        # terminal state constraint
        #############################
        opti.subject_to(x[:, N] == xref[:, N])

        #############################
        # positivity constraints on dual multipliers
        #############################
        for k in range(N + 1):
            opti.subject_to(l[:, k] >= 0)
            opti.subject_to(mu[:, k] >= 0)
            # opti.subject_to(Topt[k] > 0)
            opti.subject_to(opti.bounded(0.8, Topt[k], 1.2))
            # opti.subject_to(opti.bounded(0.05, Topt[k] * Ts, 10))
            # opti.subject_to(s[i] >= 0)

        #############################
        # obstacle avoidance constraint
        #############################
        for k in range(0, N + 1):
            n = 0
            for i in range(nObs):
                Aobs_ith = casadi.MX(vObs[i] - 1, 2)
                bobs_ith = casadi.MX(vObs[i] - 1, 1)
                l_ith = l[n: n + vObs[i] - 1, :]
                mu_ith = mu[i * 4: (i + 1) * 4, :]

                m = n
                for j in range(vObs[i] - 1):
                    Aobs_ith[j, 0] = AObs[n, 0]
                    Aobs_ith[j, 1] = AObs[n, 1]
                    bobs_ith[j, 0] = bObs[n, 0]
                    n += 1

                # norm(Aobs_ith.T * lambda) <= 1
                # Aobs_ith_T = Aobs_ith.T
                # l = lambda = l[i:vObs[i], k].T
                # (Aobs_ith_T[0] * l) ** 2 + (Aobs_ith_T[1] * l) ** 2
                # Aobs_ith_T = Aobs_ith.T
                # opti.subject_to((Aobs_ith_T[0] @ l[m: m + vObs[i] - 1, k].T) ** 2 + (
                #         Aobs_ith_T[1] @ l[m: m + vObs[i] - 1, k].T) ** 2 <= 1)

                cost1 = 0.0
                cost2 = 0.0
                for j in range(vObs[i] - 1):
                    cost1 += Aobs_ith[j, 0] * l_ith[j, k]
                    cost2 += Aobs_ith[j, 1] * l_ith[j, k]
                opti.subject_to(cost1 ** 2 + cost2 ** 2 <= 1)

                # G.T * mu + R.T * Aobs_ith.T * lambda == 0
                # G = [[1, 0], [0, 1], [-1, 0], [0, -1]]
                # mu_ith = mu[i * 4: (i+1) * 4].T
                # R = [ [cos(x[2, k]), -sin(x[2, k])], [sin(x[2, k]), cos(x[2, k])]]
                # Aobs_ith_T = Aobs_ith.T
                # l = lambda = l[i:vObs[i], k].T
                G = casadi.MX(4, 2)
                G[0, 0] = 1
                G[0, 1] = 0
                G[1, 0] = 0
                G[1, 1] = 1
                G[2, 0] = -1
                G[2, 1] = 0
                G[3, 0] = 0
                G[3, 1] = -1

                R = casadi.MX(2, 2)
                R[0, 0] = casadi.cos(x[2, k])
                R[0, 1] = -casadi.sin(x[2, k])
                R[1, 0] = casadi.sin(x[2, k])
                R[1, 1] = casadi.cos(x[2, k])  # x[2, k-1] + Ts * u[1, k-1]

                # R[0, 0] = casadi.cos(x[2, k-1] + Ts * u[1, k-1])
                # R[0, 1] = -casadi.sin(x[2, k-1] + Ts * u[1, k-1])
                # R[1, 0] = casadi.sin(x[2, k-1] + Ts * u[1, k-1])
                # R[1, 1] = casadi.cos(x[2, k-1] + Ts * u[1, k-1])

                # G_R = casadi.MX(2, 1)
                # G_R = G.T @ mu[i * 4: (i + 1) * 4, k] + R.T @ (Aobs_ith_T @ l[m: m + vObs[i] - 1, k])
                # opti.subject_to(G_R[0] == 0)
                # opti.subject_to(G_R[1] == 0)

                cost_G1 = 0.0
                cost_G1 = 0.0
                cost_G1 = (mu_ith[0, k] - mu_ith[2, k]) + np.cos(x[2, k]) * cost1 + np.sin(x[2, k]) * cost2
                cost_G2 = (mu_ith[1, k] - mu_ith[3, k]) - np.sin(x[2, k]) * cost1 + np.cos(x[2, k]) * cost2
                opti.subject_to(cost_G1 == 0)
                opti.subject_to(cost_G2 == 0)

                # -g' * mu + (AObs_ith * t - bObs_ith) * lambda > 0
                # g' = [L/2, W/2, L/2, W/2]
                # mu_ith = mu[i * 4: (i+1) * 4].T
                # t = [[v * cos(theta)], [v * sin(theta)]] = [[x[0, k+1] - x[0, k]], [x[1, k+1] - x[1, k]]]
                # l = lambda = l[i:vObs[i], k].T
                g = casadi.MX(1, 4)
                L = ego[0] + ego[2]
                W = ego[1] + ego[3]
                g[0, 0] = L / 2
                g[0, 1] = W / 2
                g[0, 2] = L / 2
                g[0, 3] = W / 2

                offset = (ego[0] + ego[2]) / 2 - ego[2]

                t = casadi.MX(2, 1)
                # t[0, 0] = x[0, k-1] +  Ts * u[0, k-1] * casadi.cos(x[2, k-1]) + Ts * u[0, k-1] * casadi.cos(x[2, k-1]) * (L / 2) # x[0, k + 1] - x[0, k]
                # t[1, 0] = x[1, k-1] +  Ts * u[0, k-1] * casadi.sin(x[2, k-1]) + Ts * u[0, k-1] * casadi.cos(x[2, k-1]) * (L / 2)# x[1, k + 1] - x[1, k]

                t[0, 0] = x[0, k] + casadi.cos(
                    x[2, k]) * offset  # + casadi.cos(x[2, k]) * (L / 2)  # x[0, k + 1] - x[0, k]
                t[1, 0] = x[1, k] + casadi.sin(
                    x[2, k]) * offset  # + casadi.sin(x[2, k]) * (L / 2)  # x[1, k + 1] - x[1, k]

                # v = x[3, k] + Ts / 2 * u[1, k]
                # t[0, 0] = L / 2 + x[0, k] + Ts * (v * casadi.cos(x[2, k]))  # x[0, k + 1] - x[0, k]
                # t[1, 0] = L / 2 + x[1, k] + Ts * (v * casadi.cos(x[2, k])) # x[1, k + 1] - x[1, k]
                # opti.subject_to(-g @ mu[i * 4: (i+1) * 4] + (Aobs_ith @ t - bobs_ith).T @ l[m: m + vObs[i] - 1, k] >= dmin)
                # opti.subject_to((Aobs_ith @ t - bobs_ith).T @ l[m: m + vObs[i] - 1, k] >= dmin)

                cost_g = 0.0
                for j in range(4):
                    cost_g += g[j] * mu_ith[j, k]
                cost_b = 0.0
                for j in range(vObs[i] - 1):
                    cost_b += bobs_ith[j] * l_ith[j, k]

                cost_dis = -cost_g + (x[0, k] + np.cos(x[2, k]) * offset) * cost1 + \
                           (x[1, k] + np.sin(x[2, k]) * offset) * cost2 - cost_b
                opti.subject_to(cost_dis >= dmin)
                # opti.subject_to(cost_dis >= s[i, k])

        # opti_setting = {'ipopt.max_iter': 2000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
        #                 'ipopt.acceptable_obj_change_tol': 1e-6}

        # opti_setting = {'ipopt.max_iter': 5000}

        x_Opt = []
        u_Opt = []
        Ts_opt = 0
        feas = False
        try:
            # opti.solver('ipopt', opti_setting)
            opti.solver('ipopt')
            sol = opti.solve()
            x_Opt = np.asarray(sol.value(x))
            u_Opt = np.asarray(sol.value(u))
            Ts_opt = sol.value(Topt[0] * Ts)
            # for i in range(np.size(xOpt, 0)):
            #     x_Opt[0, i] = x_Opt[0, i]  + casadi.cos(x_Opt[2, i]) * (L / 2 + 1)
            #     x_Opt[1, i] = x_Opt[1, i] + casadi.cos(x_Opt[2, i]) * (L / 2 + 1)

            print("full dimension")
            print(sol.value(Topt * Ts))
            print(u_Opt)
            feas = True
        except:
            print(feas)
            x_Opt = np.asarray(opti.debug.value(x))
            u_Opt = np.asarray(opti.debug.value(u))
            # print(x_Opt)
            # print(u_Opt)
            print(opti.debug.value(Topt))
            x_Opt = []
            u_Opt = []

        return x_Opt, u_Opt, feas, Ts_opt

    def obca_mpc6(self, Ts, P, Q, R, N, x0, xL, xU, uL, uU, xref, nObs, vObs, AObs, bObs, dmin, ego, u0, uOpt, terminal_set):

        # set nx, nu
        nx = 3  # state z = [x, y, theta]
        nu = 2  # control input u = [v, w]

        # set R
        R1 = R[0]
        R2 = R[1]

        # optimization model
        opti = casadi.Opti()

        # state variable
        x = opti.variable(nx, N + 1)

        # control input variable
        u = opti.variable(nu, N)

        # dual variable
        l = opti.variable(np.size(AObs, 0), N + 1)  # l = lambda
        mu = opti.variable(nObs * 4, N + 1)

        # objective fxn
        def obj_rule(x, u, xref):
            cost_x = 0
            cost_u = 0
            cost_acc = 0  # acc = acceleration
            cost_terminal = 0.0

            for t in range(N):
                for i in range(nx):
                    for j in range(nx):
                        if t < N:
                            cost_x += (x[i, t] - xref[i, t]) * Q[i, j] * (x[j, t] - xref[j, t])

            for t in range(N):
                for i in range(nu):
                    for j in range(nu):
                        if t < N:
                            cost_u += (u[i, t]) * R1[i, j] * (u[j, t])
                            # cost_u += (u[i, t] - uOpt[i, t]) * R1[i, j] * (u[j, t] - uOpt[j, t])
                        if t < (N - 1):
                            cost_acc += ((u[i, t + 1] - u[i, t]) / Ts) * R2[i, j] * ((u[j, t + 1] - u[j, t]) / Ts)
                        if t == 0:
                            cost_acc += ((u[i, 0] - u[i]) / Ts) * R2[i, j] * ((u[j, 0] - u[j]) / Ts)

            for i in range(nx):
                for j in range(nx):
                    cost_terminal += (x[i, N] - xref[i, N]) * P[i, j] * (x[j, N] - xref[j, N])

            return cost_x + cost_u + cost_acc + cost_terminal #

        opti.minimize(obj_rule(x, u, xref))

        #############################
        # dynamic fxn
        #############################
        for k in range(N):
            opti.subject_to(x[0, k + 1] == x[0, k] + Ts * (u[0, k] * casadi.cos(x[2, k])))
            opti.subject_to(x[1, k + 1] == x[1, k] + Ts * (u[0, k] * casadi.sin(x[2, k])))
            opti.subject_to(x[2, k + 1] == x[2, k] + Ts * u[1, k])

        #############################
        # state constraint
        #############################
        for i in range(nx - 1):
            opti.subject_to(opti.bounded(xL[i], x[i, :], xU[i]))

        #############################
        # input constraint
        #############################
        for i in range(nu):
            opti.subject_to(opti.bounded(uL[i], u[i, :], uU[i]))

        #############################
        # acceleration constraint
        #############################
        for k in range(N):
            if k == 0:
                acc = (u0[0] - u[0, k]) / Ts
                angle_acc = (u0[1] - u[1, k]) / Ts
                opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

            else:
                acc = (u[0, k - 1] - u[0, k]) / Ts
                angle_acc = (u[1, k - 1] - u[1, k]) / Ts
                opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

        #############################
        # initial state constraint
        #############################
        opti.subject_to(x[:, 0] == x0)
        # opti.subject_to(u[:, 0] == u0)

        #############################
        # terminal state constraint
        #############################
        # opti.subject_to(x[0, N] == xref[0, N])
        # opti.subject_to(x[1, N] == xref[1, N])
        # opti.subject_to(opti.bounded(terminal_set[0][0], x[0, N], terminal_set[0][1]))
        # opti.subject_to(opti.bounded(terminal_set[1][0], x[1, N], terminal_set[1][1]))
        opti.subject_to(x[0, N] >= terminal_set[0, 0])
        opti.subject_to(opti.bounded(terminal_set[1, 0], x[1, N], terminal_set[1, 1]))

        # opti.subject_to(opti.bounded(20, x[0, N], 39))
        # opti.subject_to(opti.bounded(1, x[1, N], 9))
        # opti.subject_to(opti.bounded(xref[2, N]- np.pi/4, x[2, N], xref[2, N] + np.pi/4))

        #############################
        # positivity constraints on dual multipliers
        #############################
        for k in range(N + 1):
            opti.subject_to(l[:, k] >= 0)
            opti.subject_to(mu[:, k] >= 0)

        #############################
        # obstacle avoidance constraint
        #############################
        n = 0
        for k in range(0, N + 1):
            for i in range(nObs):
                Aobs_ith = casadi.MX(vObs[i] - 1, 2)
                bobs_ith = casadi.MX(vObs[i] - 1, 1)
                l_ith = l[n: n + vObs[i] - 1, :]
                mu_ith = mu[i * 4: (i+1) * 4, :]

                for j in range(vObs[i] - 1):
                    Aobs_ith[j, 0] = AObs[n, 0]
                    Aobs_ith[j, 1] = AObs[n, 1]
                    bobs_ith[j, 0] = bObs[n, 0]
                    n += 1

                # norm(Aobs_ith.T * lambda) <= 1
                cost1 = 0.0
                cost2 = 0.0
                for j in range(vObs[i] - 1):
                    cost1 += Aobs_ith[j, 0] * l_ith[j, k]
                    cost2 += Aobs_ith[j, 1] * l_ith[j, k]
                opti.subject_to(cost1 ** 2 + cost2 ** 2 <= 1)

                # G.T * mu + R.T * Aobs_ith.T * lambda == 0
                cost_G1 = 0.0
                cost_G1 = 0.0
                cost_G1 = (mu_ith[0, k] - mu_ith[2, k]) + np.cos(x[2, k]) * cost1 + np.sin(x[2, k]) * cost2
                cost_G2 = (mu_ith[1, k] - mu_ith[3, k]) - np.sin(x[2, k]) * cost1 + np.cos(x[2, k]) * cost2
                opti.subject_to(cost_G1 == 0)
                opti.subject_to(cost_G2 == 0)

                # -g' * mu + (AObs_ith * t - bObs_ith) * lambda > 0
                g = casadi.MX(1, 4)
                L = ego[0] + ego[2]
                W = ego[1] + ego[3]
                g[0, 0] = L / 2
                g[0, 1] = W / 2
                g[0, 2] = L / 2
                g[0, 3] = W / 2

                offset = (ego[0] + ego[2]) / 2 - ego[2]

                t = casadi.MX(2, 1)
                t[0, 0] = x[0, k]  + casadi.cos(x[2, k]) * offset
                t[1, 0] = x[1, k]  + casadi.sin(x[2, k]) * offset

                cost_g = 0.0
                for j in range(4):
                    cost_g += g[j] * mu_ith[j, k]
                cost_b = 0.0
                for j in range(vObs[i] - 1):
                    cost_b += bobs_ith[j] * l_ith[j, k]

                cost_dis = -cost_g + (x[0, k] + np.cos(x[2, k]) * offset) * cost1 + \
                           (x[1, k] + np.sin(x[2, k]) * offset) * cost2 - cost_b
                opti.subject_to(cost_dis >= dmin)

        opti_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6} # 'ipopt.print_level': 0

        # opti_setting = {'ipopt.max_iter': 1000}

        x_Opt = []
        u_Opt = []
        feas = False
        try:
            print("========= Runing: full dimension  with fixTime =========")
            # opti.solver('ipopt')
            opti.solver('ipopt', opti_setting)
            sol = opti.solve()
            x_Opt = np.asarray(sol.value(x))
            u_Opt = np.asarray(sol.value(u))

            feas = True
        except:
            x_Opt = np.asarray(opti.debug.value(x))
            u_Opt = np.asarray(opti.debug.value(u))
            # print(x_Opt)
            # print(u_Opt)
            # print(feas)

        return x_Opt, u_Opt, feas, Ts

    def obca_mpc8(self, Ts, P, Q, R, N, x0, xL, xU, uL, uU, xref, nObs, vObs, AObs, bObs, dmin, ego, u0, uOpt):

        # set nx, nu
        nx = 3  # state z = [x, y, theta]
        nu = 2  # control input u = [v, w]

        # set R
        R1 = R[0]
        R2 = R[1]

        # optimization model
        opti = casadi.Opti()

        # state variable
        x = opti.variable(nx, N + 1)

        # control input variable
        u = opti.variable(nu, N)

        # dual variable
        l = opti.variable(np.size(AObs, 0), N + 1)  # l = lambda
        mu = opti.variable(nObs * 4, N + 1)


        # objective fxn
        def obj_rule(x, u, xref):
            cost_x = 0
            cost_u = 0
            cost_acc = 0  # acc = acceleration
            cost_terminal = 0.0

            for t in range(N):
                for i in range(nx):
                    for j in range(nx):
                        if t < N:
                            cost_x += (x[i, t] - xref[i, t]) * Q[i, j] * (x[j, t] - xref[j, t])

            for t in range(N):
                for i in range(nu):
                    for j in range(nu):
                        if t < N:
                            cost_u += (u[i, t]) * R1[i, j] * (u[j, t])
                            # cost_u += (u[i, t] - uOpt[i, t]) * R1[i, j] * (u[j, t] - uOpt[j, t])
                        if t < (N - 1):
                            cost_acc += ((u[i, t + 1] - u[i, t]) / Ts) * R2[i, j] * ((u[j, t + 1] - u[j, t]) / Ts)
                        if t == 0:
                            cost_acc += ((u[i, 0] - u[i]) / Ts) * R2[i, j] * ((u[j, 0] - u[j]) / Ts)

            for i in range(nx):
                for j in range(nx):
                    cost_terminal += (x[i, N] - xref[i, N]) * P[i, j] * (x[j, N] - xref[j, N])

            return cost_x + cost_u + cost_acc + cost_terminal

        opti.minimize(obj_rule(x, u, xref))

        #############################
        # dynamic fxn
        #############################
        for k in range(N):
            opti.subject_to(x[0, k + 1] == x[0, k] + Ts * (u[0, k] * casadi.cos(x[2, k])))
            opti.subject_to(x[1, k + 1] == x[1, k] + Ts * (u[0, k] * casadi.sin(x[2, k])))
            opti.subject_to(x[2, k + 1] == x[2, k] + Ts * u[1, k])

        #############################
        # state constraint
        #############################
        for i in range(nx - 1):
            opti.subject_to(opti.bounded(xL[i], x[i, :], xU[i]))

        #############################
        # input constraint
        #############################
        for i in range(nu):
            opti.subject_to(opti.bounded(uL[i], u[i, :], uU[i]))

        #############################
        # acceleration constraint
        #############################
        for k in range(N):
            if k == 0:
                acc = (u0[0] - u[0, k]) / Ts
                angle_acc = (u0[1] - u[1, k]) / Ts
                opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

            else:
                acc = (u[0, k - 1] - u[0, k]) / Ts
                angle_acc = (u[1, k - 1] - u[1, k]) / Ts
                opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

        #############################
        # initial state constraint
        #############################
        opti.subject_to(x[:, 0] == x0)
        # opti.subject_to(u[:, 0] == u0)

        #############################
        # terminal state constraint
        #############################


        #############################
        # positivity constraints on dual multipliers
        #############################
        for k in range(N + 1):
            opti.subject_to(l[:, k] >= 0)
            opti.subject_to(mu[:, k] >= 0)

        #############################
        # obstacle avoidance constraint
        #############################
        n = 0
        for k in range(0, N + 1):
            for i in range(nObs):
                Aobs_ith = casadi.MX(vObs[i] - 1, 2)
                bobs_ith = casadi.MX(vObs[i] - 1, 1)
                l_ith = l[n: n + vObs[i] - 1, :]
                mu_ith = mu[i * 4: (i+1) * 4, :]

                for j in range(vObs[i] - 1):
                    Aobs_ith[j, 0] = AObs[n, 0]
                    Aobs_ith[j, 1] = AObs[n, 1]
                    bobs_ith[j, 0] = bObs[n, 0]
                    n += 1


                # norm(Aobs_ith.T * lambda) <= 1
                cost1 = 0.0
                cost2 = 0.0
                for j in range(vObs[i] - 1):
                    cost1 += Aobs_ith[j, 0] * l_ith[j, k]
                    cost2 += Aobs_ith[j, 1] * l_ith[j, k]
                opti.subject_to(cost1 ** 2 + cost2 ** 2 <= 1)

                # G.T * mu + R.T * Aobs_ith.T * lambda == 0
                cost_G1 = 0.0
                cost_G1 = 0.0
                cost_G1 = (mu_ith[0, k] - mu_ith[2, k]) + np.cos(x[2, k]) * cost1 + np.sin(x[2, k]) * cost2
                cost_G2 = (mu_ith[1, k] - mu_ith[3, k]) - np.sin(x[2, k]) * cost1 + np.cos(x[2, k]) * cost2
                opti.subject_to(cost_G1 == 0)
                opti.subject_to(cost_G2 == 0)

                # -g' * mu + (AObs_ith * t - bObs_ith) * lambda > 0
                g = casadi.MX(1, 4)
                L = ego[0] + ego[2]
                W = ego[1] + ego[3]
                g[0, 0] = L / 2
                g[0, 1] = W / 2
                g[0, 2] = L / 2
                g[0, 3] = W / 2

                offset = (ego[0] + ego[2]) / 2 - ego[2]

                t = casadi.MX(2, 1)
                t[0, 0] = x[0, k]  + casadi.cos(x[2, k]) * offset
                t[1, 0] = x[1, k]  + casadi.sin(x[2, k]) * offset

                cost_g = 0.0
                for j in range(4):
                    cost_g += g[j] * mu_ith[j, k]
                cost_b = 0.0
                for j in range(vObs[i] - 1):
                    cost_b += bobs_ith[j] * l_ith[j, k]

                cost_dis = -cost_g + (x[0, k] + np.cos(x[2, k]) * offset) * cost1 + \
                           (x[1, k] + np.sin(x[2, k]) * offset) * cost2 - cost_b
                opti.subject_to(cost_dis >= dmin)

        opti_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6} # 'ipopt.print_level': 0

        # opti_setting = {'ipopt.max_iter': 1000}

        x_Opt = []
        u_Opt = []
        feas = False
        try:
            print("========= Runing: full dimension  with fixTime & No terminal constraint =========")
            # opti.solver('ipopt')
            opti.solver('ipopt', opti_setting)
            sol = opti.solve()
            x_Opt = np.asarray(sol.value(x))
            u_Opt = np.asarray(sol.value(u))

            feas = True
        except:
            x_Opt = np.asarray(opti.debug.value(x))
            u_Opt = np.asarray(opti.debug.value(u))
            # print(x_Opt)
            # print(u_Opt)
            # print(feas)

        return x_Opt, u_Opt, feas, Ts

    def obca_mpc_dyn(self, Ts, P, Q, R, N, x0, xL, xU, uL, uU, xref, nObs, vObs, AObs, bObs, dmin, ego, u0, lObs, obs_info):
        # set nx, nu
        nx = 3  # state z = [x, y, theta]
        nu = 2  # control input u = [v, w]

        # set R
        R1 = R[0]
        R2 = R[1]

        # optimization model
        opti = casadi.Opti()

        # state variable
        x = opti.variable(nx, N + 1)

        # control input variable
        u = opti.variable(nu, N)
        opti.set_initial(u[0, 0], 0.1)
        opti.set_initial(u[1, 0], 0.001)

        # dual variable
        l = opti.variable(np.size(AObs, 0), N + 1)  # l = lambda
        mu = opti.variable(nObs * 4, N + 1)

        # slack variable
        # s = opti.variable(nObs, N + 1)

        # timescale variable
        Topt = opti.variable(N + 1)
        opti.set_initial(Topt, 1)

        # # vertex variable
        v1_x = opti.variable(sum(vObs) - nObs, N + 1)
        v1_y = opti.variable(sum(vObs) - nObs, N + 1)
        v2_x = opti.variable(sum(vObs) - nObs, N + 1)
        v2_y = opti.variable(sum(vObs) - nObs, N + 1)

        for i in range(nObs):
            for j in range(vObs[i] - 1):
                opti.set_initial(v1_x, lObs[i][j][0])
                opti.set_initial(v1_y, lObs[i][j][1])
                opti.set_initial(v2_x, lObs[i][j + 1][0])
                opti.set_initial(v2_y, lObs[i][j + 1][1])

        # objective fxn
        def obj_rule(x, u, xref, Topt):
            cost_x = 0
            cost_u = 0
            cost_acc = 0  # acc = acceleration
            cost_terminal = 0.0
            cost_t = 0.0
            cost_s = 0.0

            for t in range(N):
                for i in range(nx):
                    for j in range(nx):
                        if t < N:
                            cost_x += (x[i, t] - xref[i, t]) * Q[i, j] * (x[j, t] - xref[j, t])

            for t in range(N):
                for i in range(nu):
                    for j in range(nu):
                        if t < N:
                            cost_u += (u[i, t]) * R1[i, j] * (u[j, t])
                        if t < (N - 1):
                            cost_acc += ((u[i, t + 1] - u[i, t]) / (Topt[t] * Ts)) * R2[i, j] * ((u[j, t + 1] - u[j, t]) / (Topt[t] * Ts))
                        if t == 0:
                            cost_acc += ((u[i, 0] - u[i]) / (Topt[0] * Ts)) * R2[i, j] * ((u[j, 0] - u[j]) / (Topt[0] * Ts))

            for i in range(nx):
                for j in range(nx):
                    cost_terminal += (x[i, N] - xref[i, N]) * P[i, j] * (x[j, N] - xref[j, N])

            for t in range(N + 1):
                cost_t += 0.5 * Topt[t] + 1 * Topt[t] ** 2

            # for t in range(N + 1):
            #     for i in range(nObs):
            #         cost_s += s[i, t]
            #     cost_s = 0.5 * cost_s

            return cost_x + cost_u + cost_acc + cost_terminal + cost_t  # + cost_s

        opti.minimize(obj_rule(x, u, xref, Topt))

        #############################
        # dynamic fxn
        #############################
        for k in range(N):
            opti.subject_to(x[0, k + 1] == x[0, k] + (Topt[k] * Ts) * (u[0, k] * casadi.cos(x[2, k])))
            opti.subject_to(x[1, k + 1] == x[1, k] + (Topt[k] * Ts) * (u[0, k] * casadi.sin(x[2, k])))
            opti.subject_to(x[2, k + 1] == x[2, k] + (Topt[k] * Ts) * u[1, k])
            opti.subject_to(Topt[k] == Topt[k + 1])

        #############################
        # state constraint
        #############################
        for i in range(nx - 1):
            opti.subject_to(opti.bounded(xL[i], x[i, :], xU[i]))

        #############################
        # input constraint
        #############################
        for i in range(nu):
            opti.subject_to(opti.bounded(uL[i], u[i, :], uU[i]))

        #############################
        # acceleration constraint
        #############################
        for k in range(N):
            if k == 0:
                acc = (u0[0] - u[0, k]) / (Topt[k] * Ts)
                angle_acc = (u0[1] - u[1, k]) / (Topt[k] * Ts)
                opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

            else:
                acc = (u[0, k - 1] - u[0, k]) / (Topt[k] * Ts)
                angle_acc = (u[1, k - 1] - u[1, k]) / (Topt[k] * Ts)
                opti.subject_to(opti.bounded(-0.6, acc, 0.6))
                opti.subject_to(opti.bounded(-casadi.pi / 6, angle_acc, casadi.pi / 6))

        #############################
        # initial state constraint
        #############################
        opti.subject_to(x[:, 0] == x0)
        # opti.subject_to(u[0, 0] == 0.6)
        # opti.subject_to(u[1, 0] == casadi.pi/6)

        #############################
        # terminal state constraint
        #############################
        opti.subject_to(x[:, N] == xref[:, N])

        #############################
        # positivity constraints on dual multipliers
        #############################
        for k in range(N + 1):
            opti.subject_to(l[:, k] >= 0)
            opti.subject_to(mu[:, k] >= 0)
            # opti.subject_to(Topt[k] > 0)

            dis = np.sqrt((xref[0, N] - x0[0]) ** 2 + (xref[1, N] - x0[1]))
            max_Topt = dis / (N * uU[0] * Ts) + 1
            opti.subject_to(opti.bounded(0.0001, Topt[k], max_Topt))
            # opti.subject_to(opti.bounded(0.8, Topt[k], 1.2))
            # opti.subject_to(opti.bounded(0.05, Topt[k] * Ts, 10))
            # opti.subject_to(s[i] >= 0)

        n1 = 0
        for i in range(nObs):
            for j in range(vObs[i] - 1):
                opti.subject_to(v1_x[n1, 0] == lObs[i][j][0])
                opti.subject_to(v1_y[n1, 0] == lObs[i][j][1])
                opti.subject_to(v2_x[n1, 0] == lObs[i][j + 1][0])
                opti.subject_to(v2_y[n1, 0] == lObs[i][j + 1][1])
                n1 += 1

        #############################
        # obstacle avoidance constraint
        #############################

        for k in range(0, N):
            n = 0
            for i in range(nObs):
                Aobs_ith = casadi.MX(vObs[i] - 1, 2)
                bobs_ith = casadi.MX(vObs[i] - 1, 1)
                l_ith = l[n: n + vObs[i] - 1, :]
                mu_ith = mu[i * 4: (i+1) * 4, :]

                # calc Aobs and bobs inside mpc
                m = n
                for j in range(vObs[i] - 1):
                    # # extract two vertices
                    # v1 = lObs[i][j]  # vertex 1
                    # v2 = lObs[i][j + 1]  # vertex 2
                    #
                    # # update the vertices with predict obstacle movement
                    # v1[0] = v1[0] + (Topt[k] * Ts) * obs_info[i][5] * np.cos(obs_info[i][2]) * k
                    # v1[1] = v1[1] + (Topt[k] * Ts) * obs_info[i][5] * np.sin(obs_info[i][2]) * k
                    # v2[0] = v2[0] + (Topt[k] * Ts) * obs_info[i][5] * np.cos(obs_info[i][2]) * k
                    # v2[1] = v2[1] + (Topt[k] * Ts) * obs_info[i][5] * np.sin(obs_info[i][2]) * k

                    # v1 = casadi.MX(1, 2)
                    # v2 = casadi.MX(1, 2)
                    #
                    # v1[0, 0] = lObs[i][j][0]
                    # v1[0, 1] = lObs[i][j][1]
                    # v2[0, 0] = lObs[i][j + 1][0]
                    # v2[0, 1] = lObs[i][j + 1][1]

                    # update the vertices with predict obstacle movement
                    # opti.subject_to(v1[0, 0] == v1[0, 0] + (Topt[k] * Ts) * obs_info[i][5] * casadi.cos(obs_info[i][2]) * k)
                    # opti.subject_to(v1[0, 1] == v1[0, 1] + (Topt[k] * Ts) * obs_info[i][5] * casadi.sin(obs_info[i][2]) * k)
                    # opti.subject_to(v2[0, 0] == v2[0, 0] + (Topt[k] * Ts) * obs_info[i][5] * casadi.cos(obs_info[i][2]) * k)
                    # opti.subject_to(v2[0, 1] == v2[0, 1] + (Topt[k] * Ts) * obs_info[i][5] * casadi.sin(obs_info[i][2]) * k)

                    # opti.subject_to(v1_x[n, k + 1] == v1_x[n, k] + (Topt[k] * Ts) * obs_info[i][5] * casadi.cos(obs_info[i][2]) * k)
                    # opti.subject_to(v1_y[n, k + 1] == v1_y[n, k] + (Topt[k] * Ts) * obs_info[i][5] * casadi.sin(obs_info[i][2]) * k)
                    # opti.subject_to(v2_x[n, k + 1] == v2_x[n, k] + (Topt[k] * Ts) * obs_info[i][5] * casadi.cos(obs_info[i][2]) * k)
                    # opti.subject_to(v2_y[n, k + 1] == v2_y[n, k] + (Topt[k] * Ts) * obs_info[i][5] * casadi.sin(obs_info[i][2]) * k)
                    v1_x[n, k + 1] = v1_x[n, k] + (Topt[k] * Ts) * obs_info[i][5] * casadi.cos(obs_info[i][2])
                    v1_y[n, k + 1] = v1_y[n, k] + (Topt[k] * Ts) * obs_info[i][5] * casadi.sin(obs_info[i][2])
                    v2_x[n, k + 1] = v2_x[n, k] + (Topt[k] * Ts) * obs_info[i][5] * casadi.cos(obs_info[i][2])
                    v2_y[n, k + 1] = v2_y[n, k] + (Topt[k] * Ts) * obs_info[i][5] * casadi.sin(obs_info[i][2])


                    # v1[0, 0] = v1[0, 0] + Ts * obs_info[i][5] * casadi.cos(obs_info[i][2]) * k
                    # v1[0, 1] = v1[0, 1] + Ts * obs_info[i][5] * casadi.sin(obs_info[i][2]) * k
                    # v2[0, 0] = v2[0, 0] + Ts * obs_info[i][5] * casadi.cos(obs_info[i][2]) * k
                    # v2[0, 1] = v2[0, 1] + Ts * obs_info[i][5] * casadi.sin(obs_info[i][2]) * k

                    # v1[0, 0] = v1[0, 0] + (Topt[k] * Ts) * obs_info[i][5] * casadi.cos(obs_info[i][2]) * k
                    # v1[0, 1] = v1[0, 1] + (Topt[k] * Ts) * obs_info[i][5] * casadi.sin(obs_info[i][2]) * k
                    # v2[0, 0] = v2[0, 0] + (Topt[k] * Ts) * obs_info[i][5] * casadi.cos(obs_info[i][2]) * k
                    # v2[0, 1] = v2[0, 1] + (Topt[k] * Ts) * obs_info[i][5] * casadi.sin(obs_info[i][2]) * k


                    # find hyperplane passing through v1 and v2
                    # a = (v2[1] - v1[1]) / (v2[0] - v1[0])
                    # b = v1[1] - a * v1[0]
                    #
                    # Ab_1 = casadi.MX(1, 3)
                    # Ab_1[0, 0] = 1
                    # Ab_1[0, 1] = 0
                    # Ab_1[0, 2] = v1[0]
                    #
                    # Ab_2 = casadi.MX(1, 3)
                    # Ab_2[0, 0] = -1
                    # Ab_2[0, 1] = 0
                    # Ab_2[0, 2] = -v1[0]
                    #
                    # Ab_3 = casadi.MX(1, 3)
                    # Ab_3[0, 0] = 0
                    # Ab_3[0, 1] = 1
                    # Ab_3[0, 2] = v1[1]
                    #
                    # Ab_4 = casadi.MX(1, 3)
                    # Ab_4[0, 0] = 0
                    # Ab_4[0, 1] = -1
                    # Ab_4[0, 2] = -v1[1]
                    #
                    # Ab_5 = casadi.MX(1, 3)
                    # Ab_5[0, 0] = -a
                    # Ab_5[0, 1] = 1
                    # Ab_5[0, 2] = b
                    #
                    # Ab_6 = casadi.MX(1, 3)
                    # Ab_6[0, 0] = a
                    # Ab_6[0, 1] = -1
                    # Ab_6[0, 2] = -b

                    # Ab_tmp = casadi.MX(1, 3)
                    # Ab_tmp = casadi.if_else(casadi.logic_and(v1[0] == v2[0], v2[1] < v1[1]), Ab_1,
                    #                        casadi.if_else(casadi.logic_and(v1[0] == v2[0], v2[1] >= v1[1]), Ab_2,
                    #                        casadi.if_else(casadi.logic_and(v1[1] == v2[1], v1[0] < v2[0]), Ab_3,
                    #                        casadi.if_else(casadi.logic_and(v1[1] == v2[1], v1[0] >= v2[0]), Ab_4,
                    #                        casadi.if_else(casadi.logic_and(v1[1] != v2[1], v1[0] < v2[0]), Ab_5, Ab_6)))))
                    # Ab_tmp = casadi.if_else(v1[0] == v2[0], casadi.if_else(v2[1] < v1[1], Ab_1, Ab_2),
                    #                          casadi.if_else(v1[1] == v2[1], casadi.if_else(v1[0] < v2[0], Ab_3, Ab_4),
                    #                                         casadi.if_else(v1[0] < v2[0], Ab_5, Ab_6)))

                    # Ab_tmp1 = casadi.if_else(v1[0, 0] == v2[0, 0], casadi.if_else(v2[0, 1] < v1[0, 1], 1, -1),
                    #                         casadi.if_else(v1[0, 1] == v2[0, 1], 0,
                    #                         casadi.if_else(v1[0, 0] < v2[0, 0], -(v2[0, 1] - v1[0, 1]) / (v2[0,0] - v1[0,0]),
                    #                                       (v2[0, 1] - v1[0, 1]) / (v2[0, 0] - v1[0, 0]))))
                    #
                    # Ab_tmp2 = casadi.if_else(v1[0, 0] == v2[0, 0], 0,
                    #                         casadi.if_else(v1[0, 1] == v2[0, 1], casadi.if_else(v1[0, 0] < v2[0, 0], 1, -1),
                    #                         casadi.if_else(v1[0, 0] < v2[0, 0], 1, -1)))
                    #
                    # Ab_tmp3 = casadi.if_else(v1[0, 0] == v2[0, 0], casadi.if_else(v2[0, 1] < v1[0, 1], v1[0, 0], -v1[0, 0]),
                    #                         casadi.if_else(v1[0, 1] == v2[0, 1], casadi.if_else(v1[0, 0] < v2[0, 0], v1[0, 1], -v1[0, 1]),
                    #                         casadi.if_else(v1[0, 0] < v2[0, 0], v1[0, 1] + Ab_tmp2 * v1[0, 0], -v1[0, 1] + Ab_tmp2 * v1[0, 0])))

                    Ab_tmp1 = casadi.if_else(v1_x[n, k] == v2_x[n, k], casadi.if_else(v2_y[n, k] < v1_y[n, k], 1, -1),
                                             casadi.if_else(v1_y[n, k] == v2_y[n, k], 0,
                                             casadi.if_else(v1_x[n, k] < v2_x[n, k], -(v2_y[n, k] - v1_y[n, k]) / (
                                                                                       v2_x[n, k] - v1_x[n, k]),
                                                                           (v2_y[0, 1] - v1_y[n, k]) / (
                                                                                       v2_x[n, k] - v1_x[n, k]))))

                    Ab_tmp2 = casadi.if_else(v1_x[n, k] == v2_x[n, k], 0,
                                             casadi.if_else(v1_y[n, k] == v2_y[n, k],
                                                            casadi.if_else(v1_x[n, k] < v2_x[n, k], 1, -1),
                                                            casadi.if_else(v1_x[n, k] < v2_x[n, k], 1, -1)))

                    Ab_tmp3 = casadi.if_else(v1_x[n, k] == v2_x[n, k], casadi.if_else(v2_y[n, k] < v1_y[n, k], v1_x[n, k], -v1_x[n, k]),
                                             casadi.if_else(v1_y[n, k] == v2_y[n, k],
                                                            casadi.if_else(v1_x[n, k] < v2_x[n, k], v1_y[n, k], -v1_y[n, k]),
                                                            casadi.if_else(v1_x[n, k] < v2_x[n, k], v1_y[n, k] + Ab_tmp2 * v1_x[n, k],
                                                                           -v1_y[n, k] + Ab_tmp2 * v1_x[n, k])))

                    # if v1[0] == v2[0]:  # perpendicular hyperplane, not captured by general formula
                    #     if v2[1] < v1[1]:
                    #         A_tmp = [1, 0]
                    #         b_tmp = v1[0]
                    #     else:
                    #         A_tmp = [-1, 0]
                    #         b_tmp = -v1[0]
                    # elif v1[1] == v2[
                    #     1]:  # horizontal hyperplane, captured by general formula but included for numerical stability
                    #     if v1[0] < v2[0]:
                    #         A_tmp = [0, 1]
                    #         b_tmp = v1[1]
                    #     else:
                    #         A_tmp = [0, -1]
                    #         b_tmp = -v1[1]
                    # else:  # general formula for non-horizontal and non-vertical hyperplanes
                    #     a = (v2[1] - v1[1]) / (v2[0] - v1[0])
                    #     b = v1[1] - a * v1[0]
                    #
                    #     if v1[0] < v2[0]:  # v1 --> v2 (line moves right)
                    #         A_tmp = [-a, 1]
                    #         b_tmp = b
                    #     else:  # v2 <-- v1 (line moves left)
                    #         A_tmp = [a, -1]
                    #         b_tmp = -b

                    # store vertices
                    # Aobs_ith[j, 0] = Ab_tmp[0, 0]
                    # Aobs_ith[j, 1] = Ab_tmp[0, 1]
                    # bobs_ith[j, 0] = Ab_tmp[0, 2]
                    Aobs_ith[j, 0] = Ab_tmp1[0, 0]
                    Aobs_ith[j, 1] = Ab_tmp2[0, 0]
                    bobs_ith[j, 0] = Ab_tmp3[0, 0]
                    n += 1

                # norm(Aobs_ith.T * lambda) <= 1
                cost1 = 0.0
                cost2 = 0.0
                for j in range(vObs[i] - 1):
                    cost1 += Aobs_ith[j, 0] * l_ith[j, k]
                    cost2 += Aobs_ith[j, 1] * l_ith[j, k]
                opti.subject_to(cost1 ** 2 + cost2 ** 2 <= 1)

                # G.T * mu + R.T * Aobs_ith.T * lambda == 0
                cost_G1 = 0.0
                cost_G1 = 0.0
                cost_G1 = (mu_ith[0, k] - mu_ith[2, k]) + np.cos(x[2, k]) * cost1 + np.sin(x[2, k]) * cost2
                cost_G2 = (mu_ith[1, k] - mu_ith[3, k]) - np.sin(x[2, k]) * cost1 + np.cos(x[2, k]) * cost2
                opti.subject_to(cost_G1 == 0)
                opti.subject_to(cost_G2 == 0)

                # -g' * mu + (AObs_ith * t - bObs_ith) * lambda > 0
                g = casadi.MX(1, 4)
                L = ego[0] + ego[2]
                W = ego[1] + ego[3]
                g[0, 0] = L / 2
                g[0, 1] = W / 2
                g[0, 2] = L / 2
                g[0, 3] = W / 2

                offset = (ego[0] + ego[2]) / 2 - ego[2]

                t = casadi.MX(2, 1)

                t[0, 0] = x[0, k]  + casadi.cos(x[2, k]) * offset
                t[1, 0] = x[1, k]  + casadi.sin(x[2, k]) * offset

                cost_g = 0.0
                for j in range(4):
                    cost_g += g[j] * mu_ith[j, k]
                cost_b = 0.0
                for j in range(vObs[i] - 1):
                    cost_b += bobs_ith[j] * l_ith[j, k]

                cost_dis = -cost_g + (x[0, k] + np.cos(x[2, k]) * offset) * cost1 + \
                           (x[1, k] + np.sin(x[2, k]) * offset) * cost2 - cost_b

                opti.subject_to(cost_dis >= dmin)
                # opti.subject_to(cost_dis >= s[i, k])

            # # update obstacle vertex
            # for i in range(nObs):
            #     for j in range(vObs[i] - 1):
            #         print(Topt[k])
            #         lObs[i][j][1] = lObs[i][j][0] + (Topt[k] * Ts) * obs_info[i][5] * np.cos(obs_info[i][2])
            #         lObs[i][j][1] = lObs[i][j][1] + (Topt[k] * Ts) * obs_info[i][5] * np.sin(obs_info[i][2])
            #
            # obs_model = obstacleModel()
            # AObs, bObs = obs_model.obstacle_H_Represent(nObs, vObs, lObs)
            # lObs_list.append(lObs)


        # opti_setting = {'ipopt.max_iter': 2000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
        #                 'ipopt.acceptable_obj_change_tol': 1e-6}

        # opti_setting = {'ipopt.max_iter': 5000}

        x_Opt = []
        u_Opt = []
        feas = False
        try:
            # opti.solver('ipopt', opti_setting)
            opti.solver('ipopt')
            sol = opti.solve()
            x_Opt = np.asarray(sol.value(x))
            u_Opt = np.asarray(sol.value(u))
            print("dyn obstacle mpc ====================")
            # print(sol.value(Topt * Ts))
            # print(opti.debug.value(Topt))
            # print("v1_x ====================")
            # print(opti.debug.value(v1_x))
            # print("v1_y ====================")
            # print(opti.debug.value(v1_y))
            # print("v2_x ====================")
            # print(opti.debug.value(v2_x))
            # print("v2_y ====================")
            # print(opti.debug.value(v2_y))
            Ts_opt = np.asarray(sol.value(Topt[0]) * Ts)
            feas = True
        except:
            # print(feas)
            x_Opt = np.asarray(opti.debug.value(x))
            u_Opt = np.asarray(opti.debug.value(u))
            print("dyn obstacle mpc Failll====================")
            # print(opti.debug.value(Topt))
            # print(opti.debug.value(v1_x))
            # print(opti.debug.value(v1_y))
            # print(opti.debug.value(v2_x))
            # print(opti.debug.value(v2_y))
            # print(opti.debug.value(Ab_tmp1))
            # print(opti.debug.value(Ab_tmp2))
            # print(opti.debug.value(Ab_tmp3))
            # print(opti.debug.value(Topt[k] * Ts))
            # print(opti.debug.value(k))
            # print(opti.debug.value(n))
            Ts_opt = np.asarray(opti.debug.value(Topt[0]) * Ts)

        return x_Opt, u_Opt, feas, Ts_opt


