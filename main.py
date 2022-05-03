from scipy.integrate import odeint
import numpy as np
from matplotlib import pyplot as plt

class ODE:
    def __init__(self, beta=3e-8, eta=0.5, alpha=0.25, nu=0.2, rI=0.027, dI=0.003, rH=0.049, dH=0.001, mu=1/14,
                 lambda_=0.1, omega=0, nS=10, nE=2):
        self.beta = beta
        self.eta = eta
        self.alpha = alpha
        self.nu = nu
        self.rI = rI
        self.dI = dI
        self.rH = rH
        self.dH = dH
        self.mu = mu
        self.lambda_ = lambda_
        self.omega = omega
        self.nS = nS
        self.nE = nE

    def equation(self, y_list, t):
        S1, S2, E1, E2, I1, I2, H, R1, R2, R3 = y_list

        return np.array([
            -self.beta * S1 * (self.eta * E1 + I1) + self.mu * S2 - min(self.omega * self.nS * self.lambda_ * I1, S1),
            min(self.omega * self.nS * self.lambda_ * I1, S1) - self.mu * S2,
            self.beta * S1 * (self.eta * E1 + I1) - min(self.omega * self.nE * self.lambda_ * I1, E1) - self.alpha * E1,
            min(self.omega * self.nE * self.lambda_ * I1, E1) - self.alpha * E2,
            self.alpha * E1 - self.lambda_ * I1 - self.rI * I1 - self.dI * I1,
            self.alpha * E2 + self.lambda_ * I1 - self.nu * I2 - self.rI * I2 - self.dI * I2,
            self.nu * I2 - self.rH * H - self.dH * H,
            self.rI * (I1 + I2),
            self.rH * H,
            self.dI * (I1 + I2) + self.dH * H
        ], dtype=np.float64)


def draw(popu, start, interval):
    model_prev = ODE(omega=0)
    color = ['red', 'green', 'gold', 'blue', 'pink', 'indigo', 'black']
    for i, day in enumerate(start):
        model_q = ODE(omega=1)
        t_prev = np.linspace(0, day, day)


        result_prev = odeint(model_prev.equation, [popu, 0, 1, 0, 0, 0, 0, 0, 0, 0], t_prev)
        plt.plot(t_prev, (popu * np.ones(day)) - result_prev[:, 0] - result_prev[:, 1], label='t1={}'.format(day), color=color[i])

        t_q = np.linspace(day, interval, interval - day)
        result_q = odeint(model_q.equation, [result_prev[-1, i] for i in range(0, 10)], t_q)
        plt.plot(t_q, popu * np.ones(interval - day) - result_q[:, 0] - result_q[:, 1], color=color[i])
    t_no_q = np.linspace(0, interval, interval)
    result_no_q = odeint(model_prev.equation, [popu, 0, 1, 0, 0, 0, 0, 0, 0, 0], t_no_q)
    plt.plot(t_no_q, (popu * np.ones(interval)) - result_no_q[:, 0] - result_no_q[:, 1], label='No quarantine', color=color[-1])


    plt.legend()
    # plt.grid()
    plt.show()

if __name__ == '__main__':
    popu = 10000000
    start = [15, 35, 55, 75, 85, 95]
    interval = 400

    draw(popu, start, interval)