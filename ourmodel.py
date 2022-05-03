from scipy.integrate import odeint
import numpy as np
from matplotlib import pyplot as plt

class ODE:
    def __init__(self, beta=3e-8, eta=0.5, alpha=0.25, nu=0.2, rI=0.027, dI=0.003, rH=0.049, dH=0.001, mup=1/14, mus=1/14,
                 lambda_=0.1, omega=0, nS=10, nE=2, phi=0, Qmax=np.inf, theta=1):
        self.beta = beta
        self.eta = eta
        self.alpha = alpha
        self.nu = nu
        self.rI = rI
        self.dI = dI
        self.rH = rH
        self.dH = dH
        self.mup = mup
        self.mus = mus
        self.lambda_ = lambda_
        self.omega = omega
        self.nS = nS
        self.nE = nE
        self.phi = phi
        self.Qmax=Qmax
        self.theta=theta

    def equation(self, y_list, t):
        Sn, Sp, Ss, En, Ep, Es, In, Ip, Is, IH, Rr, Rd = y_list

        return np.array([
            -self.beta * (Sn * (self.eta * En +In)+self.phi*Sn*(self.eta*Es+Is))+self.mup*Sp+self.mus*Ss-min(
                min(self.omega*self.nS*self.lambda_*In, self.Qmax-Sp-Ep-Ip)+self.omega*self.phi*self.nS*self.lambda_*Is, Sn),
            self.theta*min(min(self.omega*self.nS*self.lambda_*In, self.Qmax-Sp-Ep-Ip)+self.omega*self.phi*self.nS*
                           self.lambda_*Is, Sn)-self.mup*Sp,
            -self.phi*self.beta*(Sn*(self.eta*En+In)+self.phi*Ss*(self.eta*Es+Is))+(1-self.theta)*min(min(self.omega*
              self.nS*self.lambda_*In, self.Qmax-Sp-Ep-Ip)+self.omega*self.phi*self.nS*self.lambda_*Is, Sn)-self.mus*Ss,
            self.beta*(Sn*(self.eta*En+In)+self.phi*Ss*(self.eta*Es+Is))-min(min(self.omega*self.nS*self.lambda_*In, self.Qmax-Sp-Ep-Ip)+self.omega*self.phi*self.nE*self.lambda_*Is, En)-self.alpha*Es,
            self.theta*min(min(self.omega*self.nE*self.lambda_*In, self.Qmax-Sp-Ep-Ip)+self.omega*self.phi*self.nE*self.lambda_*Is, En)-self.alpha*Ep,
            self.phi*self.beta*(Sn*(self.eta*En+In)+self.phi*Ss*(self.eta*Es+Is))+(1-self.theta)*min(min(self.omega*self.nE*self.lambda_*In, self.Qmax-Sp-Ep-Ip)+self.omega*self.phi*self.nE*self.lambda_*Is, En),
            self.alpha*En-self.lambda_*In-self.rI*In-self.dI*In,
            self.alpha*Ep+self.theta*self.lambda_*In-self.nu*Ip-self.rI*Ip-self.dI*Ip,
            self.alpha*Es+(1-self.theta)*self.lambda_*In-self.nu*Is-self.rI*Is-self.dI*Is,
            self.nu*Ip+self.nu*Is-self.rH*IH+self.dH*IH,
            self.rI*In+self.rI*Ip+self.rI*Is+self.rH*IH,
            self.dI*In+self.dI*Ip+self.dI*Is+self.dH*IH


            # -self.beta * S1 * (self.eta * E1 + I1) + self.mu * S2 - min(self.omega * self.nS * self.lambda_ * I1, S1),
            # min(self.omega * self.nS * self.lambda_ * I1, S1) - self.mu * S2,
            # self.beta * S1 * (self.eta * E1 + I1) - min(self.omega * self.nE * self.lambda_ * I1, E1) - self.alpha * E1,
            # min(self.omega * self.nE * self.lambda_ * I1, E1) - self.alpha * E2,
            # self.alpha * E1 - self.lambda_ * I1 - self.rI * I1 - self.dI * I1,
            # self.alpha * E2 + self.lambda_ * I1 - self.nu * I2 - self.rI * I2 - self.dI * I2,
            # self.nu * I2 - self.rH * H - self.dH * H,
            # self.rI * (I1 + I2),
            # self.rH * H,
            # self.dI * (I1 + I2) + self.dH * H
        ], dtype=np.float64)


def min(a, b):
    if a > b:
        return b
    else:
        return a


def draw(popu, start, interval):
    model_prev = ODE(omega=0)
    color = ['red', 'green', 'gold', 'blue', 'pink', 'indigo', 'black']
    for i, day in enumerate(start):
        model_q = ODE(omega=1)
        t_prev = np.linspace(0, day, day)


        result_prev = odeint(model_prev.equation, [popu, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], t_prev)
        plt.plot(t_prev, (popu * np.ones(day)) - result_prev[:, 0] - result_prev[:, 1] - result_prev[:, 2], label='t1={}'.format(day), color=color[i])

        t_q = np.linspace(day, interval, interval - day)
        result_q = odeint(model_q.equation, [result_prev[-1, i] for i in range(0, 12)], t_q)
        plt.plot(t_q, popu * np.ones(interval - day) - result_q[:, 0] - result_q[:, 1] - result_q[:, 2], color=color[i])
    t_no_q = np.linspace(0, interval, interval)
    result_no_q = odeint(model_prev.equation, [popu, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], t_no_q)
    plt.plot(t_no_q, (popu * np.ones(interval)) - result_no_q[:, 0] - result_no_q[:, 1] - result_no_q[:, 2], label='No quarantine', color=color[-1])


    plt.legend()
    # plt.grid()
    plt.show()

if __name__ == '__main__':
    popu = 10000000
    start = [15, 35, 55, 75, 85, 95]
    interval = 400

    draw(popu, start, interval)