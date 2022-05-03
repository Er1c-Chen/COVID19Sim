import numpy as np
from matplotlib import pyplot as plt

class ODE:
    def __init__(self, beta=3e-8, eta=0.5, alpha=4, nu=0.2, rI=0.027, dI=0.003, rH=0.049, dH=0.001, mu=1 / 14,
                 lambda_=0.1, omega=0, nS=10, nE=2, population=50000, interval=400, start=30):
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
        self.population = population
        self.interval = interval
        self.start = start

    def calculate(self):
        delta_t = 0.001
        S1, S2, E1, E2, I1, I2, H, R1, R2, R3 = [self.population, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        S1_cum = [S1]
        S2_cum = [S2]
        E1_cum = [E1]
        E2_cum = [E2]
        I1_cum = [I1]
        I2_cum = [I2]
        H_cum = [H]
        R1_cum = [R1]
        R2_cum = [R2]
        R3_cum = [R3]

        for i in range((self.interval*1000)-1):
            S1_d = -self.beta * S1 * (self.eta * E1 + I1) + self.mu * S2 - min(
                self.omega * self.nS * self.lambda_ * I1, S1)
            S2_d = min(self.omega * self.nS * self.lambda_ * I1, S1) - self.mu * S2
            E1_d = self.beta * S1 * (self.eta * E1 + I1) - min(self.omega * self.nE * self.lambda_ * I1,
                                                               E1) - self.alpha * E1
            E2_d = min(self.omega * self.nE * self.lambda_ * I1, E1) - self.alpha * E2
            I1_d = self.alpha * E1 - self.lambda_ * I1 - self.rI * I1 - self.dI * I1
            I2_d = self.alpha * E2 + self.lambda_ * I1 - self.nu * I2 - self.rI * I2 - self.dI * I2
            H_d = self.nu * I2 - self.rH * H - self.dH * H
            R1_d = self.rI * (I1 + I2)
            R2_d = self.rH * H
            R3_d = self.dI * (I1 + I2) + self.dH * H


            S1 = S1 + S1_d * delta_t
            S2 = S2 + S2_d * delta_t
            E1 = E1 + E1_d * delta_t
            E2 = E2 + E2_d * delta_t
            I1 = I1 + I1_d * delta_t
            I2 = I2 + I2_d * delta_t
            H  = H + H_d * delta_t
            R1 = R1 + R1_d * delta_t
            R2 = R2 + R2_d * delta_t
            R3 = R3 + R3_d * delta_t

            S1_cum.append(S1)
            S2_cum.append(S2)
            E1_cum.append(E1)
            E2_cum.append(E2)
            I1_cum.append(I1)
            I2_cum.append(I2)
            H_cum.append(H)
            R1_cum.append(R1)
            R2_cum.append(R2)
            R3_cum.append(R3)

        return [S1_cum, S2_cum, E1_cum, E2_cum, I1_cum, I2_cum, H_cum, R1_cum, R2_cum, R3_cum]


population = 6000000

interval = 400
start = 30
popu = population + np.ones(interval*1000)
model = ODE(population=population, interval=interval, start=start)
res = model.calculate()
print(res[0])
t = np.linspace(0, interval, interval*1000)
plt.plot(t, popu - res[1] - res[0], label='infected')
plt.legend()
plt.show()

