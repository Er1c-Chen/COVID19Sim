import numpy as np
from matplotlib import pyplot as plt
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController, SaveAt
import jax.numpy as jnp


class ODE:
    def __init__(self, beta=3e-8, eta=0.5, alpha=4, nu=0.2, rI=0.027, dI=0.003, rH=0.049, dH=0.001, mu=1/14,
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

    def equation(self, t, y, args):
        S1, S2, E1, E2, I1, I2, H, R1, R2, R3 = y

        return jnp.array([
            -self.beta * S1 * (self.eta * E1 + I1) + self.mu * S2 - self.omega * self.nS * self.lambda_ * I1,
            self.omega * self.nS * self.lambda_ * I1 - self.mu * S2,
            self.beta * S1 * (self.eta * E1 + I1) - self.omega * self.nE * self.lambda_ * I1 - self.alpha * E1,
            self.omega * self.nE * self.lambda_ * I1 - self.alpha * E2,
            self.alpha * E1 - self.lambda_ * I1 - self.rI * I1 - self.dI * I1,
            self.alpha * E2 + self.lambda_ * I1 - self.nu * I2 - self.rI * I2 - self.dI * I2,
            self.nu * I2 - self.rH * H - self.dH * H,
            self.rI * (I1 + I2),
            self.rH * H,
            self.dI * (I1 + I2) + self.dH * H
        ])

    def solve(self, popu, days, q_day):
        population = popu * np.ones(days)
        term = ODETerm(self.equation)
        solver = Dopri5()
        stepsize_controller = PIDController(rtol=1e-3, atol=1e-3)
        saveat = SaveAt(ts=list(range(days)))
        sol = diffeqsolve(term, solver, t0=0, t1=days, dt0=0.1, y0=jnp.array([popu, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
                          saveat=saveat, stepsize_controller=stepsize_controller)
        print(sol.ts)
        print(sol.ys)
        plt.plot(sol.ts, population - sol.ys[:, 0] - sol.ys[:, 1], label='infected')
        plt.legend()
        plt.show()

model = ODE()
model.solve(1000000, 40, 1)