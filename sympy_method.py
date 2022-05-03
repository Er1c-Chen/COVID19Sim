from sympy import Function, Eq
from sympy import dsolve, Derivative, symbols
from sympy.abc import t
import numpy as np

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

    def solve(self):
        S1, S2, E1, E2, I1, I2, H, R1, R2, R3 = symbols('S1 S2 E1 E2 I1 I2 H R1 R2 R', cls=Function)

        eq = (Eq(Derivative(S1(t), t), -self.beta * S1(t) * (self.eta * E1(t) + I1(t)) + self.mu * S2(t) - self.omega * self.nS * self.lambda_ * I1(t)),
            Eq(Derivative(S2(t), t), self.omega * self.nS * self.lambda_ * I1(t) - self.mu * S2(t)),
            Eq(Derivative(E1(t), t), self.beta * S1(t) * (self.eta * E1(t) + I1(t)) - self.omega * self.nE * self.lambda_ * I1(t) - self.alpha * E1(t)),
            Eq(Derivative(E2(t), t), self.omega * self.nE * self.lambda_ * I1(t) - self.alpha * E2(t)),
            Eq(Derivative(I1(t), t), self.alpha * E1(t) - self.lambda_ * I1(t) - self.rI * I1(t) - self.dI * I1(t)),
            Eq(Derivative(I2(t), t), self.alpha * E2(t) + self.lambda_ * I1(t) - self.nu * I2(t) - self.rI * I2(t) - self.dI * I2(t)),
            Eq(Derivative(H(t), t), self.nu * I2(t) - self.rH * H(t) - self.dH * H(t)),
            Eq(Derivative(R1(t), t), self.rI * (I1(t) + I2(t))),
            Eq(Derivative(R2(t), t), self.rH * H(t)),
            Eq(Derivative(R3(t), t), self.dI * (I1(t) + I2(t)) + self.dH * H(t)))

        popu = 10
        con = {S1(0):popu, S2(0):0, E1(0):1, E2(0):0, I1(0):0, I2(0):0, H(0):0, R1(0):0, R2(0):0, R3(0):0}
        res = dsolve(eq, ics=con)
        print(res)



model = ODE()
model.solve()