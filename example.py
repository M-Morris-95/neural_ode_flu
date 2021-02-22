from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
import torch

def func(t, z):
    return t

class Oscillation:
    def __init__(self, km):
        self.mat = torch.Tensor([[0, 1],
                                 [-km, 0]])

    def solve(self, t, x0, dx0):
        y0 = torch.cat([x0, dx0])
        out = odeint(self.func, y0, t)
        return out

    def func(self, t, y):
        # print(t)
        out = y @ self.mat  # @Is a matrix product
        return out


x0 = torch.Tensor([1])
dx0 = torch.Tensor([0])

t = torch.linspace(0, 4 * np.pi, 1000)
solver = Oscillation(1)
out = solver.solve(t, x0, dx0)