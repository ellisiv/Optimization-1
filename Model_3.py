import matplotlib.pyplot as plt
import numpy as np
from Method_1_BFGS import constructproblem
from Method_1_BFGS import r1
from Method_1_BFGS import rxy
from Method_1_BFGS import plot_solution
from Method_1_BFGS import f1
from Method_1_BFGS import generate_noise
from Method_1_BFGS import generate_points
from Method_1_BFGS import grad1
from Method_1_BFGS import BFGS_model_1
from Method_1_BFGS import convergence_plot
from Model_2 import BFGS_model_2
from Model_2 import grad2
from Model_2 import f2
from Model_2 import rxy_tilde

def construct_circle_params(x):

    c = np.array([x[0], x[1]])
    rho = x[2]
    d = np.array([x[3], x[4]])
    sigma = x[5]

    return c, rho, d, sigma


def r3(zi, center, radius):
    return np.linalg.norm((zi - center), 2) ** 2 - radius ** 2

"""
def rxy(A, c, x, y):
    return (x - c[0]) * (A[0, 0] * (x - c[0]) + A[0, 1] * (y - c[1])) + (y - c[1]) * (A[1, 0] * (x - c[0]) + A[1, 1] * (y - c[1])) - 1
"""


def f3(x, z, a, b):
    c, rho, d, sigma = construct_circle_params(x)
    sum = 0

    for i in range(len(z)):
        if i in a and i not in b:
            sum += max(r3(z[i], c, rho), 0) ** 2 + min(r3(z[i], d, sigma), 0) ** 2
        elif i in a:
            sum += max(r3(z[i], c, rho), 0) ** 2 + max(r3(z[i], d, sigma), 0) ** 2
        elif i in b:
            sum += min(r3(z[i], c, rho), 0) ** 2 + max(r3(z[i], d, sigma), 0) ** 2
        else:
            sum += min(r3(z[i], c, rho), 0) ** 2 + min(r3(z[i], d, sigma), 0) ** 2
    return sum


def grad3(x, z, inner):
    g1 = np.zeros((2, 2))
    g2 = np.zeros(2)
    A, c = constructproblem(x)
    for i in range(len(z)):
        r = r1(z[i], A, c)
        if (i in inner and r > 0) or ((i not in inner) and r < 0):
            g1 += 2 * r * np.outer((z[i] - c), (z[i] - c))
            g2 += -4 * r * (z[i] - c) @ A
    g = np.array([g1[0, 0], g1[0, 1], g1[1, 1], g2[0], g2[1]])
    return g