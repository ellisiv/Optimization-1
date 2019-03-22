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
from Method_1_BFGS import BFGS
from Model_2 import grad2
from Model_2 import f2
from Model_2 import rxy_tilde

def method_1_unconstrained_nice_problem():
    x = [3, 1, 3, 0, 0]
    x0 = [10, 10, 10, 10, 10]

    z, inner = generate_points(x, size=300)
    noise = generate_noise(z, 10 ** -1)

    plot_solution(x0, z, inner, rxy, 0)

    xf, nf, gradsf = BFGS(x0, points, inner, 0, gradient_decent=0)
    plot_solution(xf, points, inner, rxy, nf)
    convergence_plot(gradsf, 1)

    return 0


def method_2_unconstrained_nice_problem():

    return 0