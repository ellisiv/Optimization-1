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
from Method_1_BFGS import linesearch_wolfe
from Method_1_BFGS import BFGS

def f1_penalty(x, z, inner, my, constraints):
    f = f1(x, z, inner)
    f += my / 2 * constraints(x)
    return f

def quadratic_penalty_method()







