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
from Constrained_m1 import c1, c2, c3, c4, c5
from Constrained_m1 import B_func, beta_optimization, grad_B, convergence_plot_constr


def model_1_BFGS_unconstrained_nice_problem():
    x = [3, 1, 3, 0, 0]
    x0 = [1, 0, 1, 0, 0]

    z, inner = generate_points(x, size=500, scale=0.7)
    z = generate_noise(z, 1 * 10 ** -1)

    plot_solution(x0, z, inner, rxy, 0, Metode=1)

    xf, nf, gradsf = BFGS_model_1(x0, z, inner, TOL=10 ** (-10), gradient_decent=0)
    plot_solution(xf, z, inner, rxy, nf, Metode=1)
    convergence_plot(gradsf, 1)

    return 0


def model_2_BFGS_unconstrained_nice_problem():
    x = [3, 1, 3, 0, 0]
    x0 = [1, 0, 1, 0, 0]

    z, inner = generate_points(x, size=500, scale=0.5)
    z = generate_noise(z, 1 * 10 ** -1)

    plot_solution(x0, z, inner, rxy_tilde, 0, Metode=2)

    xf, nf, gradsf = BFGS_model_2(x0, z, inner, TOL=10 ** (-10), gradient_decent=0)
    plot_solution(xf, z, inner, rxy_tilde, nf, Metode=2)
    convergence_plot(gradsf, 2)

    return 0


def model_2_BFGS_unconstrained_not_so_nice_problem():
    x = [0.008, 1, 0.008, 0, 0]
    x0 = [1, 0, 1, 0, 0]

    z, inner = generate_points(x, size=500)
    z = generate_noise(z, 1 * 10 ** -1)

    plot_solution(x0, z, inner, rxy_tilde, 0, Metode=2)

    xf, nf, gradsf = BFGS_model_2(x0, z, inner, TOL=10 ** (-10), gradient_decent=0)
    plot_solution(xf, z, inner, rxy_tilde, nf, Metode=2)
    convergence_plot(gradsf, 2)

    return 0


def model_1_Grad_Decent_unconstrained_nice_problem():
    x = [3, 1, 3, 0, 0]
    x0 = [1, 0, 1, 0, 0]

    z, inner = generate_points(x, size=500, scale=0.7)
    z = generate_noise(z, 1 * 10 ** -1)

    plot_solution(x0, z, inner, rxy, 0, Metode=1)

    xf, nf, gradsf = BFGS_model_1(x0, z, inner, TOL=10 ** (-6), gradient_decent=1)
    plot_solution(xf, z, inner, rxy, nf, Metode=1)
    convergence_plot(gradsf, 1)
    return 0


def model_2_Grad_Decent_unconstrained_nice_problem():
    x = [3, 1, 3, 0, 0]
    x0 = [1, 0, 1, 0, 0]

    z, inner = generate_points(x, size=500, scale=0.7)
    z = generate_noise(z, 1 * 10 ** -1)

    plot_solution(x0, z, inner, rxy_tilde, 0, Metode=2)

    xf, nf, gradsf = BFGS_model_2(x0, z, inner, TOL=10 ** (-6), gradient_decent=1)
    plot_solution(xf, z, inner, rxy_tilde, nf, Metode=2)
    convergence_plot(gradsf, 2)
    return 0


def model_2_BFGS_constrained_nice_problem():
    x = [3, 1, 3, 0, 0]

    c = [c1, c2, c3, c4, c5]

    x0 = np.array([4, 1, 3, 0, 0])
    points, inner = generate_points(x, size=300, scale=1)

    points = generate_noise(points, 2 * 10 ** (-1))
    plot_solution(x0, points, inner, rxy_tilde, 0, 2)

    xf, itr, b_vals = beta_optimization(x0, B_func, grad_B, f2, grad2, 1, c, points, inner, 0.1, 1000, n=0, TOL=10 ** (-6))

    plot_solution(xf, points, inner, rxy_tilde, np.sum(itr), 2)
    convergence_plot_constr(b_vals)
    return 0


def Model_2_constrained_not_so_nice_problem():
    x = [3, 1, 3, 0, 0]

    c = [c1, c2, c3, c4, c5]

    x0 = np.array([4, 1, 3, 0, 0])
    points, inner = generate_points(x, size=300, scale=1)

    points = generate_noise(points, 2 * 10 ** (-1))
    plot_solution(x0, points, inner, rxy_tilde, 0, 2)

    xf, itr, b_vals = beta_optimization(x0, B_func, grad_B, f2, grad2, 1, c, points, inner, 0.1, 1000, n=0, TOL=10 ** (-6))

    plot_solution(xf, points, inner, rxy_tilde, itr, 2)
    convergence_plot_constr(b_vals)
    return 0


#model_1_BFGS_unconstrained_nice_problem()
#model_2_BFGS_unconstrained_nice_problem()
#model_2_BFGS_unconstrained_not_so_nice_problem()
#model_1_Grad_Decent_unconstrained_nice_problem()
#model_2_Grad_Decent_unconstrained_nice_problem()
model_2_BFGS_constrained_nice_problem()



