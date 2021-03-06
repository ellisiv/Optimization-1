from Method_1_BFGS import constructproblem
import numpy as np
from Method_1_BFGS import plot_solution
from Method_1_BFGS import generate_noise
from Method_1_BFGS import generate_points
from Method_1_BFGS import convergence_plot

def r2(zi, A, b):
    zib = np.matmul(zi.transpose(), A)
    return np.matmul(zib, zi) - np.matmul(zi.T, b) - 1


def rxy_tilde(A, b, x, y):
    return x * (A[0, 0] * x + A[0, 1] * y) + y * (A[1, 0] * x + A[1, 1] * y) - x * b[0] - y * b[1] - 1


def f2(x, z, inner):
    A, b = constructproblem(x)
    sum = 0
    for i in range(len(z)):
        if i in inner:
            sum += (max(r2(z[i], A, b), 0)) ** 2
        else:
            sum += (min(r2(z[i], A, b), 0)) ** 2
    return sum


def grad2(x, z, inner):
    g1 = np.zeros((2, 2))
    g2 = np.zeros(2)
    A, b = constructproblem(x)
    for i in range(len(z)):
        r = r2(z[i], A, b)
        if (i in inner and r > 0) or (i not in inner and r < 0):
            g1 += 2 * r * np.outer(z[i], z[i])
            g2 += -2 * r * z[i].T
    g = np.array([g1[0, 0], g1[0, 1], g1[1, 1], g2[0], g2[1]])
    return g


def Wolfe_2(z, inner, p, x, c1=10 ** -4, c2=0.9):
    alpha = 1
    amax = np.infty
    amin = 0
    k = 0
    while ((f2((x + alpha * p), z, inner) > f2(x, z, inner) + c1 * alpha * np.matmul(grad2(x, z, inner).T, p)) or
            (np.matmul(grad2(x + alpha * p, z, inner).T, p) < c2 * np.matmul(grad2(x, z, inner).T, p))) and k < 20:
        if f2(x + alpha * p, z, inner) > f2(x, z, inner) + c1 * alpha * np.matmul(grad2(x, z, inner).T, p):
            amax = alpha
            alpha = (amax + amin) / 2
            k += 1
        else:
            k += 1
            amin = alpha
            if amax == np.infty:
                alpha = alpha * 2
            else:
                alpha = (amax + amin)/2
    return alpha


def BFGS_model_2(x, z, inner, TOL, n=0, gradient_decent=0):
    H = np.eye(5)
    xnew = x
    funks = np.zeros(0)
    while 1 / len(z) * np.linalg.norm(grad2(xnew, z, inner), 2) > TOL:  # skalerer med antall punkter
        p = - np.matmul(H, grad2(xnew, z, inner))
        alpha = Wolfe_2(z, inner, p, xnew)
        xold = xnew
        xnew = xnew + alpha * p
        if not gradient_decent:
            s = xnew - xold
            y = grad2(xnew, z, inner) - grad2(xold, z, inner)
            rho = 1 / np.matmul(y.T, s)

            if n == 0:
                H = np.matmul(y.T, s) / np.matmul(y.T, y) * H

            temp1 = np.outer(s, y)
            temp2 = np.outer(y, s)
            temp3 = np.outer(s, s)

            H = (np.eye(5) - rho * temp1) @ H @ (np.eye(5) - rho * temp2) + rho * temp3
        print('n = ', n, "\t x=", xnew)
        n += 1
        funks = np.append(funks, f2(xnew, z, inner))

    return xnew, n-1, funks

if __name__ == '__main__':
    #x = [0.01, 1, 0.1, 0, 0] #kult problem

    x = [3, 1, 3, 0, 0]

    x0 = np.array([5, 1, 0.1, 0, 0])
    points, inner = generate_points(x, size=500)

    Af, cf = constructproblem(x0)

    points = generate_noise(points, 2 * 10 ** (-1))

    plot_solution(x0, points, inner, rxy_tilde, 0, Metode=2)

    xf, nf, gradients = BFGS_model_2(x0, points, inner, TOL=10 ** (-10), gradient_decent=0)
    plot_solution(xf, points, inner, rxy_tilde, nf, Metode=2)
    convergence_plot(gradients, 2)


