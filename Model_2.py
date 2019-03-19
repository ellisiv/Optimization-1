from Method_1_BFGS import constructproblem
import numpy as np
from Method_1_BFGS import plot_solution
from Method_1_BFGS import generate_noise
from Method_1_BFGS import generate_points

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


def BFGS_M2(x, z, inner, n=0):
    H = np.eye(5)
    xnew = x
    while 1 / len(z) * np.linalg.norm(grad2(xnew, z, inner), 2) > 10 ** (-6) and n < 100:  # skalerer med antall punkter
        p = - np.matmul(H, grad2(xnew, z, inner))
        alpha = Wolfe_2(z, inner, p, xnew)
        xold = xnew
        xnew = xnew + alpha * p
        s = xnew - xold
        y = grad2(xnew, z, inner) - grad2(xold, z, inner)
        rho = 1 / np.matmul(y.T, s)

        if n == 0:
            H = np.matmul(y.T, s) / np.matmul(y.T, y) * H

        if rho > 10 ** 8:
            print(n + 1, "restart")
            return BFGS_M2(xnew, z, inner, n=n+1)

        temp1 = np.outer(s, y)
        temp2 = np.outer(y, s)
        temp3 = np.outer(s, s)

        H = (np.eye(5) - rho * temp1) @ H @ (np.eye(5) - rho * temp2) + rho * temp3
        print(grad2(xnew, z, inner))
        print('n = ', n, "\t x=", xnew)
        n += 1

    return xnew

if __name__ == '__main__':
    x = [-1, -2, 1, 0, 0]

    x0 = np.array([0, 1, 0, 0, 0])
    points, inner = generate_points(x, size=300)

    Af, cf = constructproblem(x0)

    points = generate_noise(points, 10 ** (-2))

    plot_solution(x0, points, inner, rxy_tilde)

    xf = BFGS_M2(x0, points, inner, 0)
    plot_solution(xf, points, inner, rxy_tilde)

