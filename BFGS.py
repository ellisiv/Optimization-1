import numpy as np
import matplotlib.pyplot as plt

def general_wolfe_linesearch(x, p, f, gradf, c1=10 ** -4, c2=0.9):
    alpha = 1
    amax = np.infty
    amin = 0
    k = 0
    while ((f(x + alpha * p) > f(x) + c1 * alpha * np.matmul(gradf(x).T, p)) or
            (np.matmul(gradf(x + alpha * p).T, p) < c2 * np.matmul(gradf(x).T, p))) and k < 20:
        if f(x + alpha * p) > f(x) + c1 * alpha * np.matmul(gradf(x).T, p):
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


def general_BFGS(x, f, gradf, n=0, TOL=10**(-6)):
    H = np.eye(len(x))
    xnew = x
    while np.linalg.norm(gradf(xnew), 2) > TOL and n < 100:
        p = - np.matmul(H, gradf(xnew))
        alpha = general_wolfe_linesearch(xnew, p, f, gradf)
        xold = xnew
        xnew = xnew + alpha * p
        s = xnew - xold
        y = gradf(xnew) - gradf(xold)
        rho = 1 / np.matmul(y.T, s)
        if n == 0:
            H = np.matmul(y.T, s) / np.matmul(y.T, y) * H
        if rho > 10 ** 12:
            print(n, "restart")
            return general_BFGS(xnew, f, gradf, n=n+1)

        temp1 = np.outer(s, y)
        temp2 = np.outer(y, s)
        temp3 = np.outer(s, s)

        H = (np.eye(5) - rho * temp1) @ H @ (np.eye(5) - rho * temp2) + rho * temp3
        print('n = ', n, "\t x=", xnew)
        n += 1
    return xnew



