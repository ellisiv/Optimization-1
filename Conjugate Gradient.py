import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg


def r1(zi, A, c):
    zic = np.matmul((zi - c).transpose(), A)
    return np.dot(zic, (zi - c)) - 1



def f1(x, z, inner):
    A = np.zeros((2,2))
    A[0] = [x[0], x[1]]
    A[2] = [x[1], x[2]]
    c = x[-2:]
    sum = 0
    for i in range(len(z)):
        if i in inner:
            sum += (max(r1(z[i], A, c), 0)) ** 2
        else:
            sum += min(r1(z[i], A, c), 0) ** 2
    return sum


def r2(zi, A, b):
    zib = np.matmul(zi.transpose(), A)
    return np.dot(zib, zi) - np.dot(zi.transpose(), b) - 1


def f2(x, z, inner):
    A = np.zeros((2, 2))
    A[0] = [x[0], x[1]]
    A[2] = [x[1], x[2]]
    b = x[-2:]
    sum = 0
    for i in range(len(z)):
        if i in inner:
            sum += (max(r2(z[i], A, b), 0)) ** 2
        else:
            sum += (min(r2(z[i], A, b), 0)) ** 2
    return sum


def grad1(x, z, inner):
    g1 = np.zeros((2, 2))
    g2 = np.zeros(2)
    A = np.zeros((2, 2))
    A[0] = [x[0], x[1]]
    A[1] = [x[1], x[2]]
    c = x[-2:]
    for i in range(len(z)):
        r = r1(z[i], A, c)
        if (i in inner and r > 0) or (i not in inner and r < 0):
            g1 += 2 * r * np.matmul((z[i] - c), (z[i] - c).transpose())
            g2 += -4 * r * np.dot(np.array([1, 2]).transpose(), np.eye(2))
    return g1, g2

def grad2(x, z, inner):
    g1 = np.zeros((2, 2))
    g2 = np.zeros(2)
    A = np.zeros((2, 2))
    A[0] = [x[0], x[1]]
    A[1] = [x[1], x[2]]
    b = x[-2:]
    for i in range(len(z)):
        r = r2(z[i], A, b)
        if (i in inner and r > 0) or (i not in inner and r < 0):
            g1 += 2 * r * np.matmul(z[i], z[i].T)
            g2 += -2 * r * z[i].T
    return g1, g2


def linesearch_wolfe(z, inner, p, x, c1=10 ** -4, c2 = 0.9, text= "1"):
    alpha = 1
    amax = np.infty
    amin = 0
    if text == "1":
        while (f1((x + alpha * p), z, inner) > f1(x, z, inner) + c1 * alpha * grad1(x, z , inner)) and (np.dot(grad1(x + alpha * p, z, inner).T, p) < c2 * np.dot(grad1(x, z, inner).T, p)):
            if f1(x + alpha * p, z, inner) > f1(x, z, inner) + c1 * alpha * grad1(x, z, inner):
                amax = alpha
                alpha = (amax + amin) / 2
            else:
                amin = alpha
                if amax == np.infty:
                    alpha = alpha * 2
                else:
                    alpha = (amax + amin)/2
    else:
        while (f2(x + alpha * p, z, inner) > f2(x, z, inner) + c1 * alpha * grad2(x, z, inner)) and (np.dot(grad2(x + alpha * p, z, inner).T, p) < c2 * np.dot(grad2(x, z, inner).T, p)):
            if f2(x + alpha * p, z, inner) > f2(x, z, inner) + c1 * alpha * grad2(x, z, inner):
                amax = alpha
                alpha = (amax + amin) / 2
            else:
                amin = alpha
                if amax == np.infty:
                    alpha = alpha*2
                else:
                    alpha = (amax + amin)/2
    return alpha


def BFGS(x, z, inner, f1, f2, text):
    H = np.eye(2)
    xnew = x
    xold = np.infty
    if text == "1":
        p = -H * grad1(x, z, inner)
        alpha = linesearch_wolfe(z, inner, p, x, text="1")
        xold = xnew
        xnew = xnew + alpha * p
        s = xnew - xold
        y = grad1(xnew, z, inner) - grad1(xold, z, inner)
        ysk = np.dot(y.T)
        H =

    else:

    #initialize B0
    #compute pk = -Hk*gradf(xk) here we split up x
    #line search alpha
    #xk+1 = xk +alphakpk:
        #sk = xk+1-xk
        #yk = grad f(xk+1) - grad f(xk)
        #Bk+1 = ...
    return A, b
