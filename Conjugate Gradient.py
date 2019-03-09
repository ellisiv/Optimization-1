import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg


def r1(zi, A, c):
    zic = np.matmul((zi - c).transpose(), A)
    return np.dot(zic, (zi - c)) - 1


def f1(x, z, inner):
    A = np.zeros((2, 2))
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
    g = np.array([g1[0, 0], g1[0, 1], g1[1, 1], g2[0], g2[1]])
    return g

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
    g = np.array([g1[0,0], g1[0,1], g1[1, 1], g2[0], g2[1]])
    return g


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


def BFGS(x, z, inner, text):
    H = np.eye(2)
    xnew = x
    xold = np.array(5 * [np.infty])
    if text == "1":
        while np.linalg.norm(xnew - xold, 2) < 10 ** -8:
            p = -H * grad1(x, z, inner)
            alpha = linesearch_wolfe(z, inner, p, x, text="1")
            xold = xnew
            xnew = xnew + alpha * p
            s = xnew - xold
            y = grad1(xnew, z, inner) - grad1(xold, z, inner)
            ysk = np.dot(y.T,s)
            H =np.matmul(np.matmul((np.eye(2)- 1 / ysk * np.dot(s, y.T)), H), (np.eye(2)- 1 / ysk * np.dot(y, s.T)))
    else:
        while np.linalg.norm(xnew - xold, 2) < 10 ** -8:
            p = -H * grad2(x, z, inner)
            alpha = linesearch_wolfe(z, inner, p, x, text="2")
            xold = xnew
            xnew = xnew + alpha * p
            s = xnew - xold
            y = grad2(xnew, z, inner) - grad2(xold, z, inner)
            ysk = np.dot(y.T, s)
            H = np.matmul(np.matmul((np.eye(2) - 1 / ysk * np.dot(s, y.T)), H), (np.eye(2) - 1 / ysk * np.dot(y, s.T)))
    return x

c = np.array([2, 2])
A = 3 * np.eye(2)

x = np.array([3, 1, 3, 2, 2])

points = np.random.multivariate_normal(c, 1 * np.linalg.inv(A), size=90)

inner = []
for i in range(len(points)):
    if r1(points[i], A, c) <= 0:
        inner.append(i)

print(BFGS(x, points, inner, "1"))
