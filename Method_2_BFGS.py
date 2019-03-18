import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
from matplotlib import patches


def constructproblem(x):

    A = np.zeros((2, 2))
    A[0] = [x[0], x[1]]
    A[1] = [x[1], x[2]]

    vec = np.array([x[3], x[4]])

    return A, vec


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
            g1 += 2 * r * np.matmul(z[i], z[i].T)
            g2 += -2 * r * z[i].T
    g = np.array([g1[0,0], g1[0, 1], g1[1, 1], g2[0], g2[1]])
    return g


def linesearch_wolfe(z, inner, p, x, c1=10 ** -4, c2=0.9):
    alpha = 1
    amax = np.infty
    amin = 0
    k = 0
    while ((f2(x + alpha * p, z, inner) > f2(x, z, inner) + c1 * alpha * np.matmul(grad2(x, z, inner), p)) or
            (np.matmul(grad2(x + alpha * p, z, inner).T, p) < c2 * np.matmul(grad2(x, z, inner).T, p))) and k < 20:
        if f2(x + alpha * p, z, inner) > f2(x, z, inner) + c1 * alpha * np.matmul(grad2(x, z, inner), p):
            k += 1
            amax = alpha
            alpha = (amax + amin) / 2
        else:
            k += 1
            amin = alpha
            if amax == np.infty:
                alpha = alpha * 2
            else:
                alpha = (amax + amin) / 2
    print(k)
    return alpha


def BFGS(x, z, inner, n=0):
    H = np.eye(5)
    xnew = x
    while np.linalg.norm(grad2(xnew, z, inner), 2) > 10 ** (-5) and n < 100:
        print(n)
        p = - np.matmul(H, grad2(xnew, z, inner))
        alpha = linesearch_wolfe(z, inner, p, xnew)
        xold = xnew
        xnew = xnew + alpha * p
        s = xnew - xold
        y = grad2(xnew, z, inner) - grad2(xold, z, inner)
        rho = 1 / np.matmul(y.T, s)
        print(rho)
        if rho > 10 ** 12:
            print("restart")
            return BFGS(xnew, z, inner, n + 1)
        if n == 0:
            H = np.matmul(y.T, s) / np.matmul(y.T, y) * H
        temp1 = np.outer(s, y)
        temp2 = np.outer(y, s)
        temp3 = np.outer(s, s)

        H = (np.eye(5) - rho * temp1) @ H @ (np.eye(5) - rho * temp2) + rho * temp3
        n += 1
        #print(xnew)
    print(n)
    return xnew


def generate_points(x):
    A, b = constructproblem(x)
    c = np.zeros(2)
    points = np.random.multivariate_normal(c, np.eye(2), size=300)
    inner = []
    for i in range(len(points)):
        if r2(points[i], A, np.zeros(2)) <= 0:
            inner.append(i)
    return points, inner


x = [1, 0, 1, 0, 0]

x0 = np.array([1, 0, 1.2, 0, 0])
points, inner = generate_points(x)

def plot_solution(xf, points, funk):
    Af, cf = constructproblem(xf)

    minx = min(points[:, 0])
    maxx = max(points[:, 0])

    miny = min(points[:, 1])
    maxy = max(points[:, 1])

    x = np.arange(minx, maxx, 0.01)
    y = np.arange(miny, maxy, 0.01)

    X, Y = np.meshgrid(x, y)
    Z = funk(Af, cf, X, Y)

    plt.figure()

    plt.contour(X, Y, Z, [0], colors='black')

    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(points[inner, 0], points[inner, 1])
    plt.show()

xf = BFGS(x0, points, inner)
print(xf)


plot_solution(x0, points, rxy_tilde())

plot_solution(xf, points, rxy_tilde())

