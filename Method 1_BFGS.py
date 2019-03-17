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


def r1(zi, A, c):
    zic = np.matmul((zi - c).T, A)
    return np.matmul(zic, (zi - c)) - 1


def rxy(A, c, x, y):
    return (x - c[0]) * (A[0, 0] * (x - c[0]) + A[0, 1] * (y - c[1])) + (y - c[1]) * (A[1, 0] * (x - c[0]) + A[1, 1] * (y - c[1])) - 1


def f1(x, z, inner):
    A, c = constructproblem(x)
    sum = 0
    for i in range(len(z)):
        if i in inner:
            sum += (max(r1(z[i], A, c), 0)) ** 2
        else:
            sum += min(r1(z[i], A, c), 0) ** 2
    return sum


def grad1(x, z, inner):
    g1 = np.zeros((2, 2))
    g2 = np.zeros(2)
    A, c = constructproblem(x)
    for i in range(len(z)):
        r = r1(z[i], A, c)
        if (i in inner and r > 0) or ((i not in inner) and r < 0):
            g1 += 2 * r * np.outer((z[i] - c), (z[i] - c))
            g2 += -4 * r * (z[i].T - c) @ A
    g = np.array([g1[0, 0], g1[0, 1], g1[1, 1], g2[0], g2[1]])
    return g


def linesearch_wolfe(z, inner, p, x, c1=10 ** -4, c2=0.9):
    alpha = 1
    amax = np.infty
    amin = 0
    k = 0
    while ((f1((x + alpha * p), z, inner) > f1(x, z, inner) + c1 * alpha * np.matmul(grad1(x, z, inner).T, p)) or
            (np.matmul(grad1(x + alpha * p, z, inner).T, p) < c2 * np.matmul(grad1(x, z, inner).T, p))) and k < 20:
        if f1(x + alpha * p, z, inner) > f1(x, z, inner) + c1 * alpha * np.matmul(grad1(x, z, inner).T, p):
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


def ellipse_eqn_plot1(X, Y, A, c):
    return A[0][0]*(X-c[0])**2+2*A[0][1]*(X-c[0])*(Y-c[1])+A[1][1]*(Y-c[1])**2-1


def BFGS(x, z, inner, n=0):
    H = np.eye(5)
    xnew = x
    xold = np.array(5 * [np.infty])
    while np.linalg.norm(grad1(xnew, z, inner), 2) > 10 ** (-5) and n < 100:
        p = - np.matmul(H, grad1(xnew, z, inner))
        alpha = linesearch_wolfe(z, inner, p, xnew)
        xold = xnew
        xnew = xnew + alpha * p
        s = xnew - xold
        y = grad1(xnew, z, inner) - grad1(xold, z, inner)
        rho = 1 / np.matmul(y.T, s)
        if rho > 10 ** 8:
            print("restart")
            return BFGS(xnew, z, inner, n=n)
        if n == 0:
            H = np.matmul(y.T, s) / np.matmul(y.T, y) * H
        temp1 = np.outer(s, y)
        temp2 = np.outer(y, s)
        temp3 = np.outer(s, s)

        H = (np.eye(5) - rho * temp1) @ H @ (np.eye(5) - rho * temp2) + rho * temp3
        print('n = ', n, "\t x=", xnew)
        n += 1

    return xnew


def generate_points(x):
    A, c = constructproblem(x)
    points = np.random.multivariate_normal(c, 1 * np.linalg.inv(A), size=300)
    inner = []
    for i in range(len(points)):
        if r1(points[i], A, c) <= 0:
            inner.append(i)
    return points, inner


def generate_noise(z, scale):
    n = len(z)
    x = np.random.normal(0, scale, n)
    y = np.random.normal(0, scale, n)
    z[:, 0] = z[:, 0] + x
    z[:, 1] = z[:, 1] + y
    return z


def plot_solution(xf, points):
    Af, cf = constructproblem(xf)

    minx = min(points[:, 0])
    maxx = max(points[:, 0])

    miny = min(points[:, 1])
    maxy = max(points[:, 1])

    x = np.arange(minx, maxx, 0.01)
    y = np.arange(miny, maxy, 0.01)

    X, Y = np.meshgrid(x, y)
    Z = rxy(Af, cf, X, Y)

    plt.figure()

    plt.contour(X, Y, Z, [0], colors='black')

    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(points[inner, 0], points[inner, 1])
    plt.show()


x = [3, 1, -3, 0, 0]

x0 = np.array([3, -1, 3, 0, 2])
points, inner = generate_points(x)

noisy = generate_noise(points, 2 * 10 ** (-1))

Af, cf = constructproblem(x0)
plot_solution(x0, points)

xf = BFGS(x0, generate_noise(points, 10 ** (-3)), inner, 0)
plot_solution(xf, points)




