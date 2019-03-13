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
    return np.matmul(zib, zi) - np.matmul(zi.transpose(), b) - 1


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
    while ((f2(x + alpha * p, z, inner) > f2(x, z, inner) + c1 * alpha * np.matmul(grad2(x, z, inner), p)) or \
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


def make_ellipse(A, b):
    eval, evec = np.linalg.eig(A)
    principal_axes = np.sqrt(1 / eval)

    vec1 = evec[0, :]
    angle = - np.arctan(vec1[0] / vec1[1]) * 180 / np.pi

    c = 1 / 2 * np.linalg.inv(A) @ b
    c = 1 / 2 * np.linalg.inv(A) @ b

    e1 = patches.Ellipse((c[0], c[1]), 2 * principal_axes[0], 2 * principal_axes[1],
                         angle=angle, linewidth=2, fill=False, zorder=2)
    return e1


def BFGS(x, z, inner):
    H = np.eye(5)
    xnew = x
    xold = np.array(5 * [np.infty])
    n = 0
    Af, bf = constructproblem(x)
    plt.figure()
    plt.scatter(z[:, 0], z[:, 1])
    plt.scatter(z[inner, 0], z[inner, 1])
    ax = plt.axes()
    ax.add_patch(make_ellipse(Af, bf))
    plt.show()
    while np.linalg.norm(grad2(xnew, z, inner), 2) > 10 ** (-3) and n < 100:
        p = - np.matmul(H, grad2(xnew, z, inner))
        #print(p)
        alpha = linesearch_wolfe(z, inner, p, xnew)
        xold = xnew
        xnew = xnew + alpha * p
        s = xnew - xold
        y = grad2(xnew, z, inner) - grad2(xold, z, inner)
        #print(s)
        rho = 1 / np.matmul(y.T, s)
        #print(rho)
        if n == 0:
            H = np.matmul(y.T, s) / np.matmul(y.T, y) * H

        temp1 = np.outer(s, y)
        temp2 = np.outer(y, s)
        temp3 = np.outer(s, s)

        H = (np.eye(5) - rho * temp1) @ H @ (np.eye(5) - rho * temp2) + rho * temp3
        #H = np.matmul(np.matmul((np.eye(5) - rho * np.matmul(s, y.T)), H), (np.eye(5) - rho * np.matmul(y, s.T))) + rho * np.matmul(s, s.T)
        print(H)
        n += 1
    print(n)
    return xnew


def generate_points(x):
    A, b = constructproblem(x)
    c = 1 / 2 * np.linalg.inv(A) @ b
    points = np.random.multivariate_normal(c, 1 * np.linalg.inv(A), size=500)
    inner = []
    for i in range(len(points)):
        if r2(points[i], A, b) <= 0:
            inner.append(i)
    return points, inner


c = np.array([2, 2])
A = 3 * np.eye(2)
A[0, 1], A[1, 0] = 1, 1
x = [3, 1, 3, 2, 2]

x0 = np.array([3, 1, 3, 1.6, 2])
points, inner = generate_points(x)

def plot_solution(x, z):
    Af, bf = constructproblem(x)
    plt.figure()
    plt.scatter(z[:, 0], z[:, 1])
    plt.scatter(z[inner, 0], z[inner, 1])
    ax = plt.axes()
    ax.add_patch(make_ellipse(Af, bf))
    plt.show()

xf = BFGS(x0, points, inner)
print(xf)

plot_solution(xf, points)

