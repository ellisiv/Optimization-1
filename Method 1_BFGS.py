import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
from matplotlib import patches


def r1(zi, A, c):
    zic = np.matmul((zi - c).T, A)
    return np.matmul(zic, (zi - c)) - 1


def f1(x, z, inner):
    A = np.zeros((2, 2))
    A[0] = [x[0], x[1]]
    A[1] = [x[1], x[2]]
    c = [x[3], x[4]]
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
    A = np.zeros((2, 2))
    A[0] = [x[0], x[1]]
    A[1] = [x[1], x[2]]
    c = [x[3], x[4]]
    for i in range(len(z)):
        r = r1(z[i], A, c)
        if (i in inner and r > 0) or ((i not in inner) and r < 0):
            g1 += 2 * r * np.matmul((z[i] - c), (z[i] - c).transpose())
            g2 += -4 * r * (z[i].T - c) @ A

    g = np.array([g1[0, 0], g1[0, 1], g1[1, 1], g2[0], g2[1]])
    return g


def linesearch_wolfe(z, inner, p, x, c1=10 ** -4, c2=0.9):
    alpha = 1
    amax = np.infty
    amin = 0
    k = 0
    while (f1((x + alpha * p), z, inner) > f1(x, z, inner) + c1 * alpha * np.matmul(grad1(x, z, inner).T, p)) or \
            (np.matmul(grad1(x + alpha * p, z, inner).T, p) < c2 * np.matmul(grad1(x, z, inner).T, p)):
        if k > 20:
            return alpha
        elif f1(x + alpha * p, z, inner) > f1(x, z, inner) + c1 * alpha * np.matmul(grad1(x, z, inner).T, p):
            #print(alpha)
            amax = alpha
            alpha = (amax + amin) / 2
        else:
            #print(alpha)
            amin = alpha
            if amax == np.infty:
                alpha = alpha * 2
            else:
                alpha = (amax + amin)/2

    #print(alpha)
    return alpha


def get_ellipse(A, c):
    eigen_values, eigen_vectors = np.linalg.eig(A)
    principal_axes = np.sqrt(1 / eigen_values)

    vec1 = eigen_vectors[0, :]
    angle = np.arctan(vec1[0] / vec1[1])

    e1 = patches.Ellipse((c[0], c[1]), 2 * principal_axes[0], 2 * principal_axes[1],
                         angle=-angle * 180 / np.pi, linewidth=2, fill=False, zorder=2)
    return e1


def make_ellipse(A, c):
    eval, evec = np.linalg.eig(A)
    principal_axes = np.sqrt(1 / eval)

    vec1 = evec[0, :]
    angle = - np.arctan(vec1[0] / vec1[1]) * 180 / np.pi

    e1 = patches.Ellipse((c[0], c[1]), 2 * principal_axes[0], 2 * principal_axes[1],
                         angle=angle, linewidth=2, fill=False, zorder=2)
    return e1


def BFGS(x, z, inner):
    H = np.eye(5)
    xnew = x
    xold = np.array(5 * [np.infty])
    n = 0
    Af = np.zeros((2, 2))
    Af[0] = [xnew[0], xnew[1]]
    Af[1] = [xnew[1], xnew[2]]
    cf = xnew[-2:]
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(points[inner, 0], points[inner, 1])
    ax = plt.axes()
    ax.add_patch(make_ellipse(Af, cf))
    plt.show()
    while np.linalg.norm(grad1(xnew, z, inner), 2) > 10 ** (-5) and n < 100:
        p = - np.matmul(H, grad1(xnew, z, inner))
        #print(p)
        alpha = linesearch_wolfe(z, inner, p, xnew)
        xold = xnew
        xnew = xnew + alpha * p
        s = xnew - xold
        y = grad1(xnew, z, inner) - grad1(xold, z, inner)
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

c = np.array([2, 2])
A = 3 * np.eye(2)
A[0, 1], A[1, 0] = 1, 1
x = [3, 1, 3, 2, 2]

x0 = np.array([3, 1, 3, 1.6, 2])

def generate_points(x):
    points = np.random.multivariate_normal(c, 1 * np.linalg.inv(A), size=500)
    inner = []
    for i in range(len(points)):
        if r1(points[i], A, c) <= 0:
            inner.append(i)
    return points, inner



points, inner = generate_points(x)

xf = BFGS(x0, points, inner)
Af = np.zeros((2, 2))
Af[0] = [xf[0], xf[1]]
Af[1] = [xf[1], xf[2]]
cf = xf[-2:]



plt.figure()
plt.scatter(points[:, 0], points[:, 1])
plt.scatter(points[inner, 0], points[inner, 1])
ax = plt.axes()
ax.add_patch(get_ellipse(Af, cf))
plt.show()


