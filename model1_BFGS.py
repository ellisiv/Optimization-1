import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
from matplotlib import patches


def constructproblem(x):
    """
    Constructing A and b or c from the vector x which we are optimizing
    """

    A = np.zeros((2, 2))
    A[0] = [x[0], x[1]]
    A[1] = [x[1], x[2]]

    vec = np.array([x[3], x[4]])

    return A, vec


def r1(x, zi):
    """
    :param x: vector containing entries of A and c
    :param zi: data point
    :return: r1(A, c): a measure of distance from the data point to the ellipse constructed by A, c
    """
    A, c = constructproblem(x)
    zic = np.matmul((zi - c).T, A)
    return np.matmul(zic, (zi - c)) - 1


def f1(x, z):
    """
    :param x: vector containing entries of A and c
    :param z: list of data points on the form [x, y, {0, 1}] where 0 means point classified outside ellipse and
                1 means classified inside ellipse
    :return: measure of how well the points are classified. A form of total distance to the right side of the ellipse
                for each point
    """
    f = 0
    for zi in z:
        if zi[3] == 1:
            f += max(r1(x, zi), 0) ** 2
        else:
            f += min(r1(x, zi), 0) ** 2
    return f


def grad_f1(x, z):
    """
    :param x: vector containing entries of A and c
    :param z: list of data points on the form [x, y, {0, 1}] where 0 means point classified outside ellipse and
                1 means classified inside ellipse
    :return: gradient in point x, that is gradient for the current ellipse
    """
    gA = np.zeros((2, 2))
    gc = np.zeros(2)
    A, c = constructproblem(x)
    for zi in z:
        r = r1(x, zi)
        if (zi[3] == 1 and r > 0) or (zi[3] == 0 and r < 0):
            gA += 2 * r * np.matmul((zi - c), (zi - c).transpose())
            gc += -4 * r * np.matmul((zi.T - c), A)
    return np.array([gA[0, 0], gA[0, 1], gA[1, 1], gc[0], gc[1]])


def linesearch_wolfe(x, z, p, c1=10 ** -4, c2=0.9):
    """
    :param z: list of data points on the form [x, y, {0, 1}] where 0 means point classified outside ellipse and
                1 means classified inside ellipse
    :param p: decent direction
    :param x: vector containing entries of A and c
    :param c1:
    :param c2:
    :return: a step length satisfying the Wolfe conditions with parameters c1 and c2
    """
    alpha = 1
    amax = np.infty
    amin = 0
    while (f1((x + alpha * p), z) > f1(x, z) + c1 * alpha * np.matmul(grad_f1(x, z).T, p)) or \
            (np.matmul(grad_f1(x + alpha * p, z).T, p) < c2 * np.matmul(grad_f1(x, z).T, p)):
        if f1(x + alpha * p, z) > f1(x, z) + c1 * alpha * np.matmul(grad_f1(x, z).T, p):
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
    return alpha

def BFGS(x, z):
    xnew = x
    xold = np.array(5 * [np.infty])
    H = np.eye(5)
    n = 0
    while np.linalg.norm(grad_f1(xnew, z), 2) > 10 ** (-3) and n < 20:
        p = - np.matmul(H, grad_f1(xnew, z))
        print(p)
        alpha = linesearch_wolfe(xnew, z, p)
        xold = xnew
        xnew = xnew + alpha * p
        s = xnew - xold
        y = grad_f1(xnew, z) - grad_f1(xold, z)
        rho = 1 / np.matmul(y.T, s)
        """
        if n == 0:
            H = np.matmul(y.T, s) / np.matmul(y.T, y) * H
        H = np.matmul(np.matmul((np.eye(5) - rho * np.matmul(s, y.T)), H), (np.eye(5) - rho * np.matmul(y, s.T))) \
            + rho * np.matmul(s, s.T)
        """
        n += 1
    return xnew


def make_ellipse(A, c):
    eval, evec = np.linalg.eig(A)
    principal_axes = np.sqrt(1 / eval)

    vec1 = evec[0, :]
    angle = - np.arctan(vec1[0] / vec1[1]) * 180 / np.pi

    e1 = patches.Ellipse((c[0], c[1]), 2 * principal_axes[0], 2 * principal_axes[1],
                         angle=angle, linewidth=2, fill=False, zorder=2)
    return e1





