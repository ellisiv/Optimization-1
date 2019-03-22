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
            g2 += -4 * r * (z[i] - c) @ A
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


def BFGS(x, z, inner, n=0):
    H = np.eye(5)
    xnew = x
    grads = np.zeros(0)
    while 1 / len(z) * np.linalg.norm(grad1(xnew, z, inner), 2) > 10 ** (-6) and n < 100: #skalerer med antall punkter
        p = - np.matmul(H, grad1(xnew, z, inner))
        alpha = linesearch_wolfe(z, inner, p, xnew)
        xold = xnew
        xnew = xnew + alpha * p
        s = xnew - xold
        y = grad1(xnew, z, inner) - grad1(xold, z, inner)
        rho = 1 / np.matmul(y.T, s)
        if n == 0:
            H = np.matmul(y.T, s) / np.matmul(y.T, y) * H
        """
        if rho > 10 ** 12:
            print(n, "restart")
            return BFGS(xnew, z, inner, n=n+1)
        """
        temp1 = np.outer(s, y)
        temp2 = np.outer(y, s)
        temp3 = np.outer(s, s)

        H = (np.eye(5) - rho * temp1) @ H @ (np.eye(5) - rho * temp2) + rho * temp3
        print('n = ', n, "\t x=", xnew)
        grads = np.append(grads, np.linalg.norm(grad1(xnew, z, inner), 2))
        n += 1
    return xnew, n - 1, grads


def generate_points(x, size=300):
    A, c = constructproblem(x)
    points = np.random.multivariate_normal(c, 1 * np.linalg.inv(A), size=size)
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


def plot_solution(xf, points, inner, funk, n):
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
    if n == 0:
        plt.title("Initial guess")
    else:
        #plt.title("Ferdig")
        plt.title('Finished after n= {}'.format(n) + ' iterations')
    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(points[inner, 0], points[inner, 1])
    plt.show()


def convergence_plot(grads, method=1):
    n = len(grads)
    x_grid = np.linspace(0, n - 1, n)
    plt.figure()
    plt.plot(x_grid, grads, label=r'$\nabla f_{}$'.format(method))
    plt.title('Convergence plot for method {}'.format(method))
    plt.show()


if __name__ == '__main__':
    x = [3, 1, 3, 0, 0]

    x0 = np.array([2, 0, 1, 0, 2])
    points, inner = generate_points(x, size=500)

    Af, cf = constructproblem(x0)

    points = generate_noise(points, 2 * 10 ** (-1))
    plot_solution(x0, points, inner, rxy, 0)

    xf, nf, gradsf = BFGS(x0, points, inner, 0)
    plot_solution(xf, points, inner, rxy, nf)
    convergence_plot(gradsf, 1)




