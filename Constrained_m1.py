import matplotlib.pyplot as plt
import numpy as np
from Method_1_BFGS import constructproblem
from Method_1_BFGS import plot_solution
from Method_1_BFGS import generate_noise
from Method_1_BFGS import generate_points
from Method_1_BFGS import convergence_plot
from Method_1_BFGS import r1
from Model_2 import grad2
from Model_2 import f2
from Model_2 import rxy_tilde


def B_func(x, f, beta, constraints, z, inner, gamma1 = 1, gamma2 = 10):    
    B = f(x, z, inner)
    
    for i in range(len(constraints)):
        if constraints[i](x, gamma1, gamma2) <= 0:
            B += np.infty
        else: 
            B -= beta*np.log(constraints[i](x, gamma1, gamma2))
    
    return B

#Definer constraint-funksjonene:

def c1(x,g1,g2):
    return x[0] - g1

def c2(x,g1,g2):
    return g2 - x[0]

def c3(x,g1,g2):
    return x[2] - g1

def c4(x,g1,g2):
    return g2 - x[2]

def c5(x,g1,g2):
    return np.sqrt(np.abs(x[0] * x[2])) - np.sqrt(g1 **2 + x[1] **2)


def grad_B(x, beta, gradf, z, inner, g1 = 1, g2 = 10):
    A = np.zeros((2,2))
    A[0,0] = 1/(x[0] - g1) - 1/(g2 - x[0]) + (x[2]/2 * 1/np.sqrt(np.abs(x[0] * x[2]))) * 1/(np.sqrt(np.abs(x[0] * x[2])) - np.sqrt(g1 **2 + x[1] **2))
    A[1,1] = 1/(x[2] - g1) - 1/(g2 - x[2]) + (x[1]/2 * 1/np.sqrt(np.abs(x[0] * x[1]))) * 1/(np.sqrt(np.abs(x[0] * x[2])) - np.sqrt(g1 **2 + x[1] **2))
    A[0,1] = -x[1] * 1/np.sqrt((g1 **2 + x[1] **2)) * 1/(np.sqrt(np.abs(x[0] * x[2])) - np.sqrt(g1 **2 + x[1] **2))
    
    g = np.array([A[0,0], A[0,1], A[0,0], 0, 0])
    
    return gradf(x, z, inner) - beta * g

def wolfe_constr(x, p, B, gradB, f, gradf, beta, constraints, z, inner, g1 = 1, g2 = 10, c1=10 ** -4, c2=0.9):
    alpha = 1
    amax = np.infty
    amin = 0
    k = 0
    while ((B(x + alpha * p, f, beta, constraints, z, inner, g1, g2) > B(x, f, beta, constraints, z, inner, g1, g2) + c1 * alpha * np.matmul(gradB(x, beta, gradf, z, inner, g1, g2).T, p)) or
            (np.matmul(gradB(x + alpha * p, beta, gradf, z, inner, g1, g2).T, p) < c2 * np.matmul(gradB(x, beta, gradf, z, inner, g1, g2).T, p))) and k < 20:
        if B(x + alpha * p, f, beta, constraints, z, inner, g1, g2) > B(x, f, beta, constraints, z, inner, g1, g2) + c1 * alpha * np.matmul(gradB(x, beta, gradf, z, inner, g1, g2).T, p):
            amax = alpha
            alpha = (amax + amin) / 2
            k += 1
        elif B(x + alpha * p, f, beta, constraints, z, inner, g1, g2) == np.infty:
            
            alpha = 0.1 * alpha
            print("Alpha lowered to keep within boundary")
            k += 1
        else:
            k += 1
            amin = alpha
            if amax == np.infty:
                alpha = alpha * 2
            else:
                alpha = (amax + amin)/2
    return alpha


def BFGS_constr(x, B, gradB, f, grad_f, beta, constraints, z, inner, g1, g2, n=0, TOL=10**(-6)):
    H = np.eye(len(x))
    xnew = x
    #B_vals = np.zeros(0)
    #B_vals = np.append(B_vals, B(xnew, f, beta, constraints, z, inner, g1, g2))
    
    f_vals = np.zeros(0)
    f_vals = np.append(f_vals, f(xnew, z, inner))
    
    while np.linalg.norm(gradB(xnew, beta, grad_f, z, inner, g1, g2), 2) > TOL and n < 50:
        p = - np.matmul(H, gradB(xnew, beta, grad_f, z, inner, g1, g2))
        alpha = wolfe_constr(xnew, p, B, gradB, f, grad_f, beta, constraints, z, inner, g1, g2)
        xold = xnew
        xnew = xnew + alpha * p
        s = xnew - xold
        #print("grad B: ", np.linalg.norm(gradB(xnew, beta, grad_f, z, inner, g1, g2), 2))
        y = gradB(xnew, beta, grad_f, z, inner, g1, g2) - gradB(xold, beta, grad_f, z, inner, g1, g2)
        rho = 1 / np.matmul(y.T, s)
        if n == 0:
            H = np.matmul(y.T, s) / np.matmul(y.T, y) * H
        temp1 = np.outer(s, y)
        temp2 = np.outer(y, s)
        temp3 = np.outer(s, s)

        H = (np.eye(5) - rho * temp1) @ H @ (np.eye(5) - rho * temp2) + rho * temp3
        print("n = ", n, "\t x = ", xnew)
        n += 1
        
        #B_vals = np.append(B_vals, B(xnew, f, beta, constraints, z, inner, g1, g2))
        f_vals = np.append(f_vals, f(xnew, z, inner))
        
    return xnew, n - 1, f_vals                              

def beta_optimization(x, B, gradB, f, grad_f, beta, constraints, z, inner, g1, g2, n=0, TOL=10**(-6)):
    xnew = x
    beta_new = beta
    #b_val_list = np.zeros(0)
    f_val_list = np.zeros(0)
    iter_sum = 0
    
    while np.linalg.norm(gradB(xnew, beta_new, grad_f, z, inner, g1, g2), 2) > np.max([10 **(-6), beta_new]) and n < 10:
        #Tried with TOL = beta_new
        print("betaløkke,", n)
        beta_old = beta_new
<<<<<<< HEAD
        xnew, itr, f_vals = BFGS_constr(xnew, B, gradB, f, grad_f, beta_old, constraints, z, inner, g1, g2, 0, 10 **3 * TOL)
        beta_new = 0.1 * beta_old #annen oppdatering? 
        iter_sum += itr
=======
        xnew, itr, f_vals = BFGS_constr(xnew, B, gradB, f, grad_f, beta_old, constraints, z, inner, g1, g2, n, 10 **3 * TOL)
        beta_new = 0.5 * beta_old #annen oppdatering? 
        iter_list = np.append(iter_list, itr)
>>>>>>> master
        f_val_list = np.append(f_val_list, f_vals[-1])
        n += 1
        
    return xnew, iter_sum, f_val_list

def convergence_plot_constr(grads):
    n = len(grads)
    x_grid = np.linspace(0, n - 1, n)
    plt.figure()
    #plt.yscale('log')
    #plt.ylim(min(grads), max(grads))
    plt.loglog(x_grid, grads, label=r'B(x;\beta)')
    plt.title('Convergence plot for constrained optimization')
    plt.legend()
    plt.show()

def generate_points_constr(x, size=300, scale=1):
    A, c = constructproblem(x)
    points = np.random.uniform(-3, 3, ((size,2)))
    inner = []
    for i in range(len(points)):
        if r1(points[i], A, c) <= 0:
            inner.append(i)
    return points, inner


if __name__ == '__main__':
    #x = [3, 1, 3, 0, 0]
    x = [3, 1, 0.2, 0, 0]
    #x = [0.008, 1, 0.008, 0, 0] #not so nice problem
    
    #x = [0.008, 1, 0.008, 0, 0]
    
    c = [c1, c2, c3, c4, c5]
    
    x0 = np.array([4, 1, 3, 0, 0])
<<<<<<< HEAD
    points, inner = generate_points(x, size=300)
=======
    points, inner = generate_points_constr(x, size = 70)
>>>>>>> master


    points = generate_noise(points, 2 * 10 ** (-1))
    plot_solution(x0, points, inner, rxy_tilde, 0, 2)
    
    print(B_func(x, f2, 0.1, c, points, inner))
    print(B_func(x0, f2, 0.1, c, points, inner))

<<<<<<< HEAD
    xf, itr, b_vals = beta_optimization(x0, B_func, grad_B, f2, grad2, 1, c, points, inner, 0.1, 1000, n=0, TOL=10**(-6))
    print(xf)
    print(itr)
    print(b_vals)
    print("len(b_vals): ", len(b_vals))


=======
    #xf = BFGS_constr(x0, B_func, grad_B, f2, grad2, 5, c, points, inner, 0.1, 1000, n = 0, TOL = 10 **(-3)) #general_BFGS(x, f, gradf, n=0, TOL=10**(-6))
    
    xf, itr, b_vals = beta_optimization(x0, B_func, grad_B, f2, grad2, 2, c, points, inner, 0.1, 1000, n=0, TOL=10**(-6))
            
>>>>>>> master
    convergence_plot_constr(b_vals)
    plot_solution(xf, points, inner, rxy_tilde, np.sum(itr), 2)
<<<<<<< HEAD
=======
    
>>>>>>> master

    
    



