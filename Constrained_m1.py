import matplotlib.pyplot as plt
import numpy as np
from Method_1_BFGS import constructproblem
from Method_1_BFGS import r1
from Method_1_BFGS import rxy
from Method_1_BFGS import plot_solution
from Method_1_BFGS import f1
from Method_1_BFGS import generate_noise
from Method_1_BFGS import generate_points
from Method_1_BFGS import grad1
from Model_2 import grad2
from Model_2 import f2
from Model_2 import rxy_tilde
#from Method_1_BFGS import linesearch_wolfe
#from Method_1_BFGS import BFGS
#from BFGS import general_wolfe_linesearch
#from BFGS import general_BFGS


def B_func(x, f, beta, constraints, z, inner, gamma1 = 1, gamma2 = 1000):
    #Helene sin start for å kontruere funksjonen B(x, beta)
    #f: funksjonen vi tar utgangspunkt i, f1 eller f2
    #x: vektor med A og b eller c
    #constraints: constraintene dine, i en liste (foreløpig er det det du har kodet med)
    # I: ant constraints
    # beta: betaverdien
    #z, inner: z-verider for å construere f
    
    B = f(x, z, inner)
    
    for i in range(len(constraints)):
        if constraints[i](x, gamma1, gamma2) <= 0:
            B += np.infty
            print("inf!")
        else: 
            B -= beta*np.log(constraints[i](x, gamma1, gamma2))
            #print("i: ",i, " ", constraints[i](x, gamma1, gamma2))
    
    #Den skal returnere et tall som skal minimeres 
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

def grad_B(x, beta, gradf, z, inner, g1 = 1, g2 = 1000):
    x0 = np.float64(x[0])
    x1 = np.float64(x[1])
    x2 = np.float64(x[2])
    g = gradf(x, z, inner)
    if x0*x2 < 0:
        print("æææ")
    g -= beta * (1/(x0 - g1) + 1/(g2 - x0) + 1/(x2 - g1) + 1/(g2 - x2) + 1/2 * (1/np.sqrt(np.abs(x0*x2))) * (x2 + x0) - x1/np.sqrt(g1 **2 + x1 **2))
    return g

def wolfe_constr(x, p, B, gradB, f, gradf, beta, constraints, z, inner, g1 = 1, g2 = 1000, c1=10 ** -4, c2=0.9):
    alpha = 1
    amax = np.infty
    amin = 0
    k = 0
    while ((B(x + alpha * p, f, beta, constraints, z, inner, g1, g2) > B(x, f, beta, constraints, z, inner, g1, g2) + c1 * alpha * np.matmul(gradB(x, beta, gradf, z, inner, g1, g2).T, p)) or
            (np.matmul(gradB(x + alpha * p, beta, gradf, z, inner, g1, g2).T, p) < c2 * np.matmul(gradB(x, beta, gradf, z, inner, g1, g2).T, p))) and k < 20:
        #print("I wolfe_constr while")
        if B(x + alpha * p, f, beta, constraints, z, inner, g1, g2) > B(x, f, beta, constraints, z, inner, g1, g2) + c1 * alpha * np.matmul(gradB(x, beta, gradf, z, inner, g1, g2).T, p):
            amax = alpha
            alpha = (amax + amin) / 2
            k += 1
        elif B(x + alpha * p, f, beta, constraints, z, inner, g1, g2) == np.infty:
            alpha = 0.1 * alpha
            print("Måtte senke alpha!")
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
    #print("TOL: ", TOL)
    #print("BFGS_constr blir hvertfall kjørt!!")
    H = np.eye(len(x))
    xnew = x
    while np.linalg.norm(gradB(xnew, beta, grad_f, z, inner, g1, g2), 2) > TOL and n < 100:
        p = - np.matmul(H, gradB(xnew, beta, grad_f, z, inner, g1, g2))
        alpha = wolfe_constr(xnew, p, B, gradB, f, grad_f, beta, constraints, z, inner, g1, g2)
        xold = xnew
        xnew = xnew + alpha * p
        s = xnew - xold
        y = gradB(xnew, beta, grad_f, z, inner, g1, g2) - gradB(xold, beta, grad_f, z, inner, g1, g2)
        rho = 1 / np.matmul(y.T, s)
        if n == 0:
            H = np.matmul(y.T, s) / np.matmul(y.T, y) * H
        '''
        if rho > 10 ** 12:
            #print(n, "restart")
            return BFGS_constr(xnew, B, gradB, f, grad_f, beta, constraints, z, inner, g1, g2, n=n+1, TOL = TOL)
        '''
        temp1 = np.outer(s, y)
        temp2 = np.outer(y, s)
        temp3 = np.outer(s, s)

        H = (np.eye(5) - rho * temp1) @ H @ (np.eye(5) - rho * temp2) + rho * temp3
        
        #print('n = ', n, "\t x=", xnew)
        n += 1
    return xnew                              

def beta_run_through(betalist):
    #Ganske banal funksjon for å undersøke betaene  
    
    x = [2, 5, 3, 0, 0]
    c = [c1,c2,c3,c4,c5]
    
    x0 = np.array([3, 1, 4, 0, 0])
    
    points, inner = generate_points(x)
    
    Af, cf = constructproblem(x0)
    
    points = generate_noise(points, 2 * 10 ** (-1))
    plot_solution(x0, points, inner, rxy_tilde)
    
    for beta in betalist:
        
        print("Betaen vår er nå: ", beta)
        
        xf = BFGS_constr(x0, B_func, grad_B, f2, grad2, beta, c, points, inner, 0.1, 1000, n = 0, TOL = 10 **(-3)) #general_BFGS(x, f, gradf, n=0, TOL=10**(-6))
        print("For beta = ", beta, "x er: ", xf)
        plot_solution(xf, points, inner, rxy_tilde)
    
    return 0

def beta_optimization(x, B, gradB, f, grad_f, beta, constraints, z, inner, g1, g2, n=0, TOL=10**(-6)):
    xnew = x
    beta_new = beta
    
    while np.linalg.norm(gradB(xnew, beta_new, grad_f, z, inner, g1, g2), 2) > TOL and n < 20:
        print("Betaløkke!!!!:)), nr: ", n)
        beta_old = beta_new
        xnew = BFGS_constr(xnew, B, gradB, f, grad_f, beta_old, constraints, z, inner, g1, g2, n, TOL)
        beta_new = 0.1 * beta_old #annen oppdatering? 
        n += 1
        
    return xnew

if __name__ == '__main__':
        
    #----------------------------------- Uttesting --------------------------------------#
    '''
    x = [3,5,1,7,0]
    c = [c1,c2,c3,c4,c5]
    z, inner = generate_points(x)
    
    #print(c[4](x,2,100))
    
    print(B_func(f1, x, 2, c, z, inner, 1, 10000))
    
    #------------------------------------------------------------------------------------#
    '''
    x = [-3, 0, 1, 0, 2]
    
    c = [c1,c2,c3,c4,c5]
    
    x0 = np.array([2, 0, 1, 0, 0])
    points, inner = generate_points(x, size=100)

    Af, cf = constructproblem(x0)

    points = generate_noise(points, 2 * 10 ** (-2))
    #plot_solution(x0, points, inner, rxy_tilde)
    
    print(B_func(x,f2,0.1,c,points, inner))
    print(B_func(x0,f2,0.1,c,points, inner))

    #xf = BFGS_constr(x0, B_func, grad_B, f2, grad2, 5, c, points, inner, 0.1, 1000, n = 0, TOL = 10 **(-3)) #general_BFGS(x, f, gradf, n=0, TOL=10**(-6))
    
    xf = beta_optimization(x0, B_func, grad_B, f2, grad2, 1, c, points, inner, 0.1, 1000, n=0, TOL=10**(-6))
    print(xf)
    plot_solution(xf, points, inner, rxy_tilde)


    
    



