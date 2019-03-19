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
from Method_1_BFGS import linesearch_wolfe
from Method_1_BFGS import BFGS
from BFGS import general_wolfe_linesearch
from BFGS import general_BFGS

#Ellisiv sin idé for penalty:
def f1_penalty(x, z, inner, my, constraints):
    f = f1(x, z, inner)
    f += my / 2 * constraints(x)
    return f

def quadratic_penalty_method():
    return 0

def construct_B(f, x, beta, constraints, I, z, inner, gamma1, gamma2):
    #Helene sin start for å kontruere funksjonen B(x, beta)
    #f: funksjonen vi tar utgangspunkt i, f1 eller f2
    #x: vektor med A og b eller c
    #constraints: constraintene dine, i en liste (foreløpig er det det du har kodet med)
    # I: ant constraints
    # beta: betaverdien
    #z, inner: z-verider for å construere f
    
    B = f(x, z, inner)
    
    for i in range(I):
        if constraints[i](x, gamma1, gamma2) <= 0:
            B += np.inf
            #print("inf!: ",i)
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
    return x[1] - g1

def c4(x,g1,g2):
    return g2 - x[1]

def c5(x,g1,g2):
    return np.sqrt(x[0] * x[1]) - np.sqrt(g1 **2 + x[2] **2)




#----------------------------------- Uttesting --------------------------------------#
x = [3,5,1,7,0]
c = [c1,c2,c3,c4,c5]
z, inner = generate_points(x)

#print(c[4](x,2,100))

print(construct_B(f1, x, 2, c, 5, z, inner, 1, 10000))

#------------------------------------------------------------------------------------#






