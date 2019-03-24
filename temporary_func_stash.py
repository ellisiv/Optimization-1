# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:55:49 2019

@author: helen
"""

#Bare midlertidig lagrer en funskjon du ikke bruker lenger

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