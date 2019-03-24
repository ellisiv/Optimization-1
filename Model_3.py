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
from Method_1_BFGS import BFGS_model_1
from Method_1_BFGS import convergence_plot
from Model_2 import BFGS_model_2
from Model_2 import grad2
from Model_2 import f2
from Model_2 import rxy_tilde

def construct_circle_params(x):

    c = np.array([x[0], x[1]])
    rho = x[2]
    d = np.array([x[3], x[4]])
    sigma = x[5]

    return c, rho, d, sigma


def r3(zi, center, radius):
    return np.linalg.norm((zi - center), 2) ** 2 - radius ** 2

'''
def rxy(A, c, x, y):
    return (x - c[0]) * (A[0, 0] * (x - c[0]) + A[0, 1] * (y - c[1])) + (y - c[1]) * (A[1, 0] * (x - c[0]) + A[1, 1] * (y - c[1])) - 1
'''

def rxy3(x,y,c,rho):
    return (x - c[0]) **2 + (y - c[1]) **2 -rho**2

def f3(x, z, a, b):
    c, rho, d, sigma = construct_circle_params(x)

    sum = 0
    for i in range(len(z)):
        if i in a:
            sum += max(r3(z[i], c, rho), 0) ** 2 + min(r3(z[i], d, sigma), 0) ** 2
        elif i in b:
            sum += min(r3(z[i], c, rho), 0) ** 2 + max(r3(z[i], d, sigma), 0) ** 2
        else:
            sum += min(r3(z[i], c, rho), 0) ** 2 + min(r3(z[i], d, sigma), 0) ** 2
    return sum


def grad3_tull(x, z, inner):
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


def grad3(x,z,inner_a, inner_b):
    c, rho, d, sigma = construct_circle_params(x)
    g1 = np.zeros(2)
    g2 = 0
    g3 = np.zeros(2)
    g4 = 0
    
    for i in range(len(z)):
        ra = r3(z[i], c, rho) 
        rb = r3(z[i], d, sigma)
        
        if (i in inner_a):
            g1 = g1 -4 * np.max(ra,0) * (z[i] - c)
            g2 = g2 - 4 * rho * np.max(ra,0)
            g3 = g3 - 4 * np.min(rb,0) * (z[i] - d)
            g4 = g4 - 4 * sigma * np.min(rb,0)
        
        if (i in inner_b):
            g1 = g1 -4 * np.min(ra,0) * (z[i] - c)
            g2 = g2 - 4 * rho * np.min(ra,0)
            g3 = g3 - 4 * np.max(rb,0) * (z[i] - d)
            g4 = g4 - 4 * sigma * np.max(rb,0)
        
        else:
            g1 = g1 -4 * np.min(ra,0) * (z[i] - c)
            g2 = g2 - 4 * rho * np.min(ra,0)
            g3 = g3 - 4 * np.min(rb,0) * (z[i] - d)
            g4 = g4 - 4 * sigma * np.min(rb,0)
    
    g = np.array(g1[0], g1[1], g2, g3[0], g3[1], g4)
    return g

def generate_points3(x, size=300):
    c, rho, d, sigma = construct_circle_params(x)
    #points = np.random.multivariate_normal(c, 1 * np.linalg.inv(A), size=size) #endre til ish uniform 
    points = np.random.uniform(-3,3,((size,2))) #Evt endre hvor stort området skal være i stedet for -2, 2
    inner_a = []
    inner_b = []
    for i in range(len(points)):
        #Trenger to ekstra if-setninger, én for den andre sirkelen og én hvis den er i begge
        #Lag litt omstendelig med fire if-setninger først 
        if ((r3(points[i], c, rho) <= 0) and (r3(points[i], d, sigma) >= 0)):
            print("bare a")
            inner_a.append(i)
        elif ((r3(points[i], d, sigma) <= 0) and (r3(points[i], c, rho) >= 0)):
            print("bare b")
            inner_b.append(i)
        elif ((r3(points[i], d, sigma) <= 0) and (r3(points[i], c, rho) <= 0)):
            #tilfeldig utvelgelse:
            a = np.random.uniform(0,1)
            print(a)
            if a < 0.5:
                inner_a.append(i)
            else:
                inner_b.append(i)
                
    return points, inner_a, inner_b #returner inner_a og inner_b feks. 

def plot_solution3(xf, points, inner_a, inner_b, funk, n, Metode): #bør ta inn inner_a og inner_b
    c, rho, d, sigma = construct_circle_params(xf) #endret

    minx = min(points[:, 0]) #kan også settes til -3, 3 osv, siden det er der vi har definert punktene våre
    maxx = max(points[:, 0]) #men kanskje like greit å bare beholde

    miny = min(points[:, 1])
    maxy = max(points[:, 1])

    x = np.arange(minx, maxx, 0.01)
    y = np.arange(miny, maxy, 0.01)

    X, Y = np.meshgrid(x, y)
    #Z = funk(Af, cf, X, Y) #denne bør endres, her har du faktisk sirkelen din
    Z_a = funk(X,Y, c, rho)
    Z_b = funk(X,Y, d, sigma)
    #du vil ha to av dem, og det vil være f3 som sendes inn? 
    #Du må lage en rxy3 
    #tidligere har rxy tatt inn A, c, x, y
    #altså bør rxy3 ta inn c, rho, x, y -> tror det er sirkelligningen (eller er ganske sikker)

    plt.figure()

    plt.contour(X, Y, Z_a, [0], colors='red')
    plt.contour(X, Y, Z_b, [0], colors='green')
    if n == 0:
        plt.title("Initial guess")
    else:
        #plt.title("Ferdig")
        plt.title('Metode {}'.format(Metode)+' finished after n= {}'.format(n) + ' iterations')
    plt.scatter(points[:, 0], points[:, 1], c = 'blue')
    plt.scatter(points[inner_a, 0], points[inner_a, 1], c = 'red')#legg til en til av denne. 
    plt.scatter(points[inner_b, 0], points[inner_b, 1], c = 'green')
    plt.show()
    
if __name__ == '__main__':
    
    x = [1,0,2,0,1,1.5]
    
    points, inner_a, inner_b = generate_points3(x, size=300)
    
    x0 = np.array([4, 1, 3, 0, 0])
    
    plot_solution(x0, points, inner_a, rxy_tilde, 0, 2)
    plot_solution(x0, points, inner_b, rxy_tilde, 0, 2)
    plot_solution3(x, points, inner_a, inner_b, rxy3, 0, 3)
