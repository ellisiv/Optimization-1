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

x = np.linspace(-2, 2, 20)
plt.figure()
plt.scatter(-1, 0, color='red')
plt.scatter(0, 0, color='blue')
plt.scatter(1, 0, color='red')
#plt.plot(x, x * 0, color='black')
plt.xlim(-1.1, 1.1)
plt.axis('off')
plt.show()
