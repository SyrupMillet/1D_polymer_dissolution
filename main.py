import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from output import out_class
from parameters import Parameters
from scalarSolver import mesh, Timer, mbScalarSolver

# Define the main polymer parameters
L0 = 1    # Initial polymer length [m]
nps = 10    # number of numerical points

# Define the time parameters
timer = Timer(0.0, 0.5, 0.0001, 0.0001)

# Define the parameters
para = Parameters()





