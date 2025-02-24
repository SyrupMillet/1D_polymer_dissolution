import numpy as np
from output import out_class
from parameters import Parameters
from scalarSolver import Timer, MovingBoundarySolver

# Define the main polymer parameters
L0 = 1    # Initial polymer length [m]
nps = 10    # number of numerical points


# Define the time parameters
timer = Timer(0.0, 0.5, 0.001, 0.001)

# Define the parameters
para = Parameters()


# initailize the inside diffusion domain
solver1 = MovingBoundarySolver(nps, 0, L0, timer)
# initiate phi1 field
solver1.y.fill(0.0)
# Get phi1Minus at S_right
phi1Minus = para.getPhi1Minus()
solver1.setBoundaryCondition(\
    {"type":"Neumann","value":0.0},\
    {"type":"Dirichlet","value":phi1Minus})
# Initiate the velocity field and diffusion coefficient field
v1 = np.zeros_like(solver1.x_f)
Dcoef1 = np.zeros_like(solver1.x_f)
Dcoef1.fill(0.5)
solver1.updateVelocity(v1)
solver1.updateDiffusionCoefficient(Dcoef1)
solver1.updateBoundaryVelocity(0.0, 0.0)
solver1.updateBoundaryPosition()
solver1.applyBC()

# print inital field
print(f"t = {timer.cur_time}")
print(f"x_f = {solver1.x_f}")
print(f"phi1 = {solver1.y[1:-1]}")
phi2 = 1-solver1.y[1:-1]
print(sum(phi2*(solver1.x_f[1:]-solver1.x_f[:-1])))
print(" \n ")


while not timer.isEnd():

    # get phi1 inflow flux at S
    phi1Inflow = solver1.getResidual(timer.cur_time, solver1.y[1:-1])[2]


    solver1.step()


    # update moving boundary velocity
    solver1.updateBoundaryVelocity(0.0,-phi1Inflow)
    solver1.updateBoundaryPosition()

    # apply boundary condition
    solver1.applyBC()

    timer.increment()




    

# get integral of 1-phi1 in solver1
phi1 = solver1.y[1:-1]
print("phi1: ", phi1)
phi2 = 1-phi1
phi2Int = np.sum(phi2*(solver1.x_f[1:]-solver1.x_f[:-1]))
print("phi2Int: ", phi2Int)