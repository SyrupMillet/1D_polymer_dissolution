import numpy as np
from output import out_class
from parameters import Parameters
from scalarSolver import Timer, MovingBoundarySolver

# Define the main polymer parameters
L1 = 0.005    # Initial polymer length [m]
nps1 = 20    # number of numerical points
L2 = 0.05*L1    # length of diffusion layer [m]
nps2 = 5    # number of numerical points in diffusion layer


# Define the time parameters
timer = Timer(0.0, 2400, 0.05, 100)

# Define the parameters
para = Parameters()


# initailize the inside diffusion domain
solver1 = MovingBoundarySolver(nps1, 0, L1, timer)
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
# Calculate the diffusion coefficient field from Para
for i in range(nps1+1):
    Dcoef1[i] = para.getD12(0.0)
Dcoef1[-1] = para.getD12(phi1Minus)
solver1.updateVelocity(v1)
solver1.updateDiffusionCoefficient(Dcoef1)
solver1.updateBoundaryVelocity(0.0, 0.0)
solver1.updateBoundaryPosition()
solver1.applyBC()

# initialize the outside diffusion layer domain
solver2 = MovingBoundarySolver(nps2, L1, L1+L2 , timer)
# initiate phi2 field
solver2.y.fill(0.0)
v2 = np.zeros_like(solver2.x_f)
Dcoef2 = np.zeros_like(solver2.x_f)
for i in range(nps2+1):
    Dcoef2[i] = para.getDp()
solver2.updateVelocity(v2)
solver2.updateDiffusionCoefficient(Dcoef2)
solver2.updateBoundaryVelocity(0.0, 0.0)
solver2.updateBoundaryPosition()
solver2.setBoundaryCondition(\
    {"type":"Neumann","value":0.0},\
    {"type":"Dirichlet","value":0.0})
solver2.applyBC()

# Get some parameters
trep = para.getRepTime((1-phi1Minus))
Kd = -para.getKd(1-phi1Minus)

phi2OutInt = 0.0

while not timer.isEnd():

    # get phi1 inflow flux at S-
    phi1Inflow = solver1.getResidual(timer.cur_time, solver1.y[1:-1])[2]
    # get phi2 inflow flux at S+
    phi2Inflow = solver2.getResidual(timer.cur_time, solver2.y[1:-1])[1]
    # get phi2 outflow flux at S+\delta
    phi2Outflow = solver2.getResidual(timer.cur_time, solver2.y[1:-1])[2]

    solver1.step()
    solver2.step()

    # combine the flux to get S moving velocity
    vs = -phi1Inflow - phi2Inflow/phi1Minus

    # update moving boundary velocity
    solver1.updateBoundaryVelocity(0.0,vs)
    solver1.updateBoundaryPosition()

    solver2.updateBoundaryVelocity(vs,vs)
    solver2.updateBoundaryPosition()

    # update diffusion coefficient in solver1
    for i in range(nps1+1):
        Dcoef1[i] = para.getD12(0.5*(solver1.y[i]+solver1.y[i+1]))
    solver1.updateDiffusionCoefficient(Dcoef1)

    # update domain velocity in solver2
    v2.fill(vs)
    solver2.updateVelocity(v2)
    # update boundary condition in solver2
    if (timer.cur_time > trep):
        # Get left boundary phi2 in solver2
        phi2Left = 0.5*(solver2.y[0]+solver2.y[1])
        if (phi2Left < (1-phi1Minus)):
            # phi2 hasnt reached the equilibrium value at S, the boundary flux equal to disentanglement rate
            solver2.setBoundaryCondition(\
                {"type":"Neumann","value":Kd},\
                {"type":"Dirichlet","value":0.0})
        else:
            # phi2 has reached the equilibrium value at S
            solver2.setBoundaryCondition(\
                {"type":"Dirichlet","value":(1-phi1Minus)},\
                {"type":"Dirichlet","value":0.0})

    # apply boundary condition
    solver1.applyBC()
    solver2.applyBC()

    # Compute total amount of polymer in inside domain
    phi2 = 1.0 - solver1.y[1:-1]
    IntPhi2 = np.sum(phi2*(solver1.x_f[1:]-solver1.x_f[:-1]))
    # Compute total amount of polymer in outside domain
    IntPhi2_2 = np.sum(solver2.y[1:-1]*(solver2.x_f[1:]-solver2.x_f[:-1]))
    # Compute total amount of polymer flow out
    phi2OutInt += phi2Outflow*timer.dt

    phi2totalratio = (IntPhi2+IntPhi2_2+phi2OutInt)/L1


    timer.increment()

    if (timer.isWrite()):
        print(f"t = {timer.cur_time}")
        print(solver1.D)
        print(solver2.D)

print(solver1.S_right/0.005)
print(phi2OutInt/L1)

# plot the result
import matplotlib.pyplot as plt
plt.plot(solver2.x_c, solver2.y[1:-1], label='phi1')
plt.legend()
plt.show()