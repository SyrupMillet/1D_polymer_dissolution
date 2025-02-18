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

# Define inside diffusion solver
solver1 = mbScalarSolver(nps=nps, x_left=0.0, x_right=L0, timer=timer)

# initiate the scalar field
solver1.phi.fill(0.0)

# define inside diffusion residue function
def f1(t:float, phi1:np.ndarray) -> np.ndarray:
    lenphi = solver1.msh.nps_tot
    flux = np.zeros(solver1.msh.nps_tot-1)
    for i in range(lenphi-1):
        #interpolate the volume fraction
        phi_int = (phi1[i]+phi1[i+1])/2
        # get the diffusion coefficient
        # D12 = para.getD12(phi)
        D12 = 0.5
        flux[i] = -D12*(phi1[i+1]-phi1[i])/(solver1.msh.xc_[i+1]-solver1.msh.xc_[i])
    res = np.zeros(lenphi)
    res[0] = flux[0]/(solver1.msh.xc_[1]-solver1.msh.xc_[0])*(1-phi1[0])
    res[lenphi-1] = -flux[lenphi-2]/(solver1.msh.xc_[lenphi-1]-solver1.msh.xc_[lenphi-2])
    for i in range(1,lenphi-1):
        res[i] = (flux[i-1]-flux[i])/(solver1.msh.xc_[i]-solver1.msh.xc_[i-1])
    return res

solver1.setResidualFunction(f1)

# Define the boundary conditions
phi1_bc_left = {'type':'neumann', 'value':0.0}
phi1_bc_right = {'type':'dirichlet', 'value':1.0}
solver1.setBoundaryConditions(phi1_bc_left, phi1_bc_right)
solver1.applyBoundaryConditions()

print("Start solving the diffusion equation")
print(solver1.phi)

while (not(timer.isEnd())):

    # get the flux of phi1 across the right boundary
    flux = 0
    D12 = 0.5
    flux = -D12*(solver1.phi[-1]-solver1.phi[-2])/(solver1.msh.xc_[-1]-solver1.msh.xc_[-2])

    solver1.solve()
    solver1.mvb1(V_right=-1*flux)


    timer.increment()
    if (timer.isWrite()):
        print("Time: ", timer.cur_time)
        print("phi1: ",solver1.phi)
        phi2 = np.ones(solver1.msh.nps_tot) - solver1.phi
        phi2[0] = 0.0 ; phi2[-1] = 0.0
        # print(sum(phi2))
        print(solver1.x_right)

print(solver1.phi)