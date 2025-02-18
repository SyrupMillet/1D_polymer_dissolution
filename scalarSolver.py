import numpy as np
from scipy.integrate import solve_ivp
from types import FunctionType as function

class mesh:

    # Define Mesh properties
    # number of cells inside the domain
    nps : int 
    # number of total cells, including ghost cells
    nps_tot : int
    # domain length
    L0 : float

    # Number of ghost cells on each side
    nghost : int = 1

    # vertex positions, length = nps+2*nghost+1
    xv_ : np.array
    # cell cennter positions, length = nps+2*nghost
    xc_ : np.array
    # cell size, length = nps+2*nghost
    dx_init : float
    dx_ : np.array
    inidx_ : np.array   # store content of L0 in the cell

    def __init__(self, nps:int, x_left:float, x_right:float):
        self.nps = nps
        self.nps_tot = nps + self.nghost*2
        self.L0 = x_right - x_left
        self.xv_ = np.zeros(nps+self.nghost*2+1)
        self.xc_ = np.zeros(nps+self.nghost*2)
        self.dx_ = np.zeros(nps+self.nghost*2)
        self.dx_init = self.L0/nps

        dx = self.L0/nps
        for i in range(nps+self.nghost*2+1):
            self.xv_[i] = x_left + (i-self.nghost)*dx
        for i in range(nps+self.nghost*2):
            self.xc_[i] = (self.xv_[i]+self.xv_[i+1])/2
            self.dx_[i] = self.xv_[i+1]-self.xv_[i]
        self.inidx_ = self.dx_.copy()

    # expand or shrink the mesh for one cell
    # boundary: 'left' or 'right'
    # direction: 'expand' or 'shrink'
    def adjustMesh(self, boundary:str, direction:str):
        if direction=='expand':
            a = 1
        elif direction=='shrink':
            a = -1
        else:
            print("Direction is not defined")
        self.nps += a; self.nps_tot += a; self.L0 += a*self.dx_
        xv_old = self.xv_.copy()
        if boundary == 'left':
            if a == 1:
                self.xv_ = np.insert(self.xv_, 0, self.xv_[0]-self.dx_)
            elif a == -1:
                self.xv_ = np.delete(self.xv_, [0])
        elif boundary == 'right':
            if a == 1:
                self.xv_ = np.append(self.xv_, self.xv_[-1]+self.dx_)
            elif a == -1:
                self.xv_ = np.delete(self.xv_, [-1])
        else:
            print("Boundary is not defined")

        self.xc_ = (self.xv_[:-1]+self.xv_[1:])/2
        self.dx_ = self.xv_[1:]-self.xv_[:-1]

    # def reconstructFromdx(self, lrFix:str):
    #     # if fix left boundary position
    #     if lrFix == 'left':
    #         xv_new = np.zeros(self.nps+self.nghost*2)
    #         xv_new[1] = self.xv_[1] ; xv_new[0] = self.xv_[1] - self.dx_[0]
    #         for i in range(2, self.nps+self.nghost*2):
    #             xv_new[i] = xv_new[i-1] + self.dx_[i-1]
    #         for i in range(self.nps+self.nghost*2):
    #             self.xc_[i] = (self.xv_[i]+self.xv_[i+1])/2
    #     # if fix right boundary position
    #     elif lrFix == 'right':
    #         xv_new = np.zeros(self.nps+self.nghost*2)
    #         xv_new[-2] = self.xv_[-2] ; xv_new[-1] = self.xv_[-2] + self.dx_[-1]
    #         for i in range(self.nps+self.nghost*2-2, -1, -1):
    #             xv_new[i] = xv_new[i+1] - self.dx_[i]
    #         for i in range(self.nps+self.nghost*2):
    #             self.xc_[i] = (self.xv_[i]+self.xv_[i+1])/2
    #     else:
    #         print("Boundary is not defined")

# A Timer class to record the time
class Timer:
    # Start time
    start_time : float = 0.0
    # End time
    end_time : float = 0.0
    # Time step
    dt : float = 0.0
    # current time
    cur_time : float = 0.0
    # write interval
    write_interval : float = 0.0

    # CFL list, write by user
    CFL_list : list = []

    def __init__(self, start_time:float,end_time:float, dt:float, write_interval:float):
        self.start_time = start_time
        self.end_time = end_time
        self.dt = dt
        self.write_interval = write_interval

        self.cur_time = start_time

    def increment(self):
        # check CFL
        if self.CFL_list is not None:
            cfl = max(self.CFL_list)
            if cfl > 1:
                print("CFL number is larger than 1, please reduce the time step")
                self.dt = self.dt/cfl
        self.CFL_list = []

        self.cur_time += self.dt

    def isEnd(self) -> bool:
        return self.cur_time >= self.end_time
    
    def isWrite(self):
        multi = int(1/self.dt+0.5)
        return (int(self.cur_time*multi+0.5)%int(self.write_interval*multi+0.5) == 0)


# This is a scalar solver class to solver 1-D scalar equation with moving bounndary
class mbScalarSolver:
    # Define mesh
    msh : mesh
    # Define Timer
    timer : Timer
    # Number of cells
    nps : int

    # Define initial properties
    init_x_left : float
    init_x_right : float

    # Define working arrays
    # Scalar field, length = nps+2*nghost
    phi : np.array
    phi_old : np.array
    # Residual of the scalar field, length = nps+2*nghost
    # res : np.array
    # left boundary position
    x_left : float
    # right boundary position
    x_right : float

    # Define boundary conditions
    # left boundary condition       {type: 'dirichlet' or 'neumann', value: float}
    bc_left : dict = None
    # right boundary condition      {type: 'dirichlet' or 'neumann', value: float}
    bc_right : dict = None

    # function to get the residual of the scalar field
    getResidual : callable = None

    def __init__(self, nps:int, x_left:float, x_right:float, timer:Timer):
        self.nps = nps
        self.x_left = x_left
        self.x_right = x_right

        self.init_x_left = x_left
        self.init_x_right = x_right

        self.timer = timer

        # Initialize mesh
        self.msh = mesh(nps, x_left, x_right)

        # Initialize scalar field
        self.phi = np.zeros(nps+self.msh.nghost*2)

    # Set callable function to get the residual of the scalar field
    def setResidualFunction(self, func:function):
        self.getResidual = func

    # Set boundary conditions
    def setBoundaryConditions(self, bc_left:dict, bc_right:dict):
        self.bc_left = bc_left
        self.bc_right = bc_right

    # Apply boundary conditions
    def applyBoundaryConditions(self):
        if self.bc_left is None:
            print("Left boundary condition is not set")
        if self.bc_right is None:
            print("Right boundary condition is not set")

        if self.bc_left['type'] == 'dirichlet':
            # set the value of left ghost cell from the left boundary condition and left boundary position
            value = self.bc_left['value']
            a = (self.msh.xc_[1]-self.msh.xc_[0])/(self.msh.xc_[1]-self.x_left)
            self.phi[0] = self.phi[1] + a*(value - self.phi[1])
        elif self.bc_left['type'] == 'neumann':
            # set the value of left ghost cell from the left boundary condition and left boundary position
            value = self.bc_left['value']
            self.phi[0] = self.phi[1] - value*(self.msh.xc_[1]-self.msh.xc_[0])
        else:
            print("Left boundary condition type is not defined")

        if self.bc_right['type'] == 'dirichlet':
            # set the value of right ghost cell from the right boundary condition and right boundary position
            value = self.bc_right['value']
            a = (self.msh.xc_[-1]-self.msh.xc_[-2])/(self.msh.xc_[-1]-self.x_right)
            self.phi[-1] = a*(value - self.phi[-2])+self.phi[-2]
        elif self.bc_right['type'] == 'neumann':
            # set the value of right ghost cell from the right boundary condition and right boundary position
            value = self.bc_right['value']
            self.phi[-1] = self.phi[-2] + value*(self.msh.xc_[-1]-self.msh.xc_[-2])
        else:
            print("Right boundary condition type is not defined")

    # Solve the scalar field for a time step
    def solve(self):
        curtime = self.timer.cur_time
        dt = self.timer.dt
        self.phi_old = self.phi.copy()
        sol = solve_ivp(self.getResidual, [curtime , curtime +dt], self.phi, method='RK45', t_eval=[curtime+dt])
        self.phi = sol.y[:,0]

    # V_right is the velocity of the right boundary
    def mvb1(self, V_right:float):
        right_old = self.x_right
        right_new = right_old + V_right*self.timer.dt
        self.timer.CFL_list.append(V_right*self.timer.dt/self.msh.dx_[-1])

        # check if the right boundary is out of the center of ghost cell, if yes, expand the mesh
        if (right_new > self.msh.xc_[-1]):
            phi1 = self.phi[-2] ; xc1 = self.msh.xc_[-2]
            phiS = self.bc_right['value'] ; xcS = right_new
            xc2 = self.msh.xc_[-1]
            # use interpolate to get the value of phi2 at xc2, its in between phi1 and phiS
            phi2 = phi1 + (phiS-phi1)*(xc2-xc1)/(xcS-xc1)
            self.phi[-1] = phi2
            # expand the mesh and add new ghost cell to phi
            self.msh.adjustMesh('right', 'expand')
            self.phi = np.append(self.phi, 0)
            self.x_right = right_new
        # check if the right boundary is inside the center of first cell in the domain, if yes, shrink the mesh
        elif (right_new < self.msh.xc_[-2]):
            self.msh.adjustMesh('right', 'shrink')
            self.phi = np.delete(self.phi, -1)
            self.x_right = right_new
        else:
            self.x_right = right_new

    
        # Apply boundary conditions 
        self.applyBoundaryConditions()
            