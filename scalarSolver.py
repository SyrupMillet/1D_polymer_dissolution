import numpy as np

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
        # if self.CFL_list is not None:
        #     cfl = max(self.CFL_list)
        #     if cfl > 1:
        #         print("CFL number is larger than 1, please reduce the time step")
        #         self.dt = self.dt/cfl
        self.CFL_list = []

        self.cur_time += self.dt

    def isEnd(self) -> bool:
        return self.cur_time >= self.end_time
    
    def isWrite(self):
        if (self.dt < 1.0):
            multi = int(1/self.dt+0.5)
            return (int(self.cur_time*multi+0.5)%int(self.write_interval*multi+0.5) == 0)
        else:
            return (int(self.cur_time)%int(self.write_interval) == 0)


class MovingBoundarySolver:
    """
    Using ALE method to solve the scalar transport equation with moving boundary

    Initial PDE:
        ∂y/∂t + v(x,t) ∂y/∂x = D_12(ξ,t) ∂²y/∂x²
        x from S_left(t) to S_right(t)
    Let:
        x(ξ,t) = S_left(t) + ξ [S_right(t) - S_left(t)] ; ξ ∈ [0,1]
    Then:
        ∂Y/∂t + u_eff(ξ,t) ∂Y/∂ξ = D_eff(ξ,t) ∂²Y/∂ξ²
    where:
        J = S_right(t) - S_left(t)
        mesh velocity : w(ξ,t) = v_left(t) + ξ [v_right(t)-v_left(t)]
        u_eff = (v(x,t)-w(ξ,t)) / J
        D_eff = D(x,t) / J^2

    - boundary condition bc_left, bc_right: dictionary format, must contain "type" ("Dirichlet" or "Neumann")
            and "value" (function, receive current time t and return boundary value or boundary derivative)
    
    """

    nps : int                   # number of points in the domain
    timer : Timer = None        # Timer class

    # Initial condition
    S_left0 : float = 0.0       # initial left boundary
    S_right0 : float = 1.0      # initial right boundary

    # Working Arrays
    y : np.ndarray = None       # scalar field
    y_old : np.ndarray = None   # scalar field at previous time step

    ksi_c : np.ndarray = None   # cell-centered points
    ksi_f : np.ndarray = None   # cell-faced points
    dksi : np.ndarray = None    # cell size

    x_c : np.ndarray = None     # cell-centered physical points
    x_f : np.ndarray = None     # cell-faced physical points

    v : np.ndarray = None       # velocity field, cell-faced
    D : np.ndarray = None       # diffusion coefficient field, cell-faced
    
    S_left : float              # left boundary
    S_right : float             # right boundary
    V_left : float = 0.0            # left velocity
    V_right : float = 0.0            # right velocity

    # Boundary conditions
    # in form of {"type": "Dirichlet", "value": 1.0}
    # "type" can be "Dirichlet" or "Neumann"
    # if Neumann, "value" is D*dy/dx
    bc_left : dict = None       # left boundary condition
    bc_right : dict = None      # right boundary condition

    def __init__(self,
                 nps:int,
                 S_left0:float,
                 S_right0:float,
                 timer:Timer):
        self.nps = nps
        self.timer = timer
        self.S_left0 = S_left0  # initial left boundary
        self.S_right0 = S_right0 # initial right boundary
        self.S_left = S_left0 ; self.S_right = S_right0

        self.y = np.zeros(nps+3)    # scalar field add two ghost points
        self.y_old = np.zeros(nps+3) # scalar field at previous time step
        self.ksi_f = np.linspace(0,1,nps+1)     # excluding ghost points

        self.dksi = 1/nps
        self.x_f = np.zeros(nps+1)

        self.v = np.zeros(nps+1) 
        self.D = np.zeros(nps+1)

        # compute physical points
        self.updatePhysicalPoints()

    def updatePhysicalPoints(self):
        self.x_f = self.S_left + self.ksi_f*(self.S_right-self.S_left)
        self.x_c = 0.5*(self.x_f[1:] + self.x_f[:-1])

    def setBoundaryCondition(self, bc_left:dict, bc_right:dict):
        self.bc_left = bc_left
        self.bc_right = bc_right

    def updateDiffusionCoefficient(self, D:np.ndarray):
        if (D.shape[0] != self.nps+1):
            raise ValueError("Diffusion coefficient array size mismatch")
        self.D = D

    def updateVelocity(self, v:np.ndarray):
        if (v.shape[0] != self.nps+1):
            raise ValueError("Velocity array size mismatch")
        self.v = v

    def updateBoundaryVelocity(self, V_left:float, V_right:float):
        self.V_left = V_left
        self.V_right = V_right

    # if is Nuemann BC, value is D*dy/dx 
    def applyBC(self):
        if (self.bc_left==None) or (self.bc_right==None):
            raise ValueError("Boundary conditions not set")
        
        if self.bc_left["type"] == "Dirichlet":
            value = self.bc_left["value"]
            self.y[0] = value
            self.y[1] = value
        elif self.bc_left["type"] == "Neumann":
            value = self.bc_left["value"]
            # convert dy/dx to dy/dξ
            dksi = self.dksi
            S = self.S_right - self.S_left
            D = self.D[0]
            self.y[0] = self.y[2] + 2*dksi*value*S/D
        else:
            raise ValueError("Unknown boundary condition type")
        
        if self.bc_right["type"] == "Dirichlet":
            value = self.bc_right["value"]
            self.y[-2] = value
            self.y[-1] = value
        elif self.bc_right["type"] == "Neumann":
            value = self.bc_right["value"]
            # convert dy/dx to dy/dξ
            dksi = self.dksi
            S = self.S_right - self.S_left
            D = self.D[-1]
            self.y[-1] = self.y[-3] - 2*dksi*value*S/D
        else:
            raise ValueError("Unknown boundary condition type")
        
        
    def getResidual(self):
        # Prepare parameters
        S_m = self.S_right - self.S_left

        um = (1-self.ksi_f)*self.V_left + self.ksi_f*self.V_right
        u_eff = (self.v - um)/S_m

        D_eff = self.D/S_m**2
        
        y = self.y # including ghost points
        
        # Compute residual
        diffuseRes = D_eff*(y[2:] - 2*y[1:-1] + y[:-2])/(self.dksi**2)
        advectRes = u_eff*(y[2:] - y[:-2])/(2*self.dksi)
        residual = -advectRes + diffuseRes

        # Compute flux
        fluxr = (3*y[-2] - 4*y[-3] + y[-4])/(2*self.dksi)/S_m
        fluxl = (3*y[1] - 4*y[2] + y[3])/(2*self.dksi)/S_m

        return residual, fluxl, fluxr



    def updateBoundaryPosition(self):
        dt = self.timer.dt
        # update boundary position
        self.S_left = self.S_left + self.V_left*dt
        self.S_right = self.S_right + self.V_right*dt
        # update physical points
        self.updatePhysicalPoints()

    
    def step(self):
        curtime = self.timer.cur_time
        dt = self.timer.dt

        # rememeber old value
        self.y_old = np.copy(self.y)

        y = self.y[1:-1]

        # explicit Euler
        self.y[1:-1] = y + dt*self.getResidual()[0]


    def getCFL(self):
        S_m = self.S_right - self.S_left
        dt = self.timer.dt
        dksi = self.dksi
        return 2*dt/(dksi**2*S_m**2)



if __name__ == "__main__":
    """
    Here is an example to stefan problem
        dU/dt = d²U/dx², 0<x<S(t), t>0
    with boundary conditions:
        dU/dx = -exp(t), at x=0
        U = 0, at x=S(t)
    and initial
        U = 0, 0<x<S(0)

    The boundary moves with
        dS/dt = -dU/dx|_{x=S(t)}
        S(t=0) = 0

    The exact solution is
        U = exp(t-x)-1, 0<x<S(t), 0<t<1
        S(t) = t
    
    """
    L1 = 0.0005    # Initial length
    nps1 = 20    # number of numerical points

    # Define the time parameters
    timer = Timer(0.0, 0.5, 0.0000000001, 0.01)

    # initailize the inside diffusion domain
    solver1 = MovingBoundarySolver(nps1, 0, L1, timer)
    # initiate field
    solver1.y.fill(0.0)

    # Initiate the velocity field and diffusion coefficient field
    v1 = np.zeros_like(solver1.x_f)
    Dcoef1 = np.zeros_like(solver1.x_f)

    Dcoef1.fill(1.0)

    solver1.updateVelocity(v1)
    solver1.updateDiffusionCoefficient(Dcoef1)

    solver1.updateBoundaryVelocity(0.0, 0.0)
    solver1.updateBoundaryPosition()

    solver1.setBoundaryCondition(\
        {"type":"Neumann","value":(np.exp(timer.cur_time))},\
        {"type":"Dirichlet","value":0.0})
    solver1.applyBC()

    # print initial CFL
    print(f"CFL = {solver1.getCFL()}")

    while not timer.isEnd():
        if timer.isWrite():
            print(f"t = {timer.cur_time}")
            print(f"x = {solver1.x_f}")
            print(f"y = {solver1.y[1:-1]}")

        # Get dU/dx at S
        dUdx = solver1.getResidual()[2]

        solver1.step()

        solver1.updateBoundaryVelocity(0.0, -dUdx)

        solver1.updateBoundaryPosition()

        solver1.setBoundaryCondition(\
        {"type":"Neumann","value":(np.exp(timer.cur_time))},\
        {"type":"Dirichlet","value":0.0})
        solver1.applyBC()

        # dynamic time step
        S_m = solver1.S_right - solver1.S_left
        dt = 0.25*solver1.dksi**2*S_m**2
        timer.dt = dt
        timer.increment()

    # print final result and compare with exact solution
    print(f"t = {timer.cur_time}")
    print(f"x = {solver1.x_f}")
    print(f"y = {solver1.y[1:-1]}")
    print(f"Exact = {np.exp(timer.cur_time-solver1.x_f)-1}")
    
