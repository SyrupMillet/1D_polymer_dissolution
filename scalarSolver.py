import numpy as np
from scipy.integrate import solve_ivp

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
        multi = int(1/self.dt+0.5)
        return (int(self.cur_time*multi+0.5)%int(self.write_interval*multi+0.5) == 0)


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

    - 边界条件 bc_left, bc_right：字典格式，必须包含 "type"（"Dirichlet" 或 "Neumann"）
         以及 "value"（函数，接收当前时间 t 返回边界值或边界导数）
    
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
    dksi_c : np.ndarray = None  # cell-centered cell size, consider ghost cells
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

        self.y = np.zeros(nps+2)    # scalar field add two ghost points
        self.y_old = np.zeros(nps+2) # scalar field at previous time step
        self.ksi_f = np.linspace(0,1,nps+1)     # excluding ghost points
        self.ksi_c = 0.5*(self.ksi_f[1:] + self.ksi_f[:-1])
        self.dksi = self.ksi_f[1:] - self.ksi_f[:-1]
        temp = np.zeros(nps+2) ; temp[1:-1] = self.ksi_c
        temp[0] = 2*self.ksi_f[0] - self.ksi_c[0] ; temp[-1] = 2*self.ksi_f[-1] - self.ksi_c[-1]
        self.dksi_c = temp[1:] - temp[:-1]
        self.x_f = np.zeros(nps+1)
        self.x_c = np.zeros(nps)
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

    def applyBC(self):
        if (self.bc_left==None) or (self.bc_right==None):
            raise ValueError("Boundary conditions not set")
        
        if self.bc_left["type"] == "Dirichlet":
            value = self.bc_left["value"]
            self.y[0] = 2*value - self.y[1]
        elif self.bc_left["type"] == "Neumann":
            value = self.bc_left["value"]
            # convert dy/dx to dy/dξ
            dksi = self.ksi_f[1] - self.ksi_f[0]
            J = self.S_right - self.S_left
            D = self.D[0]
            self.y[0] = self.y[1] - value*dksi*J/D
        else:
            raise ValueError("Unknown boundary condition type")
        
        if self.bc_right["type"] == "Dirichlet":
            value = self.bc_right["value"]
            self.y[-1] = 2*value - self.y[-2]
        elif self.bc_right["type"] == "Neumann":
            value = self.bc_right["value"]
            # convert dy/dx to dy/dξ
            dksi = self.ksi_f[-1] - self.ksi_f[-2]
            J = self.S_right - self.S_left
            D = self.D[-1]
            self.y[-1] = self.y[-2] + value*dksi*J/D
        else:
            raise ValueError("Unknown boundary condition type")
        
    def getResidual(self, t:float, y:np.ndarray):
        # The input y is the scalar field inside domain with length nps
        y_ = np.zeros(self.nps+2) ; y_[1:-1] = y
        y_[0] = self.y[0] ; y_[-1] = self.y[-1]
        # Compute residual
        # ∂Y/∂t = D_eff(ξ,t) ∂²Y/∂ξ² - u_eff(ξ,t) ∂Y/∂ξ

        # Prepare parameters
        J = self.S_right - self.S_left
        um = (1-self.ksi_f)*self.V_left + self.ksi_f*self.V_right
        u_eff = (self.v - um) / J
        D_eff = self.D / J**2
        
        # Conpute Flux
        # 1st order numerical method to compute diffusive flux across cell faces
        # len(flux) = n+1
        flux = D_eff*(y_[1:] - y_[:-1])/self.dksi_c
        # 1st order numerical method to compute advective flux across cell faces
        # interpolate to get cell-faced y
        flux = flux - u_eff*0.5*(y_[1:] + y_[:-1])
        # Compute residual
        # len(residual) = nps
        residual = (flux[1:] - flux[:-1])/self.dksi

        return residual
    
    def step(self):
        curtime = self.timer.cur_time
        dt = self.timer.dt

        # update boundary position
        self.S_left = self.S_left + self.V_left*dt
        self.S_right = self.S_right + self.V_right*dt
        self.updatePhysicalPoints()

        # rememeber old value
        self.y_old = np.copy(self.y)

        # Get in-domain y
        y = self.y.copy()[1:-1]
        # Solve the PDE
        sol = solve_ivp(self.getResidual, [curtime, curtime+dt], y, method='RK45', t_eval=[curtime+dt])
        
        # Update y
        self.y[1:-1] = sol.y[:,0]

        # Apply boundary conditions
        self.applyBC()


        




if __name__ == "__main__":
    N = 50
    timer = Timer(0,0.5,0.00005,0.1)
    solver = MovingBoundarySolver(N,0,1,timer)
    solver.setBoundaryCondition({"type":"Neumann","value":0.0},{"type":"Dirichlet","value":1.0})
    solver.updateVelocity(np.zeros(N+1))
    D = np.ones(N+1)
    solver.updateDiffusionCoefficient(D)
    solver.updateBoundaryVelocity(0.0,0.0)
    while not timer.isEnd():
        if timer.isWrite():
            print(f"t = {timer.cur_time}")
        solver.step()
        timer.increment()
    print(solver.x_c)
    print(solver.y[1:-1])
    
