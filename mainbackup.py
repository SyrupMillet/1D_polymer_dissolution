import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from output import out_class

# Define the main polymer parameters
L0 = 0.01    # Initial polymer length [cm]
nps = 25    # number of numerical points

# Define constants
rho_p = 1.05 # polymer density   [g/cm^3]
MW_cri = 38000.0 # critical molecular weight [g/mol]
MW = 400000.0 # molecular weight [g/mol]
Flory_para = 0.49 # Flory interaction parameter \X
MV_sol = 89.4 # solvent molar volume [cm^3/mol] \V1_bar

critical_gel_vf = 0.18 # critical gel volume fraction
kd = 2.15E-4 # disentanglement rate constant [1/s]
alpha = 2
beta = 12

R = 8.314 # gas constant [J/(mol*K)]
T = 300.0 # temperature [K]

# Maxwell model parameters
E = 1
miu = 0.1

# Diffusion coefficient parameters
D0 = 1.1E-10    # [cm^2/s]
ad = 20

# Define the time parameters
start_time = 0.0
end_time = 50
dt = 0.0001
out_dt = 0.5

# Calculate v1,eff
# solve the equation 31
# Verified with figure 2
def f1(v1):
    a = np.log(v1) + (1-v1) + Flory_para*(1-v1)**2
    v2 = 1-v1
    para = MV_sol*rho_p*(2.0/MW_cri-1/MW)
    a += para*(2/v2-v2)
    return a
# there should be a root in the range of (0,1)
v1eff = brentq(f1, 0.001, 0.999)
v2eff = 1-v1eff

print("v1eff: ", v1eff)
print("v2eff: ", v2eff)

# Calculate the disentanglement time
td = kd*(MW/MW_cri)**alpha*v2eff**beta
print("td: ", td)
tau_diff = L0**2/(D0*np.exp(ad*v1eff))
print("tau_diff: ", tau_diff)

def getEffectiveSolventVF() -> float:
    return v1eff

def getDiffusionCoefficient(v1) -> float:   # D12 [cm^2/s]
    D12 = D0*np.exp(ad*max(min(v1,1.0),0.0))
    return D12

def getStressConstitutive(D12, v1) -> float:
    return D12*MV_sol*v1/(R*T*(1-v1)*(1-2*Flory_para*v1))

def getv1residual(v1_old, sigma_xx_old,  x_act, xs_ind) -> np.ndarray:
    leny = len(v1_old)
    y = v1_old.copy()
    sigma = sigma_xx_old.copy()
    x = np.zeros(len(y))
    for i in range(len(y)):
        x[i] = (x_act[i]+x_act[i+1])/2
    # get flux across the cell interface, length = nps+1(len(y)-1)
    flux = np.zeros(leny-1)
    for i in range(leny-1):
        #interpolate the volume fraction
        v1_inter = (y[i]+y[i+1])/2
        # get the diffusion coefficient
        D12_inter = getDiffusionCoefficient(v1_inter)
        StressCoeff_inter = getStressConstitutive(D12_inter, v1_inter)

        flux[i] = -D12_inter*(y[i+1]-y[i])/(x[i+1]-x[i]) - StressCoeff_inter*(sigma[i+1]-sigma[i])/(x[i+1]-x[i])
    
    # get the residual
    res = np.zeros(nps+2)
    for i in range(1, nps+1):
        res[i] = (flux[i-1]-flux[i])/(x[i]-x[i-1])
    res[0] = flux[0]/(x[1]-x[0])
    res[nps+1] = flux[nps-1]/(x[nps+1]-x[nps])

    # for cells larger than xs_ind, set the residual to zero
    for i in range(xs_ind+1, nps+2):
        res[i] = 0.0
    
    return res

def getStressResidual(sigma_xx_old,xs_ind, v1=0, v1_res=0) -> np.ndarray:
    y = sigma_xx_old.copy()
    res = -1.0*y/(miu/E) + E/(1-v1)**2*v1_res
    # for cells larger than xs_ind, set the residual to zero
    for i in range(xs_ind+1, nps+2):
        res[i] = 0.0
    return 0.0      # ignore stress residual now

def main():
    # ==== Define working parameters ====
    xs_ind = nps    # cell index of the solvent-gel interface position
    xg_ind = nps    # cell index of the gel-glassy interface position
    # ==== Define working arrays ====
    X_uc = np.zeros(nps+3) # Undeformed coordinates of the polymer , vertex position
    x_act = np.zeros(nps+3) # actual positions of the polymer , vertex position
    dx_act = np.zeros(nps+2) # actual cell size
    v1_sol = np.zeros(nps+2) # solvent volume fraction
    release_clock = np.zeros(nps+2) # release clock for disenaglement
    release_clock_running = [False for i in range(nps+2)] # release clock running flag
    sigma_xx = np.zeros(nps+2) # stress in the polymer

    # ==== Initialize ====
    dx_act = [L0/nps for i in range(nps+2)]
    X_uc[0] = 0.0 - dx_act[0]
    for i in range(1, nps+3):
        X_uc[i] = X_uc[i-1] + dx_act[i-1]
    x_act = X_uc.copy()

    # apply the boundary conditions to v1_sol, sigma_xx
    v1eff = getEffectiveSolventVF()
    v1_sol[xs_ind+1] = 2.0*v1eff-v1_sol[xs_ind]
    v1_sol[0] = v1_sol[1]       # dv1(x=0)/dx = 0
    sigma_xx[xs_ind+1] = 0.0 - sigma_xx[xs_ind]
    sigma_xx[0] = sigma_xx[1]    # d(sigma_xx)/dx = 0
    dx_act[xs_ind+1] = dx_act[xs_ind]
    dx_act[0] = dx_act[1]

    # ==== Output ====
    # Define the output
    otp = out_class("csv")
    otp.addOutput("position.csv")
    otp.addOutput("x_act.csv")
    otp.addOutput("solvf.csv")
    otp.addOutput("sigma_xx.csv")
    otp.addOutput("release_clock.csv")

    otp.addDataList("time", 0)
    otp.addDataList("nondim_time", 0)
    otp.addDataList("normed_xs", 0)
    otp.addDataList("normed_xg", 0)
    otp.addDataList("normed_thickness", 0)
    otp.addDataList("v1", nps+2)
    otp.addDataList("x_act", nps+3)
    otp.addDataList("sigma_xx", nps+2)
    otp.addDataList("release_clock", nps+2)

    otp.bindDataToFile("position.csv", "time","nondim_time", "normed_xs", "normed_xg","normed_thickness")
    otp.bindDataToFile("solvf.csv", "time", "v1")
    otp.bindDataToFile("x_act.csv", "time", "x_act")
    otp.bindDataToFile("sigma_xx.csv", "time", "sigma_xx")
    otp.bindDataToFile("release_clock.csv", "time", "release_clock")
    otp.initOutput()
    
    # ==== Time loop ====
    cur_time = start_time
    while cur_time < end_time:
        ##print('[ Time ]: ',cur_time)
        # remenber the old state
        v1_sol_old = v1_sol.copy()
        sigma_xx_old = sigma_xx.copy()

        # implicitlt Solve the diffusion equation
        # Verified with 1-D diffusion problem
        # get v1 residual dv1/dt from last time step
        v1_res = getv1residual(v1_sol_old, sigma_xx_old, x_act, xs_ind)
        # solve stress equation
        def f(t, y):
            return getStressResidual(y, xs_ind, v1_sol_old, v1_res)
        sol = solve_ivp(f, [cur_time, cur_time+dt], sigma_xx_old, method='RK45', t_eval=[cur_time+dt])
        sigma_xx = sol.y[:,0]
        # Solve the diffusion equation
        def f(t, y):
            return getv1residual(y, sigma_xx, x_act, xs_ind)
        sol = solve_ivp(f, [cur_time, cur_time+dt], v1_sol_old, method='BDF', t_eval=[cur_time+dt])
        v1_sol = sol.y[:,0]

        # apply the boundary conditions
        v1eff = getEffectiveSolventVF()
        v1_sol[xs_ind+1] = 2.0*v1eff-v1_sol[xs_ind]
        v1_sol[0] = v1_sol[1]       # dv1(x=0)/dx = 0
        sigma_xx[xs_ind+1] = 0.0 - sigma_xx[xs_ind]
        sigma_xx[0] = sigma_xx[1]    # d(sigma_xx)/dx = 0

        # update the release clock
        for i in range(1, nps+1):
            if (v1_sol[i] > critical_gel_vf):
                release_clock_running[i] = True
        for i in range(1, nps+1):
            if release_clock_running[i]:
                release_clock[i] += dt
            else:
                release_clock[i] = 0.0

        # check if release time reach the threshold, if so move the xs
        # move xs based on the release clock
        for i in range(1, nps+1):
            if release_clock[i] > td:
                xs_ind = i
                break

        # update x
        for i in range(1, xs_ind+1):
            dx_act[i] = (L0/nps)/(1.0-v1_sol[i])
        dx_act[xs_ind+1] = dx_act[xs_ind]
        dx_act[0] = dx_act[1]
        x_act[0] = 0.0 - dx_act[0]
        for i in range(1, nps+3):
            x_act[i] = x_act[i-1] + dx_act[i-1]

        # output
        multi = int(1/dt+0.5)
        if (int(cur_time*multi+0.5)%int(out_dt*multi+0.5) == 0):
            otp.appendData("time", format(cur_time, ".2f"))
            otp.appendData("nondim_time", cur_time*D0*np.exp(ad*v1eff)/L0**2)
            otp.appendData("normed_xs", x_act[xs_ind+1]/L0)
            xg_ind = nps
            for i in range(0, nps+2):
                if (release_clock_running[i]):
                    xg_ind = i
                    break
            otp.appendData("normed_xg", x_act[xg_ind+1]/L0)
            otp.appendData("normed_thickness", -(x_act[xg_ind+1]-x_act[xs_ind+1])/L0)
            otp.appendData("v1", v1_sol)
            otp.appendData("x_act", x_act)
            otp.appendData("sigma_xx", sigma_xx)
            otp.appendData("release_clock", release_clock)
            otp.updateOutput()

        # Update time
        cur_time += dt
        



if __name__ == "__main__":
    main()