from scipy.optimize import brentq
import scipy.optimize as opt
import numpy as np

class Parameters:
    # constants
    T = 300.0
    R = 8.314
    NA = 6.022E23

    # Solvent Molar Volume [cm^3/mol]
    smv = 89.4
    # polymer density [g/cm^3]
    rho_p = 1.05
    # critical molecular weight [g/mol]
    MW_cri = 38000.0
    # molecular weight [g/mol]
    MW = 52000.0
    Flory_para = 0.49
    xn = 554

    # Deffusion coefficients
    # Monomer molecular weight of styrene [g/mol]
    MW_mon = 104.15
    # average bond length [m]
    abl = 0.154E-9

    D0 = 4.0E-14    # [m^2/s]
    ad = 7

    # Solvent viscosity
    eta1 = 0.4E-3   # [Pa.s]


    def __init__(self):
        # Degree of polimerization
        self.N_dp = self.MW/self.MW_mon
        # Radius of gyration [m]
        self.rg = 0.408248*self.abl*self.N_dp**0.5
        # D2 coefficient [m^2/s]
        self.A = 4.8157E-4*self.T/self.eta1*self.N_dp**(-2.4)

    # solve chemical potential equilibrium equation at the gel-solvent interface S
    def getPhi1Minus(self) -> float:
        v1 = self.smv
        rho2 = self.rho_p
        mc = self.MW_cri
        m = self.MW
        x = self.Flory_para
        def f1(phi1):
            phi2 = 1-phi1
            a = np.log(phi1) + (1-1/self.xn)*(1-phi1) + x*(1-phi1)**2 + v1*rho2*phi2*(2.0/mc-1/m)*(2/phi2-phi2)
            return a
        return brentq(f1, 0.001, 0.999)
    
    # Get inside region diffusion coefficient [m^2/s]
    def getD12(self, phi) -> float:
        # if (phi < self.cri_frac):
        #     return self.D0*np.exp(self.ad*phi)
        # else:
        #     return self.A/(1-phi)**1.9
        return self.D0*np.exp(self.ad*phi)
    
    # Get outside region diffusion coefficient [m^2/s]
    def getDp(self) -> float:
        Dp = 1.1648E-14*self.T/(self.eta1*self.N_dp**0.5) # in [m^2/s]
        return Dp
    
    # Get reptation time [s]
    def getRepTime(self, phi2) -> float:
        trep = 0.01368*self.eta1/self.T*phi2**1.9*self.N_dp**3.4

        # Boltzmann constant [J/K]
        kb = 1.38E-23

        return trep
    
    # Get disentanglement rate [m/s]
    def getKd(self, phi2) -> float:
        kd = 4.5958E-9*self.T/self.eta1*phi2**(-1.9)*self.N_dp**(-2.9)
        return kd


# test
if __name__ == "__main__":
    para = Parameters()
    a = para.getPhi1Minus()
    print(a)
    print(para.getRepTime((1-a)))