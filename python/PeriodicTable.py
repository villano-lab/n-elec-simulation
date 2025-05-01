"""
PeriodicTable.py - module defining Element class, and defining a few useful ones
"""
#### module imports
import numpy as np
import scipy.constants as co
from numpy import tanh

m_n = co.physical_constants['neutron mass energy equivalent in MeV'][0]
m_p = co.physical_constants['proton mass energy equivalent in MeV'][0]
k = co.physical_constants['Boltzmann constant in eV/K'][0]
gram = 1/co.physical_constants['electron volt-kilogram relationship'][0]/1e9 # MeV/c^2 per gram
amu = co.physical_constants['atomic mass constant energy equivalent in MeV'][0]

class Element:
    
    def __init__(self, Z, A, masses = None, percent_abuns = np.array([1.]), capture_sxns = None, recoil_sxns = None, binding_energies = None, ionization_energies = None, E_ehpair = 0., avg_capture_sxn_sig = 0., bandgap = 0., bandgap_list = None, bandgap_fxn = None, rho = 0):
        self.Z = Z # atomic number
        self.A = A # mass numbers

        # abundances of each isotope (proportions)
        self.abuns = percent_abuns/100 if self.checkshape(percent_abuns, True) else None 
        
        # nuclear masses [MeV]
        self.masses = masses if self.checkshape(masses, True) else None 

        # total thermal neutron capture cross sections [barns]
        self.capture_sxns = capture_sxns if self.checkshape(capture_sxns, True) else None 

        # average capture cross section [cm^2]
        self.capture_sxn = self.avg(self.capture_sxns)/1e24

        # total elastic neutron scatter cross sections [barns]
        self.recoil_sxns = recoil_sxns if self.checkshape(recoil_sxns, True) else None

        # neutron separation energies [MeV]
        self.binding_energies = binding_energies if self.checkshape(masses, True) else None

        # ionization energies [eV]
        self.ionization_energies = np.array([ionization_energies]) if np.isscalar(ionization_energies) else ionization_energies

        self.E_ehpair = E_ehpair # average e/h pair energy [eV]

        self.bandgap = bandgap # band gap at room temperature [eV]

        self.rho = rho # solid density (natural abundances) (g/cm^3)

        # list of temperature [K], bandgap [eV] pairs
        if bandgap_list:
            self.bandgap_list = bandgap_list 
        else:
            self.bandgap_list = [[298, self.bandgap]]

        # function of the form Eg = f(T) giving band gap in eV as a function of temperature T in Kelvin
        if bandgap_fxn:
            self.bandgap_fxn = bandgap_fxn

        
        self.avg_capture_sxn_sig = avg_capture_sxn_sig # uncertainty in the average thermal neutron radiative capture cross section [barns]
        
        # [MeV]
        if self.binding_energies is not None and self.masses is None:
            self.masses = self.A*amu - self.binding_energies#self.Z*m_p + (self.A - self.Z)*m_n - self.binding_energies
        
        # MeV
        self.mass = self.avg(self.masses)
        
        # number density [cm^-3]
        self.rho_n = rho*gram/self.mass

        # energy transfer constants (unitless)
        self.alph = {str(A): 2*A/(A + 1)**2 for A in self.A}


    def avg(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.shape != self.abuns.shape:
            raise Exception('Error in Element.avg(): data must have same shape as Element.abuns') 
        return (self.abuns*data).sum()


    def checkshape(self, obj, allow_none = False, fatal = True):
        if allow_none and obj is None:
            return True
        if hasattr(obj, 'shape'):
            if self.A.shape != obj.shape:
                if fatal:
                    raise Exception('Error in Element object. Shape mismatch.')
                else:
                    return False
        elif not np.is_scalar(obj):
            if fatal:
                raise Exception('Error in Element object. obj must be scalar or ndarray, but is ' + str(type(obj)))
            else:
                return False
        return True
    
    
    # end Element class


def Si_bandgap_fit(T):
    """
    returns bandgap (eV) as a function of temperature T in silicon

    using function proposed by O'Donnell and Chen, Appl. Phys. Lett. 58, 2924 (1991); https://doi.org/10.1063/1.104723, using parameters given therein

    NOTE: valid for temperatures between 0 and 300 K
    """
    Eg0 = 1.17
    S = 1.49
    hwbar  = 0.0255
    return Eg0 - S*hwbar*(1/tanh(hwbar/2/k/T) - 1)

# create element objects
Si = Element(Z = 14,
             A = np.array([28, 29, 30]),
             binding_energies = np.array([21.4927, 21.895, 24.4329]),
             percent_abuns = np.array([92.223, 4.685, 3.092]),
             capture_sxns = np.array([0.177, 0.101, 0.107]),
             ionization_energies = 8.1517,
             E_ehpair = 3.8,
             avg_capture_sxn_sig = 0.003,
             bandgap = 1.12,
             bandgap_fxn = Si_bandgap_fit,
             rho = 2.33
             )


Ge = Element(Z = 32,
             A = np.array([70, 72, 73, 74, 76]),
             binding_energies = np.array([70.5618, 72.5859, 71.2975, 73.4224, 73.2128]),
             percent_abuns = np.array([20.57, 27.45, 7.75, 36.5, 7.73]),
             capture_sxns = np.array([3.15, 0.98, 15, 0.34, 0.06]),
             avg_capture_sxn_sig = 0.0001,
             )



