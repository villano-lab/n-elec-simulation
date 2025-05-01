"""
ProjectUtilities - utility functions such as sampling and trapezoidal integration for project
"""


# imports
import PeriodicTable as pt
import numpy as np
import scipy.constants as co
from sys import float_info

# explicit imports
from numpy import sqrt, pi, exp, concatenate, array, zeros, inf
from scipy.stats import truncnorm



#### physical constants
amu = co.physical_constants['atomic mass constant energy equivalent in MeV'][0]
hbar = co.physical_constants['Planck constant in eV/Hz'][0]/1e6/2/np.pi # reduced Planck's constant [MeV/Hz]
k = co.physical_constants['Boltzmann constant in eV/K'][0]/1e6 # Boltzmann constant in MeV/K
m_n = co.physical_constants['neutron mass energy equivalent in MeV'][0]
m_p = co.physical_constants['proton mass energy equivalent in MeV'][0]
m_e = co.physical_constants['electron mass energy equivalent in MeV'][0]
f0 = 4/3600 # ambient thermal neutron flux at sea level (cm^-2 s^-1)
yr = 365*24*3600 # seconds per year
day = 24*3600 # seconds per day
c = co.c # speed of light [m/s]
kg = 1/co.physical_constants['electron volt-kilogram relationship'][0]/1e6 # MeV/c^2 per kg
Ethermal = co.physical_constants['Boltzmann constant in eV/K'][0]*(co.zero_Celsius + 25) # standard thermal energy at room temperature (25 Celsius) [eV]
E_hole = pt.Si.E_ehpair # average energy of e/h pair in Si [eV]
T_hvev = 0.03 # operation temperature of HVeV detectors [K]
Eg = pt.Si.bandgap_fxn(T_hvev) # band gap in silicon at HVeV temp [eV]
mHVeV = 0.93/1e3 # mass of HVeV detectors [kg] (as of 2020: https://doi.org/10.1007/s10909-020-02349-x)
hw0 = 0.063 # optical phonon energy in Si [eV]
barn = 1e-24 # cm^2 per barn



#### functions

def PDF(x, y):
    """
    normalize y(x) by its integral over x (calculated via trapezoidal integration)

    NOTE:
        inputs x and y must be numpy arrays (must support slicing, element-wise addition, and element-wise multiplication)

    PARAMETERS:
        x - (numeric array) - independent variable
        y - (numeric array) - dependent variable

    RETURNS:
        (numeric array) - y normalized by the integral of y over x
    """
    return 2*y/sum((y[:-1] + y[1:])*(x[1:] - x[:-1]))


def trap(x, y):
    """
    Return trapezoidal integral of points given by (x,y)

    NOTE:
        inputs x and y must be numpy arrays (must support slicing, element-wise addition, and element-wise multiplication)

    PARAMETERS:
        x - (numeric array) - independent variable
        y - (numeric array) - dependent variable

    RETURNS:
        (numeric array) - trapezoidal integral of (the first dimension of) y over x
    """
    return sum((y[:-1] + y[1:])*(x[1:] - x[:-1]))/2


def logtrap(x, y):
    """
    Return logarithmic trapezoidal integral of points given by (x,y) (trapezoidal integration but with points assumed to be connected by an exponential curve)

    NOTE:
        inputs x and y must be numpy arrays (must support slicing, element-wise addition, and element-wise multiplication)

    PARAMETERS:
        x - (numeric array) - independent variable
        y - (numeric array) - dependent variable

    RETURNS:
        (numeric array) - trapezoidal integral of (the first dimension of) y over x
    """
    nz = (y > 0)
    x = x[nz]
    y = y[nz]
    return ((y[1:] - y[:-1])*(x[1:] - x[:-1])/np.log(y[1:]/y[:-1])).sum()


def maxwell(E, B):
    """
    Maxwell-Boltzmann distribution of energies at temperature T = 1/kB
    """
    return 2*B*sqrt(B*E/pi)*exp(-B*E)



def sample_from_pdf(xs, pdf, sample_size, seed = None):
    """
    return an array of length sample_size containing values in the domain spanned by independent variable array xs sampled randomly according to the PDF pdf, with the random generator (numpy.random.default_rng()) seeded with the argument seed

    PARAMETERS:
        xs - (ndarray) - array of independent variable
        pdf - (ndarray) - PDF array (must be same shape as xs)
        sample_size - (int or tuple of ints) - shape of output array
        seed - (optional) - seed value for random number generator (see documentation for numpy.random.default_rng()) (defaults to None (no seed; "fresh, unpredictable entropy will be pulled from the OS" to seed BitGenerator instance (from documentation)))

    RETURNS:
        (ndarray) - array of random values (shape = sample_size)
    """
    cdf = np.zeros(pdf.shape)
    x_ = xs[0]
    p_ = pdf[0]
    for i, (x, p) in enumerate(zip(xs[1:], pdf[1:])):
        cdf[i+1] = (cdf[i] if i else 0) + (p + p_)*(x - x_)/2
        x_ = x
        p_ = p
    cdf[-1] = 1
    rng = np.random.default_rng(seed = seed)
    random_numbers = rng.random(sample_size)
    indices = np.array([np.where((cdf[1:]>=val) & (cdf[:-1]<val))[0][0] for val in random_numbers])
    return xs[indices] + (random_numbers - cdf[indices])*(xs[indices + 1] - xs[indices])/(cdf[indices+1] - cdf[indices])


def sample_from_cdf(xs, cdf, sample_size, seed = None):
    """
    return an array of length sample_size containing values in the domain spanned by independent variable array xs sampled randomly according to the CDF cdf, with the random generator (numpy.random.default_rng()) seeded with the argument seed
    PARAMETERS:
        xs - (ndarray) - array of independent variable
        cdf - (ndarray) - CDF array (must be same shape as xs)
        sample_size - (int or tuple of ints) - shape of output array
        seed - (optional) - seed value for random number generator (see documentation for numpy.random.default_rng()) (defaults to None (no seed; "fresh, unpredictable entropy will be pulled from the OS" to seed BitGenerator instance (from documentation)))
    RETURNS:
        (ndarray) - array of random values (shape = sample_size)
    """
    rng = np.random.default_rng(seed = seed)
    random_numbers = rng.random(sample_size)
    indices = np.array([np.where((cdf[1:]>=val) & (cdf[:-1]<val))[0][0] for val in random_numbers])
    return xs[indices] + (random_numbers - cdf[indices])*(xs[indices + 1] - xs[indices])/(cdf[indices+1] - cdf[indices])



def histogram_to_cdf(bins, counts):
    """
    convert histogram of counts to cdf

    returned array of cdf values has the same shape as bins
    """
    cdf = np.zeros(bins.shape)
    c = 0
    for i, y in enumerate(counts):
        c += counts[i]
        cdf[i+1] = c
    cdf /= cdf[-1]
    return cdf


def interpolate_yvals(x, xk, yk):
    """
    for some xy data xk, yk, calculate the corresponding y values of data x by interpolating between nearest (xk, yk) points
    """
    i = np.array([np.where((xk[1:] >= this_x) & (xk[:-1] < this_x))[0][0] for this_x in x])
    return yk[i] + (yk[i+1] - yk[i])*(x - xk[i])/(xk[i+1] - xk[i])


def interpolate_yvals_with_equal_ends(x, xk, yk):
    """
    same as interpolate_yvals but looking for equal values at the ends of x and xk
    """
    out = np.zeros(x.shape)

    if x[0] == xk[0]:
        out[0] = yk[0]

        if x[-1] == xk[-1]: 
            out[-1] = yk[-1]
            s = slice(1, -1) # both ends equal

        else: # only start equal
            s = slice(1,None)

    elif x[-1] == xk[-1]: # only end equal
        out[-1] = yk[-1]
        s = slice(0,-1)
    
    else:
        s = slice(None)
    
    i = np.array([(np.where((xk[1:] >= this_x) & (xk[:-1] < this_x))[0][0]) for this_x in x[s]])
    out[s] = yk[i] + (yk[i+1] - yk[i])*(x[s] - xk[i])/(xk[i+1] - xk[i])
    return out





def interpolate_yval(x, xk ,yk):
    """
    same as interpolate_yvals_with_overlap() but x is a scalar
    """
    if x <= xk[0] or x > xk[-1]:
        return 0
    i = np.where((xk[1:] >= x) & (xk[:-1] < x))[0][0]
    return yk[i] + (yk[i+1] - yk[i])*(x - xk[i])/(xk[i+1] - xk[i])


def interpolate_yvals_with_overlap(x, xk, yk):
    """
    same as interpolate_yvals() but x outside the bounds of xk return zero
    """
    #print(x.shape, xk.shape, yk.shape)
    i_list = [((np.where((xk[1:] >= this_x) & (xk[:-1] < this_x))[0][0]) if (this_x > xk[0] and this_x <= xk[-1]) else -9999) for this_x in x]
    #print([(yk[i+1]-yk[i]) for i in i_list if i >= 0])
    out_list = [(0. if i == -9999 else (yk[i] + (yk[i+1] - yk[i])*(this_x - xk[i])/(xk[i+1] - xk[i]))) for i, this_x in zip(i_list, x)]
    #print(out_list)
    #out = np.array(out_list, dtype = float)
    out = np.array(out_list)
    return out


def min_En(Er_thresh, iso = '28'):
    """
    return minimum neutron energy required to produce a recoil with energy Er_thresh against silicon of isotope iso
    """
    return Er_thresh/2/pt.Si.alph[iso]

def max_Er(En, iso = '28'):
    """
    return maximum recoil energy produced by neutron of energy En against silicon of isotope iso
    """
    return 2*pt.Si.alph[iso]*En



def add_truncated_gaussian_noise(x_samples, sigma = None, sig = None, seed = None):
    """
    The same as add_gaussian_noise(), but values are constrained above zero
    """
    if sig is None:
        sig = sigma(x_samples)
    return truncnorm.rvs(-x_samples/sig, np.inf, loc = x_samples, scale = sig, size = len(x_samples), random_state = seed)


def add_truncated_gaussian_noise_constant_width(x_samples, sig, seed = None):
    """
    The same as add_gaussian_noise_constant_width(), but values are constrained above zero
    """
    return truncnorm.rvs(-x_samples/sig, np.inf, loc = x_samples, scale = sig, size = len(x_samples), random_state = seed)



def endf_float(word):
    if '-' in word[1:]:
        demarc = word.find('-', 1)
    elif '+' in word:
        demarc = word.find('+')
    else:
        return float(word)
    return float(word[:demarc] + 'e' + word[demarc:])


def weighted_histogram(X, bins, w):
    """
    return modified histogram values for array of values X and bins bins, with the "counting" of points in each bin equal to the sum of the weights w corresponding to the data points in X
    """
    s = np.argsort(X)
    X = X[s]
    w = w[s]
    n = 0
    for n, x in enumerate(X):
        if x >= bins[0]: 
            break
    out = np.zeros(len(bins) - 1)
    for i, b in enumerate(bins[1:-1]):
        while X[n] < b:
            out[i] += w[n]
            n += 1
            if n >= len(X):
                break
        if n >= len(X):
            break
    if n < len(X):
        while X[n] <= bins[-1]:
            out[-1] += w[n]
            n += 1
            if n >= len(X):
                break
    return out
        


def trap_bins(bins, x, y):
    """
    integrate (x,y) data over bins such that the height of output[i] is the trapezoidal integral of the (x,y) data between bins[i] and bins[i+1]
    """
    bin_widths = bins[1:] - bins[:-1]
    output = np.zeros(bin_widths.shape)

    for i, (bmin, bmax, db) in enumerate(zip(bins[:-1], bins[1:], bin_widths)):
        
        idx = np.where((bmin < x)*(x < bmax))[0]

        x_eval = list(x[idx])
        y_eval = list(y[idx])

        id_less = np.where((x < bmin))[0]

        if len(id_less) > 0:
            id = id_less[-1]
            x_eval = [bmin] + x_eval
            y_eval = [y[id] + (bmin - x[id])*(y[id+1] - y[id])/(x[id+1] - x[id])] + y_eval

        id_greater = np.where((bmax < x))[0]
        if len(id_greater) > 0:
            id = id_greater[0]
            x_eval = x_eval + [bmax]
            y_eval = y_eval + [y[id-1] + (bmax - x[id-1])*(y[id] - y[id-1])/(x[id] - x[id-1])]

        output[i] = trap(np.array(x_eval), np.array(y_eval))

    return output





