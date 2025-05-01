"""
CaptureCrossSections - module containing functions and utilities for generating samples of neutron captures
"""

# imports
import numpy as np
import pickle
import PeriodicTable as pt
import pandas as pd
import ProjectUtilities as U
from ProjectUtilities import trap

# average energy of e/h pair in silicon [eV]
E_hole = pt.Si.E_ehpair

# maximum capturable neutron energy [eV]
E_cutoff = 5 

# random number generator
rng = np.random.default_rng()

# file containing pre-generated capture event yield energies
capture_file = '../data_files/capture/capture_Eic.txt'

# file containing placeholder neutron kinetic energy spectrum data
neutron_spec_file = '../data_files/neutron kinetic/placeholder_A.txt'


def capture_sample_size(exposure = 1., neutron_spec_file = neutron_spec_file):
    """
    return expected number of capture events given ambient neutron spectrum given in neutron_spec_file and an exposure time given by exposure

    exposure [kg * day]
    """
    spec = pd.read_pickle(neutron_spec_file)
    spec = spec.loc[spec['E'] < E_cutoff]
    neutron_flux = trap(spec['E'].values, spec['spec'].values) # total neutron flux rate [cm^-2 s^-1]
    capture_rate = sum([neutron_flux*pt.Si.capture_sxns[i]*pt.Si.abuns[i]/1e24 for i in range(3)]) # captures per second per particle
    return int(capture_rate*exposure*U.day*U.kg/pt.Si.mass) # number of captures



def sample_captures(sample_size, sample_file = capture_file):
    """
    return array of sampled capture events from pre-generated capture events file sample_file
    """
    with open(sample_file, 'rb') as f:
        Ers, Eic = pickle.load(f)
    indices = rng.integers(len(Ers), size = sample_size)
    return Ers[indices], Eic[indices]



def get_captures(sample_file = capture_file):
    """
    return all generated capture events (recoil energies and yields) from file sample_file
    """
    with open(sample_file, 'rb') as f:
        Ers, Eic = pickle.load(f)
    return Ers, Eic






