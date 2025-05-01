"""
ElasticCrossSections - module containing functions and other utilities for calculating cross sections of elastic neutron scatters with silicon
"""

# imports 

import numpy as np
import pickle
import PeriodicTable as pt
import lindhard as L
import pandas as pd
import ProjectUtilities as U
from ProjectUtilities import interpolate_yvals, interpolate_yval, interpolate_yvals_with_overlap, max_Er, trap, sample_from_pdf
from numpy.polynomial.legendre import legval
from numpy import where, arange, zeros, concatenate


# data file locations
#das_file_base = "../data_files/ENDF/MT2/Si_das_data_"
#sig_file_base = "../data_files/ENDF/MT2/Si_sig_data_"
#neutron_spec_file = '../data_files/neutron kinetic/placeholder_A.txt'

#uncer_file = '../data_files/ENDF/MT2/Si_sig_uncers.txt'
#inelastic_file = '../data_files/ENDF/MT3/Si_cross_sections.txt'

das_file_base = "../data_files/ENDF/MT2/Si_das_data_"
sig_file_base = "../data_files/ENDF/MT2/Si_sig_data_"
neutron_spec_file = '../data_files/neutron kinetic/placeholder_A.txt'

uncer_file = '../data_files/ENDF/MT2/Si_sig_uncers.txt'
inelastic_file = '../data_files/ENDF/MT3/Si_cross_sections.txt'

# constants
Si = pt.Si # element object
alph = Si.alph
isos = [str(A) for A in Si.A] # isotope labels

# dictionaries to hold all different types of data
E_sigs = {} # eV
sig_sigs = {} # barns
E_leg = {} # eV
A_leg = {} 
E_tab = {} # eV
P_tab = {}
cos_tab = {}


# read in data
for iso in isos:

    # sigs data
    with open(sig_file_base + iso + '.txt', 'rb') as file:
        E_sigs[iso], sig_sigs[iso] = pickle.load(file)

    # das data
    with open(das_file_base + iso + '.txt', 'rb') as file:
        E_leg[iso], A_leg[iso], E_tab[iso], P_tab[iso], cos_tab[iso] = pickle.load(file)


# sig uncertainties
with open(uncer_file, 'rb') as file:
    uncers = pickle.load(file)

with open(inelastic_file, 'rb') as file:
    E_inelastic, inelastic = pickle.load(file)


das_energy_lims = {}

for iso in isos:
    das_energy_lims[iso] = E_leg[iso][-1]
    if das_energy_lims[iso] != E_tab[iso][0]:
        raise Exception('Error in ElasticCrossSections: Discontinuity in angular distribution data. Bounds of Legendre coefficient data and tabulated data do not match.')

if all([lim == das_energy_lims['28'] for lim in das_energy_lims.values()]):
    das_lim = das_energy_lims['28']
else:
    das_lim = None
    raise Warning('Error in ElasticCrossSections: Could not assign global angular distribution energy limit. Variable set to None.')



def recoil_spectrum(Ers, Ens, flux_spectrum, iso = '28', sig_sigs = sig_sigs, E_sigs = E_sigs):
    """
    Convolute input neutron kinetic energy spectrum (Ens, flux_spectrum) with ENDF cross section data to give recoil spectrum data

    PARAMETERS:
        Ers - array of recoil energies [eV]
        Ens - array of neutron energies [eV]
        flux_spectrum - array of flux spectrum values corresponding to Ens [barn^-1 s^-1 eV^-1]
    
    RETURNS:
        (array) - recoil flux spectrum corresponding to Ers
    """
    def _f(En):
        # return f(Er, En)
        
        us = 1 - Ers/alph[iso]/En

        f = zeros(Ers.shape)

        good = (us > -1) & (us < 1) #us > 0 
        
        if En <= das_lim: # leg
            
            i = where((E_leg[iso][1:] >= En) & (E_leg[iso][:-1] < En))[0][0]
            
            a1 = A_leg[iso][i]
            a2 = A_leg[iso][i+1]

            if len(a1) > len(a2):
                a2 = concatenate((a2, zeros(len(a1) - len(a2))))
            elif len(a2) > len(a1): 
                a1 = concatenate((a1, zeros(len(a2) - len(a1))))
                        
            a = a1 + (a2 - a1)*(En - E_leg[iso][i])/(E_leg[iso][i+1] - E_leg[iso][i])

            c = a*(2*arange(len(a)) + 1)/2

            f[good] = legval(us[good], c)

            
        else: # tab
            
            i = where((E_tab[iso][1:] >= En) & (E_tab[iso][:-1] < En))[0][0]

            ut1 = cos_tab[iso][i]
            Pt1 = P_tab[iso][i]
            f_1 = interpolate_yvals(us[good], ut1, Pt1)

            ut2 = cos_tab[iso][i+1]
            Pt2 = P_tab[iso][i+1]
            f_2 = interpolate_yvals(us[good], ut2, Pt2)

            f[good] =  f_1 + (f_2 - f_1)*(En - E_tab[iso][i])/(E_tab[iso][i+1] - E_tab[iso][i])

        return f/En
        # end of _sigma_I

    in_range = (Ens > min(E_sigs[iso])) & (Ens < max(E_sigs[iso]))


    Ens = Ens[in_range]
    flux_spectrum = flux_spectrum[in_range]

    sigs = interpolate_yvals(Ens, E_sigs[iso], sig_sigs[iso])
    

    out = _f(Ens[0])*sigs[0]*flux_spectrum[0]*(Ens[1] - Ens[0]) + _f(Ens[-1])*sigs[-1]*flux_spectrum[-1]*(Ens[-1] - Ens[-2])

    for En0, En1, En2, flux1, sig1 in zip(Ens[:-2], Ens[1:-1], Ens[2:], flux_spectrum[1:-1], sigs[1:-1]):

        out += _f(En1)*sig1*flux1*(En2 - En0)

    return out/2/alph[iso]#/4/np.pi/alph[iso]




def mono_recoil_spectrum(Ers, En, total_flux, iso = '28', sig_sigs = sig_sigs):
    """
    recoil spectrum for monoenergetic neutron flux at energy En, with total flux total_flux

    PARAMETERS:
        Ers - array of recoil energies [eV]
        En - neutron energy [eV]
        total_flux - total neutron flux [barn^-1 s^-1]
    
    RETURNS:
        (array) - recoil flux spectrum corresponding to Ers
    """
    def _f(En):
        # return f(Er, En)
        
        us = 1 - Ers/alph[iso]/En

        f = zeros(Ers.shape)

        good = (us > -1) & (us < 1) #us > 0 
        
        if En <= das_lim: # leg
            
            i = where((E_leg[iso][1:] >= En) & (E_leg[iso][:-1] < En))[0][0]
            
            a1 = A_leg[iso][i]
            a2 = A_leg[iso][i+1]

            if len(a1) > len(a2):
                a2 = concatenate((a2, zeros(len(a1) - len(a2))))
            elif len(a2) > len(a1): 
                a1 = concatenate((a1, zeros(len(a2) - len(a1))))
                        
            a = a1 + (a2 - a1)*(En - E_leg[iso][i])/(E_leg[iso][i+1] - E_leg[iso][i])

            c = a*(2*arange(len(a)) + 1)/2

            f[good] = legval(us[good], c)

            
        else: # tab
            
            i = where((E_tab[iso][1:] >= En) & (E_tab[iso][:-1] < En))[0][0]

            ut1 = cos_tab[iso][i]
            Pt1 = P_tab[iso][i]
            f_1 = interpolate_yvals(us[good], ut1, Pt1)

            ut2 = cos_tab[iso][i+1]
            Pt2 = P_tab[iso][i+1]
            f_2 = interpolate_yvals(us[good], ut2, Pt2)

            f[good] =  f_1 + (f_2 - f_1)*(En - E_tab[iso][i])/(E_tab[iso][i+1] - E_tab[iso][i])

        return f/En
        # end of _sigma_I

    if (En < min(E_sigs[iso]) or (En > max(E_sigs[iso]))):
        return np.zeros(Ers.shape)


    sig = interpolate_yval(En, E_sigs[iso], sig_sigs[iso])
    

    out = _f(En)*sig*total_flux

    return out/alph[iso]#/4/np.pi/alph[iso]




def mono_emission_rate_per_particle(Ens, En, total_flux, iso = '28', sig_sigs = sig_sigs):
    """
    return neutron spectrum emitted by an atom in detector (the same as recoil_spectrum() but integrating over a line where difference in incident neutron energy and recoil energy is constant)

    Inputs parallel those of mono_recoil_spectrum(), with addition of Ens
    """
    def _f(En):
        # return f(En - En', En) <- En' = energies of outgoing neutron = Ens
        
        us = 1 - (En - Ens)/alph[iso]/En

        f = zeros(us.shape)

        good = (us > -1) & (us < 1) & (Ens < En) #us > 0 
        
        if En <= das_lim: # leg
            
            i = where((E_leg[iso][1:] >= En) & (E_leg[iso][:-1] < En))[0][0]
            
            a1 = A_leg[iso][i]
            a2 = A_leg[iso][i+1]

            if len(a1) > len(a2):
                a2 = concatenate((a2, zeros(len(a1) - len(a2))))
            elif len(a2) > len(a1): 
                a1 = concatenate((a1, zeros(len(a2) - len(a1))))
                        
            a = a1 + (a2 - a1)*(En - E_leg[iso][i])/(E_leg[iso][i+1] - E_leg[iso][i])

            c = a*(2*arange(len(a)) + 1)/2

            f[good] = legval(us[good], c)

            
        else: # tab
            
            i = where((E_tab[iso][1:] >= En) & (E_tab[iso][:-1] < En))[0][0]

            ut1 = cos_tab[iso][i]
            Pt1 = P_tab[iso][i]
            f_1 = interpolate_yvals(us[good], ut1, Pt1)

            ut2 = cos_tab[iso][i+1]
            Pt2 = P_tab[iso][i+1]
            f_2 = interpolate_yvals(us[good], ut2, Pt2)

            f[good] =  f_1 + (f_2 - f_1)*(En - E_tab[iso][i])/(E_tab[iso][i+1] - E_tab[iso][i])

        return f/En
        # end of _sigma_I

    in_range = (Ens > min(E_sigs[iso])) & (Ens < max(E_sigs[iso]))

    if (1-in_range).all():
        raise Exception('bad')

    if (En < min(E_sigs[iso])) or (En > max(E_sigs[iso])):
        return np.zeros(Ens.shape)



    sig = interpolate_yval(En, E_sigs[iso], sig_sigs[iso])
    

    out = _f(En)*sig*total_flux


    return out/alph[iso]#/4/np.pi/alph[iso]





def emission_rate_per_particle(Ens, flux_spectrum, iso = '28', sig_sigs = sig_sigs):
    """
    return neutron spectrum emitted by an atom in detector (the same as recoil_spectrum() but integrating over a line where difference in incident neutron energy and recoil energy is constant)

    Inputs parallel those of recoil_spectrum(), though note now the output has the same shape as the input flux spectrum, Ens.shape
    """
    def _f(En):
        # return f(En - En', En) <- En' = energies of outgoing neutron = Ens
        
        us = 1 - (En - Ens)/alph[iso]/En

        f = zeros(us.shape)

        good = (us > -1) & (us < 1) & (Ens < En) #us > 0 
        
        if En <= das_lim: # leg
            
            i = where((E_leg[iso][1:] >= En) & (E_leg[iso][:-1] < En))[0][0]
            
            a1 = A_leg[iso][i]
            a2 = A_leg[iso][i+1]

            if len(a1) > len(a2):
                a2 = concatenate((a2, zeros(len(a1) - len(a2))))
            elif len(a2) > len(a1): 
                a1 = concatenate((a1, zeros(len(a2) - len(a1))))
                        
            a = a1 + (a2 - a1)*(En - E_leg[iso][i])/(E_leg[iso][i+1] - E_leg[iso][i])

            c = a*(2*arange(len(a)) + 1)/2

            f[good] = legval(us[good], c)

            
        else: # tab
            
            i = where((E_tab[iso][1:] >= En) & (E_tab[iso][:-1] < En))[0][0]

            ut1 = cos_tab[iso][i]
            Pt1 = P_tab[iso][i]
            f_1 = interpolate_yvals(us[good], ut1, Pt1)

            ut2 = cos_tab[iso][i+1]
            Pt2 = P_tab[iso][i+1]
            f_2 = interpolate_yvals(us[good], ut2, Pt2)

            f[good] =  f_1 + (f_2 - f_1)*(En - E_tab[iso][i])/(E_tab[iso][i+1] - E_tab[iso][i])

        return f/En
        # end of _sigma_I

    in_range = (Ens > min(E_sigs[iso])) & (Ens < max(E_sigs[iso]))

    if (1-in_range).all():
        raise Exception('bad')


    Ens = Ens[in_range]
    flux_spectrum = flux_spectrum[in_range]

    sigs = interpolate_yvals(Ens, E_sigs[iso], sig_sigs[iso])
    

    out = _f(Ens[0])*sigs[0]*flux_spectrum[0]*(Ens[1] - Ens[0]) + _f(Ens[-1])*sigs[-1]*flux_spectrum[-1]*(Ens[-1] - Ens[-2])

    for En0, En1, En2, flux1, sig1 in zip(Ens[:-2], Ens[1:-1], Ens[2:], flux_spectrum[1:-1], sigs[1:-1]):

        out += _f(En1)*sig1*flux1*(En2 - En0)

    return out/2/alph[iso]#/4/np.pi/alph[iso]



def sample_scatters(exposure = 1., neutron_spec_file = neutron_spec_file):
    """
    return sample of average yields of scatter events in natural silicon

    exposure [kg * day]
    """
    spec = pd.read_pickle(neutron_spec_file) # neutron flux spectrum
    Ers = np.geomspace(1e-15, max_Er(max(spec['E']), '28'), 15_000) # recoil energies [eV]
    F_avg = sum([abun*recoil_spectrum(Ers, spec['E'].values, spec['spec'].values/1e24, iso) for iso, abun in zip(isos, pt.Si.abuns)]) # recoil spectrum s^-1 eV^-1
    rate = trap(Ers, F_avg) # counts per second per particle
    P_avg = F_avg/rate # PDF of recoil energies
    sample_size = int(rate*exposure*U.day*U.kg/pt.Si.mass)
    Er_sample = sample_from_pdf(Ers, P_avg, sample_size)
    yLind = L.getLindhardSi_k(0.15) # yield function 
    return Er_sample, Er_sample*yLind(Er_sample) # return recoil energies and average yields



def get_F_avg(neutron_spec_file):
    spec = pd.read_pickle(neutron_spec_file)
    Ers = np.geomspace(1e-15, max_Er(max(spec['E']), '28'), 15_000) # recoil energies [eV]
    F_avg = sum([abun*recoil_spectrum(Ers, spec['E'].values, spec['spec'].values/1e24, iso) for iso, abun in zip(isos, pt.Si.abuns)]) # recoil spectrum s^-1 eV^-1
    return Ers, F_avg


def recoil_spectrum_up(Ers, Ens, fs, iso):
    #with uncers[iso] + sig_sigs[iso] as sig_sigs[iso]:
    #    return recoil_spectrum(Ers, Ens, fs, iso)
    sig_up = {iso: (sig_sigs[iso] + uncers[iso]) for iso in sig_sigs}
    return recoil_spectrum(Ers, Ens, fs, iso, sig_up)


def recoil_spectrum_down(Ers, Ens, fs, iso):
    sig_down = {iso: (sig_sigs[iso] - uncers[iso]) for iso in sig_sigs}
    return recoil_spectrum(Ers, Ens, fs, iso, sig_down)






def recoil_spectrum_interval(Ers, Ens, flux_spectrum, iso = '28'):
    """
    Return bounds of 1-sigma confidence interval for recoil_spectrum due to uncertainties in scatter cross sections
    """
    def _f(En):
        # return f(Er, En)
        
        us = 1 - Ers/alph[iso]/En

        f = zeros(Ers.shape)

        good = (us > -1) & (us < 1) #us > 0 
        
        if En <= das_lim: # leg
            
            i = where((E_leg[iso][1:] >= En) & (E_leg[iso][:-1] < En))[0][0]
            
            a1 = A_leg[iso][i]
            a2 = A_leg[iso][i+1]

            if len(a1) > len(a2):
                a2 = concatenate((a2, zeros(len(a1) - len(a2))))
            elif len(a2) > len(a1): 
                a1 = concatenate((a1, zeros(len(a2) - len(a1))))
                        
            a = a1 + (a2 - a1)*(En - E_leg[iso][i])/(E_leg[iso][i+1] - E_leg[iso][i])

            c = a*(2*arange(len(a)) + 1)/2

            f[good] = legval(us[good], c)

            
        else: # tab
            
            i = where((E_tab[iso][1:] >= En) & (E_tab[iso][:-1] < En))[0][0]

            ut1 = cos_tab[iso][i]
            Pt1 = P_tab[iso][i]
            f_1 = interpolate_yvals(us[good], ut1, Pt1)

            ut2 = cos_tab[iso][i+1]
            Pt2 = P_tab[iso][i+1]
            f_2 = interpolate_yvals(us[good], ut2, Pt2)

            f[good] =  f_1 + (f_2 - f_1)*(En - E_tab[iso][i])/(E_tab[iso][i+1] - E_tab[iso][i])

        return f/En
        # end of _sigma_I

    in_range = (Ens > min(E_sigs[iso])) & (Ens < max(E_sigs[iso]))


    Ens = Ens[in_range]
    flux_spectrum = flux_spectrum[in_range]

    sigs_up = interpolate_yvals(Ens, E_sigs[iso], sig_sigs[iso] + uncers[iso])
    sigs_down = interpolate_yvals(Ens, E_sigs[iso], sig_sigs[iso] - uncers[iso])
    

    out_up = _f(Ens[0])*sigs_up[0]*flux_spectrum[0]*(Ens[1] - Ens[0]) + _f(Ens[-1])*sigs_up[-1]*flux_spectrum[-1]*(Ens[-1] - Ens[-2])
    out_down = _f(Ens[0])*sigs_down[0]*flux_spectrum[0]*(Ens[1] - Ens[0]) + _f(Ens[-1])*sigs_down[-1]*flux_spectrum[-1]*(Ens[-1] - Ens[-2])

    for En0, En1, En2, flux1, sig_up, sig_down in zip(Ens[:-2], Ens[1:-1], Ens[2:], flux_spectrum[1:-1], sigs_up[1:-1], sigs_down[1:-1]):

        out_up += _f(En1)*sig_up*flux1*(En2 - En0)
        out_down += _f(En1)*sig_down*flux1*(En2 - En0)

    return out_up/2/alph[iso], out_down/2/alph[iso] #/4/np.pi/alph[iso]



def get_F_avg_interval(neutron_spec_file):
    spec = pd.read_pickle(neutron_spec_file)
    Ers = np.geomspace(1e-15, max_Er(max(spec['E']), '28'), 15_000) # recoil energies [eV]
    F_high, F_low = np.array([[abun*f for f in recoil_spectrum_interval(Ers, spec['E'].values, spec['spec'].values/1e24, iso)] for iso, abun in zip(isos, pt.Si.abuns)]).sum(axis = 0)
    return Ers, F_low, F_high # s^-1 particle^-1



def get_F_avg_triple(neutron_spec_file):
    spec = pd.read_pickle(neutron_spec_file)
    Ers = np.geomspace(1e-15, max_Er(max(spec['E']), '28'), 15_000) # recoil energies [eV]
    F_high, F_low = np.array([[abun*f for f in recoil_spectrum_interval(Ers, spec['E'].values, spec['spec'].values/1e24, iso)] for iso, abun in zip(isos, pt.Si.abuns)]).sum(axis = 0)
    F_avg = sum([abun*recoil_spectrum(Ers, spec['E'].values, spec['spec'].values/1e24, iso) for iso, abun in zip(isos, pt.Si.abuns)]) # recoil spectrum s^-1 eV^-1
    return Ers, F_low, F_avg, F_high # s^-1 particle^-1





def recoil_spectrum_inelastic(Ers, Ens, flux_spectrum, iso = '28'):
    """
    Convolute input neutron kinetic energy spectrum (Ens, flux_spectrum) with ENDF cross section data to give recoil spectrum data

    PARAMETERS:
        Ers - array of recoil energies [eV]
        Ens - array of neutron energies [eV]
        flux_spectrum - array of flux spectrum values corresponding to Ens [barn^-1 s^-1 eV^-1]
    
    RETURNS:
        (array) - recoil flux spectrum corresponding to Ers
    """
    def _f(En):
        # return f(Er, En)
        
        us = 1 - Ers/alph[iso]/En

        f = zeros(Ers.shape)

        good = (us > -1) & (us < 1) #us > 0 
        
        if En <= das_lim: # leg
            
            i = where((E_leg[iso][1:] >= En) & (E_leg[iso][:-1] < En))[0][0]
            
            a1 = A_leg[iso][i]
            a2 = A_leg[iso][i+1]

            if len(a1) > len(a2):
                a2 = concatenate((a2, zeros(len(a1) - len(a2))))
            elif len(a2) > len(a1): 
                a1 = concatenate((a1, zeros(len(a2) - len(a1))))
                        
            a = a1 + (a2 - a1)*(En - E_leg[iso][i])/(E_leg[iso][i+1] - E_leg[iso][i])

            c = a*(2*arange(len(a)) + 1)/2

            f[good] = legval(us[good], c)

            
        else: # tab
            
            i = where((E_tab[iso][1:] >= En) & (E_tab[iso][:-1] < En))[0][0]

            ut1 = cos_tab[iso][i]
            Pt1 = P_tab[iso][i]
            f_1 = interpolate_yvals(us[good], ut1, Pt1)

            ut2 = cos_tab[iso][i+1]
            Pt2 = P_tab[iso][i+1]
            f_2 = interpolate_yvals(us[good], ut2, Pt2)

            f[good] =  f_1 + (f_2 - f_1)*(En - E_tab[iso][i])/(E_tab[iso][i+1] - E_tab[iso][i])

        return f/En
        # end of _sigma_I

    in_range = (Ens > min(E_sigs[iso])) & (Ens < max(E_sigs[iso]))


    Ens = Ens[in_range]
    flux_spectrum = flux_spectrum[in_range]

    sigs = interpolate_yvals(Ens, E_sigs[iso], sig_sigs[iso]) + interpolate_yvals(Ens, E_inelastic[iso], inelastic[iso])
    

    out = _f(Ens[0])*sigs[0]*flux_spectrum[0]*(Ens[1] - Ens[0]) + _f(Ens[-1])*sigs[-1]*flux_spectrum[-1]*(Ens[-1] - Ens[-2])

    for En0, En1, En2, flux1, sig1 in zip(Ens[:-2], Ens[1:-1], Ens[2:], flux_spectrum[1:-1], sigs[1:-1]):

        out += _f(En1)*sig1*flux1*(En2 - En0)

    return out/2/alph[iso]#/4/np.pi/alph[iso]






def _ang_distr(En, us):
    # return f(us) at En 
    
    f = zeros(us.shape)
    
    if En <= das_lim: # leg
        
        i = where((E_leg['28'][1:] >= En) & (E_leg['28'][:-1] < En))[0][0]
        
        a1 = A_leg['28'][i]
        a2 = A_leg['28'][i+1]

        if len(a1) > len(a2):
            a2 = concatenate((a2, zeros(len(a1) - len(a2))))
        elif len(a2) > len(a1): 
            a1 = concatenate((a1, zeros(len(a2) - len(a1))))
                    
        a = a1 + (a2 - a1)*(En - E_leg['28'][i])/(E_leg['28'][i+1] - E_leg['28'][i])

        c = a*(2*arange(len(a)) + 1)/2

        f = legval(us, c)

        
    else: # tab
        
        i = where((E_tab['28'][1:] >= En) & (E_tab['28'][:-1] < En))[0][0]

        ut1 = cos_tab['28'][i]
        Pt1 = P_tab['28'][i]
        f_1 = interpolate_yvals(us, ut1, Pt1)

        ut2 = cos_tab['28'][i+1]
        Pt2 = P_tab['28'][i+1]
        f_2 = interpolate_yvals(us, ut2, Pt2)

        f =  f_1 + (f_2 - f_1)*(En - E_tab['28'][i])/(E_tab['28'][i+1] - E_tab['28'][i])

    return f
    # end of _sigma_I



def prn(Er, En, iso = '28'):
    # return p(Er|En) for isotope iso

    f = zeros(Er.shape)

    if En > E_leg[iso][-1]:
        return f

    us = 1 - Er/alph[iso]/En 

    good = (us > -1) & (us < 1)

    if not any(good):
        return f
    
    if En <= das_lim: # leg
        
        i = where((E_leg[iso][1:] >= En) & (E_leg[iso][:-1] < En))[0][0]
        
        a1 = A_leg[iso][i]
        a2 = A_leg[iso][i+1]

        if len(a1) > len(a2):
            a2 = concatenate((a2, zeros(len(a1) - len(a2))))
        elif len(a2) > len(a1): 
            a1 = concatenate((a1, zeros(len(a2) - len(a1))))
                    
        a = a1 + (a2 - a1)*(En - E_leg[iso][i])/(E_leg[iso][i+1] - E_leg[iso][i])

        c = a*(2*arange(len(a)) + 1)/2

        f[good] = legval(us[good], c)

        
    else: # tab

        if En < E_tab[iso].min():
            print('too low')
        elif En >= E_tab[iso].max():
            print('too high')
        
        i = where((E_tab[iso][1:] >= En) & (E_tab[iso][:-1] < En))[0][0]

        ut1 = cos_tab[iso][i]
        Pt1 = P_tab[iso][i]
        f_1 = interpolate_yvals(us[good], ut1, Pt1)

        ut2 = cos_tab[iso][i+1]
        Pt2 = P_tab[iso][i+1]
        f_2 = interpolate_yvals(us[good], ut2, Pt2)

        f[good] =  f_1 + (f_2 - f_1)*(En - E_tab[iso][i])/(E_tab[iso][i+1] - E_tab[iso][i])

    return f/alph[iso]/En 
    # end of _sigma_I
    

def sigma(En, iso = 'nat'):
    # return total cross section (cm^2) at neutron energy En ()
    if iso == 'nat':
        return 1e-24*sum([abun*interpolate_yval(En, E_sigs[iso], sig_sigs[iso]) for iso, abun in zip(isos, pt.Si.abuns)])
    else:
        return 1e-24*interpolate_yval(En, E_sigs[iso], sig_sigs[iso])


def Sigma(En):
    # return macroscopic cross section (cm^-1) at neutron energy En ()
    if np.isscalar(En):
        return pt.Si.rho_n*1e-24*sum([abun*interpolate_yval(En, E_sigs[iso], sig_sigs[iso]) for iso, abun in zip(isos, pt.Si.abuns)])
    else:
        return pt.Si.rho_n*1e-24*sum([abun*interpolate_yvals_with_overlap(En, E_sigs[iso], sig_sigs[iso]) for iso, abun in zip(isos, pt.Si.abuns)])


