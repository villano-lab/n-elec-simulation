"""
ProjectFunctions - useful high-level functions
"""
#### module imports
import numpy as np
import pandas as pd
import scipy.optimize as so
import pickle

# custom modules
import PeriodicTable as pt
import ElasticCrossSections as el
import CaptureCrossSections as cap
import lindhard as L
from ProjectUtilities import *

# explicit imports
from numpy import isscalar, sqrt, histogram, arange, geomspace


#### constants and utilities

fano_file = '../data_files/lind_data/fano_D.txt' # file containing Fano factor data

pair_dir = '../data_files/ionization/'


#### functions

def load_fano(fano_file = fano_file):
    """
    load arrays of Fano factor and corresponding yield energies
    """
    with open(fano_file, 'rb') as f:
            Ef, Ff = pickle.load(f)
    Ef = np.concatenate((np.array([0]), Ef, np.array([np.inf])))
    Ff = np.concatenate((np.array([Ff[0]]), Ff, np.array([Ff[-1]])))
    return Ef, Ff

Ef, Ff = load_fano()


def get_fano(E):
    if np.isscalar(E):
        return interpolate_yval(E, Ef, Ff)
    else:
        return interpolate_yvals(E, Ef, Ff)

def hole_counts(yield_sample, fano_file = fano_file):
    """
    return number of e/h pairs measured from sample of average yield energies yield_sample [eV]
    """
    Ef, Ff = load_fano(fano_file)
    N_avg_sample = yield_sample/E_hole
    Fanos = interpolate_yvals(yield_sample, Ef, Ff)
    sigs = sqrt(N_avg_sample*Fanos)
    N_sample_floats = add_truncated_gaussian_noise(N_avg_sample, sig = sigs)
    return np.floor(N_sample_floats).astype(int)



def sample(neutron_spec_file, exposure = 1., detector_resolution = 2.5, bias = 100, fano_file = fano_file):
    """
    return sample of capture and scatter events off natural silicon detector with exposure time exposure, given ambient neutron flux data given in neutron_spec_file (return total phonon energies)
    """
    Ers_capture, Eys_capture = cap.sample_captures(sample_size = cap.capture_sample_size(neutron_spec_file = neutron_spec_file, exposure = exposure)) # sample of average capture yields

    Ers_scatter, Eys_scatter = el.sample_scatters(neutron_spec_file = neutron_spec_file, exposure = exposure) # sample of average scatter yields

    Ns = hole_counts(np.concatenate((Eys_capture, Eys_scatter)), fano_file = fano_file) # e/h pair counts

    Eps = Ns*bias + np.concatenate((Ers_capture, Ers_scatter)) # phonon energies
    return add_truncated_gaussian_noise_constant_width(Eps, sig = detector_resolution) # add detector noise
    

def sample_scatter(neutron_spec_file, exposure = 1., detector_resolution = 2.5, bias = 100, fano_file = fano_file):
    """
    return sample of scatter events off natural silicon detector with exposure time exposure, given ambient neutron flux data given in neutron_spec_file (return total phonon energies)
    """
    Ers_scatter, Eys_scatter = el.sample_scatters(neutron_spec_file = neutron_spec_file, exposure = exposure) # sample of average scatter yields

    Ns = hole_counts(Eys_scatter, fano_file = fano_file) # e/h pair counts

    Eps = Ns*bias + Ers_scatter # phonon energies
    return add_truncated_gaussian_noise_constant_width(Eps, sig = detector_resolution) # add detector noise


def sample_capture(neutron_spec_file, exposure = 1., detector_resolution = 2.5, bias = 100, fano_file = fano_file):
    """
    return sample of capture events off natural silicon detector with exposure time exposure, given ambient neutron flux data given in neutron_spec_file (return total phonon energies)
    """
    Ers_capture, Eys_capture = cap.sample_captures(sample_size = cap.capture_sample_size(neutron_spec_file = neutron_spec_file, exposure = exposure)) # sample of average capture yields

    Ns = hole_counts(Eys_capture, fano_file = fano_file) # e/h pair counts

    Eps = Ns*bias + Ers_capture # phonon energies
    return add_truncated_gaussian_noise_constant_width(Eps, sig = detector_resolution) # add detector noise


def get_neutron_flux(neutron_spec_file):
    spec = pd.read_pickle(neutron_spec_file)
    return trap(spec['E'].values, spec['spec'].values) # [cm^-2 s^-1]


def per_particle_sec_to_per_kgday(counts):
    return counts*day*kg/pt.Si.mass


def per_kgday_to_per_particle_sec(counts):
    return counts*pt.Si.mass/day/kg


def get_spectra(neutron_spec_file, bias = 100, fano_file = fano_file, bin_width = 5, detector_resolution = 2.5):
    """
    bin_width [eV]
    detector_resolution [eV]
    bias [V]
    """

    # neutron spectrum data
    spec = pd.read_pickle(neutron_spec_file)

    # scatter recoil spectrum
    Ers = geomspace(1e-15, max_Er(max(spec['E']), '28'), 15_000) # recoil energies [eV]
    F_avg = sum([abun*el.recoil_spectrum(Ers, spec['E'].values, spec['spec'].values/1e24, iso) for iso, abun in zip(el.isos, pt.Si.abuns)]) # recoil spectrum s^-1 eV^-1

    # scatter rate
    scatter_rate = trap(Ers, F_avg) # scatters per second per particle

    # capture rate
    spec_cap = spec.loc[spec['E'] < cap.E_cutoff]
    neutron_flux = trap(spec_cap['E'].values, spec_cap['spec'].values) # total neutron flux rate [cm^-2 s^-1]
    capture_rate = neutron_flux*pt.Si.avg(pt.Si.capture_sxns)/1e24 # captures per second per particle

    # pre-generated captures
    Ers_capture, Eys_capture = cap.get_captures()
    N = len(Ers_capture)

    # phonon energy of captures
    Eps_capture_nonoise = hole_counts(Eys_capture, fano_file = fano_file)*bias + Ers_capture    
    Eps_capture = add_truncated_gaussian_noise_constant_width(Eps_capture_nonoise, sig = detector_resolution)

    # sample scatter events
    P_avg = F_avg/scatter_rate
    Ers_scatter = sample_from_pdf(Ers, P_avg, N)
    yLind = L.getLindhardSi_k(0.15)

    # phonon energy of scatters
    Eys_scatter = Ers_scatter*yLind(Ers_scatter)
    Eps_scatter_nonoise = hole_counts(Eys_scatter, fano_file = fano_file)*bias + Ers_scatter
    Eps_scatter = add_truncated_gaussian_noise_constant_width(Eps_scatter_nonoise, sig = detector_resolution)

    # calculate histogram counts
    Emax = max([Eps_scatter.max(), Eps_capture.max()])
    bins = arange(0, Emax + bin_width, bin_width)
    counts_scatter, _ = histogram(Eps_scatter, bins = bins)
    counts_capture, _ = histogram(Eps_capture, bins = bins)

    # counts per kg per day per keV
    DRU_scatter = counts_scatter*scatter_rate*kg*day/pt.Si.mass/N/(bin_width/1000)
    DRU_capture = counts_capture*capture_rate*kg*day/pt.Si.mass/N/(bin_width/1000)

    return bins, DRU_scatter, DRU_capture





def get_spectra_with_stats_uncertainties(neutron_spec_file, bias = 100, fano_file = fano_file, bin_width = 5, detector_resolution = 2.5):
    """
    bin_width [eV]
    detector_resolution [eV]
    bias [V]
    """

    # neutron spectrum data
    spec = pd.read_pickle(neutron_spec_file)

    # scatter recoil spectrum
    Ers = geomspace(1e-15, max_Er(max(spec['E']), '28'), 15_000) # recoil energies [eV]
    F_avg = sum([abun*el.recoil_spectrum(Ers, spec['E'].values, spec['spec'].values/1e24, iso) for iso, abun in zip(el.isos, pt.Si.abuns)]) # recoil spectrum s^-1 eV^-1

    # scatter rate
    scatter_rate = trap(Ers, F_avg) # scatters per second per particle

    # capture rate
    spec_cap = spec.loc[spec['E'] < cap.E_cutoff]
    neutron_flux = neutron_flux = trap(spec_cap['E'].values, spec_cap['spec'].values) # total neutron flux rate [cm^-2 s^-1]
    capture_rate = neutron_flux*pt.Si.avg(pt.Si.capture_sxns)/1e24 # captures per second per particle

    # pre-generated captures
    Ers_capture, Eys_capture = cap.get_captures()
    N = len(Ers_capture)

    # phonon energy of captures
    Eps_capture_nonoise = hole_counts(Eys_capture, fano_file = fano_file)*bias + Ers_capture    
    Eps_capture = add_truncated_gaussian_noise_constant_width(Eps_capture_nonoise, sig = detector_resolution)

    # sample scatter events
    P_avg = F_avg/scatter_rate
    Ers_scatter = sample_from_pdf(Ers, P_avg, N)
    yLind = L.getLindhardSi_k(0.15)

    # phonon energy of scatters
    Eys_scatter = Ers_scatter*yLind(Ers_scatter)
    Eps_scatter_nonoise = hole_counts(Eys_scatter, fano_file = fano_file)*bias + Ers_scatter
    Eps_scatter = add_truncated_gaussian_noise_constant_width(Eps_scatter_nonoise, sig = detector_resolution)

    # calculate histogram counts
    Emax = max([Eps_scatter.max(), Eps_capture.max()])
    bins = arange(0, Emax + bin_width, bin_width)
    counts_scatter, _ = histogram(Eps_scatter, bins = bins)
    counts_capture, _ = histogram(Eps_capture, bins = bins)

    # uncertainties
    dc_scatter = N*sqrt((counts_scatter + 1)*(N - counts_scatter + 1)/(N + 3))/(N + 2)

    dc_capture = N*sqrt((counts_capture + 1)*(N - counts_capture + 1)/(N + 3))/(N + 2)

    # counts per kg per day per keV
    DRU_scatter = counts_scatter*scatter_rate*kg*day/pt.Si.mass/N/(bin_width/1000)
    DRU_capture = counts_capture*capture_rate*kg*day/pt.Si.mass/N/(bin_width/1000)

    dDRU_scatter = dc_scatter*scatter_rate*kg*day/pt.Si.mass/N/(bin_width/1000)
    dDRU_capture = dc_capture*capture_rate*kg*day/pt.Si.mass/N/(bin_width/1000)

    return bins, DRU_scatter, DRU_capture, dDRU_scatter, dDRU_capture





def get_spectra_with_stats_and_sig_uncertainties(neutron_spec_file, bias = 100, fano_file = fano_file, bin_width = 5, detector_resolution = 2.5):
    """
    bin_width [eV]
    detector_resolution [eV]
    bias [V]
    """

    # neutron spectrum data
    spec = pd.read_pickle(neutron_spec_file)

    # average scatter recoil spectrum
    Ers = geomspace(1e-15, max_Er(max(spec['E']), '28'), 15_000) # recoil energies [eV]
    F_avg = sum([abun*el.recoil_spectrum(Ers, spec['E'].values, spec['spec'].values/1e24, iso) for iso, abun in zip(el.isos, pt.Si.abuns)]) # recoil spectrum s^-1 eV^-1

    # scatter spectrum interval
    F_high_sig, F_low_sig = np.array([[abun*f for f in el.recoil_spectrum_interval(Ers, spec['E'].values, spec['spec'].values/1e24, iso)] for iso, abun in zip(el.isos, pt.Si.abuns)]).sum(axis = 0)
    F_high_flux = sum([abun*el.recoil_spectrum(Ers, spec['E'].values, (spec['spec'].values + spec['sig'].values)/1e24, iso) for iso, abun in zip(el.isos, pt.Si.abuns)])
    F_low_flux = sum([abun*el.recoil_spectrum(Ers, spec['E'].values, (spec['spec'].values - spec['sig'].values)/1e24, iso) for iso, abun in zip(el.isos, pt.Si.abuns)])
    F_high_inelastic = sum([abun*el.recoil_spectrum_inelastic(Ers, spec['E'].values, spec['spec'].values/1e24, iso) for iso, abun in zip(el.isos, pt.Si.abuns)])

    # scatter rate
    scatter_rate = trap(Ers, F_avg) # scatters per second per particle

    # capture rate
    spec_cap = spec.loc[spec['E'] < cap.E_cutoff]
    thermal_flux = trap(spec_cap['E'].values, spec_cap['spec'].values) # thermal neutron flux rate [cm^-2 s^-1]
    capture_rate = thermal_flux*pt.Si.avg(pt.Si.capture_sxns)/1e24 # captures per second per particle


    # pre-generated captures
    Ers_capture, Eys_capture = cap.get_captures()
    N = len(Ers_capture)

    # phonon energy of captures
    Ns_capture = hole_counts(Eys_capture, fano_file = fano_file)
    Eps_capture_nonoise = Ns_capture*bias + Ers_capture    
    Eps_capture = add_truncated_gaussian_noise_constant_width(Eps_capture_nonoise, sig = detector_resolution)

    # sample scatter events
    P_avg = F_avg/scatter_rate
    Ers_scatter = sample_from_pdf(Ers, P_avg, N)
    yLind = L.getLindhardSi_k(0.15)

    # phonon energy of scatters
    Eys_scatter = Ers_scatter*yLind(Ers_scatter)
    Ns_scatter = hole_counts(Eys_scatter, fano_file = fano_file)
    Eps_scatter_nonoise = Ns_scatter*bias + Ers_scatter
    Eps_scatter = add_truncated_gaussian_noise_constant_width(Eps_scatter_nonoise, sig = detector_resolution)

    # values of distributions at event energies
    Fs_mid = interpolate_yvals(Ers_scatter, Ers, F_avg)
    Fs_low_sig = interpolate_yvals(Ers_scatter, Ers, F_low_sig)
    Fs_high_sig = interpolate_yvals(Ers_scatter, Ers, F_high_sig)
    Fs_low_flux = interpolate_yvals(Ers_scatter, Ers, F_low_flux)
    Fs_high_flux = interpolate_yvals(Ers_scatter, Ers, F_high_flux)
    Fs_high_inelastic = interpolate_yvals(Ers_scatter, Ers, F_high_inelastic)

    # "weights" for events from upper/lower bound distributions
    w_low_sig = Fs_low_sig/Fs_mid
    w_high_sig = Fs_high_sig/Fs_mid
    w_low_flux = Fs_low_flux/Fs_mid
    w_high_flux = Fs_high_flux/Fs_mid
    w_high_inelastic = Fs_high_inelastic/Fs_mid

    # calculate histogram counts
    Emax = max([Eps_scatter.max(), Eps_capture.max()])
    bins = arange(0, Emax + bin_width, bin_width)
    counts_scatter, _ = histogram(Eps_scatter, bins = bins)
    counts_capture, _ = histogram(Eps_capture, bins = bins)
    counts_scatter_low_sig = weighted_histogram(Eps_scatter, bins, w_low_sig)
    counts_scatter_high_sig = weighted_histogram(Eps_scatter, bins, w_high_sig)
    counts_scatter_low_flux = weighted_histogram(Eps_scatter, bins, w_low_flux)
    counts_scatter_high_flux = weighted_histogram(Eps_scatter, bins, w_high_flux)
    counts_scatter_high_inelastic = weighted_histogram(Eps_scatter, bins, w_high_inelastic)

    # statistical uncertainties
    dc_scatter = N*sqrt((counts_scatter + 1)*(N - counts_scatter + 1)/(N + 3))/(N + 2)
    dc_capture = N*sqrt((counts_capture + 1)*(N - counts_capture + 1)/(N + 3))/(N + 2)

    k_capture = capture_rate*kg*day/pt.Si.mass/N/(bin_width/1000)
    k_scatter = scatter_rate*kg*day/pt.Si.mass/N/(bin_width/1000)

    # counts per kg per day per keV
    DRU_scatter = counts_scatter*k_scatter
    DRU_capture = counts_capture*k_capture

    dDRU_scatter_stat = dc_scatter*k_scatter
    dDRU_capture_stat = dc_capture*k_capture

    # confidence interval for scatter spectra from different sources
    DRU_scatter_low_sig = counts_scatter_low_sig*k_scatter
    DRU_scatter_high_sig = counts_scatter_high_sig*k_scatter
    DRU_scatter_low_flux = counts_scatter_low_flux*k_scatter
    DRU_scatter_high_flux = counts_scatter_high_flux*k_scatter
    DRU_scatter_high_inelastic = counts_scatter_high_inelastic*k_scatter

    rel_dDRU_capture_sig = pt.Si.avg_capture_sxn_sig/pt.Si.avg(pt.Si.capture_sxns) # relative error in capture flux due to uncertainty in average Si radiative capture cross section

    rel_dDRU_capture_flux = trap(spec_cap['E'].values, spec_cap['sig'].values)/thermal_flux # relative error in capture rate due to uncertainty in neutron spectrum

    return bins, DRU_scatter, DRU_capture, dDRU_scatter_stat, dDRU_capture_stat, DRU_scatter_low_sig, DRU_scatter_high_sig, DRU_scatter_low_flux, DRU_scatter_high_flux, DRU_scatter_high_inelastic, rel_dDRU_capture_sig, rel_dDRU_capture_flux


def get_pair_probs(pe_filename, dir = pair_dir):
    """
    function to read in pair-creation probabilities for electron scatters from file ./dir/pe_file.txt in format generated by alig_model.cpp
    """
    return pd.read_csv(dir + pe_filename + '.txt', float_precision = 'round_trip', sep = ' ').values.T



def alig_threshholds(N):
    """
    return array of threshold energies calculated in Alig model (eq 22)
    """
    Eth = np.zeros(N)
    Eth[0] = 3*Eg/2

    for n in range(1, N):
        Eth[n] = 5*Eth[n-1] + 2*Eg - 4*np.sqrt(Eth[n-1]**2 + Eg*Eth[n-1]/2)
    return Eth



def alig_gauss(x, k, o0):
    """
    fit function for high-energy extension of pair creation probabilities
    """
    E,n = x

    return k*np.exp( -((E - n*k - (k-Eg)/2)**2)/((2*n+1)*o0**2))/o0/np.sqrt((2*n+1)*np.pi)




def fit_pn(pn, En, nmin, Nmax):
    N, K = pn.shape

    ns = np.arange(nmin, Nmax)

    E_s =  np.concatenate([En for _ in range(N)])
    n_s = np.stack([ns for _ in range(K)], axis = 1).flatten()

    x = np.array([E_s, n_s])
    p = pn.flatten()
    (k, o0), pcov = so.curve_fit(alig_gauss, x, p, p0 = (3*Eg, Eg))

    E_s = []
    n_s = []
    p = []

    hw0 = (En[1:] - En[:-1]).mean()
    if (En[1:] - En[:-1]).std() > hw0/100:
        raise Exception('should probably use evenly-spaced energy grid')

    for n in range(nmin, Nmax):
        mean = n*k + (k - Eg)/2
        std = np.sqrt(n + 1/2)*o0

        i_center = (mean - En[0])/hw0

        i5 = 3*std/hw0

        l = np.floor(i_center - i5)
        r = np.ceil(i_center + i5 + 1)

        l = 0 if l < 0 else int(l)
        r = K if r > K else int(r)

        E_s += list(En[l:r])
        n_s += [n]*(r-l)

        p += list(pn[n - nmin][l:r])

    x = np.array([E_s, n_s])

    p = np.array(p)
    
    return so.curve_fit(alig_gauss, x, p, p0 = (k, o0))



def AL_pair_counts(Ers, pe_file = 'pe', pe_dir = pair_dir):
    """
    return array of e/h pair counts based on array of recoil energies Ers [eV]

    uses combined Alig/Lindhard model, with k = 0.15

    pe_dir + pe_file + '.txt' is relpath to file where alig model output data is kept
    """

    yLind = L.getLindhardSi_k(0.15)

    rng = np.random.default_rng()

    Eys = Ers*yLind(Ers) # effective electron energies

    pe = get_pair_probs(pe_file, dir = pe_dir)

    N, K = pe.shape

    Ee = np.arange(0, K*hw0, hw0)

    pairs = np.empty(Ers.size, dtype = int)

    #(k0,o0), _ = fit_pn(pe[15:50], Ee, 15, 50)
    #
    #na = np.expand_dims(np.arange(N), axis = 1)
    #nbara = (na*pe).sum(axis = 0)/pe.sum(axis = 0)
    #n2bara = (na**2*pe).sum(axis = 0)/pe.sum(axis = 0)
    #Fa = (n2bara - nbara**2)/nbara
    #
    #zmax = 5

    for i in range(Eys.size):
        if Eys[i] <= Ee[-1]:
            ps = np.array([interpolate_yval(Eys[i], Ee, pe[n]) for n in range(N)])
            pairs[i] = rng.choice(N, p = ps/ps.sum())
        else:
            if Ers[i] <= Ef[-1]:
                Fano = interpolate_yval(Ers[i], Ef, Ff)
            else:
                Fano = Ff[-1]

            #pairs[i] = int(np.floor(rng.normal(loc = Eys[i]/E_hole, scale = np.sqrt(Fano*Eys[i]/E_hole))))
            center = Eys[i]/E_hole
            width = np.sqrt(Fano*center)

            ns = np.arange(center - 5*width, center + 5*width + 1, dtype = int)
            ps = np.exp(-((ns - center)**2)/2/width**2)

            pairs[i] = rng.choice(ns, p = ps/ps.sum())

    return pairs

flux_scales = {
    'SNOLABSi': 5.2789118728349384e-06,
    'SNOLABGe': 5.102096706341595e-06,
    'DAMIC': 0.004716102802350275,
    'EDW': 0.00017807813639789135,
    'CRESST': 0.0013464792222913468
}