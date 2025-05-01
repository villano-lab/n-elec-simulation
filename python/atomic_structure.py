import numpy as np

#matplotlib for plotting
import matplotlib as mpl
from matplotlib import pyplot as plt

def cross_section(amplitude, energy, shell_energies, electron_number):
    cross_section = 0
    total_e = np.sum(electron_number)  # Total electron number
    
    # Case where energy is below the first shell energy
    if energy < shell_energies[0]:
        cross_section = 0
    # Case where energy is above the last shell energy
    elif energy >= shell_energies[-1]:
        cross_section = 1
    else:
        # Loop through shell energies and calculate cross-section
        for i in range(len(shell_energies) - 1):
            if shell_energies[i] <= energy < shell_energies[i + 1]:
                cross_section = np.sum(electron_number[:i + 1]) / total_e
                break

    # Multiply by amplitude
    cross_section *= amplitude
    return cross_section

def mod_and_plot(Er, dRdE, shell_energies, electron_number, plot=False, xmin = 0, xmax = 20, ymin = 1e-3, ymax = 1e3):
    modified_dRdE = []
    if(plot):
      fig, ax = plt.subplots(1, 1, figsize = (9, 6))
    for i in range(len(dRdE)):
        cross_sect = cross_section(dRdE[i], Er[i]*1000, shell_energies, electron_number)
        modified_dRdE.append(cross_sect)
    if plot:
        ax.plot(Er*1000, drde, 'r--', label='Unmodified')
        ax.plot(Er*1000, modified_dRdE, label='Modified Low Energy Spectrum')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_yscale('log')
        ax.set_xlabel(r'recoil energy [keV]')
        ax.set_ylabel('rate [events kg$^{-1}$ day$^{-1}$ keV$^{-1}$]')
        ax.yaxis.grid(True,which='minor',linestyle='--')
        ax.legend(loc='lower left',prop={'size':15})
        ax.yaxis.grid(True,which='minor',linestyle='--')
        ax.grid(True)
    else:
        return modified_dRdE
