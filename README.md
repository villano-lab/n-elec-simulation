# n-elec-scattering
Neutron-electron scattering as a low-energy DM background

## Overview

## Important Files

The `python` directory contains some sets of useful functions, many of which are self-explanatory.

* `neut-e-calc.nb` checks the cross-sectiion calculations with a different method.

### Numbered Directories

### Organization to do
* Rename notebooks so that there is a clear order
* Write instructions for running notebooks
* Create READMEs in each directory that lacks one
* Describe purpose of each notebook
* Highlight python functions
* Write an overview of the repository
* Sign thy name in the contributors list.

### Considerations
* Should we move all the data to one folder, instead of being split like this?
* Do figures and paper_figures need to stay split? (If yes, maybe the important figures can go top-level?)
* Have any notebooks become 'defunct'/do any no longer contain unique information that is relevant to the paper? (If yes we should move these, either to their own directory or to another branch or repository, or just delete them)
* Are any of the python libraries used in other repostiories of ours? (If yes we should make them packages to prevent de-syncing.)
* Can any of our directories become hidden or nested (e.g. mplstyles)?
* Are any files extraneous? (What is _MACOSX directory, that seems like it may be an accidental push)
* Is there some way we can automate testing to ensure that our reorganization doesn't break anything?
* Are the notebooks adequately documented/commented/annotated?

#### Specific Suggestions
(I can implement these if they're what we want to do, just don't know what's what here so didn't want to jump to conclusions)
* RGTC appears to be an external library of some kind. Remove the RGTC directories and zip. If they are needed, use whatever is the equivalent of a pip requirements file to make sure they get included.
* Move important figures to top-level numbered directories.
* lindhard.py and other copied libraries should be uploaded to pypi.
    * Maybe I should upload my mplstyles as well? they seem to be getting reused a lot. =^.^=
    * Should we make a fork of ENDF6 (was it modified) or can we simply create cloning instructions?
* Combine useful notebooks into a single directory numbered "0" and then number the notebooks.
* Combine all data into one directory, possibly with subdirectories for organization
* Remove the 'papers' directory, instead referencing the papers in this README.

# References
1. SuperCDMS Collaboration, October 4, 2016. Projected Sensitivity of the SuperCDMS SNOLAB experiment. [arXiv:1610.00006](https://arxiv.org/abs/1610.00006), [DOI:10.1103/PhysRevD.95.082002](https://doi.org/10.1103/PhysRevD.95.082002).
2. A.J. Biffl and A.N. Villano, July 11, 2024. Analytical computation of neutron recoil spectra for rare-event backgrounds.

# Contributors

Kitty C. Mickelson

*Last updated 4 September 2024, v?*
