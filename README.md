This repository contains the code used to carry out my PhD research in Characterising Exoplanet Atmospheres using High-Resolution Transmission Spectroscopy. 

My pipeline takes the raw echelle spectra as output from the Spectrograph's OPERA pipeline. The desired data fields (Flux, wavelength, errors etc) are extracted. 

The data is then reduced with a correction for blaze varations, undesired spectral features are filtered out with the PCA-based SysRem algorithm, Cosmic Rays are removed
with an upper Median Absolute Deviation filter, then the processed data are cross-correlated with atmospheric models, phase-folded and plotted in various ways to reveal any
atmospheric signals present for a range of chemical species. 
