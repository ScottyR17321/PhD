#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:06:15 2021

@author: scott
"""

# Blaze correction code for all spectra

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.ascii as ascii
from astropy.io import fits
from astropy.stats import sigma_clip
import glob
import scipy
from scipy.interpolate import interp1d

# declare function to calculate MAD of an array
def mad(arr, axis=None, keepdims=True):
    median = np.median(arr, axis=axis, keepdims=True)
    mad = np.median(np.abs(arr-median),axis=axis, keepdims=keepdims)
    return mad

# spec_ref is the reference spectrum, with the highest S/N
spec_ref = fits.open('/Users/scott/Downloads/2020/N20201011G0038.m.fits')
# spec0 is the spectrum with the wavelength grid we want to interpolate everything to
spec0 = fits.open('/Users/scott/Downloads/2020/N20201011G0038.m.fits')
# this is the list of all spectra we will be blaze-correcting
filelist=glob.glob('/Users/scott/Downloads/2020/N*.fits')
filelist.sort()

# These values are the sae for all spectra
order = spec_ref[0].data[0,:] # order number of given data point
n = spec_ref[0].data[1,:] # number of elements in each order
eindex = spec_ref[0].data[3,:] #index of element in each order

# get spectral data for our refernce interpolation spectrum
tell0 = spec0[0].data[5,:] # telluric-corrected wavelength from file 0
uc_flux0 = spec0[0].data[12,:] # uncorrected flux from file 0
uc_err0 = np.sqrt(spec0[0].data[13,:]) # error on uncorrected flux

# get spectral data for our reference blaze correction spectrum
tell_ref = spec_ref[0].data[5,:] # telluric-corrected wavelength from highest SNR file
uc_flux_ref = spec_ref[0].data[12,:] # uncorrected flux from highest SNR file
uc_err_ref = np.sqrt(spec_ref[0].data[13,:]) # error on uncorrected flux


binsize = 100 # binsize is 100 pixels for this purpose
nbins = 4608 // binsize # number of bins is number of pixels (4608) floor-divided by the bin size
for n,file in enumerate(filelist):
    filename = file[28:-7] # This isolates the filename from the filepath,  trimming out first 28 and last 7 characters
    spec_n = fits.open(file)
    tell_n = spec_n[0].data[5,:] # telluric-corrected wavelength from file n in filelist
    uc_flux_n = spec_n[0].data[12,:] # uncorrected flux from file n in filelist
    uc_err_n = np.sqrt(spec_n[0].data[13,:])
    
    # interpolating the wavelength grid of the current spectrum
    uc_flux_n_int = interp1d(tell_n, uc_flux_n, bounds_error=False, kind='linear', fill_value=0) #interpolated uc flux
    uc_flux_n_int_new = uc_flux_n_int(tell0) # interpolate flux to tell0 wavelengths
    
    uc_err_n_int = interp1d(tell_n, uc_err_n, bounds_error=False, kind='linear', fill_value=0) #interpolated uc error
    uc_err_n_int_new = uc_err_n_int(tell0) # interpolate errors to tell0 wavelengths
    
    # interpolating the wavelength grid of the highest SNR reference spectrum to tell0
    uc_flux_ref_int = interp1d(tell_ref, uc_flux_ref, bounds_error=False, kind='linear', fill_value=0) #interpolated uc flux
    uc_flux_ref_int_new = uc_flux_ref_int(tell0)
    
    uc_err_ref_int = interp1d(tell_ref, uc_err_ref, bounds_error=False, kind='linear', fill_value=0) #interpolated uc error
    uc_err_ref_int_new = uc_err_ref_int(tell0) # interpolate errors to tell0 wavelengths
    
    # now that the two spectra are on the same wavelength grid, take the ratio of the two. 
    ratio = uc_flux_n_int_new / uc_flux_ref_int_new
    ratio_err = ratio * np.sqrt((uc_err_n_int_new/uc_flux_n_int_new)**2 + (uc_err_ref_int_new/uc_flux_ref_int_new)**2)

    for o in np.unique(order):
        idx = [order==o]
        ratio_n = ratio[idx]# get the ratio of the two spectra for a single order
        bin_width = np.size(ratio_n) // nbins # get the width (in number of elements) of each bin
        ratio_n2 = np.resize(ratio_n, [nbins, bin_width]) # resize array so that each row is one bin
    
        # Manually find median and mad of the unclipped ratio array
        # binned_median_unclipped = np.median(ratio_n2,axis=1) # find median of each row of the array
        # binned_mad = mad(ratio_n2, axis=1)
    
        # Automatically clip out values greater than 3 sigma
        ratio_n_clipped = sigma_clip(ratio_n2, sigma=3, maxiters=1, axis=1)
        median_clipped = np.median(ratio_n_clipped, axis=1)
    
        # create x-axis array for the midpoints of the bins
        bin_midpoints = np.linspace(63, 5823, 46)
    
        poly3 = np.poly1d(np.polyfit(bin_midpoints, median_clipped, 3))
        poly4 = np.poly1d(np.polyfit(bin_midpoints, median_clipped, 4))
        # plt.plot(bin_midpoints, poly3(bin_midpoints))
        # plt.plot(bin_midpoints, poly4(bin_midpoints))
    
        uc_flux_n_poly3 = uc_flux_n_int_new / poly3(tell0)
        uc_flux_n_poly4 = uc_flux_n_int_new / poly4(tell0)
        
        uc_err_n_poly3 = uc_err_n_int_new / poly4(tell0)
        
        np.save('/Users/scott/Documents/Data/2020 (First Data Set of PhD)/Blaze-Corrected/BC ' + filename + 'Order' + str(int(o)), uc_flux_n_poly3[idx]) # Save each order's data as a file
        np.save('/Users/scott/Documents/Data/2020 (First Data Set of PhD)/Blaze-Corrected/Error BC ' + filename + 'Order' + str(int(o)), uc_err_n_poly3[idx])
        
        plt.plot(tell0[idx], uc_flux_n_int_new[idx], 'r-')
        plt.plot(tell0[idx], uc_flux_n_poly3[idx], 'g-')
        plt.plot(tell0[idx], uc_flux_n_poly4[idx], 'b-')
        plt.legend(['Uncorrected interpolated flux', '3rd order blaze correction', '4th order blaze correction'])
    plt.savefig('/Users/scott/Documents/Plots/Blaze Correction 02 Nov 2021/' + str(n) + '.png')
    plt.close()
