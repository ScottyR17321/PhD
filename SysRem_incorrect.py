#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:47:03 2021

@author: scott
"""

# SysRem algorithm from Tamuz, Mazeh, Zucker (2013)

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.ascii as ascii
from astropy.io import fits
import glob
import scipy
from scipy.interpolate import interp1d

# Get list of files
filelist=glob.glob('/Users/scott/Downloads/2020/N*.fits')
filelist.sort()

# Show headers & descriptions of all the columns
hdu = fits.open(filelist[0])
# print(repr(hdu[0].header))

# Get data that doesn't change between the files (Fluxes were interpolated to this tell)
n = hdu[0].data[1,:]
order = hdu[0].data[0,:]
order_unique = np.unique(order)
tell = hdu[0].data[5,:]

flux_var = hdu[0].data[13,:] # variance of uncorrected flux of file 0038

a = np.zeros(169) # empty array to fill with airmass for each frame

# populate the airmass array
for i,file in enumerate(filelist):
    hdu_n = fits.open(file)
    a[i] = hdu_n[0].header['AIRMASS']

# read in Blaze-Corrected flux for this order
filelist_BC = glob.glob('/Users/scott/Documents/Data/2020 (First Data Set of PhD)/Blaze-Corrected/BC N20201011G*Order*.npy')
filelist_BC.sort()

mean_flux_BC = np.load('/Users/scott/Documents/Data/2020 (First Data Set of PhD)/Blaze-Corrected/mean_flux_BC.npy')

# Empty arrays for the iterative a_n and c_n calues
c_n = np.zeros(35)
a_n = np.zeros(35)
# This loop represents one iteration of the SysRem algorithm
for i, file in enumerate(filelist_BC):
    flux = np.load(file)
    idx = (order==order_unique[i])
    flux_var_n = flux_var[idx] # get flux variance for this specific order
    tell_n = tell[idx]
    
    # calculate residuals
    res_n = flux - np.mean(flux)
    
    # best-fit c_i from eqn2 of the paper
    c_n[i] = np.sum((res_n  * a[i]) / flux_var_n)/np.sum(a[i]**2 / flux_var_n)
    
    # minimising a_j from eqn4 of the paper
    a_n[i] = np.sum((res_n * c_n[i]) / flux_var_n)/np.sum(c_n[i]**2 / flux_var_n)
    
    # # calculate gradient of best linear fit - this is c_n
    # fit = np.polyfit(tell_n, res_n, 1)
    # c_n[i] = fit[0]
    

