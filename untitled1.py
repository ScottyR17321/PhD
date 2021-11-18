#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 01:30:53 2021

@author: scott
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.ascii as ascii
from astropy.io import fits
import glob
import scipy
from scipy.interpolate import interp1d

# create filelist
filelist_BC = glob.glob('/Users/scott/Documents/Data/2020 (First Data Set of PhD)/Blaze-Corrected/BC N20201011G*Order*.npy')
filelist_BC.sort()

# open first fits file in the data and get the order column
hdu = fits.open('/Users/scott/Downloads/2020/N20201011G0038.m.fits')
order = hdu[0].data[0,:]
order_unique = np.unique(order)

fluxes = np.zeros([5915, 5910]) # declare 2d array to store each saved numpy array

# loop over filelist to populate 2d array with each saved 1d array of flux values
for i,file in enumerate(filelist_BC):
    flux = np.load(file)
    fluxes[i, 0:np.size(flux)] = flux
    
# declare 3d array so we can reshape (sort of) the 2d array above, to separate by order and file
fluxes_new = np.zeros([35, 169, 5910])

# loop over each order to populate an array containing orders x frames x fluxes
# the result is an array that we can average every [n, :, :] slice to get the average flux for each wavelength value for each order
for i, order in enumerate(order_unique):
    fluxes_new[i,:,:] = fluxes[i::35, :]
    
# declare 2d array to contain mean flux values for each order and wavelength index per order
mean_flux_BC = np.zeros([35, 5910])
for i in range (0, 35):
    for j in range(0, 5910):
        mean_flux_BC[i, j] = np.mean(fluxes_new[i, :, j])
        
# # save numpy array containing mean flux values for each wavelength index of each order
# np.save('/Users/scott/Documents/Data/2020 (First Data Set of PhD)/Blaze-Corrected/mean_flux_BC' , mean_flux_BC)
    
