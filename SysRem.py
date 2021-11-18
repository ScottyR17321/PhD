#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:02:13 2021

@author: scott
"""

# SysRem algorithm

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.ascii as ascii
from astropy.io import fits
import glob
import scipy
from scipy.interpolate import interp1d

# open residuals and variance values from saved files (calculated in SysRem_getting_values.py)
residuals = np.load('/Users/scott/Documents/Data/2020 (First Data Set of PhD)/SysRem/residuals.npy')
var = np.load('/Users/scott/Documents/Data/2020 (First Data Set of PhD)/UC_flux_variance.npy')
var[var==0] = 'nan'

# get filelist for airmasses
filelist=glob.glob('/Users/scott/Downloads/2020/N*.fits')
filelist.sort()

hdu = fits.open(filelist[0])
order = hdu[0].data[0,:]
order_unique = np.unique(order)
n = hdu[0].data[1,:]
hdu.close()

a = np.zeros(169) # empty array to fill with airmass for each frame

# populate the airmass array
for i,file in enumerate(filelist):
    hdu_n = fits.open(file)
    a[i] = hdu_n[0].header['AIRMASS']
    
# empty arrays for a_n and c_n values
c_n = np.zeros([35,5910])
a_n = np.zeros([35,169])
a_n[0:35, :] = np.copy(a) # make a copy that we will later change in a loop

for i in range(0,35): # loop over each order - each has its own calculations
    for j in range(0,np.size(filelist)): # loop over each file
        for k in range(0, 5910): # number of elements in each order
    
            c_n = np.nansum((residuals[i,j,k] * a_n[i,j]) / var[i,j,k])/np.nansum(a_n[i,j]**2 / var[i,j,k])

            a_n[i, j] = np.nansum((residuals[i,j,k] * c_n[i, k]) / var[i,j,k])/np.nansum(c_n[i, k]**2 / var[i,j,k])
        


        