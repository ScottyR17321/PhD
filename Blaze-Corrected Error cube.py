#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 18:12:59 2021

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
filelist = glob.glob('/Users/scott/Documents/Data/2020 (First Data Set of PhD)/Blaze-Corrected/Error BC N20201011G*Order*.npy')
filelist.sort()

hdu = fits.open('/Users/scott/Documents/Data/2020 (First Data Set of PhD)/N20201011G0038.m.fits')
order = hdu[0].data[0,:]
order_unique = np.unique(order)

err = np.zeros([5915, 5910]) # array told hold variance values
for i,file in enumerate(filelist):
    errors = np.load(file)
    err[i, 0:np.size(errors)] = errors

err_new = np.zeros([35, 169, 5910])
for i, order in enumerate(order_unique):
    err_new[i,:,:] = err[i::35, :]
    
# save numpy array containing the blaze-corrected error values in the format calculated above
np.save('/Users/scott/Documents/Data/2020 (First Data Set of PhD)/Blaze-Corrected/Errors_BC', err_new)