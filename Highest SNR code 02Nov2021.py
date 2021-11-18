#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:04:44 2021

@author: scott
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.ascii as ascii
from astropy.io import fits
import glob
import scipy
from scipy.interpolate import interp1d

# Determine highest SNR Spectra

filelist=glob.glob('/Users/scott/Downloads/2020/N*.fits')
filelist.sort()

hdu = fits.open(filelist[38])
print(repr(hdu[0].header)) # print the description of each column/header
order=hdu[0].data[0,:]
order_unique = np.unique(order)

sn = np.ndarray([35, 169])

for i,file in enumerate(filelist):
    for j, o in enumerate(order_unique):
        hdu_n = fits.open(file)
        SNR_n = hdu_n[0].header['SNR' + str(int(o))].split()
        sn[j,i] = float(SNR_n[0])
        
# Looks like order 28 usually has the highest SNR
x = np.where(sn==120.35) # [6,38] is highest SNR - File 0076 has highest S/N