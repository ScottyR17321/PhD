#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: scott
"""

import numpy as np
from astropy.io import fits
import glob
"""
create data cube from list of spectra

dimensions by default are npix x norders x nframes
"""

# filelist0=glob.glob('/Users/scott/Downloads/2020/N*.fits')
# filelist0.sort()

# # hdu = fits.open('/Users/scott/Downloads/2020/N20201011G0038.m.fits')
# hdu = fits.open(filelist0[0])

# print(repr(hdu[0].header)) # get header for fits file

def data_cube(filelist0):
    filelist = filelist0.copy()
    filelist.sort()
    hdu = fits.open(filelist0[0])

    # get dimensions for each dimension of cube e.g. npix, norder, nframe
    order = hdu[0].data[0,:]
    nelements = hdu[0].data[1,:]
    order_unique = np.unique(order)
    
    npix = int(np.max(nelements)) # max number of pixels from all orders, as an integer
    norder = order_unique.size
    nframe = len(filelist)
    
    
    UC_fluxes = np.zeros([npix,norder,nframe],dtype="float32")
    UC_errors = np.zeros([npix,norder,nframe],dtype="float32")
    tell_cube = np.zeros([npix,norder,nframe],dtype="float32")
    tell_rvel_cube = np.zeros([npix,norder,nframe],dtype="float32")
    wave_cube = np.zeros([npix,norder,nframe],dtype="float32")
    
    
    for k,filename in enumerate(filelist): # k is frame number
        hdu_n = fits.open(filename)
        order_unique_n = np.unique(hdu_n[0].data[0,:])
        uc_flux_n = hdu_n[0].data[12,:]
        uc_var_n = hdu_n[0].data[13,:]
        tell_n = hdu[0].data[5,:]
        rvel_n = hdu[0].data[6,:]
        wave_n = hdu[0].data[4,:]
        for i,o in enumerate(order_unique_n): # i is order number
            idx = (order == o)
            UC_fluxes[0:np.sum(idx),i,k]=uc_flux_n[idx].copy()
            UC_errors[0:np.sum(idx),i,k]=np.sqrt(uc_var_n[idx].copy())
            tell_cube[0:np.sum(idx),i,k]=tell_n[idx].copy()
            wave_cube[0:np.sum(idx),i,k]=wave_n[idx].copy()
            # tell_rvel_cube[0:np.sum(idx),i,k]=tell_n[idx].copy() + rvel_n[idx].copy()
            
        hdu_n.close()
    return(UC_fluxes,UC_errors,tell_cube,wave_cube)
    
# np.save('/Users/scott/Documents/Data/2020 (First Data Set of PhD)/UC_fluxes.npy', UC_fluxes)

