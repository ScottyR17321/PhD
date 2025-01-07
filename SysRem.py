#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# define function for sysrem algorithm 

def SysRem(flux, uncertainty, airmass, nIter): 
    
    # set 0-values to NaN (if the order icontains 5907 elements, the last 3 will be 0 - to make 5910)
    # this is due to the dimensions of my data cube - not all orders contain the same number of pixels
    # flux[flux==0] = 'nan'
    # uncertainty[uncertainty==0] = 'nan'
    
    idx = (~np.isfinite(flux) + (flux<=0)) + (~np.isfinite(uncertainty) + (uncertainty<=1e-5))
    flux[idx] = 1e-10
    uncertainty[idx] = 1e10
    
    # convert to magnitudes
    mag = -2.5*np.log10(flux)
    err = 2.5/np.log(10) * uncertainty / flux
    err[err<1e-6] = 1e10
    err[err>1e10] = 1e10
    
    # calculate residuals
    mag_residual = mag - np.nanmean(mag, axis=0)
    
    # locate the NaN indices in residual and err, so we can negate the effect of these values
    # set the residuals to 0 so they sum to 0
    # set the errors to a large number so their squared reciprocals sum to 0 
    
    # idx = (~np.isfinite(mag) + (~np.isfinite(err) + (err<=0))
    # mag_residual[idx] = 0
    
    # empty arrays to hold a_n and c_n values for each iteration
    a_n = np.zeros(mag.shape[1])
    c_n = np.zeros(mag.shape[0])

    # empty arrays to hold converged a_n and c_n values for each iteration
    a_n_comb = np.zeros([mag.shape[1], nIter])
    c_n_comb = np.zeros([mag.shape[0], nIter])
    
    mag_residual_comb = np.zeros([mag.shape[0], mag.shape[1], nIter+1]) # a_n x c_n x nIter
    mag_residual_comb[:,:,0] = mag_residual#
    
    c_conv=1e-3
    a_conv=1e-3
    test = mag_residual
    for i in range(0, nIter): #
        # empty arrays to hold a_n and c_n values for each iteration
        a_n = np.copy(airmass) # populate with airmass from the instrument ###
        c_n = np.zeros(mag.shape[0]) # this one doesn't need to be populated beforehand
        
        c_n_prev = 1e-6 # limits to ensure this loops at least once
        a_n_prev = 1e-6 # limits to ensure this loops at least once
        print(i) # print iteration number
        n=0
        diff_c = np.sum(np.abs(c_n-c_n_prev)) / np.sum(np.abs(c_n_prev))
        diff_a = np.sum(np.abs(a_n-a_n_prev)) / np.sum(np.abs(a_n_prev))
        
        # while loop to run code until convergence limits are reached
        while(((diff_c > c_conv) or (diff_a > a_conv)) and (n < 1e4)):
            n=n+1
            c_n_prev = c_n
            a_n_prev = a_n
            
            # calculating c and a values with equations 2 & 4 from Tamuz(2005) paper "Correcting systematic effects..."
            # print(np.sum(a_n[np.newaxis,:]**2 / err**2)) # # # # # test line
            c_n = np.sum((mag_residual * a_n[np.newaxis,:] /err**2), axis=1) / np.sum(a_n[np.newaxis,:]**2 / err**2, axis=1)
            a_n = np.sum((mag_residual * c_n[:,np.newaxis] /err**2), axis=0) / np.sum(c_n[:,np.newaxis]**2 / err**2, axis=0)
            
            # print(np.sum(np.abs(c_n - c_n_prev)))###
            # print(np.sum(np.abs(a_n - a_n_prev))) ###
            diff_c = np.sum(np.abs(c_n-c_n_prev)) / np.sum(np.abs(c_n_prev))
            diff_a = np.sum(np.abs(a_n-a_n_prev)) / np.sum(np.abs(a_n_prev))
            
            
            
        a_n_comb[:,i] = a_n # storing the a_n values at the current iteration for future reference
        c_n_comb[:,i] = c_n # storing the c_n values at the current iteration for future reference
        
        model = a_n[np.newaxis,:] * c_n[:,np.newaxis] # model for systematic we want to remove - this is just a * c
        
        mag_residual = mag_residual - model # removing the model from the residuals
        mag_residual_comb[:,:,i] = mag_residual # storing the residuals at the current iteration for future reference
        
    return(c_n_comb, a_n_comb, mag_residual_comb, test) # returning arrays of c and a values, as well as the residuals after removing each model
