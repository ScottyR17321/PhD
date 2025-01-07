#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Processing Pipeline for ESPaDOnS Transmission Spectra

This script is for WASP-76b data. Can be adapted for other targets from ExoGemS

@author: scott
"""
import os 
os.chdir("/Users/scott/Documents/Python/code")
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from mad_clip import mad_clip
import glob
from scipy.interpolate import interp1d
from data_cube import data_cube
from highest_SNR import highest_SNR
from blaze_correction import blaze_correction
from SysRem import SysRem
from scipy.optimize import curve_fit
from PyAstronomy import pyasl
import random
import scipy.signal as signal
from astropy.convolution import convolve
from astropy.convolution import Gaussian1DKernel

font = {'family' : 'normal',
        'weight' : 'normal',
        'size' : 9}
matplotlib.rc('font',**font)

c = 299792.458 # c in km/s

# Observatory location
obs_long = 155.471667 # observatory longitude
obs_lat = 19.828294 # observatory latitude
obs_alt = 4204 # observatory altitude


# stellar parameters for WASP-76
r_star_s = 1.756 # stellar radius in R_sun | Ehrenreich et al. (2020)
r_sun_j = 9.731 # R_sun in units of R_jup - CONSTANT
r_star_j = r_star_s * r_sun_j # stellar radius in units of R_jup


#declare functions used only in this script

def vac2air(wave_vac): # Converts wavelengths (in nm) vacuum > air (from VALD)
    wave_vac = wave_vac * 10 # convert nm to angstroms
    
    s = 1e4/wave_vac # Wavelengths must be in Angstroms for calculations
    n_vac2air = 1 + 0.0000834254 + (0.02406147 / (130-s**2)) + (0.00015998/(38.9-s**2))
    wave_air = wave_vac / n_vac2air
    return(wave_air/10) # returning wavelengths in nm

def orbital_phase(jd,tc,period): # Calculates orbital phase for planet
    phase = (jd-tc) / period # orbital phase equation
    phi = phase - np.round(phase) # fold orbital phase to be between -0.5 & +0.5
    return(phi)

# determines which frames are in-transit & out-of-transit
# read in (system scale,phase,inclination,Rp,Rs) in solar units
def transit_frames(a_Rs,phi,i,rad_p_s,Rs): 
    z = np.sqrt(((a_Rs)*np.sin(2 * np.pi * phi))**2 + (a_Rs * np.cos(2 * np.pi * i / 360) * np.cos(2 * np.pi * phi))**2) # projected impact parameter z, using values at top of script
    
    # using projected impact parameter (z) to determine in-transit frames
    lim14 = 1 + (rad_p_s / Rs) # impact parameter limit - transit points 1 & 4 (planet touches stellar disk)
    lim23 = 1 - (rad_p_s / Rs) # impact parameter limit - transit points 2 & 3 (planet fully overlaps stellar disk)
    
    in_t14 = (z < lim14)  # locate in-transit spectra (z < lim for in-transit frames)
    out_t14 = (z >= lim14)
    in_t23 = (z < lim23)  
    out_t23 = (z >= lim23)
    return(in_t14,out_t14,in_t23,out_t23)

# subtracts envelope in disrete bins from a model spectrum, returns model wavelengths 
# and enelope-subtracted transit depths 
def remove_envelope(model_w,model_r,binsize=20000):
    bins = np.size(model_w)//binsize
    remainder=np.size(model_w) - bins * binsize # remainder that doesn't fit inside the bins
    if remainder == 0:
        
        model_r_bin = model_r.reshape(bins,binsize) # trim remainder and reshape in to bins x binsize
        model_r_binmin = model_r_bin.min(axis=1) # get min r per bin
        model_r_env_bin = model_r_bin - model_r_binmin[:,np.newaxis] # removing envelope
        model_r_env = model_r_env_bin.reshape(bins*binsize) # re-reshape to 1-dimensional model
        model_depth_env = model_r_env**2 / r_star_j**2

        return model_w, model_depth_env
    else:
        model_r_bin = model_r[:-remainder].reshape(bins,binsize) # trim remainder and reshape in to bins x binsize
        model_r_binmin = model_r_bin.min(axis=1) # get min r per bin
        model_r_env_bin = model_r_bin - model_r_binmin[:,np.newaxis] # removing envelope
        model_r_env = model_r_env_bin.reshape(bins*binsize) # re-reshape to 1-dimensional model
        model_depth_env = model_r_env**2 / r_star_j**2

        return model_w[:-remainder], model_depth_env
    
#%%
    
target_list = ['WASP-76','WASP-140','WASP-177']

for target in target_list:
    

    if target == 'WASP-76':
        # WASP-76b parameters | Deibert et al (2021) Table 1
        Rs = 1.756 # stellar radius in solar radii
        a_Rs = 4.08 # system scale a/R_star
        inclination = 89.623 # inclination in degrees
        rad_p = 1.854 # planetary radius in jupiter radii
        rad_p_s = 0.19052202 # planetary radius in solar radii
        r_ratio = rad_p_s/Rs # planet-star radius ratio
        # WASP-76b orbit/transit parameters | values are from deibert et al 2021
        t_mid = 2459135.419700356 # mid-transit time in JD (Lucy calculated this 24-4-23)
        tc = 2458080.626165 # epoch of mid-transit (BJD) from past transit
        period = 1.80988198 # orbital period of WASP-76b in days
        
        Kp = 196.52 # Planetary RV semi-amplitude km/s (Ehrenreich 2020)
        V_sys = -1.0733 # systemic velocity km/s
        
        ingress = 16
        egress = 137
        
        order_low=0
        order_high=0
        
        # good_orders = [0,1,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26] # Ca
        # good_orders = range(27) # Fe
        # good_orders = range(13) # FeH
        # good_orders = range(35)

        # WASP-76 good orders
        # good_orders = 'Mg',[4,21,22,27,28]
        # good_orders = 'K',[0,1,3,4,5,6,7,9,10,11,17,20,21,22,23,24,25,26,27]
        # good_orders = 'VO',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        # good_orders = 'TiO',[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        # good_orders = 'H2O',[4,5,6,7,8,9,10]
        # good_orders = 'Na',[3,4,6,7,14,15,18,22,23,24,25,26,27,28]
        # good_orders = 'Fe',[0,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18]
        # good_orders = 'FeH',[0,1,2,3,4,5,6,7,8,9,10,11,12]
        # good_orders = 'Ca',[0,1,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
        
        
    if target == 'WASP-140':
        # WASP-140b parameters 
        Rs = 0.870 # stellar radius in solar radii
        a_Rs = 7.98 # system scale a/R_star
        inclination = 83.3 # inclination in degrees
        rad_p = 1.44 # planetary radius in jupiter radii
        rad_p_s = 0.14798068030007194 # planetary radius in solar radii
        r_ratio = rad_p_s/Rs # planet-star radius ratio
        r_orbit = 0.0323 # orbital radius in AU
        # WASP-140b orbit/transit parameters
        tc = 2456912.35105 # epoch of mid-transit (BJD) from past transit
        period = 2.2359835 # orbital period of WASP-140b in days
        ingress = 11
        egress = 48
        # WASP-140 has no literature Kp - calculate manually
        G = 6.6743*1e-11 # big G in SI units
        Mstar_SI = 0.90 * 1.989e+30 # stellar mass in kg = M(solar) * Msun(Kg)
        Mp_SI = 2.44 * 1.899e+27 # planet mass in kg = M(jup) * Mjup(Kg)
        R_orbit_SI = r_orbit * 1.496e+11 # orbital radius in m
        
        Kp = np.sin(np.deg2rad(inclination)) * np.sqrt(G * Mstar_SI / R_orbit_SI) / 1000 # divide by 1000 to get in km/s
        V_sys = 2.125 # systemic velocity km/s (+ve = redshift)
        
        order_low=0
        order_high=0
        
        # WASP-140 good orders
        # good_orders = 'Mg',[1,2,3,4,5,6]
        # good_orders = 'K',[7,10,17,20,26]
        # good_orders = 'VO',[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
        # good_orders = 'TiO',[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
        # good_orders = 'H2O',[0,1,3,4,5,6,7,8,9,10,11]
        # good_orders = 'Na',[6,16,17]
        # good_orders = 'Fe',[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20]
        # good_orders = 'FeH',[0,1,2,3,4,5,6,7,8,9,10,11,12]
        # good_orders = 'Ca',[4,5,9,10,11,12,13,14,15,16,17,18,19]
        
    if target == 'WASP-177':
        # WASP-177 system parameters 
        Rs = 0.885 # stellar radius in solar radii
        r_sun_j = 9.731 # R_sun in units of R_jup - CONSTANT
        r_star_j = Rs * r_sun_j # stellar radius in units of R_jup
        r_orbit = 0.03957 # orbital radius in AU
        a_Rs = 9.61 # system scale a/R_star
        inclination = 84.14 # inclination in degrees
        rad_p = 1.58 # planetary radius in jupiter radii
        rad_p_s = 0.16236769088480116 # planetary radius in solar radii
        r_ratio = rad_p_s/Rs # planet-star radius ratio
        # WASP-177b orbit/transit parameters 
        tc = 2457994.37140 #+tenmins # epoch of mid-transit (BJD) from past transit
        period = 3.071722 # orbital period of WASP-76b in days
        ingress = 18
        egress = 74
        # WASP-177 has no literature Kp - calculate manually
        G = 6.6743*1e-11 # big G in SI units
        Mstar_SI = 0.876 * 1.989e+30 # stellar mass in kg
        Mp_SI = 0.508 * 1.899e+27 # planet mass in kg
        R_orbit_SI = r_orbit * 1.496e+11 # orbital radius in m
        
        Kp = np.sin(np.deg2rad(inclination)) * np.sqrt(G * Mstar_SI / R_orbit_SI) / 1000 # divide by 1000 to get in km/s
        V_sys = -7.1434 # systemic velocity km/s
        
        order_low=0
        order_high=0
        
        # good_orders = [4,5,9,10,12,13,14,15,18,19,21] # Ca
        # good_orders = [6,16,23,24,26,27] # Na
        
        # WASP-177 good orders
        # good_orders = 'Mg',[27,28]
        # good_orders = 'K',[7,10,11,16,17,20,21,22,23,24,25,26,27]
        # good_orders = 'VO',[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        # good_orders = 'TiO',[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,27,28,29,30,31]
        # good_orders = 'H2O',[0,1,4,5,6,7,8,9,10,11]
        # good_orders = 'Na',[6,16,23,24,26,27]
        # good_orders = 'Fe',[0,2,3,4,5,6,7,9,12,13,14,15,16,17]
        # good_orders = 'FeH',[0,1,2,3,4,5,6,7,9,10,11,12]
        # good_orders = 'Ca',[4,5,9,10,12,13,14,15,18,19,21]
        
    
    
    # list = ['H2O_1,0%', 'H2O_0,1%', 'H2O_0,01%', 'H2O_0,001%', 'H2O_1e-04%','H2O_1e-05%', 'H2O_1e-06%', 'H2O_1e-07%']
    # list = ['H2O_1e-05%', 'H2O_1e-06%', 'H2O_1e-07%']
    
    # list2 = ['TiO_1,0%', 'TiO_0,1%', 'TiO_0,01%', 'TiO_0,001%', 'TiO_1e-04%']
    # list2 = ['TiO_1e-05%', 'TiO_1e-06%', 'TiO_1e-07%']
    # list3 = ['VO_1,0%', 'VO_0,1%', 'VO_0,01%', 'VO_0,001%', 'VO_1e-04%']
    # list3 = ['VO_1e-05%', 'VO_1e-06%', 'VO_1e-07%']
    # list4 = ['Mg_1,0%', 'Mg_0,1%', 'Mg_0,01%', 'Mg_0,001%', 'Mg_1e-04%', 'Mg_1e-05%', 'Mg_1e-06%', 'Mg_1e-07%']
    # list5 = ['Al_1,0%', 'Al_0,1%', 'Al_0,01%', 'Al_0,001%', 'Al_1e-04%', 'Al_1e-05%', 'Al_1e-06%', 'Al_1e-07%']
    # mantis_list = ['CoI_2000','CoI_2500']
    
    # CCF_list0 = ['Mg_0,001%']
    # CCF_list = ['Mg_0,1%','K_0,1%','VO_0,1%','TiO_0,1%','H2O_0,1%','Na_0,1%','Fe_0,1%','FeH_0,1%','Ca_0,1%']
    # CCF_list2 = ['Mg_0,1%','K_0,1%','VO_0,1%','TiO_0,1%','H2O_0,1%','Na_0,1%','Fe_0,1%','FeH_0,1%','Ca_0,1%']
    model_list = ['Mg_0,001%','K_0,001%','VO_0,001%','TiO_0,001%','H2O_0,001%','Na_0,001%','Fe_0,001%','FeH_0,001%','Ca_0,001%']
    model_type = 'pRT' # 'pRT' or 'mantis'

    
    for m,item in enumerate(model_list):
    
        #%% get list of .fits files and sort
        
        # These are the WASP-76 .fits files from the telescope
        filelist=glob.glob('/Users/scott/Documents/Data/'+target+'/fits/N*.fits')
        filelist.sort()
        
        hdu = fits.open(filelist[0])
        # print(hdu[0].header)
        print(repr(hdu[0].header))
        
        # hrv = np.load('/Users/scott/Documents/Data/'+target+'/Helio_RV.npy')
        # get RA and Dec from header
        ra = float(hdu[0].header['RA'])
        dec = float(hdu[0].header['DEC'])
        
        #%% Data cubes from filelist
        ## turn filelist in to data cube for flux, error, wavelength
        
        # The data-cube script has indices for the various columns of data in each .fits file
        flux,error,wavelength,waveUC = data_cube(filelist) # dimensions: npix,norder,nframe
        
        trim = 10   # number of values to trim from the end of each order. longest-5910, shortest-5903
                    # A value of 7 or higher makes all orders the same length
                    # these are telluric-corrected wavelengths
                    
        npix=np.size(flux,axis=0)-trim
        norder=np.size(flux,axis=1)
        nframe=np.size(flux,axis=2)
        
        wavelength = wavelength[:-trim,:,:]
        flux=flux[:-trim,:,:]
        error=error[:-trim,:,:]
        
        #%% Phase, dates and velocity corrections
        
        a = np.zeros(nframe) # airmass for each frame
        jd = np.zeros(nframe) # julian date (UTC)
        baryvel = np.zeros(nframe) # projected barycentric velocity
        helvel = np.zeros(nframe) # projected heliocentric velocity
        barycor = np.zeros(nframe) # barycentric correction
        hjd = np.zeros(nframe) # heliocentric julian date
        p_beg = np.zeros(nframe)
        p_end = np.zeros(nframe)
        
        tellrv = np.zeros(nframe) # heliocentric julian date
        
        for i,file in enumerate(filelist):
            hdu = fits.open(file)
            
            tellrv[i] = float(hdu[0].header['TELLRV'])
            p_beg[i] = float(hdu[0].header['EPRSPBEG'])
            p_end[i] = float(hdu[0].header['EPRSPEND'])
            
            exptime = float(hdu[0].header['EXPTIME']) # exposure time for frame
            a[i] = hdu[0].header['AIRMASS'] # mean airmass for observation
            jd[i] = float(hdu[0].header['MJDATE']) + 2400000.5 + (0.5*exptime)/86400 # modified julian date - converted to julian date in middle of exposure
            helvel[i],baryvel[i] = pyasl.baryCorr(jd[i], ra, dec, deq=0.0) # barycentric correction
            barycor[i],hjd[i] = pyasl.helcorr(obs_long,obs_lat,obs_alt,ra,dec,jd[i]) # heliocentric correction
            hdu.close()
        
        phi = orbital_phase(jd,tc,period) # get phase, jd read-in from files along with airmass earlier
        in_t,out_t,in_t23,out_t23 = transit_frames(a_Rs,phi,inclination,rad_p_s,Rs) #the ones labelled '23' are between transit contact points 2 and 3
        
    
        Vp = Kp * np.sin(2 * np.pi * phi) + V_sys # calculate radial velocities of planet in barycentric frame
        
        trail = Vp-barycor
        
        #%% Model-injection 
        test_type ='injection-recovery' #label to add to saved filenames
        # model_type = 'mantis' # 'pRT' or 'mantis' ---------------------------------------
        
        # model_name_inj = 'H2O_0,1%' # model being injected
        model_name_inj = item # item = current model from VMR list at top of this script
        
        vshift_inj = Vp - barycor
        z = 1 + (vshift_inj / c)
        flux_inj = np.zeros(flux.shape)
    
        ### Convolving models with gaussian kernel FWHM ~5km/s
        res = np.zeros(35)
        for o in range(35):
            res[o] = float(hdu[0].header['SPCRES'+str(o+22)][:5])
        res_mean = np.mean(res)
    
        
        # model_r_conv = convolve(model_r,kernel2, normalize_kernel=True, boundary='extend') # convolve model with kernel
        
        if model_type == 'pRT':
            model_r = np.load('/Users/scott/Documents/Data/'+target+'/pRT_models/'+model_name_inj+'/radii.npy')
            model_w = np.load('/Users/scott/Documents/Data/'+target+'/pRT_models/'+model_name_inj+'/wavelengths.npy')
            model_w = vac2air(model_w * 1e3) # convert to nm and perform vacuum correction
            fwhm = 299792.458 / res_mean # c (km/s) / mean resolution
            sigma = fwhm/2.355 # convery fwhm to stddev
            # scale=np.mean( (model_w[1:]-model_w[:-1])/(model_w[1:]+model_w[:-1])*2 * 3e5)
            kernel = Gaussian1DKernel(stddev=sigma) # set up 1d gaussian kernel
            model_r_conv = convolve(model_r, kernel, normalize_kernel=True, boundary='extend') # convolve model with kernel
            model_w_env,model_depth_env = remove_envelope(model_w,model_r,binsize=25000)
            
            
        if model_type == 'mantis':
            model_depth = np.load('/Users/scott/Documents/Data/mantis_models/'+model_name_inj+'/depths.npy')
            model_w = np.load('/Users/scott/Documents/Data/mantis_models/'+model_name_inj+'/wavelengths.npy')
            model_w = vac2air(model_w) # convert to nm and perform vacuum correction
            fwhm = 299792.458 / res_mean # c (km/s) / mean resolution
            sigma = fwhm/2.355 # convery fwhm to stddev
            # scale=np.mean((model_w[1:]-model_w[:-1])/(model_w[1:]+model_w[:-1])*2 * 3e5)
            kernel = Gaussian1DKernel(stddev=sigma) # set up 1d gaussian kernel
            model_depth_conv = convolve(model_depth, kernel, normalize_kernel=True, boundary='extend') # convolve model with kernel
            model_w_env,model_depth_env = remove_envelope(model_w,model_depth_conv,binsize=25000)
            model_depth_env = np.sqrt(model_depth_env * r_star_j**2) # mantis models are already given in transit depths, this undoes the conversion to transit depth done in the remove_envelope function
            
        
        # model_w_env,model_depth_env = remove_envelope(model_w,model_r_conv,binsize=25000)
        model_depth_interp = interp1d(model_w_env, model_depth_env, bounds_error=False, kind='linear', fill_value=0)
        
        
        for f in range(nframe):
            model_interp = model_depth_interp(wavelength[:,:,f]/z[f])
        
            flux_inj[:,:,f] = flux[:,:,f] * (1 - model_interp)
        
        #%% find highest SNR spectrum before blaze correction
        high_SNR_idx,sn = highest_SNR(filelist)
        sn_mean_order = np.mean(sn,axis=1)
        sn_mean_frame = np.mean(sn,axis=0)
        
        #%% Blaze-variation correction
        
        if test_type == 'raw data':
            fluxBC,errorBC = blaze_correction(flux,error,wavelength,high_SNR=high_SNR_idx) # -----
        elif test_type == 'injecton-recovery':
            fluxBC,errorBC = blaze_correction(flux_inj,error,wavelength,high_SNR=high_SNR_idx) # -----
        #%% SysRem systematics removal
        nIter = 4 #----------------------------------------------------------------
        fluxSR = np.zeros([npix,norder,nframe,nIter+1]) # array to store the fluxes after sysrem
        
        for i in range(0,norder):
            c_n_comb, a_n_comb, mag_residual_comb, test = SysRem(fluxBC[:,i,:], errorBC[:,i,:], a, nIter)
            fluxSR[:,i,:,:] = 10**(-0.4*mag_residual_comb[:,:,:]) # convert final sysrem values back to fluxes, store in array
            print(i)
            
        
        #%% remove cosmic rays from the data
        fluxCR = np.zeros(fluxSR.shape)
        for k in range(fluxSR.shape[0]):
            print(k)
            for i in range(fluxSR.shape[1]):
                for j in range(fluxSR.shape[3]):
                    fluxCR[k,i,:,j] = mad_clip(fluxSR[k,i,:,j], cliplow=False) #cliplow=false clips only positive spikes (like cosmic rays)
            
        np.save('/Users/scott/Documents/Data/'+target+'/fluxCR_'+str(nIter)+'.npy',fluxCR)
        #%%
        
        # # Set up range of velocities to shift data by in CCF
        vsys_lim = 500 # magnitude of velocity shifts in km/s
        vsys_n = 2*vsys_lim+1 # number of shifts for 1 km/s resolution
        vshift = np.linspace(-vsys_lim,vsys_lim,vsys_n) # velocity shifts in km/s for CCF
        z = 1 + (vshift / c)  # (1 + z) factor for shifting wavelengths, Wrest = Wobserved / (1+z)
        
        
        #%% Monstrosity to select good orders. There has to be a better way of doing this
        
        if target == 'WASP-76':
            if m==0:
                good_orders = [4,21,22,27,28]
            elif m==1:
                good_orders = [0,1,3,4,5,6,7,9,10,11,17,20,21,22,23,24,25,26,27]
            elif m==2:
                good_orders = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
            elif m==3:
                good_orders = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
            elif m==4:
                good_orders = [4,5,6,7,8,9,10]
            elif m==5:
                good_orders = [3,4,6,7,14,15,18,22,23,24,25,26,27,28]
            elif m==6:
                good_orders = [0,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18]
            elif m==7:
                good_orders = [0,1,2,3,4,5,6,7,8,9,10,11,12]
            elif m==8:
                good_orders = [0,1,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]              
        
        elif target == 'WASP-140':
            if m==0:
                good_orders = [1,2,3,4,5,6]
            elif m==1:
                good_orders = [7,10,17,20,26]
            elif m==2:
                good_orders = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
            elif m==3:
                good_orders = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
            elif m==4:
                good_orders = [0,1,3,4,5,6,7,8,9,10,11]
            elif m==5:
                good_orders = [6,16,17]
            elif m==6:
                good_orders = [0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20]
            elif m==7:
                good_orders = [0,1,2,3,4,5,6,7,8,9,10,11,12]
            elif m==8:
                good_orders = [4,5,9,10,11,12,13,14,15,16,17,18,19]
                
        elif target == 'WASP-177':
            if m==0:
                good_orders = [27,28]
            elif m==1:
                good_orders = [7,10,11,16,17,20,21,22,23,24,25,26,27]
            elif m==2:
                good_orders = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
            elif m==3:
                good_orders = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,27,28,29,30,31]
            elif m==4:
                good_orders = [0,1,4,5,6,7,8,9,10,11]
            elif m==5:
                good_orders = [6,16,23,24,26,27]
            elif m==6:
                good_orders = [0,2,3,4,5,6,7,9,12,13,14,15,16,17]
            elif m==7:
                good_orders = [0,1,2,3,4,5,6,7,9,10,11,12]
            elif m==8:
                good_orders = [4,5,9,10,12,13,14,15,18,19,21]



        
        #%% Cross-Correlation trails for multiple orders
        
        # species,model_name = 'H2O','H2O_0,1%'
        species,model_name = item,item #-------------------------------------------
        newpath = '/Users/scott/Documents/Data/'+target+'/Plots/'+species+' plots/CCF Trails'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            
        # model_r = np.load('/Users/scott/Documents/Data/WASP-76/pRT_models/'+model_name+'/radii.npy')
        # model_w = np.load('/Users/scott/Documents/Data/WASP-76/pRT_models/'+model_name+'/wavelengths.npy')
        # model_w = vac2air(model_w * 1e3) # convert to nm and perform vacuum correction
        # model_w_env,model_depth_env = remove_envelope(model_w,model_r,binsize=25000)
        
        if model_type == 'pRT':
            model_r = np.load('/Users/scott/Documents/Data/'+target+'/pRT_models/'+model_name_inj+'/radii.npy')
            model_w = np.load('/Users/scott/Documents/Data/'+target+'/pRT_models/'+model_name_inj+'/wavelengths.npy')
            model_w = vac2air(model_w * 1e3) # convert to nm and perform vacuum correction
            model_w_env,model_depth_env = remove_envelope(model_w,model_r,binsize=25000)
            
        if model_type == 'mantis':
            model_r = np.load('/Users/scott/Documents/Data/mantis_models/'+model_name_inj+'/depths.npy')
            model_w = np.load('/Users/scott/Documents/Data/mantis_models/'+model_name_inj+'/wavelengths.npy')
            model_w = vac2air(model_w) # convert to nm and perform vacuum correction
            model_w_env,model_depth_env = remove_envelope(model_w,model_r,binsize=25000)
            model_depth_env = np.sqrt(model_depth_env * r_star_j**2) # mantis models are already given in transit depths, this undoes the conversion to transit depth done in the remove_envelope function
            
            
    
        
        # lower and upper limits for orders to include

        
        ccf_sum = np.zeros([nframe,vsys_n]) # array to hold CCF output
        ccf_order = np.zeros([nframe,vsys_n]) 
        ccf_order_medsub = np.zeros([nframe,vsys_n]) 
        vshift = np.linspace(-vsys_lim,vsys_lim,vsys_n)
        nIter_used = 3 #-----------------------------------------------------------
        
        for order in good_orders:
        # for order in range(order_low,order_high+1):
        
            print(str(order)) # print current order to show progress
            idx = (model_w_env > wavelength[0,order,0]) & (model_w_env < wavelength[-trim,order,0])
        
            model_depth_order = model_depth_env[idx]
            model_w_order = model_w_env[idx]
            model_depth_interp = interp1d(model_w_order, model_depth_order, bounds_error=False, kind='linear', fill_value=0)
            ccf_data = fluxCR[:,order,:,nIter_used] # dimension 5910x169
            
            for i in np.arange(ccf_sum.shape[1]): # loop over 1001 velocity shift values
            
                # shift model by the velocity change for all v_shifts, then interpolate back to the rest frame
                model_depth_interp_shift = model_depth_interp(wavelength[:,order,0]/z[i])
                model_depth_interp_shift = model_depth_interp_shift[:,np.newaxis]
                ccf_sum[:,i] += np.sum((ccf_data-np.mean(ccf_data)) * (model_depth_interp_shift - np.mean(model_depth_interp_shift)),axis=0)
                
                ccf_order[:,i] = np.sum((ccf_data-np.mean(ccf_data)) * (model_depth_interp_shift - np.mean(model_depth_interp_shift)),axis=0)
            ccf_order_medsub = ccf_order - np.median(ccf_order,axis=1)[:,np.newaxis]
            plt.figure()
            plt.pcolormesh(vshift,phi,ccf_order,shading='auto')
            plt.axhline(y=phi[ingress], color='r', linestyle='--') # horizontal lines to mark in-transit phases
            plt.axhline(y=phi[egress], color='r', linestyle='--')
            plt.xlabel('RV (km/s)')
            plt.ylabel('Orbital Phase')
            plt.title('Combined CCF |'+model_name+' [Order '+str(order)+']')
            plt.xlim([-200,200])
            plt.colorbar()
            plt.savefig('/Users/scott/Documents/Data/'+target+'/Plots/'+species+' plots/CCF Trails/'+test_type+' '+model_name+' '+str(22+order)+' '+str(nIter_used)+'iter.png')
                
            plt.close('all')

            plt.figure()
            plt.pcolormesh(vshift,phi,ccf_order_medsub,shading='auto')
            plt.axhline(y=phi[ingress], color='r', linestyle='--') # horizontal lines to mark in-transit phases
            plt.axhline(y=phi[egress], color='r', linestyle='--')
            plt.xlabel('RV (km/s)')
            plt.ylabel('Orbital Phase')
            plt.title('Combined CCF |'+model_name+' [Order '+str(order)+']')
            plt.xlim([-200,200])
            plt.colorbar()
            plt.savefig('/Users/scott/Documents/Data/'+target+'/Plots/'+species+' plots/CCF Trails/median_subtracted '+test_type+' '+model_name+' '+str(22+order)+' '+str(nIter_used)+'iter.png')
            
            plt.close('all')

        plt.figure()
        plt.pcolormesh(vshift,phi,ccf_sum, shading='auto')
        # plt.plot(trail,phi,color='w',linestyle='--')
        plt.axhline(y=phi[ingress], color='r', linestyle='--') # horizontal lines to mark in-transit phases
        plt.axhline(y=phi[egress], color='r', linestyle='--')
        plt.xlabel('RV (km/s)')
        plt.ylabel('Orbital Phase')
        plt.title('Combined CCF |'+model_name+' [Orders '+str(22+order_low)+'-'+str(22+order_high)+']')
        plt.xlim([-200,200])
        plt.colorbar()
        plt.savefig('/Users/scott/Documents/Data/'+target+'/Plots/'+species+' plots/CCF Trails/'+test_type+' '+model_name+' fixed?'+str(22+order_low)+'-'+str(22+order_high)+' '+str(nIter_used)+'iter.png')
        
        
        ccf_sum_medsub = ccf_sum - np.median(ccf_sum,axis=0)
        plt.figure()
        plt.pcolormesh(vshift,phi,ccf_sum_medsub, shading='auto')
        # plt.plot(trail,phi,color='w',linestyle='--')
        plt.axhline(y=phi[ingress], color='r', linestyle='--') # horizontal lines to mark in-transit phases
        plt.axhline(y=phi[egress], color='r', linestyle='--')
        plt.xlabel('RV (km/s)')
        plt.ylabel('Orbital Phase')
        plt.title('Combined CCF |'+model_name+' [Orders '+str(22+order_low)+'-'+str(22+order_high)+']')
        plt.xlim([-200,200])
        plt.colorbar()
        plt.savefig('/Users/scott/Documents/Data/'+target+'/Plots/'+species+' plots/CCF Trails/median_subtracted '+test_type+' '+model_name+' fixed?'+str(22+order_low)+'-'+str(22+order_high)+' '+str(nIter_used)+'iter.png')
        #%% Phase-folding CCFs (see Deibert et al (2021) - Figure 6)
        
        # separate in-transit phase and ccf
        phi_it = phi[ingress:egress]
        ccf_it = ccf_sum[ingress:egress]
        
        ccf_oot = ccf_sum[np.r_[0:ingress,egress:nframe],:] # the np.r term allows indexing two regions of this array
        phi_oot = phi[np.r_[0:ingress,egress:nframe]]
        
        
        # Kp and Vsys ranges
        K_p = np.linspace(0,300,301) # rv semi-amplitude
        vsys_range = np.linspace(-70,70,141) # systemic velocity
        
        fold_map = np.zeros([141,301]) # Vsys x Kp array
        fold_map_oot = np.zeros([141,301]) # Vsys x Kp array
    
        # ccf_sum = np.zeros([301,141])
        
        
        for pidx,p in enumerate(phi): #loop over in-transit phases 121
        
            ccf_int_obj = interp1d(vshift+barycor[pidx], ccf_sum_medsub[pidx,:], axis=0, bounds_error=False, kind='linear', fill_value=0)
            
            for kidx,k in enumerate(K_p): # loop over range of k_p values 301
                
                vp = k * np.sin(2 * np.pi * p) 
                
                ccf_int = ccf_int_obj(vp + V_sys + vsys_range)
            
                fold_map[:,kidx] += ccf_int
        
            
        plt.figure()
        plt.pcolormesh(vsys_range,K_p,np.transpose(fold_map), shading='auto')
        plt.axhline(Kp,color='r',linestyle='--')
        plt.axvline(0,color='r',linestyle='--')
        plt.xlabel('V_sys (km/s)')
        plt.ylabel('Kp')
        plt.title('Folded CCF in-transit | '+model_name+' [Orders '+str(22+order_low)+'-'+str(22+order_high)+']'+str(nIter_used)+' iter')
        plt.xlim([-70,70])
        plt.colorbar()
        plt.savefig('/Users/scott/Documents/Data/'+target+'/Plots/'+species+' plots/median subtracted '+test_type+' '+model_name+' '+str(22+order_low)+'-'+str(22+order_high)+' '+str(nIter_used)+'iter')
        
        #%% Planet rest-frame plots 
        
        newpath1 = '/Users/scott/Documents/Data/'+target+'/Plots/'+species+' plots/Planet RF plots'
        if not os.path.exists(newpath1):
            os.makedirs(newpath1)
        
        # Kp and Vsys ranges
        K_p = np.linspace(0,300,301) # rv semi-amplitude
        vsys_range = np.linspace(-70,70,141) # systemic velocity
        
        rest_map = np.zeros([nframe,1001]) # Vp x vshift array
        rest_map_oot = np.zeros([48,1001]) # Vp x vshift array
        
        # As a test for removing stellar contamination, subtracting mean OOT CCF, scaled by transit depth
        ccf_data_test = ccf_sum - (1-r_ratio**2)*ccf_oot.mean(axis=0) 
        
        # ccf_res_rest = np.zeros([169,1001])
        # ccf_sum = np.zeros([301,141])
        
        for pidx,p in enumerate(phi): #loop over 169 phases
        
            ccf_int_obj = interp1d(vshift+barycor[pidx], ccf_sum_medsub[pidx,:], axis=0, bounds_error=False, kind='linear', fill_value=0)
            
            for vidx,v in enumerate(vshift): # loop over range of vshift values
                
                ccf_rest = ccf_int_obj(vshift + Vp[pidx])
                
                rest_map[pidx,:] = ccf_rest
                
        # # parameters for binning data
        # binsize=2 # phases per bin
        # nbins=np.size(filelist)// binsize # number of bins that fit within observations
        # remainder=np.size(filelist) - nbins*binsize # remainder that doesn't fit inside the bins
        
        # # trimming and reshaping data for binning
        # rest_map_bin = rest_map[:-remainder,:].reshape(nbins,binsize, 1001, 1).sum(3).sum(1) 
        # #rest map=(m,n) >>> reshape(m_bins, m//m_bins ,n_bins, n//n_bins), sum along m//m_bins axes
        # # trimming and reshaping phase data for binning
        # phi_bin = phi[:-remainder].reshape(nbins,binsize).mean(axis=1) 
        
        plt.figure()
        plt.pcolormesh(vshift,phi,rest_map, shading='auto')
        plt.xlabel('RV (km/s)')
        plt.ylabel('Phase')
        plt.title('Planet Rest-Frame CCF '+model_name+' '+str(22+order_low)+'-'+str(22+order_high)+'] '+str(nIter_used)+' iter')
        # plt.title('Planet Rest-Frame CCF Ca_1 '+str(binsize)+' phases per bin ['+str(22+order_low)+'-'+str(22+order_high)+'] '+str(nIter)+' iter')
        plt.xlim([-50,50])
        plt.axhline(y=phi[ingress], color='r', linestyle='--') # horizontal lines to mark in-transit phases
        plt.axhline(y=phi[egress], color='r', linestyle='--')
        plt.axvline(x=vshift[500],color='w',linestyle='--')
        plt.colorbar()
        plt.savefig('/Users/scott/Documents/Data/'+target+'/Plots/'+species+' plots/Planet RF plots/median subtracted '+test_type+' '+model_name+' '+str(22+order_low)+'-'+str(22+order_high)+' '+str(nIter_used)+'iter.png')
        # plt.savefig('/Users/scott/Documents/Plots/Iron plots/Planet RF plots/'+model_name+' '+str(binsize)+' phases per bin'+str(22+order_low)+'-'+str(22+order_high)+' '+str(nIter)+'iter.png')
        
        
        #%% Phase-scrambling
        
        # separate in-transit phase and ccf
        phi_it = phi[ingress:egress]
        ccf_it = ccf_sum[ingress:egress]
        
        ccf_oot = ccf_sum[np.r_[0:ingress,egress:nframe],:] # the np.r term allows indexing two regions of this array
        phi_oot = phi[np.r_[0:ingress,egress:nframe]]
        
        
        # Kp and Vsys ranges
        K_p = np.linspace(0,300,301) # rv semi-amplitude
        vsys_range = np.linspace(-70,70,141) # systemic velocity
        
        fold_map_scram = np.zeros([141,301,1000]) # Vsys x Kp array
        
        # ccf_sum = np.zeros([301,141])
        
        # phi_scram = np.copy(phi) # create copy of phase array, labelled for scrambling
        
        for i in range(1000):
            
            idx = np.random.randint(phi.shape[0],size=np.sum(in_t)) # draw 121 random frame indices
            phi_scram = phi[idx].copy() # phases from random frame indices
            
            print(i)
            for pidx,p in enumerate(phi_scram): #loop over in-transit phases 121
            
                ccf_int_obj = interp1d(vshift+barycor[pidx], ccf_sum_medsub[pidx,:], axis=0, bounds_error=False, kind='linear', fill_value=0)
                
                for kidx,k in enumerate(K_p): # loop over range of k_p values 301
                    
                    vp = k * np.sin(2 * np.pi * p) 
                    
                    ccf_int = ccf_int_obj(vp + V_sys + vsys_range)
                
                    fold_map_scram[:,kidx,i] += ccf_int
        
        
        sd = np.std(fold_map_scram,axis=2)
        var = np.var(fold_map_scram,axis=2)
        sd[sd==0] = 1e10
        # np.save('/Users/scott/Documents/Data/WASP-76/significance/Test2-10 SD ' +test_type+' '+ model_name+' '+str(nIter),sd)
        
        
        newpath = '/Users/scott/Documents/Data/'+target+'/plots/Injection recovery plots/'+str(nIter_used)+' iters'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        np.save('/Users/scott/Documents/Data/'+target+'/plots/Injection recovery plots/'+str(nIter_used)+' iters/'+test_type+model_name+'.npy',fold_map)
        np.save('/Users/scott/Documents/Data/'+target+'/plots/Injection recovery plots/'+str(nIter_used)+' iters/'+test_type+model_name+'_sd.npy',sd)
        
        plt.figure()
        plt.pcolormesh(vsys_range,K_p[:],np.transpose(-fold_map[:,:]/sd[:,:]), shading='auto')
        plt.axhline(Kp,color='r',linestyle='--')
        plt.axvline(0,color='r',linestyle='--')
        plt.xlabel('V_sys (km/s)')
        plt.ylabel('Kp')
        plt.title('Folded CCF in-transit | '+model_name+' [Orders '+str(22+order_low)+'-'+str(22+order_high)+']'+str(nIter_used)+' iter')
        plt.xlim([-70,70])
        plt.colorbar()
        plt.savefig('/Users/scott/Documents/Data/'+target+'/Plots/'+species+' plots/median subtracted significance'+test_type+' '+model_name+' '+str(22+order_low)+'-'+str(22+order_high)+' '+str(nIter_used)+'iter')
        
        
        plt.figure()
        plt.pcolormesh(vsys_range,K_p[10:],np.transpose(-fold_map[:,10:]/sd[:,10:]), shading='auto')
        plt.axhline(Kp,color='r',linestyle='--')
        plt.axvline(0,color='r',linestyle='--')
        plt.xlabel('V_sys (km/s)')
        plt.ylabel('Kp')
        plt.title('Folded CCF in-transit | '+model_name+' [Orders '+str(22+order_low)+'-'+str(22+order_high)+']'+str(nIter_used)+' iter')
        plt.xlim([-70,70])
        plt.colorbar()
        plt.savefig('/Users/scott/Documents/Data/'+target+'/Plots/'+species+' plots/median subtracted significance -10Kp '+test_type+' '+model_name+' '+str(22+order_low)+'-'+str(22+order_high)+' '+str(nIter_used)+'iter')
        
        plt.figure()
        plt.pcolormesh(vsys_range,K_p[30:],np.transpose(-fold_map[:,30:]/sd[:,30:]), shading='auto')
        plt.axhline(Kp,color='r',linestyle='--')
        plt.axvline(0,color='r',linestyle='--')
        plt.xlabel('V_sys (km/s)')
        plt.ylabel('Kp')
        plt.title('Folded CCF in-transit | '+model_name+' [Orders '+str(22+order_low)+'-'+str(22+order_high)+']'+str(nIter_used)+' iter')
        plt.xlim([-70,70])
        plt.colorbar()
        plt.savefig('/Users/scott/Documents/Data/'+target+'/Plots/'+species+' plots/median subtracted significance -30Kp '+test_type+' '+model_name+' '+str(22+order_low)+'-'+str(22+order_high)+' '+str(nIter_used)+'iter')
        #%% Writing data to .fits
        ## These two lines of code will write any numpy array to a .fits file
        
        # hdu = fits.PrimaryHDU(barycor)
        # hdu.writeto('/Users/scott/BERV_WASP76.fits',overwrite=True)
        
        #%% rms of data for each iteration
        # nIter=10
        # fluxCR = np.load('/Users/scott/Documents/Data/WASP-76/fluxCR_10.npy')
        # y = fluxCR[1500:4500,:,:,:]
        
        # ysq = y**2
        # ysq_mean = np.mean(ysq,axis=0)
        # ysq_mean_sqrt = np.sqrt(ysq_mean)
        
        
        # # rms = np.sqrt(np.mean(y**2,axis=0))
        # # rms_mean = np.mean(rms[:,:,:],axis=0)
        
        # rms = np.std(y,axis=0)
        # rms_mean = np.mean(rms[:,:,:],axis=1)
        
        # for o in range(norder):
        #     plt.figure()
        #     plt.plot(range(nIter),rms_mean[o,:-1])
        #     plt.title('Order' + str(o))
        #     # plt.ylim(0.03,0.06)
        #     plt.xlabel('# SysRem iterations')
        #     plt.ylabel('RMS (middle-half of order)')
        #     plt.savefig('/Users/scott/Documents/Data/WASP-76/plots/SysRem/mean_RMS_order' + str(o) + '.png')
        #     plt.close()
        
        
        # order=34
        # data = fluxCR[:,order,:,:-1]
        # hdu = fits.PrimaryHDU(data)
        # hdu.writeto('/Users/scott/order_'+str(order)+'_cube.fits',overwrite=True)
        
        # hdu = fits.primaryHDU()
    
    
    
    #%% injection recovery subplots
    # species = ['H2O','TiO','VO','Mg','Al']
    # species = ['TiO','VO','Mg']
    species = ['H2O']
    species = ['Mg','K','VO','TiO','H2O','Na','Fe','FeH','Ca']

    VMRs = ['1,0%','0,1%','0,01%','0,001%','1e-04%','1e-05%','1e-06%','1e-07%'] # VMRs as labelled in files
    
    VMR_labels = [1e-02,1e-03,1e-04,1e-05,1e-06,1e-07,1e-08,1e-09] # decimal VMRs to display on axes
    # niter = 3 # number SysRem iterations chosen to plot
    # niter2 = 6
    
    fig, axes = plt.subplots(2,len(VMRs))
    test_type = ''
    for i,s in enumerate(species):
        if s == 'H2O':
            niter = 3
        else:
            niter = 3
        for j,v in enumerate(VMRs):
            print(s+' '+v)
            data = np.load('/Users/scott/Documents/Data/'+target+'/plots/Injection recovery plots/'+str(niter)+' iters/'+test_type+s+'_'+v+'.npy')
            sd_data = np.load('/Users/scott/Documents/Data/'+target+'/plots/Injection recovery plots/'+str(niter)+' iters/'+test_type+s+'_'+v+'_sd.npy')
            axes[i,j].pcolormesh(vsys_range,K_p[10:],np.transpose(-data[:,10:]/sd_data[:,10:]), shading='auto')
            
            axes[i,j].axhline(Kp,0,0.4,color='r',linestyle='--') # horizontal line with gap
            axes[i,j].axhline(Kp,0.6,1.0,color='r',linestyle='--')
            
            axes[i,j].axvline(0,0,0.5,color='r',linestyle='--') # vertical line with gap
            axes[i,j].axvline(0,0.8,1.0,color='r',linestyle='--')
            
            axes[-1,j].set_xlabel(str(VMR_labels[j]))
            axes[i,0].set_ylabel(s)
            cb = plt.pcolormesh(vsys_range,K_p[10:],np.transpose(-data[:,10:]/sd_data[:,10:]), shading='auto')
    # fig.colorbar(cb)
    
    plt.figure()
    plt.pcolormesh(vsys_range,K_p[10:],np.transpose(-data[:,10:]/sd_data[:,10:]), shading='auto')
    plt.colorbar()
    plt.xlabel('Vsys')
    plt.ylabel('Kp')
    plt.title('H2O 1% VMR')
    plt.axhline(Kp,0,0.4,color='r',linestyle='--')
    plt.axhline(Kp,0.6,1.0,color='r',linestyle='--')
    plt.axvline(0,0,0.5,color='r',linestyle='--') # vertical line with gap
    plt.axvline(0,0.8,1.0,color='r',linestyle='--')
    plt.savefig('/Users/scott/Documents/Data/'+target+'/Plots/Injection recovery plots/'+str(nIter_used)+'iter')
    # # plt.close('all')
    
    
    
    # #%%
    # # tellurics at 6280, 6870A, same thing as before. plot the centroids
    # # 7600A - order 8, 6900A - order 11
    # order=7
    
    # plt.figure()
    # plt.xlabel('wavelength')
    # plt.ylabel('flux')
    # # plt.xlim([588.8,589.2])
    
    # line = 768.2
    # width = 0.1
    # centroids = np.zeros(nframe)
    # def Gauss(x, y0, a, x0, sigma):
    #     return y0 + a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    # for f in range(nframe): #np.linspace(0,nframe,13):
    #     # if f==15 or f==164:
    #     #     continue
    #     f=int(f)
    #     # wavelength1 = wavelength[:-trim,:,:]
    #     idx = (wavelength[:,order,0] > (line-width)) & (wavelength[:,order,0] < (line+width))
    #     x = wavelength[idx,order,f]
    #     y = fluxBC[idx,order,f]
    #     # plt.plot(x,y,label='frame'+str(f))
    #     # plt.xlim([58,589.8])
    
    #     x_grid = np.linspace(line-width,line+width,1001)
        
    #     mean = sum(x * y) / sum(y)
    #     sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    #     popt,pcov = curve_fit(Gauss, x, y, p0=[5000,4000, mean, sigma],maxfev=5000) 
    #     centroids[f] = popt[2]
    #     # p0 holds guesses for last N-1 parameters from 'Gauss' function i.e. guesses for y0=baseline y value,a=height of peak,x0=x-position of peak,sigma=S.D
    
    #     if f%19==0:
    #         plt.plot(x_grid, Gauss(x_grid, *popt), '-', label=('frame'+str(f)))
    #     else:
    #         plt.plot(x_grid, Gauss(x_grid, *popt), '-')
    #     print(f)
    #     plt.legend(loc='upper right')
    #     plt.xlim([line-0.5,line+0.5])
    #     # plt.ylim([750,1500])
    # plt.title(target+' tellurics '+str(line*10)+'A')
    # plt.savefig('/Users/scott/'+target+' tellurics '+str(line*10)+'A.png')
    
    # # # print(popt[2])
    # plt.figure()
    # plt.plot(range(nframe),centroids)
    # plt.title(target+' Centroids tellurics '+str(line*10)+'A')
    # plt.xlabel('frame')
    # plt.ylabel('centroid position (nm)')
    # plt.savefig('/Users/scott/'+target+' centroids tellurics '+str(line*10)+'A.png')
    
    # # plotting in velocity space
    # plt.figure()
    # vshift = ((centroids/centroids[0]) - 1 ) * c
    # plt.plot(range(nframe),vshift*1000)
    # plt.xlabel('frame #')
    # plt.ylabel('velocity shift in m/s (relative to frame 0)')
    # plt.savefig('/Users/scott/'+target+' centroids tellurics (velocity) '+str(line*10)+'A.png')
    
    # plt.figure()
    # plt.plot(range(nframe),p_beg)
    # plt.plot(range(nframe),p_end)
    # plt.legend(['rp beginning','rp end'])
    # plt.xlabel('frame #')
    # plt.ylabel('ESPaDOnS relative pressure (mb)')
    # plt.title('ESPaDOnS relative pressure at beginning and end of all obs ('+target+')')
    
    plt.close('all')