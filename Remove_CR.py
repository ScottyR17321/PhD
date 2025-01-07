#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:54:10 2022

@author: scott
"""

# Removing cosmic rays from the data

# load in packages
import numpy as np
import matplotlib.pyplot as plt
from mad_clip import mad_clip

# load in data cube of blaze-corrected and sys-rem'd flux data (10 iterations)

Fluxes_BC_SR = np.load('/Users/scott/Documents/Data/2020 (First Data Set of PhD)/SysRem/Test2SR_Fluxes_BC_SR-2_6iter.npy')

# we want to average across each wavelength values for each order across all frames, 
# and remove all values more than 5 MAD or ~3 sigma higher than the mean/median. 
# This means for each 35x5910 echelle spectrum, we average across the time domain (169 frames)

Fluxes_BC_SR_CR = np.zeros([35,169,5910])

for i in range(np.size(Fluxes_BC_SR,axis=0)):
        for k in range(np.size(Fluxes_BC_SR,axis=2)):
            Fluxes_BC_SR_CR[i,:,k] = mad_clip(Fluxes_BC_SR[i,:,k], cliplow=False)
            
np.save('/Users/scott/Documents/Data/2020 (First Data Set of PhD)/Cosmic Ray Removal/Test2SR_Fluxes_BC_SR_CR-2_6iter', Fluxes_BC_SR_CR)
            
            

            
        




    