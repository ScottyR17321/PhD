#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:48:15 2022

@author: scott
"""

import numpy as np

def mad_clip(arr, mad_limit=5, axis=None, keepdims=True, cliphigh=True, cliplow=True):
    arr2 = np.array(arr) # ensures this is a numpy array
    median = np.median(arr2, axis=axis, keepdims=True) # calculate median value of array
    dev = arr2-median # calculate deviations from the median
    mad = np.median(np.abs(dev),axis=axis, keepdims=keepdims) # MAD = median of the absolute deviations
    if cliphigh==True:
        arr2[(dev > (mad_limit * mad))] = median # clip outliers above the median
    if cliplow==True:
        arr2[(dev < (mad_limit * mad))] = median # clip outliers below the median
    return arr2    