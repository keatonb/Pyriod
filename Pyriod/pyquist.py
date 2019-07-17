#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module contains functions for calculating the canidate intrinsic 
frequencies and amplitudes that may underlie an observed sub-Nyquist signal,
or the sub-Nyquist frequencies and amplitudes that would be measured from
some intrinsic super-Nyquist signal.

This code was developed for the analysis of pulsating white dwarfs observed
at long-cadence by the Kepler K2 mission. If this code is helpful to you,
consider citing our original work:
    
    Bell, K. J., Hermes, J. J., Vanderbosch, Z., et al. 2017, ApJ, 851, 24
    arXiv:1710.10273 
    http://adsabs.harvard.edu/abs/2017ApJ...851...24B
    
That paper includes a thorough discussion of important considerations for 
dealing with super-Nyquist signals.

This pyquist module lives at https://github.com/keatonb/pyquist

@author: bell@mps.mpg.de
"""

from __future__ import division
from __future__ import absolute_import
import numpy as np


def subfreq(freq, fnyq=1.):
    """Return sub-Nyquist frequencies given super-Nyquist frequencies
    
    Args:
        freq: super-Nyq frequencies (fraction of fnyq)
        fnyq: Nyquist frequency (default 1.0)
            
    Returns:
        sub-Nyquist frequency measured for intrinsic freq relative to 
            optionally specified fnyq
    """
    #check whether freq is iterable or single value
    freq = np.asarray(freq,dtype=np.float64)
    scalar_input = False
    if freq.ndim == 0:
        freq = freq[None]  # newaxis
        scalar_input = True
    
    #for iterable, apply to each element
    if not scalar_input:
        return np.fromiter((subfreq(f,fnyq) for f in freq), freq.dtype)
    else: #return subNyquist alias for scalar input
        freq = freq / fnyq
        rem = freq % 1.
        if np.floor(freq) % 2 == 0:            #even bounces
            return rem*fnyq
        else:                                  #odd bounces
            return (1.-rem)*fnyq


def superfreq(freq, bounces=1, fnyq=1.):
    """Return super-Nyquist frequencies given sub-Nyquist frequency
    
    Args:
        freq: observed sub-Nyquist frequency as fraction of fnyq
        bounces: number of bounces off the Nyqust to compute (iterable; default 1)
        fnyq: Nyquist frequency (default 1.0)
        
    Returns:
        underlying frequencies if freq was observed after bouncing each of n
            times off the Nyquist (relative to Nyquist unless fnyq specified)
    """
    #check whether n is iterable or single value
    bounces = np.asarray(bounces,dtype=np.float64)
    scalar_input = False
    if bounces.ndim == 0:
        bounces = bounces[None]  # newaxis
        scalar_input = True
    
    #for iterable bounces, apply for each
    if not scalar_input:
        return np.fromiter((superfreq(freq,b,fnyq) for b in bounces), bounces.dtype)
    else: #return superNyquist candidates for scalar bounce
        #calculate for relative frequency
        freq = freq / fnyq
        if bounces % 2 == 0:                #even bounces
            return (bounces+freq)*fnyq
        else:                               #odd bounces
            return (1.+bounces-freq)*fnyq
        

def subamp(freq, fnyq=1.):
    """Return observed relative amplitudes given intrinsic frequencies
    
    Args:
        freq: intrinsic frequency as fraction of fnyq
        fnyq: Nyquist frequency (default 1.0)
        
    Returns:
        observed relative amplitudes for signal of frequency freq relative to fnyq
    """
    #check whether freq is iterable or single value
    freq = np.asarray(freq,dtype=np.float64)
    scalar_input = False
    if freq.ndim == 0:
        freq = freq[None]  # newaxis
        scalar_input = True
        
    #for iterable, apply to each element
    if not scalar_input:
        return np.fromiter((subamp(f,fnyq) for f in freq), freq.dtype)
    else: #return observed amplitude for scalar input
        if (freq > 0) and (freq/fnyq % 2) == 0: #if even multiple of Nyquist
            return np.float64(0) #np.sinc acts weird here
        else:
            return np.abs(np.sinc(0.5*freq/fnyq))
    
    
def superamp(freq, fnyq=1.):
    """Return intrinsic amplitudes relative to observed given intrinsic frequencies
    
    Args:
        freq: intrinsic frequency as fraction of fnyq
        fnyq: Nyquist frequency (default 1.0)
        
    Returns:
        intrinsic amplitudes relative to observed at frequency freq relative to fnyq
    """
    return 1./subamp(freq,fnyq)
