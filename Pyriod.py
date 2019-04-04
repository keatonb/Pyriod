#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This is Pyriod, a Python package for selecting and fitting sinusoidal signals 
to astronomical time series data.

Written by Keaton Bell

For more, see https://github.com/keatonb/Pyriod

---------------------

Long list of todos:
    - Validate that time, flux, fluxerr, etc. have same length
    - Allow (encourage) lightkurve objects to be passed
"""

from __future__ import division, print_function

import numpy as np
import pandas as pd
from astropy.stats import LombScargle
import lightkurve as lk
from lmfit import Model, Parameters

#Definition of the basic model we fit
def sin(x, freq, amp, phase):
    """for fitting to time series"""
    return amp*np.sin(2.*np.pi*(freq*x+phase))

class Pyriod(object):
    """Time series periodic analysis class.
    
    Attributes
    ----------
    time : array-like
        Time values
    flux : array-like
        Flux values (normalized, mean subtracted)
    
	Future Development
	----------
	Include flux uncertainties, units, etc.
    """
    def __init__(self, lc=None, time=None, flux=None):
        if lc is None and time is None and flux is None:
            raise ValueError('lc or time amd flux are required')
        if lc is not None:
            if lk.lightcurve.LightCurve not in type(lc).__mro__:
                raise ValueError('lc must be lightkurve object')
            else:
                self.lc = lc
        else:
            self.lc = lk.LightCurve(time=time, flux=flux)
        
        #Frequency resolution will be important for fitting
        self.fres = 1./(self.lc.time[-1]-self.lc.time[0])
        
        #Hold signal phases, frequencies, and amplitudes in Pandas DF
        self.columns = ['freq','fixfreq','amp','fixamp','phase','fixphase']
        self.values = pd.DataFrame(columns=self.columns)
        self.uncertainties = pd.DataFrame(columns=self.columns[::2]) #not yet used
        
        
    def compute_periodogram(self, on='original', osample=10, nyqfactor=1.):
        """Computes Lomb-Scargle periodogram and associated details
        
        To be replaced with lightkurve periodogram when amplitude normalization
        is implemented.
        """
        time = np.copy(self.lc.time)
        flux = np.copy(self.lc.flux)
        time -= time[0]
        
        c = np.median(np.diff(time))
        nyq = 1./(2.*c)
        df = (1./time[-1])
        
        freq = np.arange(df, nyq * nyqfactor, df/osample)
        
        ls = LombScargle(time,flux)
        amp = np.sqrt(ls.power(freq,normalization='psd')*4./time.size)
            
        return {"freq":freq, "amp":amp}

    def add_signal(self, freq, amp=None, phase=None, fixfreq=False, 
                   fixamp=False, fixphase=False):
        if amp is None:
            amp = 1.
        if phase is None:
            phase = 0.5
        
        newvalues = [freq,fixfreq,amp,fixamp,phase,fixphase]
        
        self.values = self.values.append(dict(zip(self.columns,newvalues)),ignore_index=True)
        
    def fit_model(self):
        """ 
        Update model to include current signals from DataFrame.
        
        Improve fit once with all frequencies fixed, then allow to vary.
        """
        #Set up lmfit model for fitting
        signals = {} #empty dict to be populated
        freqkeys = [] #prefixes for each signal
        params = Parameters()
        
        #first with frequencies fixed
        for i in range(len(self.values)):
            prefix = 'f{}'.format(i+1)
            freqkeys.append(prefix)
            signals[prefix] = Model(sin,prefix=prefix)
            params.update(signals[prefix].make_params())
            params[prefix+'freq'].set(self.values.freq[i], vary=False)
            params[prefix+'amp'].set(self.values.amp[i], vary=~self.values.fixamp[i])
            params[prefix+'phase'].set(self.values.phase[i], vary=~self.values.fixphase[i])
        
        #model is sum of sines
        model = np.sum([signals[freqkey] for freqkey in freqkeys])
        
        #compute fixed-frequency fit
        result = model.fit(self.lc.flux, params, x=self.lc.time)
        
        #refine, allowing freq to vary (unless fixed by user)
        params = result.params
        for i,freqkey in enumerate(freqkeys):
            params[freqkey+'freq'].set(vary=~self.values.fixphase[i])
            params[freqkey+'amp'].set(result.params[freqkey+'amp'].value)
            params[freqkey+'phase'].set(result.params[freqkey+'phase'].value)
        result = model.fit(self.lc.flux, params, x=self.lc.time)
        
        self.update_values(result.params)
        
    def update_values(self,params):
        #update dataframe of params with new values from fit
        #also rectify and negative amplitudes or phases outside [0,1)
        print (params.keys())
        for i in range(len(self.values)):
            prefix = 'f{}'.format(i+1)
            self.values['freq'][i] = params[prefix+'freq'].value
            self.values['amp'][i] = params[prefix+'amp'].value
            self.values['phase'][i] = params[prefix+'phase'].value
            #rectify
            if self.values['amp'][i] < 0:
                self.values['amp'][i] *= -1.
                self.values['phase'][i] -= 0.5
            self.values['phase'][i] %= 1.
        
        #TODO: also update uncertainties
        
        
        
