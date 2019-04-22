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
    
    
    
        
# Distinguish clicks with drag motions
# From ImportanceOfBeingErnest
# https://stackoverflow.com/questions/48446351/distinguish-button-press-event-from-drag-and-zoom-clicks-in-matplotlib
    
    
"""

from __future__ import division, print_function
 
import numpy as np
import itertools
import pandas as pd
from astropy.stats import LombScargle
from scipy.interpolate import interp1d
import lightkurve as lk
from lmfit import Model, Parameters
#from IPython.display import display #needed?
import matplotlib.pyplot as plt 
import ipywidgets as widgets
import qgrid

plt.ioff()

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
    id_generator = itertools.count(0)
    def __init__(self, lc=None, time=None, flux=None):
        self.id = next(self.id_generator)
        
        #Store light curve as LightKurve object
        if lc is None and time is None and flux is None:
            raise ValueError('lc or time amd flux are required')
        if lc is not None:
            if lk.lightcurve.LightCurve not in type(lc).__mro__:
                raise ValueError('lc must be lightkurve object')
            else:
                self.lc = lc
        else:
            self.lc = lk.LightCurve(time=time, flux=flux)
        
        ### Time series widget suite ###
        self.lcfig,self.lcax = plt.subplots(figsize=(8,3),num='Time Series ({:d})'.format(self.id))
        self.lcax.set_xlabel("time")
        self.lcax.set_ylabel("rel. variation")
        self.lcplot = self.lcax.plot(self.lc.time,self.lc.flux,marker='o',ls='None',ms=1)
        plt.tight_layout()
        
        #Frequency resolution will be important for fitting
        self.fres = 1./(self.lc.time[-1]-self.lc.time[0])
        
        #Hold signal phases, frequencies, and amplitudes in Pandas DF
        self.columns = ['freq','fixfreq','amp','fixamp','phase','fixphase']
        self.values = self.initialize_dataframe()
        
        #self.uncertainties = pd.DataFrame(columns=self.columns[::2]) #not yet used
        
        #Compute periodogram
        self.ls = self.lc.to_periodogram(normalization='amplitude',oversample_factor=10)
        
        self.interpls = interp1d(self.ls.frequency.value,self.ls.power.value)
        self._init_periodogram_widgets()
        
        #The interface for interacting with the values DataFrame:
        self.Signals = self.get_qgrid()
        
        #Set up some figs/axes for time series and periodogram plots
        self.perfig,self.perax = plt.subplots(figsize=(6,3),num='Periodogram ({:d})'.format(self.id))
        self.perax.set_xlabel("frequency")
        self.perax.set_ylabel("amplitude (mma)")
        plt.tight_layout()
        #set peak marker at highest peak
        self.marker = self.perax.plot([0],[0],c='k',marker='o')[0]
        
        
        self.update_marker(self.ls.frequency.value[np.argmax(self.ls.power.value)],
                           np.max(self.ls.power.value))
        
        self.perplot = self.perax.plot(self.ls.frequency.value,self.ls.power.value)
        #self.perfig.canvas.mpl_connect('button_press_event', self.onperiodogramclick)
        self.press= False
        self.move = False
        self.perfig.canvas.mpl_connect('button_press_event', self.onpress)
        self.perfig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.perfig.canvas.mpl_connect('motion_notify_event', self.onmove)
    
    
    def _init_timeseries_widgets(self):
        ### Time Series widget stuff  ###
        self._tstype = widgets.Dropdown(
            options=['Original', 'Residuals'],
            value='Original',
            description='Time Series to Display:',
            disabled=False,
        )
    
    def _init_periodogram_widgets(self):
        ### Periodogram widget stuff  ###
        self._pertype = widgets.Dropdown(
            options=['Original', 'Residuals', 'Model', 'Window'],
            value='Original',
            description='Periodogram to Display:',
            disabled=False,
        )
        
        self._thisfreq = widgets.BoundedFloatText(
            value=0.001,
            min=0,
            #max=np.max(freq),  #fix later
            step=None,
            description='Frequency:',
            disabled=False
        )
        
        self._thisamp = widgets.BoundedFloatText(
            value=0.001,
            min=0,
            #max=np.max(amp),
            step=None,
            description='Amplitude:',
            disabled=False
        )
        
        
        self._recalculate = widgets.Button(
            description='Recalculate',
            disabled=True,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to recalculate periodogram based on updated solution.',
            icon='refresh'
        )
        
        
        self._addtosol = widgets.Button(
            description='Add to solution',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to add currently selected values to frequency solution',
            icon='plus'
        )
        self._addtosol.on_click(self.on_button_clicked)
        
        self._snaptopeak = widgets.Checkbox(
            value=True,
            description='Snap clicks to peaks?',
            disabled=False
        )
        
        self._showperiodsolution = widgets.Checkbox(
            value=False,
            description='Show frequencies in period solution?',
            disabled=False
        )

    def add_signal(self, freq, amp=None, phase=None, fixfreq=False, 
                   fixamp=False, fixphase=False):
        if amp is None:
            amp = 1.
        if phase is None:
            phase = 0.5
        
        newvalues = [freq,fixfreq,amp,fixamp,phase,fixphase]
        
        self.values = self.values.append(dict(zip(self.columns,newvalues)),ignore_index=True)
        self.Signals.df = self.values
        
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
        result = model.fit(self.lc.flux-np.mean(self.lc.flux), params, x=self.lc.time)
        
        #refine, allowing freq to vary (unless fixed by user)
        params = result.params
        for i,freqkey in enumerate(freqkeys):
            params[freqkey+'freq'].set(vary=~self.values.fixphase[i])
            params[freqkey+'amp'].set(result.params[freqkey+'amp'].value)
            params[freqkey+'phase'].set(result.params[freqkey+'phase'].value)
        result = model.fit(self.lc.flux-np.mean(self.lc.flux), params, x=self.lc.time)
        
        self.update_values(result.params)
        
    def update_values(self,params):
        #update dataframe of params with new values from fit
        #also rectify and negative amplitudes or phases outside [0,1)
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
       
        
    def initialize_dataframe(self):
        dtypes = ['float','bool','float','bool','float','bool']
        df = pd.DataFrame(columns=self.columns).astype(dtype=dict(zip(self.columns,dtypes)))
        df.index.name = 'ID'
        
        return df
    
    
    ########## Set up *SIGNALS* widget using qgrid ##############
    
    #Set some options for how the qgrid of values should be displayed
    _gridoptions = {
            # SlickGrid options
            'fullWidthRows': True,
            'syncColumnCellResize': True,
            'forceFitColumns': False,
            'defaultColumnWidth': 150,  #control col width (all the same)
            'rowHeight': 28,
            'enableColumnReorder': False,
            'enableTextSelectionOnCells': True,
            'editable': True,
            'autoEdit': True, #double-click not required!
            'explicitInitialization': True,
            

            # Qgrid options
            'maxVisibleRows': 15,
            'minVisibleRows': 8,
            'sortable': True,
            'filterable': False,  #Not useful here
            'highlightSelectedCell': False,
            'highlightSelectedRow': True
           }
    
    _column_definitions = {"freq":      {'width': 150, 'toolTip': "mode frequency"},
                           "fixfreq":  {'width': 65, 'toolTip': "fix frequency during fit?"},
                           "amp":       {'width': 150, 'toolTip': "mode amplitude"},
                           "fixamp":   {'width': 65, 'toolTip': "fix amplitude during fit?"},
                           "phase":     {'width': 150, 'toolTip': "mode phase"},
                           "fixphase": {'width': 65, 'toolTip': "fix phase during fit?"}}
    
    
    def get_qgrid(self):
        return qgrid.show_grid(self.values, show_toolbar=True, precision = 10,
                               grid_options=self._gridoptions,
                               column_definitions=self._column_definitions)
    
    #add staged signal to frequency solution
    def on_button_clicked(self,*args):
        self.add_signal(self._thisfreq.value,self._thisamp.value)
    
    
    def onperiodogramclick(self,event):
        if self._snaptopeak.value:
            freqs = self.ls.frequency.value
            nearby = np.argwhere((freqs >= event.xdata - self.fres) & 
                                 (freqs <= event.xdata + self.fres))
            highestind = np.argmax(self.ls.power.value[nearby]) + nearby[0]
            self.update_marker(freqs[highestind],self.ls.power.value[highestind])
        else:
            self.update_marker(event.xdata,self.interpls(event.xdata))
        
    def Periodogram(self):
        display(self._pertype,self._recalculate,self._thisfreq,self._thisamp,
                self._addtosol,self._snaptopeak,self._showperiodsolution,
                self.perfig)
        
    def TimeSeries(self):
        display(self.lcfig)
        
    def update_marker(self,x,y):
        self._thisfreq.value = x
        self._thisamp.value =  y/1e6
        self.marker.set_data([x],[y])
        self.perfig.canvas.draw()
        self.perfig.canvas.flush_events()
        
        
    def onclick(self,event):
        self.onperiodogramclick(event)
    def onpress(self,event):
        self.press=True
    def onmove(self,event):
        if self.press:
            self.move=True
    def onrelease(self,event):
        if self.press and not self.move:
            self.onclick(event)
        self.press=False; self.move=False

        