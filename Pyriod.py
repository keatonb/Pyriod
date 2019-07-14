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
 
import sys
import numpy as np
import itertools
import re
#import ast
#import operator as op
import pandas as pd
#from astropy.stats import LombScargle
from scipy.interpolate import interp1d
import lightkurve as lk
from lmfit import Model, Parameters
from lmfit.models import ConstantModel
#from IPython.display import display #needed?
import matplotlib.pyplot as plt 
import ipywidgets as widgets
import qgrid
'''enable when ready to use
import logging
if sys.version_info < (3, 0):
    from io import BytesIO as StringIO
else:
    from io import StringIO
'''

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
        self._init_timeseries_widgets()
        self.lcfig,self.lcax = plt.subplots(figsize=(6,2),num='Time Series ({:d})'.format(self.id))
        self.lcax.set_xlabel("time")
        self.lcax.set_ylabel("rel. variation")
        self.lcplot, = self.lcax.plot(self.lc.time,self.lc.flux,marker='o',ls='None',ms=1)
        #Also plot the model over the time series
        dt = np.median(np.diff(self.lc.time))
        self.lcmodel_timesample = np.arange(np.min(self.lc.time),np.max(self.lc.time)+dt,dt)
        self.lcmodel_model_sampled = np.zeros(len(self.lcmodel_timesample))+np.mean(self.lc.flux)
        self.lcmodel_model_observed = np.zeros(len(self.lc.time))+np.mean(self.lc.flux)
        self.lcmodel, = self.lcax.plot(self.lcmodel_timesample,self.lcmodel_model_sampled,c='r',lw=1)
        plt.tight_layout()
        
        #Frequency resolution is important
        self.fres = 1./(self.lc.time[-1]-self.lc.time[0])
        
        #Hold signal phases, frequencies, and amplitudes in Pandas DF
        self.values = self.initialize_dataframe()
        
        #self.uncertainties = pd.DataFrame(columns=self.columns[::2]) #not yet used
        
        #Compute periodogram
        self.ls = self.lc.to_periodogram(normalization='amplitude',oversample_factor=10)/1e3
        
        self.interpls = interp1d(self.ls.frequency.value,self.ls.power.value)
        self._init_periodogram_widgets()
        
        #The interface for interacting with the values DataFrame:
        self.signals_qgrid = self.get_qgrid()
        self.signals_qgrid.on('cell_edited', self._update_values_from_qgrid)
        self._init_signals_widgets()
        
        #Set up some figs/axes for time series and periodogram plots
        self.perfig,self.perax = plt.subplots(figsize=(6,3),num='Periodogram ({:d})'.format(self.id))
        self.perax.set_xlabel("frequency")
        self.perax.set_ylabel("amplitude (mma)")
        plt.tight_layout()
        #set peak marker at highest peak
        self.marker = self.perax.plot([0],[0],c='k',marker='o')[0]
        self.signal_marker_color = 'green'
        self.signal_markers, = self.perax.plot([],[],ls='none',marker='D',fillstyle='none',c='none',ms=5)
        
        
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
            disabled=False
        )
        self._tstype.on_trait_change(self._update_lc_display)
    
    def _init_periodogram_widgets(self):
        ### Periodogram widget stuff  ###
        self._pertype = widgets.Dropdown(
            options=['Original', 'Residuals', 'Model', 'Window'],
            value='Original',
            description='Periodogram to Display:',
            disabled=False,
        )
        
        self._thisfreq = widgets.Text(
            value='',
            placeholder='',
            description='Frequency:',
            disabled=False
        )
        
        """
        .BoundedFloatText(
            value=0.001,
            min=0,
            #max=np.max(freq),  #fix later
            step=None,
            description='Frequency:',
            disabled=False
        )
        """
        
        self._thisamp = widgets.FloatText(
            value=0.001,
            #min=0,
            #max=np.max(amp),
            #step=None,
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
        self._addtosol.on_click(self._add_staged_signal)
        
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
        self._showperiodsolution.observe(self._makeperiodsolutionvisible)
        
    def _init_signals_widgets(self):
        ### Time Series widget stuff  ###
        self._refit = widgets.Button(
            description='Refine fit',
            disabled=False,
            #button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Refine fit of signals to time series',
            icon='refresh'
        )
        self._refit.on_click(self.fit_model)
        
    def add_signal(self, freq, amp=None, phase=None, fixfreq=False, 
                   fixamp=False, fixphase=False, index=None):
        if amp is None:
            amp = 1.
        if phase is None:
            phase = 0.5
        #print(self.signals_qgrid.df.dtypes)
        #list of iterables required to pass to dataframe without an index
        newvalues = [[nv] for nv in [freq,fixfreq,amp,fixamp,phase,fixphase]]
        #print(self.values.freq)
        if index == None:
            index = "f{}".format(len(self.values))
        print("Signal {} added to model with frequency {} and amplitude {}".format(index,freq,amp))
        toappend = pd.DataFrame(dict(zip(self.columns,newvalues)),columns=self.columns,
                                index=[index])
        #.astype(dtype=dict(zip(self.columns,self.dtypes)))
        #print(toappend)
        #print(toappend.dtypes)
        self.values = self.values.append(toappend,sort=False)
        #print(self.values.freq)
        #print(toappend.dtypes)
        self.signals_qgrid.df = self.values[self.columns[:6]]
        self._update_signal_markers()
        
    #operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    #             ast.Div: op.truediv,ast.USub: op.neg}
    def add_combination(self, combostr, amp=None, phase=None, fixfreq=False, 
                   fixamp=False, fixphase=False, index=None):
        combostr = combostr.replace(" ", "")
        #evaluate combostring:
        
        #replace keys with values
        parts = re.split('\+|\-|\*|\/',combostr)
        keys = set([part for part in parts if part in self.values.index])
        expression = combostr
        for key in keys:
            expression = expression.replace(key, str(self.values.loc[key,'freq']))
        #print(ast.parse(combostr, mode='eval').body)
        freqval = eval(expression)
        if amp == None:
            amp = self.interpls(freqval)/1e3
        self.add_signal(freqval,amp,index=combostr)
        #allvalid = np.all([(part in self.values.index) or [part.replace('.','',1).isdigit()] for part in parts])
        
    
    def fit_model(self, *args):
        """ 
        Update model to include current signals from DataFrame.
        
        Improve fit once with all frequencies fixed, then allow to vary.
        """
        #Set up lmfit model for fitting
        signals = {} #empty dict to be populated
        params = Parameters()
        
        #handle combination frequencies differently
        isindep = lambda key: key[1:].isdigit()
        cnum = 0
        
        prefixmap = {}
        
        #first with frequencies fixed
        for prefix in self.values.index:
            #prefix = 'f{}'.format(i+1)
            #freqkeys.append(prefix)
            if isindep(prefix):
                signals[prefix] = Model(sin,prefix=prefix)
                params.update(signals[prefix].make_params())
                params[prefix+'freq'].set(self.values.freq[prefix], vary=False)
                params[prefix+'amp'].set(self.values.amp[prefix], vary=~self.values.fixamp[prefix])
                params[prefix+'phase'].set(self.values.phase[prefix], vary=~self.values.fixphase[prefix])
                prefixmap[prefix] = prefix
            else: #combination
                useprefix = 'c{}'.format(cnum)
                signals[useprefix] = Model(sin,prefix=useprefix)
                params.update(signals[useprefix].make_params())
                parts = re.split('\+|\-|\*|\/',prefix)
                keys = set([part for part in parts if part in self.values.index])
                expression = prefix
                for key in keys:
                    expression = expression.replace(key, key+'freq')
                params[useprefix+'freq'].set(expr=expression)
                params[useprefix+'amp'].set(self.values.amp[prefix], vary=~self.values.fixamp[prefix])
                params[useprefix+'phase'].set(self.values.phase[prefix], vary=~self.values.fixphase[prefix])
                prefixmap[prefix] = useprefix
                cnum+=1
        
        #model is sum of sines
        model = np.sum([signals[prefixmap[prefix]] for prefix in self.values.index])
        
        #compute fixed-frequency fit
        result = model.fit(self.lc.flux-np.mean(self.lc.flux), params, x=self.lc.time)
        
        #refine, allowing freq to vary (unless fixed by user)
        params = result.params
        
        for prefix in self.values.index:
            if isindep(prefix):
                params[prefixmap[prefix]+'freq'].set(vary=~self.values.fixfreq[prefix])
                params[prefixmap[prefix]+'amp'].set(result.params[prefixmap[prefix]+'amp'].value)
                params[prefixmap[prefix]+'phase'].set(result.params[prefixmap[prefix]+'phase'].value)
                
        result = model.fit(self.lc.flux-np.mean(self.lc.flux), params, x=self.lc.time)
        
        self._update_values_from_fit(result.params,prefixmap)
        
        
    def _update_values_from_fit(self,params,prefixmap):
        #update dataframe of params with new values from fit
        #also rectify and negative amplitudes or phases outside [0,1)
        #isindep = lambda key: key[1:].isdigit()
        #cnum = 0
        for prefix in self.values.index:
            self.values.loc[prefix,'freq'] = float(params[prefixmap[prefix]+'freq'].value)
            self.values.loc[prefix,'amp'] = params[prefixmap[prefix]+'amp'].value
            self.values.loc[prefix,'phase'] = params[prefixmap[prefix]+'phase'].value
            #rectify
            if self.values.loc[prefix,'amp'] < 0:
                self.values.loc[prefix,'amp'] *= -1.
                self.values.loc[prefix,'phase'] -= 0.5
            self.values.loc[prefix,'phase'] %= 1.
            
        #update qgrid
        self.signals_qgrid.df = self.values[self.columns[:6]]
        #TODO: also update uncertainties
        
        self._update_values_from_qgrid()
        
        
    def _update_values_from_qgrid(self, *args):
        self.values = self.signals_qgrid.get_changed_df()
        #Update time series model displayed
        self.lcmodel_model_sampled = np.zeros(len(self.lcmodel_timesample))+np.mean(self.lc.flux)
        self.lcmodel_model_observed = np.zeros(len(self.lc.time))+np.mean(self.lc.flux)
        
        for prefix in self.values.index:
            freq = float(self.values.loc[prefix,'freq'])
            amp = float(self.values.loc[prefix,'amp'])
            phase = float(self.values.loc[prefix,'phase'])
            self.lcmodel_model_sampled += sin(self.lcmodel_timesample,
                                              freq,amp,phase)
            
            self.lcmodel_model_observed += sin(self.lc.time,
                                               freq,amp,phase)
        
        self._update_signal_markers()
        self._update_lc_display()
    
    columns = ['freq','fixfreq','amp','fixamp','phase','fixphase','combo']
    dtypes = ['object','bool','float','bool','float','bool','bool']
    
    def initialize_dataframe(self):
        df = pd.DataFrame(columns=self.columns).astype(dtype=dict(zip(self.columns,self.dtypes)))
        #df.index.name = 'ID'
        #print(df.dtypes)
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
        return qgrid.show_grid(self.values[self.columns[:6]], show_toolbar=False, precision = 10,
                               grid_options=self._gridoptions,
                               column_definitions=self._column_definitions)
    
    #add staged signal to frequency solution
    def _add_staged_signal(self, *args):
        #Is this a valid numeric frequency?
        if self._thisfreq.value.replace('.','',1).isdigit():
            self.add_signal(float(self._thisfreq.value),self._thisamp.value)
        #Is it a valid combination frequency?
        #elif
        #Otherwise issue a warning
        else:
            parts = re.split('\+|\-|\*|\/',self._thisfreq.value.replace(" ", ""))
            allvalid = np.all([(part in self.values.index) or [part.replace('.','',1).isdigit()] for part in parts])
            if allvalid and (len(parts) > 1):
                #will guess amplitude from periodogram
                self.add_combination(self._thisfreq.value)
            else:
                print("Staged frequency has invalid format: "+self._thisfreq.value)
        
    #change type of time series being displayed
    def _update_lc_display(self, *args):
        displaytype = self._tstype.value
        updatedisplay = {"Original":self._display_original_lc,
                         "Residuals":self._display_residuals_lc}
        updatedisplay[displaytype]()
        
    def _makeperiodsolutionvisible(self, *args):
        if self._showperiodsolution.value:
            self.signal_markers.set_color(self.signal_marker_color)
        else:
            self.signal_markers.set_color('none')
        self.perfig.canvas.draw()
        
    def _update_signal_markers(self):
        self.signal_markers.set_data(self.values['freq'].astype('float'),self.values['amp']*1e3)
        self.perfig.canvas.draw()
        
    def _display_original_lc(self):
        self.lcplot.set_ydata(self.lc.flux)
        self.lcmodel.set_ydata(self.lcmodel_model_sampled)
        #rescale y to better match data
        ymin = np.min([np.min(self.lc.flux),np.min(self.lcmodel_model_sampled)])
        ymax = np.max([np.max(self.lc.flux),np.max(self.lcmodel_model_sampled)])
        self.lcax.set_ylim(ymin-0.05*(ymax-ymin),ymax+0.05*(ymax-ymin))
        self.lcfig.canvas.draw()
        
    def _display_residuals_lc(self):
        self.lcplot.set_ydata(self.lc.flux-self.lcmodel_model_observed)
        self.lcmodel.set_ydata(np.zeros(len(self.lcmodel_timesample)))
        ymin = np.min(self.lc.flux-self.lcmodel_model_observed)
        ymax = np.max(self.lc.flux-self.lcmodel_model_observed)
        self.lcax.set_ylim(ymin-0.05*(ymax-ymin),ymax+0.05*(ymax-ymin))
        self.lcfig.canvas.draw()
    
    
    def onperiodogramclick(self,event):
        if self._snaptopeak.value:
            freqs = self.ls.frequency.value
            #click within either frequency resolution or 1% of displayed range
            #TODO: make this work with log frequency too
            tolerance = np.max([self.fres,0.01*np.diff(self.perax.get_xlim())])
            
            nearby = np.argwhere((freqs >= event.xdata - tolerance) & 
                                 (freqs <= event.xdata + tolerance))
            highestind = np.argmax(self.ls.power.value[nearby]) + nearby[0]
            self.update_marker(freqs[highestind],self.ls.power.value[highestind])
        else:
            self.update_marker(event.xdata,self.interpls(event.xdata))
        
    def Periodogram(self):
        display(self._pertype,self._recalculate,self._thisfreq,self._thisamp,
                self._addtosol,self._snaptopeak,self._showperiodsolution,
                self.perfig)
        
    def TimeSeries(self):
        display(self._tstype,self.lcfig)
        
    def update_marker(self,x,y):
        try:
            self._thisfreq.value = str(x[0])
        except:
            self._thisfreq.value = str(x)
        self._thisamp.value =  y/1e3
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

    def Signals(self):
        display(self._refit,self.signals_qgrid)
        
    
        
    
