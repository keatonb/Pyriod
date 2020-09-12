#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This is Pyriod, a Python package for selecting and fitting sinusoidal signals 
to astronomical time series data.

Written by Keaton Bell

For more, see https://github.com/keatonb/Pyriod

---------------------

The following code was stolen from stackoverflow:

Distinguish clicks with drag motions from ImportanceOfBeingErnest
https://stackoverflow.com/a/48452190

Capturing print output from kindall
https://stackoverflow.com/a/16571630

---------------------

Below here are just some author's notes to keep track of style decisions.

Currently overhauling the periodogram display.
Should be optional to toggle display of each type of periodogram
and their display colors
and which one the mouse is selecting on.

Names of periodograms:
    per_orig
    per_resid
    per_model
    per_sw
    per_markers

Names of associates timeseries:
    lc_orig
    lc_resid
    lc_model_sampled (evenly oversampled through gaps)
    lc_model_observed (original time samples)
TODO: rename all lc to ts

Names of plots are:
    lcplot_data,lcplot_model (different nicknames)
    perplot_orig (same nicknames)
    _perplot_orig_display toggle
    TODO: _perplot_orig_color picker widget
    
    
What to do about units:
    Time series assumed in days, frequencies computed in microHz
    TODO: Enable other units
    Different amplitude units available (relative, percent, mma, ppt, etc.)
    
TODO: Generate model light curves from lmfit model always (including initialization)

TODO: Show smoothed light curve (and when folded)

"""
#Standard imports
from __future__ import division, print_function
import sys
import os
import itertools
import re
import logging
if sys.version_info < (3, 0):
    #from io import BytesIO as StringIO
    from StringIO import StringIO
else:
    from io import StringIO
#from itertools import groupby
#from operator import itemgetter

#Third party imports
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.timeseries import LombScargle
import lightkurve as lk
from lmfit import Model, Parameters
#from lmfit.models import ConstantModel
#from IPython.display import display
from bs4 import BeautifulSoup
#import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import ipywidgets as widgets
from ipywidgets import HBox,VBox
import qgrid
from ipyfilechooser import FileChooser

#Local imports
from .pyquist import subfreq

plt.ioff()#Turn off interactive mode

#Definition of the basic model we fit
def sin(x, freq, amp, phase):
    """for fitting to time series"""
    return amp*np.sin(2.*np.pi*(freq*x+phase))

#From https://stackoverflow.com/a/16571630
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
        
class lasso_selector(object):
    """Select indices from a matplotlib collection using `LassoSelector`.
    
    Outline selected points with given color, otherwise don't outline
    
    Based on Lasso Selector Demo
    https://matplotlib.org/3.1.1/gallery/widgets/lasso_selector_demo_sgskip.html
    """
    def __init__(self, ax, collection, color='gold'):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.color = color
        
        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)
        
        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []
    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        
        ec = np.array(["None" for i in range(self.Npts)])
        ec[self.ind] = self.color
        self.collection.set_edgecolors(ec)
        self.canvas.draw_idle()
    def update(self,collection):
        self.collection = collection
        self.xys = collection.get_offsets()
    def disconnect(self):
        self.lasso.disconnect_events()
        ec = np.array(["None" for i in range(self.Npts)])
        self.collection.set_edgecolors(ec)
        self.canvas.draw_idle()

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
    def __init__(self, lc=None, time=None, flux=None, oversample_factor=5, nyquist_factor=1, amp_unit='ppt'):
        self.id = next(self.id_generator)
        self.freq_unit = u.microHertz
        time_unit = u.day
        self.freq_conversion = time_unit.to(1/self.freq_unit)
        
        self.amp_unit = amp_unit
        self.amp_conversion = {'relative':1e0, 'percent':1e2, 'ppt':1e3, 'ppm':1e6, 'mma':1e3}[self.amp_unit]
        
        ### LOG ###
        #Initialize this first and keep track of every important action taken
        self._init_log()
        
        ### TIME SERIES ###
        # Four to keep track of (called lc_nickname)
        # Original (orig), Residuals (resid), 
        # Model (oversampled: model_sampled; and observed: model_observed)
        # Each is lightkurve object
        
        #Store light curve as LightKurve object
        if lc is None and time is None and flux is None:
            raise ValueError('lc or time and flux are required')
        if lc is not None:
            if lk.lightcurve.LightCurve not in type(lc).__mro__:
                raise ValueError('lc must be lightkurve object')
            else:
                self.lc_orig = lc
        else:
            self.lc_orig = lk.LightCurve(time=time, flux=flux)
        
        #Maintain a mask of points to exclude from analysis
        self.mask = np.ones(len(self.lc_orig)) # 1 = include
        self.include = np.where(self.mask)
        
        #Establish frequency sampling
        self.dt = np.median(np.diff(self.lc_orig.time))
        self.set_frequency_sampling(oversample_factor=oversample_factor,nyquist_factor=nyquist_factor)
        
        #Initialize time series widgets and plots
        self._init_timeseries_widgets()
        self.lcfig,self.lcax = plt.subplots(figsize=(7,2),num='Time Series ({:d})'.format(self.id))
        self.lcax.set_xlabel("time")
        self.lcax.set_ylabel("rel. variation")
        self.lcax.set_position([0.13,0.22,0.85,0.76])
        self._lc_colors = {0:"bisque",1:"C0"}
        self.lcplot_data = self.lcax.scatter(self.lc_orig.time,self.lc_orig.flux,marker='o',
                                             s=5, ec='None', lw=1, c=self._lc_colors[1])
        #Define selector for masking points
        self.selector = lasso_selector(self.lcax, self.lcplot_data)
        self.lcfig.canvas.mpl_connect("key_press_event", self._mask_selected_pts)
        
        #Apply time shift to get phases to be well behaved
        self._calc_tshift()
        
        #I think this function nearly computes all the periodograms and timeshift and everything...
        #self._mask_changed()
        
        
        #Also plot the model over the time series
        time_samples = np.arange(np.min(self.lc_orig.time),
                                 np.max(self.lc_orig.time)+self.dt/oversample_factor,self.dt/oversample_factor)
        initmodel = np.zeros(len(time_samples))+np.mean(self.lc_orig.flux)
        self.lc_model_sampled = lk.LightCurve(time=time_samples,flux=initmodel)
        initmodel = np.zeros(len(self.lc_orig.time))+np.mean(self.lc_orig.flux[self.include])
        self.lc_model_observed = lk.LightCurve(time=self.lc_orig.time,flux=initmodel)
        
        self.lcplot_model, = self.lcax.plot(self.lc_model_sampled.time,
                                            self.lc_model_sampled.flux,c='r',lw=1)
        
        #And keep track of residuals time series
        self.lc_resid = self.lc_orig - self.lc_model_observed
        
        
        ### PERIODOGRAM ###
        # Four types for display
        # Original (orig), Residuals (resid), Model (model), and Spectral Window (sw)
        # Each is stored as, e.g., "per_orig", samples at self.freqs
        # Has associated plot _perplot_orig
        # Display toggle widget _perplot_orig_display
        # TODO: Add color picker _perplot_orig_color
        
        #Initialize widgets
        self._init_periodogram_widgets()
        
        #Set up some figs/axes for periodogram plots
        self.perfig,self.perax = plt.subplots(figsize=(7,3),num='Periodogram ({:d})'.format(self.id))
        self.perax.set_xlabel("frequency")
        self.perax.set_ylabel("amplitude ({})".format(self.amp_unit))
        
        #Compute and plot original periodogram
        self.compute_pers(orig=True)
        
        self.perplot_orig, = self.perax.plot(self.per_orig.frequency,self.per_orig.power.value,lw=1,c='tab:gray')
        self.perax.set_xlabel("frequency ({})".format(self.per_orig.frequency.unit.to_string()))
        self.perax.set_ylim(0,1.05*np.nanmax(self.per_orig.power.value))
        self.perax.set_xlim(np.min(self.freqs),np.max(self.freqs))
        self.perax.set_position([0.13,0.22,0.8,0.76])
        
        #Plot periodogram of sampled model and residuals
        self.perplot_model, = self.perax.plot(self.freqs,self.per_model.power.value,lw=1,c='tab:green')
        self.perplot_resid, = self.perax.plot(self.freqs,self.per_resid.power.value,lw=1,c='tab:blue')
        
        #interpolate to residual periodogram when clicks don't snap to peaks
        self.interpls = interp1d(self.freqs,self.per_resid.power.value)
        
        #Compute spectral window
        #TODO: do with lightkurve
        #May not work in Python3!!
        #self.specwin = np.sqrt(LombScargle(self.lc_orig.time*self.freq_conversion, np.ones(self.lc_orig.time.shape),
        #                                   fit_mean=False).power(self.freqs,method = 'fast'))
        #self.perplot_sw, = self.perax.plot(self.freqs,self.specwin,lw=1)
        
        #Create markers for selected peak, adopted signals
        self.marker = self.perax.plot([0],[0],c='k',marker='o')[0]
        self._signal_marker_color = 'green'
        self.signal_markers, = self.perax.plot([],[],marker='D',fillstyle='none',
                                               linestyle='None',
                                               c=self._signal_marker_color,ms=5)
        self._combo_marker_color = 'orange'
        self.combo_markers, = self.perax.plot([],[],marker='D',fillstyle='none',
                                               linestyle='None',
                                               c=self._combo_marker_color,ms=5)
        
        #self._makeperiodsolutionvisible()
        self._display_per_orig()
        self._display_per_resid()
        self._display_per_model()
        self._display_per_sw()
        self._display_per_markers()
        
        self._mark_highest_peak()
        
        #This handles clicking while zooming problems
        #self.perfig.canvas.mpl_connect('button_press_event', self._onperiodogramclick)
        self._press= False
        self._move = False
        self.perfig.canvas.mpl_connect('button_press_event', self._onpress)
        self.perfig.canvas.mpl_connect('button_release_event', self._onrelease)
        self.perfig.canvas.mpl_connect('motion_notify_event', self._onmove)
        
        ### SIGNALS ###
        
        #Hold signal phases, frequencies, and amplitudes in Pandas DF
        self.values = self._initialize_dataframe()
        
        #The interface for interacting with the values DataFrame:
        self._init_signals_qgrid()
        self.signals_qgrid = self._get_qgrid()
        self.signals_qgrid.on('cell_edited', self._qgrid_changed_manually)
        self._init_signals_widgets()
        
        self.log("Pyriod object initialized.")
        #Write lightkurve and periodogram properties to log
        self._log_lc_properties()
        self._log_per_properties()
        
        #Create some decoy figure so users don't accidentally plot over the Pyriod ones
        _ = plt.figure()
    
    ###### Run initialization functions #######
    
    
    def _init_timeseries_widgets(self):
        ### Time Series widget stuff  ###
        self._tsfig_file_location = FileChooser(
            os.getcwd(),
            filename='Pyriod_TimeSeries.png',
            #title='<b>FileChooser example</b>',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=False
        )
                
        self._save_tsfig = widgets.Button(
            description="Save",
            disabled=False,
            #button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Save currently displayed figure to file.',
            icon='save'
        )
        self._save_tsfig.on_click(self._save_tsfig_button_click)
        
        self._reset_mask = widgets.Button(
            description='Reset mask',
            disabled=False,
            tooltip='Include all points in calculations',
            icon='refresh'
        )
        self._reset_mask.on_click(self._clear_mask) 
        
        self._tstype = widgets.Dropdown(
            options=['Original', 'Residuals'],
            value='Original',
            description='Display:',
            disabled=False
        )
        self._tstype.observe(self._update_lc_display)
        
        self._fold = widgets.Checkbox(
            value=False,
            step=self.fres,
            description='Fold time series on frequency?',
        )
        self._fold.observe(self._update_lc_display)
        
        self._fold_on = widgets.FloatText(
            value=1.,
            description='Fold on freq:'
        )
        self._fold_on.observe(self._update_lc_display)
        
        self._select_fold_freq = widgets.Dropdown(
            description='Select from:',
            disabled=False,
        )
        self._select_fold_freq.observe(self._fold_freq_selected,'value')
    
    def _init_periodogram_widgets(self):
        ### Periodogram widget stuff  ###
        self._perfig_file_location = FileChooser(
            os.getcwd(),
            filename='Pyriod_Periodogram.png',
            #title='<b>FileChooser example</b>',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=False
        )
                
        self._save_perfig = widgets.Button(
            description="Save",
            disabled=False,
            #button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Save currently displayed figure to file.',
            icon='save'
        )
        self._save_perfig.on_click(self._save_perfig_button_click)
        
        self._thisfreq = widgets.Text(
            value='',
            placeholder='',
            description='Frequency:',
            disabled=False
        )
        
        
        self._thisamp = widgets.FloatText(
            value=0.001,
            description='Amplitude:',
            disabled=False
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
        
        self._show_per_markers = widgets.Checkbox(
            value=True,
            description='Signal Markers',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self._show_per_markers.observe(self._display_per_markers)
        
        #Check boxes for what to include on periodogram plot
        self._show_per_orig = widgets.Checkbox(
            value=False,
            description='Original',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self._show_per_orig.observe(self._display_per_orig)
        
        self._show_per_resid = widgets.Checkbox(
            value=True,
            description='Residuals',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self._show_per_resid.observe(self._display_per_resid)
        
        self._show_per_model = widgets.Checkbox(
            value=True,
            description='Model',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self._show_per_model.observe(self._display_per_model)
        
        self._show_per_sw = widgets.Checkbox(
            value=False,
            description='Spectral Window (disabled)',
            disabled=True,
            style={'description_width': 'initial'}
        )
        self._show_per_sw.observe(self._display_per_sw)
    
    def _init_signals_qgrid(self):
        #Set some options for how the qgrid of values should be displayed
        self._gridoptions = {
                # SlickGrid options
                'fullWidthRows': True,
                'syncColumnCellResize': True,
                'forceFitColumns': False,
                'defaultColumnWidth': 65,  #control col width (all the same)
                'rowHeight': 28,
                'enableColumnReorder': True,
                'enableTextSelectionOnCells': True,
                'editable': True,
                'autoEdit': True, #double-click not required!
                'explicitInitialization': True,
                
    
                # Qgrid options
                'maxVisibleRows': 8,
                'minVisibleRows': 8,
                'sortable': True,
                'filterable': False,  #Not useful here
                'highlightSelectedCell': False,
                'highlightSelectedRow': True
               }
        
        self._column_definitions = {"include":  {'width': 60, 'toolTip': "include signal in model fit?"},
                                    "freq":      {'width': 112, 'toolTip': "mode frequency"},
                                    "fixfreq":  {'width': 60, 'toolTip': "fix frequency during fit?"},
                                    "freqerr":  {'width': 100, 'toolTip': "uncertainty on frequency", 'editable': False},
                                    "amp":       {'width': 112, 'toolTip': "mode amplitude"},
                                    "fixamp":   {'width': 60, 'toolTip': "fix amplitude during fit?"},
                                    "amperr":  {'width': 100, 'toolTip': "uncertainty on amplitude", 'editable': False},
                                    "phase":     {'width': 112, 'toolTip': "mode phase"},
                                    "fixphase": {'width': 65, 'toolTip': "fix phase during fit?"},
                                    "phaseerr":  {'width': 100, 'toolTip': "uncertainty on phase", 'editable': False}}
    
    def _init_signals_widgets(self):
        ### Time Series widget stuff  ###
        self._refit = widgets.Button(
            description="Refine fit",
            disabled=False,
            #button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Refine fit of signals to time series',
            icon='refresh'
        )
        self._refit.on_click(self.fit_model)
        
        self._delete = widgets.Button(
            description='Delete selected',
            disabled=False,
            button_style='danger', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Delete selected rows.',
            icon='trash'
        )
        self._delete.on_click(self._delete_selected)
        
        self._signals_file_location = FileChooser(
            os.getcwd(),
            filename='Pyriod_solution.csv',
            #title='<b>FileChooser example</b>',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=False
        )
                
        self._save = widgets.Button(
            description="Save",
            disabled=False,
            #button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Save solution to csv file.',
            icon='save'
        )
        self._save.on_click(self._save_button_click)
        
        self._load = widgets.Button(
            description="Load",
            disabled=False,
            #button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Load solution from csv file.',
            icon='load'
        )
        self._load.on_click(self._load_button_click)
    
        
    def _init_log(self):
        #To log messages, use self.log() function
        self.logger = logging.getLogger('basic_logger')
        self.logger.setLevel(logging.DEBUG)
        self.log_capture_string = StringIO()
        ch = logging.StreamHandler(self.log_capture_string)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self._log = widgets.HTML(
            value='Log',
            placeholder='Log',
            description='Log:',
            layout={'height': '250px','width': '950px'}
        )
        
        self._log_file_location = FileChooser(
            os.getcwd(),
            filename='Pyriod_log.txt',
            #title='<b>FileChooser example</b>',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=False
        )
                
        self._save_log = widgets.Button(
            description="Save",
            disabled=False,
            tooltip='Save log to csv file.',
            icon='save'
        )
        self._save_log.on_click(self._save_log_button_click)
        
        self._overwrite = widgets.Checkbox(
            value=False,
            description='Overwrite?'
        )
        
        self._logbox = VBox([widgets.Box([self._log]),
                             HBox([self._save_log,self._log_file_location,self._overwrite],layout={'height': '40px'})],
                            layout={'height': '300px','width': '950px'})
        
    #Function for logging messages
    def log(self,message,level='info'):
        logdict = {
            'debug': self.logger.debug,
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical
            }
        logdict[level](message+'<br>')
        self._update_log()
        
    def _log_lc_properties(self):
        try:
            with Capturing() as output:
                self.lc_orig.show_properties()
            info = re.sub(' +', ' ', str("".join([e+' |\n' for e in output[2:]])))
            self.log("Time Series properties:"+info)
        except Exception:
            pass
        
    def _log_per_properties(self):
        try:
            with Capturing() as output:
                self.per_resid.show_properties()
            info = re.sub(' +', ' ', str("".join([e+' |\n' for e in output[3:]])))
            self.log("Periodogram properties:"+info)
        except Exception:
            pass
    
    def _next_signal_index(self,n=1):
        #Get next n unused independent signal indices
        inds = []
        i=0
        while len(inds) < n:
            if not "f{}".format(i) in self.values.index:
                inds.append("f{}".format(i))
            i+=1
        return inds
    
    def set_frequency_sampling(self, frequency = None, oversample_factor=5, nyquist_factor=1,
                                minfreq = None, maxfreq = None):
        """Set the frequency sampling for periodograms.
        
        Parameters
        ----------
        frequency : TYPE, optional
            Explicit set of frequencies to compute periodogram at. The default is None.
        oversample_factor : FLOAT, optional
            How many time more densely than the natural frequency resolution of 1/duration to sample frequencies. The default is 5.
        nyquist_factor : FLOAT, optional
            How many time beyond the approximate Nyquist frequency to sample periodograms. The default is 1. Overridden by maxfreq, if provided.
        minfreq : FLOAT
            Minimum frequency of range to use. The defualt is 1/duration.
        maxfreq : FLOAT
            Maximum frequency of range to use. The defualt is based off of nyquist_factor.

        Returns
        -------
        None.
        """
        #Frequency resolution
        self.fres = 1./np.ptp(self.lc_orig.time)
        self.oversample_factor = oversample_factor
        self.nyquist_factor = nyquist_factor
        #Compute Nyquist frequency (approximate for unevenly sampled data)
        self.nyquist = 1./(2.*self.dt*self.freq_conversion)
        #Sample the following frequencies:
        if frequency is not None:
            self.log('Using user supplied frequency sampling: ' + 
                     '{} samples between frequency {} and {}'.format(len(frequency),np.min(frequency),np.max(frequency)))
            self.freqs = frequency
        else:
            if minfreq is None:
                minfreq = self.fres
            if maxfreq is None:
                maxfreq = self.nyquist*self.nyquist_factor+self.fres/self.oversample_factor
            self.freqs = np.arange(minfreq,maxfreq,self.fres/self.oversample_factor)
        return
        
    #Functions for interacting with model fit
    def _make_all_iter(self, variables):
        """Return iterables of given variables
    
        Parameters
        ----------
        variables : list or tuple
            Set of values to returned as iterables if necessary.
            Each must have length 1 or length of first variable
        Returns
        -------
        tuple of iterable versions of input variables
        """
        #wrap all single values or strings in lists
        variables = [[v] if (not hasattr(v, '__iter__')) or (type(v) == str) else v for v in variables]
        #Get length of first variable
        nvals = len(variables[0])
        #check that all lengths are the same or 1
        if not all([len(l) in [nvals,1] for l in variables]):
            raise ValueError("Arguments passed have inconsistent lengths.")
        else:
            variables = [[v[0] for i in range(nvals)] if (len(v) == 1) else v for v in variables]
        return tuple(variables)
    
    def add_signal(self, freq, amp=None, phase=None, fixfreq=False, 
                   fixamp=False, fixphase=False, include=True, index=None):
        freq,amp,phase,fixfreq,fixamp,fixphase,include,index = self._make_all_iter([freq,amp,phase,fixfreq,fixamp,fixphase,include,index])
        colnames = ["freq","fixfreq","amp","fixamp","phase","fixphase","include"]
        newvalues = [nv for nv in [freq,fixfreq,amp,fixamp,phase,fixphase,include]]
        dictvals = dict(zip(colnames,newvalues))
        for i in range(len(freq)):
            if dictvals["amp"][i] is None:
                dictvals["amp"][i] = 1.
            else:
                dictvals["amp"][i] /= self.amp_conversion
            if dictvals["phase"][i] is None:
                dictvals["phase"][i] = 0.5
        #Replace all None indices with next available
        noneindex = np.where([ind is None for ind in index])[0]
        newindices = self._next_signal_index(n=len(noneindex))
        for i in range(len(noneindex)):
            index[noneindex[i]] = newindices[i]
        #Check that all indices are unique and none already used
        if (len(index) != len(set(index))) or any([ind in self.values.index for ind in index]):
            raise ValueError("Duplicate indices provided.")
        toappend = pd.DataFrame(dictvals,columns=self.columns,index=index)
        self.values = self.values.append(toappend,sort=False)
        self._update_freq_dropdown() #For folding time series
        displayframe = self.values.copy()#[self.columns[:-1]]
        displayframe["amp"] = displayframe["amp"] * self.amp_conversion
        self.signals_qgrid.df = displayframe.combine_first(self.signals_qgrid.df)#[self.columns[:-1]] #Update displayed values
        #self.signals_qgrid.df.columns = self.columns[:-1]
        self._update_signal_markers()
        self.log("Signal {} added to model with frequency {} and amplitude {}.".format(index,freq,amp))
    
    def _valid_combo(self,combostr):
        parts = re.split('\+|\-|\*|\/',combostr.replace(" ", "").lower())
        allvalid = np.all([(part in self.values.index) or part.replace('.','',1).isdigit() for part in parts])
        return allvalid and (len(parts) > 1)
        
    #operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    #             ast.Div: op.truediv,ast.USub: op.neg}
    def add_combination(self, combostr, amp=None, phase=None, fixfreq=False, 
                   fixamp=False, fixphase=False, include=True, index=None):
        combostr,amp,phase,fixfreq,fixamp,fixphase,include,index = self._make_all_iter([combostr,amp,phase,fixfreq,fixamp,fixphase,include,index])
        freq = np.zeros(len(combostr))
        for i in range(len(combostr)):
            combostr[i] = combostr[i].replace(" ", "").lower()
            #evaluate combostring:
            #replace keys with values
            parts = re.split('\+|\-|\*|\/',combostr[i].replace(" ", "").lower())
            keys = set([part for part in parts if part in self.values.index])
            exploded = re.split('(\+|\-|\*|\/)',combostr[i].replace(" ", "").lower())
            expression = "".join([str(self.values.loc[val,'freq']) if val in keys else val for val in exploded])
            freq[i] = eval(expression)
            if amp[i] == None:
                amp[i] = self.interpls(subfreq(freq[i],self.nyquist)[0])
        self.add_signal(list(freq),amp,phase,fixfreq,fixamp,fixphase,include,index=combostr)
        
    def fit_model(self, *args):
        """ 
        Update model to include current signals from DataFrame.
        
        Improve fit once with all freqs and amps fixed, then allow to vary.
        """
        if np.sum(self.values.include.values) == 0:
            self.log("No signals to fit.",level='warning')
            return # nothing to fit
        
        #Set up lmfit model for fitting
        signals = {} #empty dict to be populated
        params = Parameters()
        
        #handle combination frequencies differently
        isindep = lambda key: key[1:].isdigit()
        cnum = 0
        
        prefixmap = {}
        
        #first with frequencies and amplitudes fixed
        #for those specified to be included in the model
        for prefix in self.values.index[self.values.include]:
            #prefix = 'f{}'.format(i+1)
            #freqkeys.append(prefix)
                if isindep(prefix):
                    signals[prefix] = Model(sin,prefix=prefix)
                    params.update(signals[prefix].make_params())
                    params[prefix+'freq'].set(self.freq_conversion*self.values.freq[prefix], vary=False)
                    params[prefix+'amp'].set(self.values.amp[prefix], vary=False) # vary=~self.values.fixamp[prefix]
                    #Correct phase for tdiff
                    thisphase = self.values.phase[prefix] - self.tshift*self.freq_conversion*self.values.freq[prefix]
                    params[prefix+'phase'].set(thisphase, vary=~self.values.fixphase[prefix])
                    prefixmap[prefix] = prefix
                else: #combination
                    useprefix = 'c{}'.format(cnum)
                    signals[useprefix] = Model(sin,prefix=useprefix)
                    params.update(signals[useprefix].make_params())
                    parts = re.split('\+|\-|\*|\/',prefix)
                    keys = set([part for part in parts if part in self.values.index])
                    exploded = re.split('(\+|\-|\*|\/)',prefix)
                    expression = "".join([val+'freq' if val in keys else val for val in exploded])
                    params[useprefix+'freq'].set(expr=expression)
                    params[useprefix+'amp'].set(self.values.amp[prefix], vary=False) #vary=~self.values.fixamp[prefix]
                    thisphase = self.values.phase[prefix] - self.tshift*self.freq_conversion*self.values.freq[prefix]
                    params[useprefix+'phase'].set(thisphase, vary=~self.values.fixphase[prefix])
                    prefixmap[prefix] = useprefix
                    cnum+=1
        
        #model is sum of sines
        model = np.sum([signals[prefixmap[prefix]] for prefix in self.values.index[self.values.include]])
        
        #compute fixed-frequency fit
        result = model.fit(self.lc_orig.flux[self.include]-np.mean(self.lc_orig.flux[self.include]), params, x=self.lc_orig.time[self.include]+self.tshift)
        
        #refine, allowing freq to vary (unless fixed by user)
        params = result.params
        
        for prefix in self.values.index[self.values.include]:
            params[prefixmap[prefix]+'amp'].set(vary=~self.values.fixamp[prefix])
            if isindep(prefix):
                params[prefixmap[prefix]+'freq'].set(vary=~self.values.fixfreq[prefix])
                #params[prefixmap[prefix]+'amp'].set(result.params[prefixmap[prefix]+'amp'].value)
                params[prefixmap[prefix]+'phase'].set(result.params[prefixmap[prefix]+'phase'].value)
        
        result = model.fit(self.lc_orig.flux[self.include]-np.mean(self.lc_orig.flux[self.include]), params, x=self.lc_orig.time[self.include]+self.tshift)
        self.log("Fit refined.")  
        self.log("Fit properties:"+result.fit_report())
        
        self._update_values_from_fit(result.params,prefixmap)
        
        self._mark_highest_peak()#Mark highest peak in residuals
        
    def _update_values_from_fit(self,params,prefixmap):
        #update dataframe of params with new values from fit
        #also rectify and negative amplitudes or phases outside [0,1)
        #isindep = lambda key: key[1:].isdigit()
        #cnum = 0
        for prefix in self.values.index[self.values.include]:
            self.values.loc[prefix,'freq'] = float(params[prefixmap[prefix]+'freq'].value/self.freq_conversion)
            self.values.loc[prefix,'freqerr'] = float(params[prefixmap[prefix]+'freq'].stderr/self.freq_conversion)
            self.values.loc[prefix,'amp'] = params[prefixmap[prefix]+'amp'].value
            self.values.loc[prefix,'amperr'] = float(params[prefixmap[prefix]+'amp'].stderr)
            self.values.loc[prefix,'phase'] = params[prefixmap[prefix]+'phase'].value
            self.values.loc[prefix,'phaseerr'] = float(params[prefixmap[prefix]+'phase'].stderr)
            #rectify
            if self.values.loc[prefix,'amp'] < 0:
                self.values.loc[prefix,'amp'] *= -1.
                self.values.loc[prefix,'phase'] -= 0.5
            #Reference phase to t0
            self.values.loc[prefix,'phase'] += self.tshift*self.values.loc[prefix,'freq']*self.freq_conversion
            self.values.loc[prefix,'phase'] %= 1.
        self._update_freq_dropdown()
        
        #update qgrid
        #self.signals_qgrid.df = self._convert_values_to_qgrid().combine_first(self.signals_qgrid.df)[self.columns[:-1]]
        self.signals_qgrid.df = self._convert_values_to_qgrid().combine_first(self.signals_qgrid.get_changed_df())#[self.columns[:-1]]
        #self.signals_qgrid.df = self._convert_values_to_qgrid()[self.columns[:-1]]
        
        self._update_values_from_qgrid()
    
    def _convert_values_to_qgrid(self):
        tempdf = self.values.copy()#[self.columns[:-1]]
        tempdf["amp"] *= self.amp_conversion
        tempdf["amperr"] *= self.amp_conversion
        return tempdf
    
    def _convert_qgrid_to_values(self):
        tempdf = self.signals_qgrid.get_changed_df().copy().astype(dtype=dict(zip(self.columns,self.dtypes)))
        tempdf["amp"] /= self.amp_conversion
        tempdf["amperr"] /= self.amp_conversion
        return tempdf
    
    def _update_values_from_qgrid(self):# *args
        self.values = self._convert_qgrid_to_values()
        
        self._update_lcs()
        self._update_lc_display()
        self._update_signal_markers()
        self.compute_pers()
        self._update_per_plots()
        self._update_freq_dropdown()
        
    def sample_model(self,time):
        flux = np.zeros(len(time))
        for prefix in self.values.index[self.values.include]:
            freq = float(self.values.loc[prefix,'freq'])
            amp = float(self.values.loc[prefix,'amp'])
            phase = float(self.values.loc[prefix,'phase'])
            flux += sin(time,freq*self.freq_conversion,amp,phase)
        return flux
    
    def _update_lcs(self):
        #Update time series models
        meanflux = np.mean(self.lc_orig.flux[self.include])
        self.lc_model_sampled.flux = meanflux + self.sample_model(self.lc_model_sampled.time)
        #Observed is at all original times (apply mask before calculations)
        self.lc_model_observed.flux = meanflux + self.sample_model(self.lc_orig.time)
        self.lc_resid = self.lc_orig - self.lc_model_observed
    
    def _qgrid_changed_manually(self, *args):
        #note: args has information about what changed if needed
        newdf = self.signals_qgrid.get_changed_df()
        #olddf = self.signals_qgrid.df
        olddf = self._convert_values_to_qgrid()
        logmessage = "Signals table changed manually.\n"
        changedcols = []
        for key in newdf.index.values:
            if key in olddf.index.values:
                changes = newdf.loc[key][olddf.loc[key] != newdf.loc[key]]
                if len(changes) > 0:
                    logmessage += "Values changed for {}:\n".format(key)
                for change in changes.index:
                    logmessage += " - {} -> {}\n".format(change,changes[change])
                    changedcols.append(change)
            else:
                logmessage += "New row in solution table: {}\n".format(key)
                for col in newdf.loc[key]:
                    logmessage += " - {} -> {}\n".format(change,changes[change])
        self.log(logmessage)
        #self.signals_qgrid.df = self.signals_qgrid.get_changed_df().combine_first(self.signals_qgrid.df)[self.columns[:-1]]
        #self.signals_qgrid.df.columns = self.columns[:-1]
        
        #Update plots only if signal values (not what is fixed) changed
        if all([colname[:3] == 'fix' for colname in changedcols]):
            self.values = self._convert_qgrid_to_values()
        else:
            self._update_values_from_qgrid()
    
    columns = ['include','freq','fixfreq','freqerr',
               'amp','fixamp','amperr',
               'phase','fixphase','phaseerr']
    dtypes = ['bool','object','bool','float',
              'float','bool','float',
              'float','bool','float']
    
    def delete_rows(self,indices):
        self.log("Deleted signals {}".format([sig for sig in indices]))
        self.values = self.values.drop(indices)
        #self.signals_qgrid.df = self.signals_qgrid.df.drop(indices)
        self.signals_qgrid.df = self.signals_qgrid.get_changed_df().drop(indices)
    
    def _delete_selected(self, *args):
        self.delete_rows(self.signals_qgrid.get_selected_df().index)
        self._update_values_from_qgrid()

    def _initialize_dataframe(self):
        df = pd.DataFrame(columns=self.columns).astype(dtype=dict(zip(self.columns,self.dtypes)))
        return df
    
    
    #Stuff for folding the light curve on a certain frequency
    def _fold_freq_selected(self,value):
        if value['new'] is not None:
            self._fold_on.value = value['new']
        
    def _update_freq_dropdown(self):
        labels = [self.values.index[i] + ': {:.8f} '.format(self.values.freq[i]) + self.per_orig.frequency.unit.to_string() for i in range(len(self.values))]
        currentind = self._select_fold_freq.index
        if currentind == None:
            currentind = 0
        if len(labels) == 0:
            self._select_fold_freq.options = [None]
        else:
            self._select_fold_freq.options = zip(labels, self.values.freq.values)
            self._select_fold_freq.index = currentind
        
        
    ########## Set up *SIGNALS* widget using qgrid ##############
    
    
        
    
    def _get_qgrid(self):
        display_df = self.values.copy()
        display_df["amp"] *= self.amp_conversion
        display_df["amperr"] *= self.amp_conversion
        return qgrid.show_grid(display_df, show_toolbar=False, precision = 9,
                               grid_options=self._gridoptions,
                               column_definitions=self._column_definitions)
    
    #add staged signal to frequency solution
    def _add_staged_signal(self, *args):
        #Is this a valid numeric frequency?
        if self._thisfreq.value.replace('.','',1).isdigit():
            self.add_signal(float(self._thisfreq.value),self._thisamp.value)
        elif self._valid_combo(self._thisfreq.value):
            self.add_combination(self._thisfreq.value)
        else:
            self.log("Staged frequency has invalid format: {}".format(self._thisfreq.value),"error")
        
    #change type of time series being displayed
    def _update_lc_display(self, *args):
        self._display_lc(residuals = (self._tstype.value == "Residuals"))
        
    def _update_signal_markers(self):
        subnyquistfreqs = subfreq(self.values['freq'][self.values.include].astype('float'),self.nyquist)
        amps = self.values['amp'].values[self.values.include]*self.amp_conversion
        indep = np.array([key[1:].isdigit() for key in self.values.index[self.values.include]])
        
        self.signal_markers.set_data(subnyquistfreqs[np.where(indep)],amps[np.where(indep)])
        self.combo_markers.set_data(subnyquistfreqs[np.where(~indep)],amps[np.where(~indep)])
        self.perfig.canvas.draw()
        
    def _display_lc(self,residuals=False):
        lc = self.lc_orig
        if residuals:
            lc = self.lc_resid
            self.lcplot_model.set_ydata(np.zeros(len(self.lc_model_sampled.flux)))
        else:
            self.lcplot_model.set_ydata(self.lc_model_sampled.flux)
        #rescale y to better match data
        ymin = np.min(lc.flux[self.include])
        ymax = np.max(lc.flux[self.include])
        self.lcax.set_ylim(ymin-0.05*(ymax-ymin),ymax+0.05*(ymax-ymin))
        #fold if requested
        if self._fold.value:
            xdata=lc.time*self._fold_on.value*self.freq_conversion % 1.
            self.lcplot_data.set_offsets(np.dstack((xdata,lc.flux))[0])
            self.lcplot_model.set_alpha(0)
            self.lcax.set_xlim(-0.01,1.01)
        else:
            self.lcplot_data.set_offsets(np.dstack((lc.time,lc.flux))[0])
            self.lcplot_model.set_alpha(1)
            tspan = np.ptp(lc.time)
            self.lcax.set_xlim(np.min(lc.time)-0.01*tspan,np.max(lc.time)+0.01*tspan)
        self.selector.update(self.lcplot_data)
        self.lcfig.canvas.draw()
    
    def _mask_selected_pts(self,event):
        if event.key in ["backspace","delete"] and (len(self.selector.ind) > 0):
            #ranges =[]
            #for k,g in groupby(enumerate(np.sort(self.selector.ind)),lambda x:x[0]-x[1]):
            #    group = (map(itemgetter(1),g))
            #    group = list(map(int,group))
            #    ranges.append((group[0],group[-1]))
            #self.log("Masking {} points in index ranges: {}".format(len(self.selector.ind),ranges))
            self.log("Masking {} selected points.")
            self.mask[self.selector.ind] = 0
            self._mask_changed()
            
    def _clear_mask(self,b):
        self.log("Restoring all masked points.")
        self.mask[:] = 1
        self._mask_changed()
            
    def _mask_changed(self):
        self.include = np.where(self.mask)
        self.selector.ind = []
        self.lcplot_data.set_facecolors([self._lc_colors[m] for m in self.mask])
        self.lcplot_data.set_edgecolors("None")
        self._update_lcs()
        self._update_lc_display()
        #self.lcfig.canvas.draw()
        self._calc_tshift()
        
        self.compute_pers(orig=True)
        self._update_per_plots()
    
    def _calc_tshift(self,tshift=None):
        if tshift is None:
            self.tshift = -np.mean(self.lc_orig[self.include].time)
        else:
            self.tshift = tshift
    
    def compute_pers(self, orig=False):
        if orig:
            self.per_orig = self.lc_orig[self.include].to_periodogram(normalization='amplitude',freq_unit=self.freq_unit,
                                                                      frequency=self.freqs)*self.amp_conversion
        self.per_model = self.lc_model_observed[self.include].to_periodogram(normalization='amplitude',freq_unit=self.freq_unit,
                                                                             frequency=self.freqs)*self.amp_conversion
        self.per_resid = self.lc_resid[self.include].to_periodogram(normalization='amplitude',freq_unit=self.freq_unit,
                                                                    frequency=self.freqs)*self.amp_conversion
        self.interpls = interp1d(self.freqs,self.per_resid.power.value)
        self._log_per_properties()
        
    def _update_per_plots(self):
        self.perplot_orig.set_ydata(self.per_orig.power.value)
        self.perplot_model.set_ydata(self.per_model.power.value)
        self.perplot_resid.set_ydata(self.per_resid.power.value)
        self.perfig.canvas.draw()
   
    def _display_per_orig(self, *args):
        if self._show_per_orig.value:
            self.perplot_orig.set_alpha(1)
        else:
            self.perplot_orig.set_alpha(0)
        self.perfig.canvas.draw()
        
    def _display_per_resid(self, *args):
        if self._show_per_resid.value:
            self.perplot_resid.set_alpha(1)
        else:
            self.perplot_resid.set_alpha(0)
        self.perfig.canvas.draw()
        
    def _display_per_model(self, *args):
        if self._show_per_model.value:
            self.perplot_model.set_alpha(1)
        else:
            self.perplot_model.set_alpha(0)
        self.perfig.canvas.draw()
        
    def _display_per_sw(self, *args):
        #if self._show_per_sw.value:
        #    self.perplot_sw.set_alpha(1)
        #else:
        #    self.perplot_sw.set_alpha(0)
        #self.perfig.canvas.draw()
        pass #temporary
        
    def _display_per_markers(self, *args):
        if self._show_per_markers.value:
            self.signal_markers.set_alpha(1)
            self.combo_markers.set_alpha(1)
        else:
            self.signal_markers.set_alpha(0)
            self.combo_markers.set_alpha(0)
        self.perfig.canvas.draw()
    
    def _onperiodogramclick(self,event):
        if self._snaptopeak.value:
            #click within either frequency resolution or 1% of displayed range
            #TODO: make this work with log frequency too
            tolerance = np.max([self.fres,0.01*np.diff(self.perax.get_xlim())])
            
            nearby = np.argwhere((self.freqs >= event.xdata - tolerance) & 
                                 (self.freqs <= event.xdata + tolerance))
            ydata = self.perplot_resid.get_ydata()
            highestind = np.nanargmax(ydata[nearby]) + nearby[0]
            self._update_marker(self.freqs[highestind],ydata[highestind])
        else:
            self._update_marker(event.xdata,self.interpls(event.xdata))
            
    def TimeSeries(self):
        options = widgets.Accordion(children=[VBox([self._tstype,self._fold,self._fold_on,self._select_fold_freq,self._reset_mask])],selected_index=None)
        options.set_title(0, 'options')
        savefig = HBox([self._save_tsfig,self._tsfig_file_location])
        return VBox([self.lcfig.canvas,savefig,options])
       
    def Periodogram(self):
        options = widgets.Accordion(children=[VBox([self._snaptopeak,self._show_per_markers,
                        self._show_per_orig,self._show_per_resid,
                        self._show_per_model,self._show_per_sw])],selected_index=None)
        options.set_title(0, 'options')
        savefig = HBox([self._save_perfig,self._perfig_file_location])
        periodogram = VBox([HBox([self._thisfreq,self._thisamp]),
                        HBox([self._addtosol,self._refit]),
                        self.perfig.canvas,
                        savefig,
                        options])
        return periodogram
        
        
    
    #This one shows all the tabs
    def Pyriod(self):
        tstab = self.TimeSeries()
        pertab = self.Periodogram()
        signalstab = self.Signals()
        logtab = self.Log()
        tabs = widgets.Tab(children=[tstab,pertab,signalstab,logtab])
        tabs.set_title(0, 'Time Series')
        tabs.set_title(1, 'Periodogram')
        tabs.set_title(2, 'Signals')
        tabs.set_title(3, 'Log')
        return tabs
        
    def _update_marker(self,x,y):
        try:
            self._thisfreq.value = str(x[0])
        except:
            self._thisfreq.value = str(x)
        self._thisamp.value =  y
        self.marker.set_data([x],[y])
        self.perfig.canvas.draw()
        self.perfig.canvas.flush_events()
    
    def _mark_highest_peak(self):    
        self._update_marker(self.freqs[np.nanargmax(self.per_resid.power.value)],
                           np.nanmax(self.per_resid.power.value))
        
    def _onclick(self,event):
        self._onperiodogramclick(event)
    def _onpress(self,event):
        self._press=True
    def _onmove(self,event):
        if self._press:
            self._move=True
    def _onrelease(self,event):
        if self._press and not self._move:
            self._onclick(event)
        self._press=False; self._move=False

    def Signals(self):
        return VBox([HBox([self._refit,self._thisfreq,self._thisamp,self._addtosol,self._delete]),
                self.signals_qgrid,
                HBox([self._save,self._load,self._signals_file_location])])
        
    def Log(self):
        return self._logbox
    
    def _update_log(self):
        self._log.value = self.log_capture_string.getvalue()
        
    def save_solution(self,filename='Pyriod_solution.csv'):
        self.log("Writing signal solution to "+os.path.abspath(filename))
        #self.signals_qgrid.df.to_csv(filename,index_label='label')
        self._convert_values_to_qgrid().to_csv(filename,index_label='label')
        
    def _save_button_click(self, *args):
        self.save_solution(filename=self._signals_file_location.selected)
    
    def load_solution(self,filename='Pyriod_solution.csv'):
        if os.path.exists(filename):
            loaddf = pd.read_csv(filename,index_col='label')
            loaddf.index = loaddf.index.rename(None)
            logmessage = "Loading signal solution from "+os.path.abspath(filename)+".<br />"
            logmessage += loaddf.to_string().replace('\n','<br />')
            self.log(logmessage)
            self.signals_qgrid.df = loaddf
            self._update_values_from_qgrid()
        else:
            self.log("Failed to load "+os.path.abspath(filename)+". File not found.<br />",level='error')
        
    def _load_button_click(self, *args):
        self.load_solution(filename=self._signals_file_location.selected)
        
    def _save_log_button_click(self, *args):
        self.save_log(self._log_file_location.selected,self._overwrite.value)
        
    def save_log(self,filename,overwrite=False):
        logmessage = "Writing log to "+os.path.abspath(filename)
        if overwrite:
            logmessage += ", overwriting."
        self.log(logmessage)
        soup = BeautifulSoup(self._log.value, features="lxml")
        mode = {True:"w+",False:"a+"}[overwrite]
        f = open(filename, mode)
        f.write(soup.get_text().replace('|', ''))
        f.close()
        
    def save_tsfig(self,filename='Pyriod_TimeSeries.png',**kwargs):
        self.lcfig.savefig(filename,**kwargs)
        
    def _save_tsfig_button_click(self, *args):
        self.save_tsfig(self._tsfig_file_location.selected)
        
    def save_perfig(self,filename='Pyriod_Periodogram.png',**kwargs):
        self.perfig.savefig(filename,**kwargs)
        
    def _save_perfig_button_click(self, *args):
        self.save_perfig(self._perfig_file_location.selected)
        
    def close_figures(self):
        """Close all figures beloning to this class.
        
        Warning: interacting with figures will no longer work.
        """
        plt.close(self.lcfig)
        plt.close(self.perfig)