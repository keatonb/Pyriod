#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This is Pyriod, a Python package for selecting and fitting sinusoidal signals 
to astronomical time series data.

Written by Keaton Bell

For more, see https://github.com/keatonb/Pyriod

---------------------

The following code was copied from stackoverflow:

Distinguish clicks with drag motions from ImportanceOfBeingErnest
https://stackoverflow.com/a/48452190

Capturing print output from kindall
https://stackoverflow.com/a/16571630

---------------------

Major overhaul to make compatible with and to utilize new functionality of
LightKurve 2.0. Namely, lc objects are now extenctions of AstroPy TimeSeries 
objects, with user definable columns. Here are the main changes:
    - require lightkurve.LightCurve object for initialization
    - store residuals and the sampled model as columns of the lc parameter

Below here are just some author's notes to keep track of style decisions.

Names of periodogram plot objects:
    per_orig
    per_resid
    per_model
    per_sw
    per_markers

Names of timeseries:
    lc has columns "time","flux","mask","model","resid"
    lc_model_sampled (evenly oversampled through gaps)

Names of plots are:
    lcplot_data,lcplot_model (different nicknames)
    perplot_orig (same nicknames)
    _perplot_orig_display toggle
    
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
from astropy.timeseries import TimeSeries
import lightkurve as lk
from lmfit import Model, Parameters
from bs4 import BeautifulSoup
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
    lc : lightkurve.LightCurve
        Time Series to analyze, with flux and time columns
    
	
    """
    id_generator = itertools.count(0)
    def __init__(self, lc, amp_unit='ppt', freq_unit='muHz', 
                 time_unit = 'day', **kwargs):
        #Generate unique Pyriod instance ID
        self.id = next(self.id_generator)
        
        ### LOG ###
        #Initialize the log first to keep track of every important action taken
        self._init_log()
        
        #Work out the units, in a function
        self._set_units(amp_unit=amp_unit,freq_unit=freq_unit,time_unit=time_unit)
        
        #Create status widget to indicate when calculations are running
        self._status = widgets.HTML(value="")
        
        self.fit_result = None #will be replaced as we do fits
       
        ### TIME SERIES ###
        # Stored as lightkurve.LightCurve object
        # "flux" column is original data
        # "resid" is residuals
        # "model_sampled" is the model sampled as the data
        # "masked" marks points that are not to be included in the analysis
        # A separate LightCurve object holds a model that is better sampled
        
        #Input must be Lightkurve LightCurve type
        if not issubclass(type(lc),lk.LightCurve):
            raise TypeError('lc must be a lightkurve.LightCurve object.')
        self.lc = lc.copy() #copy so we don't modify original
        
        #Maintain a mask of points to exclude from analysis
        self.lc["mask"] = np.ones(len(self.lc)) # 1 = include
        self.include = np.where(self.lc["mask"])
        
        #Establish frequency sampling
        self.set_frequency_sampling(**kwargs)
        
        #Initialize time series widgets and plots
        self._init_timeseries_widgets()
        self.lcfig,self.lcax = plt.subplots(figsize=(7,2),num='Time Series ({:d})'.format(self.id))
        self.lcax.set_position([0.13,0.22,0.85,0.76])
        self._lc_colors = {0:"bisque",1:"C0"}
        self.lcplot_data = self.lcax.scatter(self.lc.time.value/self.time_to_days,self.lc.flux,marker='o',
                                             s=5, ec='None', lw=1, c=self._lc_colors[1])
        #Define selector for masking points
        self.selector = lasso_selector(self.lcax, self.lcplot_data)
        self.lcfig.canvas.mpl_connect("key_press_event", self._mask_selected_pts)
        
        #Apply time shift to get phases to be well behaved
        self._calc_tshift()
        
        #I think this function nearly computes all the periodograms and timeshift and everything...
        #self._mask_changed()
        
        #Also plot the model over the time series
        dt = np.median(np.diff(self.lc.time.value/self.time_to_days))
        tspan = (np.max(self.lc.time.value) - np.min(self.lc.time.value))/self.time_to_days
        osample = 5
        nsamples = round(osample*tspan/dt)
        time_samples = TimeSeries(time_start=np.min(lc.time),
                                  time_delta= dt * u.day / osample,
                                  n_samples=nsamples).time
        initmodel = np.zeros(nsamples)+np.mean(self.lc.flux.value)
        self.lc_model_sampled = lk.LightCurve(time=time_samples,flux=initmodel)
        
        #And store version sampled as the data as lc column
        initmodel = np.zeros(len(self.lc))+np.mean(self.lc.flux[self.include])
        self.lc["model"] = initmodel
        
        self.lcplot_model, = self.lcax.plot(self.lc_model_sampled.time.value/self.time_to_days,
                                            self.lc_model_sampled.flux,c='r',lw=1)
        
        #And keep track of residuals time series
        self.lc["resid"] = self.lc["flux"] - self.lc["model"]
        
        
        ### PERIODOGRAM ###
        # Four types for display
        # Original (orig), Residuals (resid), Model (model), 
        # and Spectral Window (sw; TODO)
        # Each is stored as, e.g., "per_orig", samples at self.freqs
        # Has associated plot _perplot_orig
        # Display toggle widget _perplot_orig_display
        # TODO: Add color picker _perplot_orig_color
        
        #Initialize widgets
        self._init_periodogram_widgets()
        
        #Set up some figs/axes for periodogram plots
        self.perfig,self.perax = plt.subplots(figsize=(7,3),num='Periodogram ({:d})'.format(self.id))
        
        #Compute and plot original periodogram
        self.compute_pers(orig=True)
        
        self.perplot_orig, = self.perax.plot(self.per_orig.frequency,self.per_orig.power.value,lw=1,c='tab:gray')
        self.perax.set_ylim(0,1.05*np.nanmax(self.per_orig.power.value))
        self.perax.set_xlim(np.min(self.freqs),np.max(self.freqs))
        self.perax.set_position([0.13,0.22,0.8,0.76])
        
        #Plot periodogram of sampled model and residuals
        self.perplot_model, = self.perax.plot(self.freqs,self.per_model.power.value,lw=1,c='tab:green')
        self.perplot_resid, = self.perax.plot(self.freqs,self.per_resid.power.value,lw=1,c='tab:blue')
        
        #interpolate to residual periodogram when clicks don't snap to peaks
        self.interpls = interp1d(self.freqs,self.per_resid.power.value)
        
        #Compute spectral window
        #TODO: do with DFT
        #self.specwin = np.sqrt(LombScargle(self.lc.time*self.freq_conversion, np.ones(self.lc.time.shape),
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
        
        #Set axis labels
        self._set_plot_labels()
        
        ### SIGNALS ###
        
        #Hold signal phases, frequencies, and amplitudes in Pandas DF
        self.stagedvalues = self._initialize_dataframe()
        self.fitvalues = self.stagedvalues.copy().drop('brute',1)
        
        
        #The interface for interacting with the values DataFrame:
        self._init_signals_qgrid()
        self.signals_qgrid = self._get_qgrid()
        self.signals_qgrid.on('cell_edited', self._qgrid_changed_manually)
        self._init_signals_widgets()
        
        self.log("Pyriod object initialized.")
        #Write lightkurve and periodogram properties to log
        self._log_lc_properties()
        self._log_per_properties()
        
        #Keep track of whether the displayed data reflect the most recent fit
        self.uptodate = True
        
        #Create some decoy figure so users don't accidentally plot over the Pyriod ones
        _ = plt.figure()
    
    ###### initialization functions #######
    
    def _set_units(self,amp_unit=None,freq_unit=None,time_unit=None):
        """Configure units to user's preferences.

        Parameters
        ----------
        amp_unit : str, optional
        freq_unit : str, optional
        time_unit : str, optional
        """
        if amp_unit is not None:
            self.amp_unit = amp_unit
            self.amp_conversion = {'relative':1e0, 'percent':1e2, 'ppt':1e3, 'ppm':1e6, 'mma':1e3}[self.amp_unit.lower()]
            self.log(f'Amplitude unit set to {amp_unit} (factor of {self.amp_conversion}).')
        if freq_unit is not None:
            muHz = u.microHertz
            perday = (1/u.day).unit
            self.freq_unit = {'muhz':muHz, 'uhz':muHz, 'microhertz':muHz, 
                              '1/d':perday, '1/day':perday, 'day':perday,
                              'days':perday, 'd':perday}[freq_unit.lower()]
            self.freq_label = {perday:"1/day",muHz:"muHz"}[self.freq_unit]
            self.log(f'Frequency unit set to {self.freq_label}.')
        if time_unit is not None:
            self.time_unit =  {'d':u.day, 'day':u.day, 'days':u.day, 'jd':u.day, 
                               'bjd':u.day, 'ut':u.day, 'utc':u.day, 
                               's':u.s, 'sec':u.s, 'secs':u.s, 'seconds':u.s, 
                               'h':u.h, 'hr':u.h, 'hour':u.h, 'hours':u.h, 
                               'm':u.min, 'min':u.min, 'mins':u.min, 'minute':u.min, 
                               'minutes':u.min, 'yr':u.yr, 'year':u.yr, 'yrs':u.yr,
                               'years':u.yr, 'epoch':u.yr}[time_unit.lower()]
            self.log(f'Input time unit set to {self.time_unit.to_string()}.')
        self.freq_conversion = self.time_unit.to(1/self.freq_unit)
        self.time_to_days = self.time_unit.to(u.day)
        
    def _set_plot_labels(self):
        #Light curve
        self.lcax.set_xlabel(f"time ({self.time_unit.to_string()})")
        self.lcax.set_ylabel("rel. variation")
        #Periodogram
        self.perax.set_ylabel(f"amplitude ({self.amp_unit})")
        self.perax.set_xlabel(f"frequency ({self.freq_label})")
        
    
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
        
        '''
        self._show_per_sw = widgets.Checkbox(
            value=False,
            description='Spectral Window (disabled)',
            disabled=True,
            style={'description_width': 'initial'}
        )
        self._show_per_sw.observe(self._display_per_sw)
        '''
    
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
                                    "freq":      {'width': 100, 'toolTip': "mode frequency"},
                                    "fixfreq":  {'width': 60, 'toolTip': "fix frequency during fit?"},
                                    "freqerr":  {'width': 90, 'toolTip': "uncertainty on frequency", 'editable': False},
                                    "amp":       {'width': 100, 'toolTip': "mode amplitude"},
                                    "fixamp":   {'width': 60, 'toolTip': "fix amplitude during fit?"},
                                    "amperr":  {'width': 90, 'toolTip': "uncertainty on amplitude", 'editable': False},
                                    "phase":     {'width': 100, 'toolTip': "mode phase"},
                                    "brute": {'width': 65, 'toolTip': "brute sample phase first during fit?"},
                                    "fixphase": {'width': 65, 'toolTip': "fix phase during fit?"},
                                    "phaseerr":  {'width': 90, 'toolTip': "uncertainty on phase", 'editable': False}}
    
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
        
        self.log(f'Initiating Pyriod instance {self.id}.')
        
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
        #If lc has metadata, put it in the log
        for key, value in self.lc.meta:
            self.log(f"{key}: {value}")
        
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
            if not "f{}".format(i) in self.stagedvalues.index:
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
            Minimum frequency of range to use. The default is 1/duration.
        maxfreq : FLOAT
            Maximum frequency of range to use. The default is based off of nyquist_factor.

        Returns
        -------
        None.
        """
        #Frequency resolution
        self.fres = self.time_to_days/(self.freq_conversion*np.ptp(self.lc.time.value))
        self.oversample_factor = oversample_factor
        self.nyquist_factor = nyquist_factor
        #Compute Nyquist frequency (approximate for unevenly sampled data)
        dt = np.median(np.diff(self.lc.time.value/self.time_to_days))
        self.nyquist = 1/(2.*dt*self.freq_conversion)
        #Sample the following frequencies:
        if frequency is not None:
            self.log('Using user supplied frequency sampling: ' + 
                     '{} samples between frequency {} and {} {}'.format(len(frequency),
                                                                        np.min(frequency),
                                                                        np.max(frequency),
                                                                        self.freq_label))
            self.freqs = frequency
        else:
            if minfreq is None:
                minfreq = self.fres
            if maxfreq is None:
                maxfreq = self.nyquist*self.nyquist_factor+0.9*self.fres/self.oversample_factor
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
                   fixamp=False, fixphase=False, include=True, brute=False, index=None):
        freq,amp,phase,fixfreq,fixamp,fixphase,include,brute,index = self._make_all_iter([freq,amp,phase,fixfreq,fixamp,fixphase,include,brute,index])
        colnames = ["freq","fixfreq","amp","fixamp","phase","brute","fixphase","include"]
        newvalues = [nv for nv in [freq,fixfreq,amp,fixamp,phase,brute,fixphase,include]]
        dictvals = dict(zip(colnames,newvalues))
        for i in range(len(freq)):
            if dictvals["amp"][i] is None:
                dictvals["amp"][i] = 1.
            else:
                dictvals["amp"][i] /= self.amp_conversion
        #Replace all None indices with next available
        noneindex = np.where([ind is None for ind in index])[0]
        newindices = self._next_signal_index(n=len(noneindex))
        for i in range(len(noneindex)):
            index[noneindex[i]] = newindices[i]
        #Check that all indices are unique and none already used
        if (len(index) != len(set(index))) or any([ind in self.stagedvalues.index for ind in index]):
            raise ValueError("Duplicate indices provided.")
        toappend = pd.DataFrame(dictvals,columns=self.columns,index=index)
        toappend = toappend.astype(dtype=dict(zip(self.columns,self.dtypes)))
        self.stagedvalues = self.stagedvalues.append(toappend,sort=False)
        self._update_freq_dropdown() #For folding time series
        displayframe = self.stagedvalues.copy()
        displayframe["amp"] = displayframe["amp"] * self.amp_conversion
        self.signals_qgrid.df = displayframe.combine_first(self.signals_qgrid.df) #Update displayed values
        self._update_signal_markers()
        self.log("Signal {} added to model with frequency {} and amplitude {}.".format(index,freq,amp))
        self._model_current(False)
    
    def _valid_combo(self,combostr):
        parts = re.split('\+|\-|\*|\/',combostr.replace(" ", "").lower())
        allvalid = np.all([(part in self.stagedvalues.index) or part.replace('.','',1).isdigit() for part in parts])
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
            keys = set([part for part in parts if part in self.stagedvalues.index])
            exploded = re.split('(\+|\-|\*|\/)',combostr[i].replace(" ", "").lower())
            expression = "".join([str(self.stagedvalues.loc[val,'freq']) if val in keys else val for val in exploded])
            freq[i] = eval(expression)
            if amp[i] == None:
                amp[i] = self.interpls(subfreq(freq[i],self.nyquist)[0])
        self.add_signal(list(freq),amp,phase,fixfreq,fixamp,fixphase,include,index=combostr)
        
    def _brute_phase_est(self,freq,amp,brute_step=0.1):
        """
        Fit single sinusoid to residuals, sampling crudely in phase.
        
        Returns rough phase estimate.
        """
        model = Model(sin)
        params = model.make_params()
        params['freq'].set(self.freq_conversion*freq, vary=False)
        params['amp'].set(amp, vary=False) 
        params['phase'].set(0.5, vary=True, min=0, max=1, brute_step=brute_step)
        result = model.fit(self.lc["resid"][self.include]-np.mean(self.lc["resid"][self.include]), 
                               params, x=(self.lc.time.value[self.include]+self.tshift)/self.time_to_days, 
                               method='brute')
        return result.params['phase'].value
        
    def fit_model(self, *args):
        """ 
        Update model to include current signals from DataFrame.
        
        Improve fit once with all freqs and amps fixed, then allow to vary.
        """
        #Indicate that a calculation is running
        self._update_status()
        
        if np.sum(self.stagedvalues.include.values) == 0:
            self.log("No signals to fit.",level='warning')
            self.fitvalues = self._initialize_dataframe().drop('brute',1) #Empty
        else: #Fit a model
            #Set up lmfit model for fitting
            signals = {} #empty dict to be populated
            params = Parameters()
            
            #handle combination frequencies differently
            isindep = lambda key: key[1:].isdigit()
            cnum = 0
            
            prefixmap = {}
            
            #Set up model to fit (for included signals only)
            #Estimate phase for new signals with _brute_phase_est
            for prefix in self.stagedvalues.index[self.stagedvalues.include]:
                if isindep(prefix):
                    signals[prefix] = Model(sin,prefix=prefix)
                    params.update(signals[prefix].make_params())
                    params[prefix+'freq'].set(self.freq_conversion*self.stagedvalues.freq[prefix],
                                              vary=~self.stagedvalues.fixfreq[prefix])
                    params[prefix+'amp'].set(self.stagedvalues.amp[prefix], 
                                             vary=~self.stagedvalues.fixamp[prefix])
                    #Correct phase for tdiff
                    thisphase = self.stagedvalues.phase[prefix] - self.tshift*self.freq_conversion*self.stagedvalues.freq[prefix]
                    if np.isnan(thisphase) or self.stagedvalues.brute[prefix]: #if new signal to fit
                        thisphase = self._brute_phase_est(self.stagedvalues.freq[prefix], self.stagedvalues.amp[prefix])
                    
                    params[prefix+'phase'].set(thisphase, min=-np.inf, max=np.inf,
                                               vary=~self.stagedvalues.fixphase[prefix])
                    prefixmap[prefix] = prefix
                else: #combination
                    useprefix = 'c{}'.format(cnum)
                    signals[useprefix] = Model(sin,prefix=useprefix)
                    params.update(signals[useprefix].make_params())
                    parts = re.split('\+|\-|\*|\/',prefix)
                    keys = set([part for part in parts if part in self.stagedvalues.index])
                    exploded = re.split('(\+|\-|\*|\/)',prefix)
                    expression = "".join([val+'freq' if val in keys else val for val in exploded])
                    params[useprefix+'freq'].set(expr=expression)
                    params[useprefix+'amp'].set(self.stagedvalues.amp[prefix], vary=~self.stagedvalues.fixamp[prefix])
                    #Correct phase for tdiff
                    thisphase = self.stagedvalues.phase[prefix] - self.tshift*self.freq_conversion*self.stagedvalues.freq[prefix]
                    if np.isnan(thisphase): #if new signal to fit
                        thisphase = self._brute_phase_est(self.stagedvalues.freq[prefix], self.stagedvalues.amp[prefix])
                    params[useprefix+'phase'].set(thisphase, min=-np.inf, max=np.inf,
                                                  vary=~self.stagedvalues.fixphase[prefix])
                    prefixmap[prefix] = useprefix
                    cnum+=1
                    
            #model is sum of sines
            model = np.sum([signals[prefixmap[prefix]] for prefix in self.stagedvalues.index[self.stagedvalues.include]])
            
            self.fit_result = model.fit(self.lc.flux[self.include]-np.mean(self.lc.flux[self.include]), 
                                        params, x=self.lc.time.value[self.include]/self.time_to_days+self.tshift)
            
            self.log("Fit refined.")  
            self.log("Fit properties:"+self.fit_result.fit_report())
            self._update_values_from_fit(self.fit_result.params,prefixmap)
        
        self._update_lcs()
        self._update_lc_display()
        self._update_signal_markers()
        self.compute_pers()
        self._update_per_plots()
        self._mark_highest_peak()#Mark highest peak in residuals
        
        self._update_status(False)#Calculation done
        self._model_current(True)#fitvalues and stagedvalues are the same
        
    def _update_values_from_fit(self,params,prefixmap):
        #update dataframe of params with new values from fit
        #also rectify and negative amplitudes or phases outside [0,1)
        #isindep = lambda key: key[1:].isdigit()
        #cnum = 0
        self.fitvalues = self.stagedvalues.astype(dtype=dict(zip(self.columns,self.dtypes))).drop('brute',1)
        for prefix in self.stagedvalues.index[self.stagedvalues.include]:
            self.fitvalues.loc[prefix,'freq'] = float(params[prefixmap[prefix]+'freq'].value/self.freq_conversion)
            self.fitvalues.loc[prefix,'freqerr'] = float(params[prefixmap[prefix]+'freq'].stderr/self.freq_conversion)
            self.fitvalues.loc[prefix,'amp'] = params[prefixmap[prefix]+'amp'].value
            self.fitvalues.loc[prefix,'amperr'] = float(params[prefixmap[prefix]+'amp'].stderr)
            self.fitvalues.loc[prefix,'phase'] = params[prefixmap[prefix]+'phase'].value
            self.fitvalues.loc[prefix,'phaseerr'] = float(params[prefixmap[prefix]+'phase'].stderr)
            #rectify
            if self.fitvalues.loc[prefix,'amp'] < 0:
                self.fitvalues.loc[prefix,'amp'] *= -1.
                self.fitvalues.loc[prefix,'phase'] -= 0.5
            #Reference phase to t0
            self.fitvalues.loc[prefix,'phase'] += self.tshift*self.fitvalues.loc[prefix,'freq']*self.freq_conversion
            self.fitvalues.loc[prefix,'phase'] %= 1.
        
        self._update_freq_dropdown()
        
        #update qgrid
        self.signals_qgrid.df = self._convert_fitvalues_to_qgrid().combine_first(self.signals_qgrid.get_changed_df())
        self._update_stagedvalues_from_qgrid()
    
    def _convert_fitvalues_to_qgrid(self):
        tempdf = self.fitvalues.copy()
        tempdf["brute"] = False
        tempdf = tempdf.astype(dtype=dict(zip(self.columns,self.dtypes)))[self.columns]
        tempdf["amp"] *= self.amp_conversion
        tempdf["amperr"] *= self.amp_conversion
        return tempdf
    
    def _convert_qgrid_to_stagedvalues(self):
        tempdf = self.signals_qgrid.get_changed_df().copy().astype(dtype=dict(zip(self.columns,self.dtypes)))
        tempdf["amp"] /= self.amp_conversion
        tempdf["amperr"] /= self.amp_conversion
        return tempdf
    
    def _update_stagedvalues_from_qgrid(self):# *args
        self.stagedvalues = self._convert_qgrid_to_stagedvalues()
        
        #self._update_lcs()
        #self._update_lc_display()
        self._update_signal_markers()
        #self.compute_pers()
        #self._update_per_plots()
        #self._update_freq_dropdown()
        
    def _model_current(self,current = True):
        """update self.uptodate to whether displayed date reflect model fit
        
        and color refine fit button accordingly
        """
        if current:
            self._refit.button_style = ''
        else:
            self._refit.button_style = 'warning'
        self.uptodate = current
        
    def sample_model(self,time):
        flux = np.zeros(len(time))
        for prefix in self.fitvalues.index[self.fitvalues.include]:
            freq = float(self.fitvalues.loc[prefix,'freq'])
            amp = float(self.fitvalues.loc[prefix,'amp'])
            phase = float(self.fitvalues.loc[prefix,'phase'])
            flux += sin(time,freq*self.freq_conversion,amp,phase)
        return flux
    
    def _update_lcs(self):
        #Update time series models
        meanflux = np.mean(self.lc.flux.value[self.include])
        self.lc_model_sampled.flux = meanflux + self.sample_model(self.lc_model_sampled.time.value/self.time_to_days)
        #Observed is at all original times (apply mask before calculations)
        self.lc["model"] = meanflux + self.sample_model(self.lc.time.value/self.time_to_days)
        self.lc["resid"] = self.lc.flux - self.lc["model"]
    
    def _qgrid_changed_manually(self, *args):
        #note: args has information about what changed if needed
        newdf = self.signals_qgrid.get_changed_df()
        olddf = self.signals_qgrid.df
        logmessage = "Signals table changed manually.\n"
        changedcols = []
        for key in newdf.index.values:
            if key in olddf.index.values:
                changes = newdf.loc[key][(olddf.loc[key] != newdf.loc[key])]
                changes = changes.dropna() #remove nans
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
        self._update_stagedvalues_from_qgrid()
    
    columns = ['include','freq','fixfreq','freqerr',
               'amp','fixamp','amperr',
               'phase','brute','fixphase','phaseerr']
    dtypes = ['bool','float','bool','float',
              'float','bool','float',
              'float','bool','bool','float']
    
    def delete_rows(self,indices):
        self.log("Deleted signals {}".format([sig for sig in indices]))
        self.stagedvalues = self.stagedvalues.drop(indices)
        self.signals_qgrid.df = self.signals_qgrid.get_changed_df().drop(indices)
    
    def _delete_selected(self, *args):
        self.delete_rows(self.signals_qgrid.get_selected_df().index)
        #Also delete associated combination frequencies
        isindep = lambda key: key[1:].isdigit()
        for key in self.signals_qgrid.df.index:
            if not isindep(key) and not self._valid_combo(key):
                self.delete_rows(key)
        #Don't update model at this point, as staged signals may crash model
        #self._update_values_from_qgrid()
        #self._mark_highest_peak()

    def _initialize_dataframe(self):
        df = pd.DataFrame(columns=self.columns).astype(dtype=dict(zip(self.columns,self.dtypes)))
        return df
    
    
    #Stuff for folding the light curve on a certain frequency
    def _fold_freq_selected(self,value):
        if value['new'] is not None:
            self._fold_on.value = value['new']
        
    def _update_freq_dropdown(self):
        labels = [self.fitvalues.index[i] + ': {:.8f} '.format(self.fitvalues.freq[i]) + self.per_orig.frequency.unit.to_string() for i in range(len(self.fitvalues))]
        currentind = self._select_fold_freq.index
        if currentind == None:
            currentind = 0
        if len(labels) == 0:
            self._select_fold_freq.options = [None]
        else:
            self._select_fold_freq.options = zip(labels, self.fitvalues.freq.values)
            self._select_fold_freq.index = currentind
        
        
    ########## Set up *SIGNALS* widget using qgrid ##############
    
    def _get_qgrid(self):
        display_df = self.stagedvalues.copy()
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
        subnyquistfreqs = subfreq(self.stagedvalues['freq'][self.stagedvalues.include].astype('float'),self.nyquist)
        amps = self.stagedvalues['amp'].values[self.stagedvalues.include]*self.amp_conversion
        indep = np.array([key[1:].isdigit() for key in self.stagedvalues.index[self.stagedvalues.include]])
        
        self.signal_markers.set_data(subnyquistfreqs[np.where(indep)],amps[np.where(indep)])
        if len(indep) > 0:
            self.combo_markers.set_data(subnyquistfreqs[np.where(~indep)],amps[np.where(~indep)])
        else:
            self.combo_markers.set_data([],[])
        self.perfig.canvas.draw()
        
    def _display_lc(self,residuals=False):
        lc = self.lc.copy()
        if residuals:
            lc = self.lc.select_flux("resid").copy()
            self.lcplot_model.set_ydata(np.zeros(len(self.lc_model_sampled.flux)))
        else:
            self.lcplot_model.set_ydata(self.lc_model_sampled.flux)
        #lc.time.value = lc.time.value/self.time_to_days
        #rescale y to better match data
        ymin = np.min(lc.flux[self.include])
        ymax = np.max(lc.flux[self.include])
        self.lcax.set_ylim(ymin-0.05*(ymax-ymin),ymax+0.05*(ymax-ymin))
        #fold if requested
        if self._fold.value:
            xdata=lc.time.value*self._fold_on.value*self.freq_conversion % 1.
            self.lcplot_data.set_offsets(np.dstack((xdata,lc.flux))[0])
            #self.lcplot_model.set_alpha(0)
            self.lcax.set_xlim(-0.01,1.01)
        else:
            self.lcplot_data.set_offsets(np.dstack((lc.time.value,lc.flux))[0])
            #self.lcplot_model.set_alpha(1)
            tspan = np.ptp(lc.time.value)
            self.lcax.set_xlim(np.min(lc.time.value)-0.01*tspan,np.max(lc.time.value)+0.01*tspan)
        self.selector.update(self.lcplot_data)
        self.lcfig.canvas.draw()
    
    def _mask_selected_pts(self,event):
        self.log(event.key,"debug")
        if event.key in ["backspace","delete"] and (len(self.selector.ind) > 0):
            #ranges =[]
            #for k,g in groupby(enumerate(np.sort(self.selector.ind)),lambda x:x[0]-x[1]):
            #    group = (map(itemgetter(1),g))
            #    group = list(map(int,group))
            #    ranges.append((group[0],group[-1]))
            #self.log("Masking {} points in index ranges: {}".format(len(self.selector.ind),ranges))
            self.log("Masking {} selected points.")
            self.lc["mask"][self.selector.ind] = 0
            self._mask_changed()
            
    def _clear_mask(self,b):
        self.log("Restoring all masked points.")
        self.lc["mask"][:] = 1
        self._mask_changed()
            
    def _mask_changed(self):
        self.include = np.where(self.lc["mask"])
        self.selector.ind = []
        self.lcplot_data.set_facecolors([self._lc_colors[m] for m in self.lc["mask"]])
        self.lcplot_data.set_edgecolors("None")
        self._update_lcs()
        self._update_lc_display()
        #self.lcfig.canvas.draw()
        self._calc_tshift()
        
        self.compute_pers(orig=True)
        self._update_per_plots()
    
    def _calc_tshift(self,tshift=None):
        if tshift is None:
            self.tshift = -np.mean(self.lc[self.include].time.value)
        else:
            self.tshift = tshift
    
    def compute_pers(self, orig=False):
        self._update_status() #indicate running calculation
        if orig:
            self.per_orig = self.lc[self.include].to_periodogram(normalization='amplitude',freq_unit=self.freq_unit,
                                                                 frequency=self.freqs)*self.amp_conversion
        self.per_model = self.lc.select_flux("model")[self.include].to_periodogram(normalization='amplitude',
                                                                                   freq_unit=self.freq_unit,
                                                                                   frequency=self.freqs)*self.amp_conversion
        self.per_resid = self.lc.select_flux("resid")[self.include].to_periodogram(normalization='amplitude',
                                                                                   freq_unit=self.freq_unit,
                                                                                   frequency=self.freqs)*self.amp_conversion
        self.interpls = interp1d(self.freqs,self.per_resid.power.value)
        self._log_per_properties()
        self._update_status(False)#Calculation complete
        
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
        return VBox([self._status,self.lcfig.canvas,savefig,options])
       
    def Periodogram(self):
        options = widgets.Accordion(children=[VBox([self._snaptopeak,self._show_per_markers,
                        self._show_per_orig,self._show_per_resid,
                        self._show_per_model])],selected_index=None)
        options.set_title(0, 'options')
        savefig = HBox([self._save_perfig,self._perfig_file_location])
        periodogram = VBox([self._status,
                            HBox([self._thisfreq,self._thisamp]),
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
        return VBox([self._status,
                     HBox([self._refit,self._thisfreq,self._thisamp,self._addtosol,self._delete]),
                     self.signals_qgrid,
                     HBox([self._save,self._load,self._signals_file_location])])
        
    def Log(self):
        return self._logbox
    
    def _update_log(self):
        self._log.value = self.log_capture_string.getvalue()
        
    def save_solution(self,filename='Pyriod_solution.csv'):
        self.log("Writing signal solution to "+os.path.abspath(filename))
        #self.signals_qgrid.df.to_csv(filename,index_label='label')
        self._convert_fitvalues_to_qgrid().to_csv(filename,index_label='label')
        
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
            self._update_stagedvalues_from_qgrid()
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
    
    def _update_status(self,calculating=True):
        if calculating:
            self._status.value = "<center><b><big><font color='red'>UPDATING CALCULATIONS...</big></b></center>"
        else:
            self._status.value = ""
            
    def close_figures(self):
        """Close all figures beloning to this class.
        
        Warning: interacting with figures will no longer work.
        """
        plt.close(self.lcfig)
        plt.close(self.perfig)