#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:01:01 2021

Widgets to let Pyriod users specify settings for computing periodogram

@author: keatonb
"""
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Box
from traitlets import Float, Int, Bool, HasTraits, observe
import astropy.units as u
import numpy as np

frequnits = ['microHz', '1 / day']
frequnitdict = {'microHz':u.microHertz, '1 / day':(1/u.day).unit}

muHz = u.microHertz
perday = (1/u.day).unit

ampunits = ['relative', 'percent', 'ppt', 'ppm']
ampunitdict = {'relative':1e0, 'percent':1e2, 'ppt':1e3, 'ppm':1e6}

_freq_widget_layout = widgets.Layout(width='200px')

class _freq_settings(HasTraits):
    nyqest = Float()
    fres = Float(.01)
    minfreq = Float()
    maxfreq = Float(100)
    freqstep = Float(0.1)
    osample = Float(5)
    nyqfactor = Float(1)
    nsamples = Int()
    frequnitind = Int(0)
    ampunitind = Int(0)
    valid = Bool(True)
    
    #ititialize
    def __init__(self,time, target_function, oversample_factor=5, nyquist_factor=1,
                 minfreq = None, maxfreq = None):
        self._init_widgets()
        self.target_function = target_function
        self.nyqfactor = nyquist_factor
        self.osample = oversample_factor
        self.from_time_array(time)
        if minfreq is None:
            self.minfreq = self.fres
        else:
            self.minfreq = minfreq
        if maxfreq is None:
            self.maxfreq = self.nyqest*nyquist_factor
        else:
            self.maxfreq = maxfreq
        self._init_widgets()
        self.recalc_frequencies()
        self.to_perday = frequnitdict[frequnits[self.frequnitind]].to(perday)
        
    
    #recalculate frequency array for changed settings
    def recalc_frequencies(self):
        self.freqs = np.arange(self.minfreq,self.maxfreq,self.freqstep)
        self.nsamples = len(self.freqs)
        self.valid = self.nsamples > 0
        if self.nsamples > 0:
            self._apply.disabled=False
            self._apply.icon='check'
        else:
            self._apply.disabled=True
            self._apply.icon='times'
    
    #Update nyquist and fres for time array
    def from_time_array(self, time):
        """Update nyquist estimate and fres from time array
        
        time is an astropy Time object from a lightkurve.LightCurve
        """
        self.fres = (1/np.ptp(time)).to(frequnitdict[frequnits[self.frequnitind]]).value
        #Compute Nyquist frequency (approximate for unevenly sampled data)
        dt = np.median(np.diff(time.value))*u.day
        self.nyqest = (1/(2.*dt)).to(frequnitdict[frequnits[self.frequnitind]]).value
        
    #changing fres changes freq step
    @observe('fres')
    def _observe_fres(self, change):
        self.freqstep = change['new']/self.osample
            
    #changing freqstep changes osample
    @observe('freqstep')
    def _observe_freqstep(self, change):
        try:
            self.osample = self.fres/change['new']
            self.recalc_frequencies()
        except:
            self.osample = self._observe_freqstep(change)
        
    #changing osample changes freqstep
    @observe('osample')
    def _observe_osample(self, change):
        try:
            self.freqstep = self.fres/change['new']
        except:
            self._observe_osample(change)
    
    #changing maxfreq changes nyqfactor
    @observe('maxfreq')
    def _observe_maxfreq(self, change):
        self.nyqfactor = change['new']/self.nyqest
        self.recalc_frequencies()
        
    #changing minfreq changes sampled frequencies
    @observe('minfreq')
    def _observe_minfreq(self,change):
        self.recalc_frequencies()
        
    #changing nyqfactor changes maxfreq
    @observe('nyqfactor')
    def _observe_nyqfactor(self, change):
        self.maxfreq = change['new']*self.nyqest
    
    #changing nyquist changes maxfreq (nyquistfactor stays)
    @observe('nyqest')
    def _observe_nyqest(self, change):
        self.maxfreq = self.nyqfactor*change['new']
    
    #changing frequnitind converts frequencies
    @observe('frequnitind')
    def _observe_frequnitind(self, change):
        try:#Sometimes this fails on a second change, so just try it again
            conversion=frequnitdict[frequnits[change['old']]].to(frequnitdict[frequnits[change['new']]])
            self.minfreq = self.minfreq*conversion
            self.nyqest = self.nyqest*conversion
            self.fres = self.fres*conversion
            self.freqs = self.freqs*conversion
            self.to_perday = frequnitdict[frequnits[change['new']]].to(perday)
        except:
            self._observe_frequnitind(change)
    
    def apply(self,b):
        if self.valid:
            self.target_function()
    
    #Initialize and connect ipywidgets!
    def _init_widgets(self):
        self._minfreq = widgets.BoundedFloatText(
            value=self.minfreq,
            description='min freq:',
            min=0,
            max=1e10,
            step=0.01,
            layout= _freq_widget_layout
        )
        widgets.link((self._minfreq, 'value'), (self, 'minfreq'))
        
        self._maxfreq = widgets.BoundedFloatText(
            value=self.maxfreq,
            min=1,
            max=1e10,
            description='max freq:',
            layout= _freq_widget_layout
        )
        widgets.link((self._maxfreq, 'value'), (self, 'maxfreq'))
        
        self._stepsize = widgets.BoundedFloatText(
            value=self.freqstep,
            step=1e-4,
            min=1e-4,
            max=1e10,
            description='freq step:',
            layout= _freq_widget_layout
        )
        widgets.link((self._stepsize , 'value'), (self, 'freqstep'))

        self._osample = widgets.BoundedFloatText(
            value=self.osample,
            step=1,
            min=1,
            max=1e7,
            description='oversample:',
            layout= _freq_widget_layout
        )
        widgets.link((self._osample , 'value'), (self, 'osample'))

        self._nyqfactor = widgets.BoundedFloatText(
            value=self.nyqfactor,
            step=0.01,
            min=0.01,
            description='nyq. factor:',
            layout= _freq_widget_layout
        )
        widgets.link((self._nyqfactor , 'value'), (self, 'nyqfactor'))

        self._frequnit = widgets.Dropdown(
            options=frequnits,
            description='freq unit:',
            index=self.frequnitind,
            layout= _freq_widget_layout
        )
        widgets.link((self._frequnit , 'index'), (self, 'frequnitind'))
        
        self._ampunit = widgets.Dropdown(
            options=ampunits,
            description='amp unit:',
            index=self.ampunitind,
            layout= _freq_widget_layout
        )
        widgets.link((self._ampunit , 'index'), (self, 'ampunitind'))

        self._nyquist = widgets.FloatText(
            value=self.nyqest,
            description=r'nyquist ~',
            disabled=True,
            layout= _freq_widget_layout    
        )
        widgets.link((self._nyquist , 'value'), (self, 'nyqest'))

        self._fres = widgets.BoundedFloatText(
            value=self.fres,
            min=0.000001,
            description=r'freq. res.',
            disabled=True,
            layout= _freq_widget_layout    
        )
        widgets.link((self._fres , 'value'), (self, 'fres'))

        self._nsamples = widgets.IntText(
            value=self.nsamples,
            description='# samples',
            disabled=True,
            layout= _freq_widget_layout
        )
        widgets.link((self._nsamples , 'value'), (self, 'nsamples'))

        self._apply = widgets.Button(
            button_style='success',
            description='Apply',
            tooltip_description="Recalculate periodograms with these settings.",
            icon='check',
            layout=widgets.Layout(width='112px',margin='0px 0px 0px 90px')
        )
        self._apply.on_click(self.apply)
        
        self._valid = widgets.Valid(
            value=True,
            readout="invalid"
        )
        widgets.link((self._valid , 'value'), (self, 'valid'))
        
        left_box = VBox([self._minfreq,self._maxfreq,self._stepsize,self._osample,self._frequnit])
        right_box = VBox([self._nyqfactor,self._nyquist,self._fres,
                          HBox([self._nsamples,self._valid]),self._ampunit])
        self.widgets = VBox([HBox([left_box, right_box]),
                                       Box([self._apply])])
        
    def show(self):
        return self.widgets