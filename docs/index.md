# Pyriod

Python implementation of basic pre-whitening period detection and fitting routines for astronomical time series data.  

Pyriod aims to provide most of the functionality of the wonderful program [Period04](https://www.period04.net/) in the Python environment for the modern astronomy workflow.

The interactive GUI runs in a Jupyter Notebook environment.

Please report any new bugs or request new features on the issues page on [GitHub](https://github.com/keatonb/Pyriod/issues).

Pyriod is written and maintained by [Keaton Bell](https://keaton.commons.gc.cuny.edu/).

Additional documentation and tutorials are under development.

## Installation

To install, use

```pip install Pyriod```

or download the latest (possibly unstable) version from [GitHub](https://github.com/keatonb/Pyriod) and run

```python setup.py install```.

## Dependencies

Pyriod requires Python 3.

You can view a list of package requirements [here](https://github.com/keatonb/Pyriod/blob/master/requirements.txt). Beyond fairly standard packages of astronomy research and plotting (Astropy, Pandas, ipywidgets, etc.), these few specific projects really made Pyriod possible, and are worth looking into if you want to understand or extend the code:

 * [lightkurve](https://docs.lightkurve.org/) -- A friendly package for Kepler & TESS time series analysis in Python.
 * [Lmfit](https://lmfit.github.io/lmfit-py/) -- Non-linear least-squares minimization and curve-fitting for Python.
 * [qgrid](https://github.com/quantopian/qgrid) -- An interactive grid for sorting, filtering, and editing DataFrames in Jupyter notebooks. Huge thanks to [qgridnext](https://github.com/zhihanyue/qgridnext/) for keeping the functionality alive and compatible with modern python/dependencies.

## Acknowledgement

This material is based upon work supported by the National
Science Foundation under Grant No. AST-2406917.

