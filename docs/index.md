# Pyriod

Python implementation of basic period detection and fitting routines for astronomical time series.  

Pyriod aims to eventually provide most of the functionality of the wonderful program [Period04](https://www.univie.ac.at/tops/Period04/) in the Python environment for the modern astronomy workflow, as well as other useful features.

The interactive GUI runs in a Jupyter Notebook (not compatible with JupyterLab).

This is a serious work in progress with many planned improvements. Please be patient, but also feel free to request new features on the issues page on [GitHub](https://github.com/keatonb/Pyriod/issues).

Pyriod is written and maintained by [Keaton Bell](http://staff.washington.edu/keatonb/).

## Installation

To install, use

```pip install Pyriod```

or download the latest (possibly unstable) version from [GitHub](https://github.com/keatonb/Pyriod) and run

```python setup.py install```.

This code uses [Qgrid](https://github.com/quantopian/qgrid) to interactively display the frequency solution. 

To display the widgets as part of the Pyriod GUI, you will need to first enable the following Jupyter notebook extensions in the terminal:
```
jupyter nbextension enable --py --sys-prefix qgrid
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix ipympl
```

## Dependencies

Pyriod should work with Python 2.7 or 3. 

You can view a list of package requirements [here](https://github.com/keatonb/Pyriod/blob/master/requirements.txt). Beyond more standard packages of astronomy research and plotting (Astropy, Pandas, ipywidgets, etc.), these few specific projects really made Pyriod possible, and are worth looking into if you want to understand or extend the code:

 * [lightkurve](https://docs.lightkurve.org/) -- A friendly package for Kepler & TESS time series analysis in Python.
 * [Lmfit](https://lmfit.github.io/lmfit-py/) -- Non-linear least-squares minimization and curve-fitting for Python.
 * [qgrid](https://github.com/quantopian/qgrid) -- An interactive grid for sorting, filtering, and editing DataFrames in Jupyter notebooks.
