# Pyriod

Python implementation of basic period detection and fitting routines for astronomical time series.

To install, use

```pip install Pyriod```

or download the latest (possibly unstable) version from GitHub and run

```python setup.py install```

Additional documentation is available at [pyriod.readthedocs.io](https://pyriod.readthedocs.io).

This code uses [Qgrid](https://github.com/quantopian/qgrid) and other Jupyter widgets to interactively display the frequency solution.  

To display the Pyriod GUI, you will need to first enable the following Jupyter notebook extensions in the terminal:
```
jupyter nbextension enable --py --sys-prefix qgrid
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix ipympl
```
Pyriod does not work in JupyterLab.

This is a serious work in progress with many planned improvements.  Please be patient, but also feel free to request new features by raising GitHub issues.
