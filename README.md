# Pyriod

Python implementation of basic period detection and fitting routines for astronomical time series.

To install, download and run

```python setup.py install```

or use

```pip install Pyriod```

This code uses [Qgrid](https://github.com/quantopian/qgrid) to interactively display the frequency solution.  

To display the Qgrid table widgets as part of the Period GUI, you will need to first enable the following Jupyter notebook extensions in the terminal:
```
jupyter nbextension enable --py --sys-prefix qgrid
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix ipympl
```

This is a serious work in progress with many planned improvements.  Please be patient, but also feel free to request new features.
