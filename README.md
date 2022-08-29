# Pyriod

Python implementation of basic period detection and fitting routines for astronomical time series.

Give it a spin before you install [with Binder](https://mybinder.org/v2/gh/keatonb/Pyriod/HEAD?filepath=examples%2FTSC2_Demo.ipynb).

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

Thanks to heroic work of [j123github](https://github.com/j123github), Pyriod can now be run in JupyerLab by running
```
jupyter labextension install @j123npm/qgrid2@1.1.4
```
See [quantopian/qgrid/#356](https://github.com/quantopian/qgrid/pull/356) for details.

This is a serious work in progress with many planned improvements.  Please be patient, but also feel free to request new features by raising GitHub issues.
