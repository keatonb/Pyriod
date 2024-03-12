# Pyriod

**IMPORTANT INSTALLATION NOTE:** you must manually install qgrid with conda from the eshard channel as described below.

Python implementation of basic period detection and fitting routines for astronomical time series.

<!---Give it a spin before you install [with Binder](https://mybinder.org/v2/gh/keatonb/Pyriod/HEAD?filepath=examples%2FTSC2_Demo.ipynb).--->

To install, use

```pip install Pyriod```

or download the latest (possibly unstable) version from GitHub and run

```python setup.py install```

Additional documentation is available at [pyriod.readthedocs.io](https://pyriod.readthedocs.io).

This code uses [Qgrid](https://github.com/quantopian/qgrid) and other Jupyter widgets to interactively display the frequency solution.  
Unfortunately Qgrid is no longer supported. [A version that works with other modern packages](https://anaconda.org/eshard/qgrid) can be installed with 

```conda install -c eshard qgrid```

Furthermore, Qgrid has specific version requirements for Jupyter (classic only, < v7) and `ipywidgets` (< v8):
```
conda install "notebook<7"
conda install "ipywidgets<8"
```

To display the Pyriod GUI, you will need to first enable the following Jupyter notebook extensions in the terminal:
```
jupyter nbextension enable --py --sys-prefix qgrid
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix ipympl
```

