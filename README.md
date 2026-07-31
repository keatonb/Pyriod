# Pyriod
[![Tests](https://github.com/keatonb/Pyriod/actions/workflows/tests.yml/badge.svg)](https://github.com/keatonb/Pyriod/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/Pyriod.svg)](https://pypi.org/project/Pyriod/)
[![DOI](https://zenodo.org/badge/176140892.svg)](https://zenodo.org/badge/latestdoi/176140892)
[![Documentation Status](https://readthedocs.org/projects/pyriod/badge/?version=latest)](https://pyriod.readthedocs.io/en/latest/)

Python implementation of basic period detection and fitting routines for astronomical time series.

Give it a spin before you install with an example notebook [on Binder](https://mybinder.org/v2/gh/keatonb/Pyriod/HEAD?filepath=examples%2FTSC2_Demo.ipynb). (This can take a while to load!)

To install, use

```pip install Pyriod```

or download the latest (possibly unstable) version from GitHub and run

```python -m pip install .```

Additional documentation is available at [pyriod.readthedocs.io](https://pyriod.readthedocs.io).

### Note on v0.5.0 refactor

Pyriod has undergone a major revision as of v0.5, separating the analysis suite
```from Pyriod import Prewhitener```
from the GUI
```from Pyriod import PyriodGUI```.
The Pyriod objects should now also take up considerably less space in memory.

Most GUI-based functionality should be work as expected with the same ```from Pyriod import Pyriod``` structure of previous versions. 
Documentation is being updated to reflect the current code, and a prewhitening tutorial based on this work is in preparation.

If you encounter any errors in the current version of Pyriod, please report them here under [Issues](https://github.com/keatonb/Pyriod/issues).


### Acknowledgments

This code uses [Qgrid](https://github.com/quantopian/qgrid) and other Jupyter widgets to interactively display the frequency solution.  
Unfortunately Qgrid is no longer supported. We are grateful to @zhihanyue for maintaining [qgridnext](https://github.com/zhihanyue/qgridnext).

This material is based upon work supported by the National Science Foundation under Grant No. AST-2406917.

### Citation

If you use this code in a published analysis, please cite record [ascl:2207.007](https://ui.adsabs.harvard.edu/abs/2022ascl.soft07007B/abstract) of the Astrophysics Source Code Library.
