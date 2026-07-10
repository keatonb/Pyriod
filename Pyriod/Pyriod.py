"""Pyriod, an astronomical prewhitening frequency analysis package.

Written by Keaton Bell

For more, see https://github.com/keatonb/Pyriod

---------------------

Pyriod includes two main classes, the core.Prewhitener and a connected gui.PyriodGUI.
This main Pyriod class can make both, if requested.

---------------------
"""

from .core import Prewhitener
from .gui import PyriodGUI

class Pyriod:
    def __init__(self, *args, gui=True, **kwargs):
        self.pw = Prewhitener(*args, **kwargs)
        self.gui = gui
        self._gui = None
        if gui:
            self._gui = PyriodGUI(self.pw)
            
    def __getattr__(self, name):
        if self._gui is not None and hasattr(self._gui, name):
            return getattr(self._gui, name)
        if hasattr(self.pw, name):
            return getattr(self.pw, name)
        raise AttributeError(name)