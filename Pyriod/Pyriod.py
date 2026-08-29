"""Pyriod, an astronomical prewhitening frequency analysis package.

Written by Keaton Bell

For more, see https://github.com/keatonb/Pyriod

---------------------

Backward-compatible Pyriod interface.

Pyriod includes two main classes, the core.Prewhitener and a connected gui.PyriodGUI.
This main Pyriod class can make both, if requested.

---------------------
"""

from .core import Prewhitener

class Pyriod:
    """Create a prewhitening analysis with an optional interactive GUI.

    This class provides the traditional Pyriod interface. It creates a
    :class:`Prewhitener` instance and, by default, an associated
    :class:`PyriodGUI`.

    Attributes and methods of the underlying objects are made accessible
    through the `Pyriod` instance for backward compatibility.

    Parameters
    ----------
    *args
        Positional arguments passed to `Prewhitener`.
    gui : bool, optional
        If True, create an interactive `PyriodGUI`. The default is True.
    **kwargs
        Keyword arguments passed to `Prewhitener`.
    """
    
    def __init__(self, *args, gui=True, **kwargs):
        self.pw = Prewhitener(*args, **kwargs)
        self.gui = gui
        self._gui = None
        if gui:
            from .gui import PyriodGUI
            self._gui = PyriodGUI(self.pw)
            
    def __getattr__(self, name):
        if self._gui is not None and hasattr(self._gui, name):
            return getattr(self._gui, name)
        if hasattr(self.pw, name):
            return getattr(self.pw, name)
        raise AttributeError(name)