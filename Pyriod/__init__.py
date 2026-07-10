# Pyriod/__init__.py

from .core import Prewhitener
from .Pyriod import Pyriod

__all__ = ["Pyriod", "Prewhitener", "PyriodGUI"]


def __getattr__(name):
    if name == "PyriodGUI":
        from .gui import PyriodGUI
        return PyriodGUI

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)