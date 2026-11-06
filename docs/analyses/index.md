# Advanced analyses

Pyriod provides optional analysis routines that operate on a
`Prewhitener` and can be used either programmatically or through
interactive GUI tabs.

The numerical analysis routines are available in `Pyriod.analyses`.
Some analyses also provide an `AnalysisGUI` interface that can be added
as an additional tab to an existing `PyriodGUI`.

For example, a spectral-window analysis can be performed either
programmatically:

```python
from Pyriod.analyses import spectral_window

frequency, amplitude = spectral_window(pw)
```

or interactively:

```
from Pyriod.analyses.gui import SpectralWindowGUI

analysis = gui.add_analysis(SpectralWindowGUI)
```

The GUI interfaces use the same `Prewhitener` as the parent `PyriodGUI`.
