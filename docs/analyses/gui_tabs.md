# Using Analysis GUI Tabs

Pyriod provides optional interactive interfaces for advanced analyses in addition to the standard Time Series, Periodogram, Signals, and Log tabs.

These analysis interfaces are subclasses of `AnalysisGUI` and can be added as new tabs to an existing `PyriodGUI`. Each analysis tab operates on the same `Prewhitener` used by the main GUI, so it has access to the current light curve, fitted solution, periodograms, masks, and other analysis state.

## Adding an analysis tab

First create a normal `Prewhitener` and `PyriodGUI`:

```python
from Pyriod import Prewhitener, PyriodGUI

pw = Prewhitener(lc)
gui = PyriodGUI(pw)
```

The complete Pyriod interface can then be displayed with:

```python
gui.Pyriod()
```

To add an optional analysis, import its GUI class and pass the class to `add_analysis()`.

For example, to add the spectral-window interface:

```python
from Pyriod.analyses.gui import SpectralWindowGUI

window_gui = gui.add_analysis(SpectralWindowGUI)
```

A new analysis tab is added to the existing Pyriod interface.

Similarly, the bootstrap significance interface can be added with:

```python
from Pyriod.analyses.gui import BootstrapSignificanceGUI

bootstrap_gui = gui.add_analysis(BootstrapSignificanceGUI)
```

`add_analysis()` returns the newly created analysis-GUI object. Keeping this reference is useful if you want to inspect or interact with the analysis programmatically.

Additional keyword arguments supplied to `add_analysis()` are passed to the analysis-GUI constructor:

```python
analysis = gui.add_analysis(MyAnalysisGUI, option=value)
```

## Shared analysis state

Analysis tabs use the same `Prewhitener` as the parent `PyriodGUI`.

For example:

```python
window_gui.pw is gui.pw
```

should refer to the same underlying analysis object.

This means that an analysis panel can use the current:

- light curve and inclusion mask,
- fitted and staged signal parameters,
- frequency sampling,
- original, model, and residual periodograms,
- amplitude and frequency units,
- significance-threshold information,
- and other state maintained by the `Prewhitener`.

The analysis panel does not create a separate copy of the primary Pyriod analysis.

Because the analysis interfaces share this state, changes made to the `Prewhitener` may affect the results of subsequently calculated advanced analyses.

## Using an analysis panel

Each analysis interface provides controls appropriate to that analysis. A typical workflow is:

1. Prepare the light curve and frequency solution using the standard Pyriod interface.
2. Add the desired analysis tab with `add_analysis()`.
3. Configure the analysis using the controls in the new tab.
4. Run the calculation.
5. Inspect the resulting plots or numerical results.
6. Modify the settings and repeat the calculation as needed.
7. Close the analysis tab when it is no longer needed.

The numerical routines underlying the GUI panels can also be called directly without creating an analysis tab. See the documentation for the individual analyses for examples of both interactive and programmatic use.

## Example: spectral window

Add the spectral-window panel with:

```python
from Pyriod.analyses.gui import SpectralWindowGUI

window_gui = gui.add_analysis(SpectralWindowGUI)
```

The panel uses the sampling of the light curve associated with `gui.pw` to calculate and display the spectral window.

The equivalent analysis can also be performed without the GUI:

```python
from Pyriod.analyses import spectral_window

frequency, amplitude = spectral_window(pw)
```

Using the numerical analysis function directly is useful for scripted analyses, automated workflows, or calculations that do not require interactive plotting.

## Example: bootstrap significance analysis

Add the bootstrap-significance panel with:

```python
from Pyriod.analyses.gui import BootstrapSignificanceGUI

bootstrap_gui = gui.add_analysis(BootstrapSignificanceGUI)
```

The panel provides controls for generating bootstrap samples and evaluating periodogram significance thresholds.

The underlying numerical analysis can also be accessed through `Pyriod.analyses` for non-interactive use.

For computationally expensive analyses, the GUI may also provide progress information and cancellation controls. Closing an analysis panel should terminate or clean up resources associated with that interface.

## Closing an analysis tab

The preferred way to programmatically close an analysis attached to a `PyriodGUI` is through `remove_analysis()`.

If you retained the object returned by `add_analysis()`:

```python
window_gui = gui.add_analysis(SpectralWindowGUI)
```

you can later remove it with:

```python
gui.remove_analysis(window_gui)
```

An analysis can also be identified by its class:

```python
gui.remove_analysis(SpectralWindowGUI)
```

or, when supported, by its class name:

```python
gui.remove_analysis("SpectralWindowGUI")
```

`remove_analysis()` closes the analysis interface and removes its tab from the parent `PyriodGUI`.

When an analysis provides a **Close** button in its interface, that button should normally be used rather than calling the analysis object's `close()` method directly. This allows the parent `PyriodGUI` to remove the corresponding tab as well as clean up the analysis resources.

## Closing an analysis directly

Individual `AnalysisGUI` objects also provide a `close()` method:

```python
window_gui.close()
```

This releases resources owned by the analysis interface, such as widget callbacks, figures, and active calculations.

For an analysis that is currently attached as a tab to a `PyriodGUI`, however, prefer:

```python
gui.remove_analysis(window_gui)
```

so that both the analysis resources and the parent tab state are updated consistently.

## Closing the complete Pyriod GUI

Closing the parent GUI also closes its attached analysis interfaces:

```python
gui.close()
```

This should be used when the complete interactive Pyriod session is no longer needed.

The associated `Prewhitener` can remain available after the GUI is closed, depending on the arguments passed to `PyriodGUI.close()`.

For example, a workflow can use the GUI interactively and then continue working directly with:

```python
pw
```

after closing the interface.

## Adding multiple analyses

Different types of analysis can be open at the same time:

```python
from Pyriod.analyses.gui import (
    BootstrapSignificanceGUI,
    SpectralWindowGUI,
)

bootstrap_gui = gui.add_analysis(BootstrapSignificanceGUI)
window_gui = gui.add_analysis(SpectralWindowGUI)
```

Each appears as a separate tab and shares the same underlying `Prewhitener`.

This makes it possible to inspect several complementary properties of the same Pyriod analysis without constructing separate `Prewhitener` objects.

## Programmatic analyses versus GUI analyses

The advanced-analysis system separates the numerical calculation from its graphical interface.

Numerical routines are available from:

```python
Pyriod.analyses
```

while interactive interfaces are available from:

```python
Pyriod.analyses.gui
```

For example:

```python
from Pyriod.analyses import spectral_window
from Pyriod.analyses.gui import SpectralWindowGUI
```

The first provides the numerical analysis, while the second provides an interactive panel for performing and displaying that analysis.

Use the numerical API when:

- running an automated or scripted analysis,
- processing many light curves,
- writing tests,
- or directly manipulating the returned results.

Use an `AnalysisGUI` when:

- exploring a light curve interactively,
- adjusting analysis settings repeatedly,
- monitoring a long calculation,
- or viewing analysis-specific plots and controls within the main Pyriod interface.

## AnalysisGUI API

All optional analysis panels are based on `AnalysisGUI`.

`AnalysisGUI` provides common infrastructure for integrating an analysis with `PyriodGUI`, including management of widgets, callbacks, busy states, figures, logging, and cleanup.

Users normally do not need to instantiate `AnalysisGUI` directly. Instead, use a specific subclass:

```python
gui.add_analysis(SpectralWindowGUI)
```

or:

```python
gui.add_analysis(BootstrapSignificanceGUI)
```

Developers who want to add a new advanced analysis can subclass `AnalysisGUI` and provide an interface for a numerical routine in `Pyriod.analyses`.

See the developer documentation for creating new analysis routines and `AnalysisGUI` subclasses.

## Recommended workflow

A typical interactive workflow is therefore:

```python
from Pyriod import Prewhitener, PyriodGUI
from Pyriod.analyses.gui import SpectralWindowGUI

# Create the primary analysis
pw = Prewhitener(lc)

# Create the interactive interface
gui = PyriodGUI(pw)
display(gui.Pyriod())

# Add an advanced analysis
window_gui = gui.add_analysis(SpectralWindowGUI)

# Work with the analysis interactively...

# Remove the analysis when finished
gui.remove_analysis(window_gui)

# Close the main GUI when finished
gui.close()
```

The `Prewhitener` remains the central analysis object throughout this workflow. The main `PyriodGUI` and optional `AnalysisGUI` tabs provide different interactive views and controls for working with that shared analysis state.
