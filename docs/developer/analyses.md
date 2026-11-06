# Writing a New Analysis

Pyriod separates optional scientific analyses from their interactive interfaces.

Numerical calculations belong in:

```text
Pyriod/analyses/
```

while optional graphical interfaces belong in:

```text
Pyriod/analyses/gui/
```

A graphical analysis should subclass `AnalysisGUI` and use a headless function from `Pyriod.analyses` to perform the scientific calculation.

This separation keeps the numerical API usable in scripts and automated workflows while allowing the same analysis to be exposed as an interactive tab in `PyriodGUI`.

## Overview

A typical new analysis requires:

```text
Pyriod/
    analyses/
        my_analysis.py
        gui/
            my_analysis.py
```

The recommended development sequence is:

1. Write and test the numerical analysis independently of the GUI.
2. Export the numerical routine from `Pyriod.analyses`.
3. Create an `AnalysisGUI` subclass that calls the numerical routine.
4. Export the GUI class from `Pyriod.analyses.gui`.
5. Add the interface to a `PyriodGUI` with `add_analysis()`.
6. Add tests for both the numerical routine and GUI lifecycle.
7. Add user and API documentation.

## 1. Write the headless analysis

Scientific calculations should not depend on widgets or on `PyriodGUI`.

A simple analysis function might look like:

```python
# Pyriod/analyses/my_analysis.py

import numpy as np


def my_analysis(
    pw,
    scale=1.0,
    progress_callback=None,
    cancel_check=None,
):
    """Calculate an example quantity from a Prewhitener.

    Parameters
    ----------
    pw : Prewhitener
        Analysis object providing the required data.
    scale : float, optional
        Example calculation parameter. Default is 1.
    progress_callback : callable or None, optional
        Callback invoked as ``progress_callback(completed, total)``.
    cancel_check : callable or None, optional
        Callable returning True when the calculation should stop.

    Returns
    -------
    x : numpy.ndarray
        Coordinates of completed calculations.
    y : numpy.ndarray
        Calculated values.
    """
    x = np.arange(100)
    values = []

    for i, value in enumerate(x):
        if cancel_check is not None and cancel_check():
            break

        values.append(scale * value)

        if progress_callback is not None:
            progress_callback(i + 1, len(x))

    return x[:len(values)], np.asarray(values)
```

The routine should be independently usable:

```python
from Pyriod.analyses.my_analysis import my_analysis

x, y = my_analysis(pw)
```

This is the core scientific API. The GUI should call this routine rather than duplicating its calculation.

### Recommended callback interface

Potentially slow analyses should, when practical, support:

```python
progress_callback(completed, total)
```

and:

```python
cancel_check()
```

This allows the numerical routine to remain independent of Jupyter while still supporting responsive GUI progress and cancellation.

Cancellation should normally be cooperative: check `cancel_check()` at safe points in the calculation and return the results completed so far when partial results are meaningful.

## 2. Export the numerical function

Add the function to:

```python
Pyriod/analyses/__init__.py
```

For example:

```python
from .my_analysis import my_analysis
```

and add it to `__all__`:

```python
__all__ = [
    # ...
    "my_analysis",
]
```

Users can then write:

```python
from Pyriod.analyses import my_analysis
```

## 3. Create the AnalysisGUI subclass

Create:

```text
Pyriod/analyses/gui/my_analysis.py
```

A minimal interface could be:

```python
from __future__ import annotations

import asyncio

import ipywidgets as widgets
import matplotlib.pyplot as plt

from .base import AnalysisGUI
from ..my_analysis import my_analysis


class MyAnalysisGUI(AnalysisGUI):
    """Interactive interface for the example analysis."""

    title = "My Analysis"

    def __init__(
        self,
        pw,
        *,
        log_updated=None,
        busy_changed=None,
        request_close=None,
    ):
        super().__init__(
            pw,
            log_updated=log_updated,
            busy_changed=busy_changed,
        )

        self._request_close_callback = request_close

        self.x = None
        self.y = None

        self._init_figure()
        self._init_widgets()
        self._init_callbacks()
        self._init_layout()
```

The `title` class attribute determines the label used for the analysis tab.

## 4. Create and track figures

Figures owned by an analysis should be registered with `_track_figure()`:

```python
def _init_figure(self):
    self.fig, self.ax = plt.subplots(figsize=(7, 3))
    self._track_figure(self.fig)
```

Tracked figures are automatically closed when the analysis interface is closed.

This avoids leaving orphaned Matplotlib figures after an analysis tab is removed.

## 5. Construct the widgets

For example:

```python
def _init_widgets(self):
    self._scale = widgets.FloatText(
        value=1.0,
        description="Scale:",
    )

    self._calculate = widgets.Button(
        description="Calculate",
        icon="refresh",
    )

    self._close = widgets.Button(
        description="Close",
        icon="times",
    )

    self._result = widgets.HTML(value="")
```

`AnalysisGUI` already creates shared status, progress, and cancellation widgets. The combined progress display is available as:

```python
self.progress_widget
```

There is normally no need for each subclass to implement its own progress bar or Cancel button.

## 6. Register callbacks through AnalysisGUI

Use `_on_click()` rather than directly calling `button.on_click()`:

```python
def _init_callbacks(self):
    self._on_click(
        self._calculate,
        self._calculate_clicked,
    )

    self._on_click(
        self._close,
        self._close_clicked,
    )
```

For trait observers, use:

```python
self._observe(
    widget,
    self._value_changed,
    names="value",
)
```

Callbacks registered through `_on_click()` and `_observe()` are automatically disconnected when `close()` is called.

If an analysis registers some other callback or external resource, an arbitrary cleanup function can be registered with:

```python
self._add_cleanup(cleanup_function)
```

These cleanup functions are executed when the analysis closes.

## 7. Construct the layout

After constructing the top-level widget, register it with `_set_widget()`:

```python
def _init_layout(self):
    controls = widgets.VBox(
        [
            self._scale,
            self._calculate,
        ]
    )

    close_row = widgets.HBox(
        [self._close],
        layout=widgets.Layout(
            justify_content="flex-end",
        ),
    )

    interface = widgets.VBox(
        [
            close_row,
            self._status,
            self.progress_widget,
            self.fig.canvas,
            self._result,
            controls,
        ]
    )

    self._set_widget(interface)
```

The top-level widget can subsequently be obtained through the base-class interface and is managed by `AnalysisGUI.close()`.

If the interface includes an interactive Matplotlib canvas, it requires the ipympl backend, normally enabled in a notebook with:

```python
%matplotlib widget
```

## 8. Run calculations without blocking the GUI

Long calculations should not run directly inside a widget callback because doing so blocks the notebook interface.

Instead, start an asynchronous task:

```python
def _calculate_clicked(self, _):
    if self.busy:
        return

    try:
        self._start_task(self._calculate())
    except Exception as exc:
        self._log_exception(
            exc,
            context="Could not start analysis",
        )
```

and perform the blocking scientific calculation in a worker thread:

```python
async def _calculate(self):
    scale = self._scale.value

    total = 100

    self._start_progress(
        total,
        message="Calculating...",
    )

    loop = asyncio.get_running_loop()

    def progress_callback(completed, requested):
        loop.call_soon_threadsafe(
            self._set_progress,
            completed,
            requested,
        )

    result = await self._run_analysis_in_thread(
        my_analysis,
        self.pw,
        scale=scale,
        progress_callback=progress_callback,
        cancel_check=self._cancel_check,
        context="My analysis failed",
    )

    if self.closed:
        return

    if result is None:
        self._fail_progress("Analysis failed.")
        return

    self.x, self.y = result

    if self.cancel_requested:
        self._abort_progress("Analysis cancelled.")
    else:
        self._finish_progress()

    self._draw_result()
```

`_run_analysis_in_thread()` redirects exceptions from the scientific calculation to the associated `Prewhitener` log rather than allowing them to disappear inside a background task.

The notebook event loop should be used to update widgets from progress callbacks executed in the worker thread.

## 9. Use the shared cancellation mechanism

The base class owns the cancellation event.

Pass:

```python
cancel_check=self._cancel_check
```

to compatible headless analysis functions.

The shared Cancel button calls the base cancellation machinery automatically.

A numerical routine should periodically check:

```python
if cancel_check is not None and cancel_check():
    break
```

The base class distinguishes between a cooperative cancellation request and closing the interface.

## 10. Use the shared progress helpers

`AnalysisGUI` provides helpers for the common calculation states.

Start:

```python
self._start_progress(
    total,
    message="Calculating...",
)
```

Update:

```python
self._set_progress(completed, total)
```

Successful completion:

```python
self._finish_progress()
```

Cancelled or intentionally stopped:

```python
self._abort_progress("Analysis cancelled.")
```

Failure:

```python
self._fail_progress("Analysis failed.")
```

Using these common methods keeps optional Pyriod analysis panels visually and behaviorally consistent.

## 11. Send messages and errors to the Pyriod log

Use:

```python
self._log("Starting my analysis.")
```

rather than calling the parent GUI directly.

Warnings and errors can specify a log level:

```python
self._log(
    "Analysis was stopped early.",
    level="warning",
)
```

For exceptions:

```python
self._log_exception(
    exc,
    context="My analysis failed",
)
```

When the analysis has been added through `PyriodGUI.add_analysis()`, log updates are connected to the main Pyriod log display.

For small synchronous operations where an exception should simply be logged and converted to `None`, use:

```python
result = self._run_analysis(
    function,
    arg,
    context="Operation failed",
)
```

## 12. Display the result

Keep plotting and presentation code in the GUI class:

```python
def _draw_result(self):
    self.ax.clear()
    self.ax.plot(self.x, self.y)
    self.ax.set_xlabel("x")
    self.ax.set_ylabel("result")
    self.fig.canvas.draw_idle()

    self._result.value = (
        f"<b>{len(self.x)}</b> values calculated."
    )
```

The headless analysis should return numerical results rather than drawing into GUI-specific figures.

## 13. Implement Close correctly

An analysis attached to a `PyriodGUI` should request that its parent remove it:

```python
def _close_clicked(self, _):
    if self._request_close_callback is not None:
        self._request_close_callback(self)
    else:
        self.close()
```

This distinction is important.

Calling:

```python
self.close()
```

releases resources owned by the analysis.

Calling the parent-provided `request_close` callback allows `PyriodGUI` to both close the analysis and remove its tab from the interface.

The base `close()` method is idempotent and automatically:

- requests cooperative cancellation,
- cancels the tracked asynchronous task,
- disconnects registered callbacks,
- runs registered cleanup functions,
- closes tracked Matplotlib figures,
- closes the top-level widget,
- and clears the busy state.

Subclasses normally should not need to replace this behavior.

## 14. Support refresh when appropriate

`AnalysisGUI` defines:

```python
def refresh(self):
    ...
```

as a lightweight synchronization hook.

Override it if the display needs to react when the associated `Prewhitener` changes:

```python
def refresh(self):
    self.ax.set_xlabel(
        f"Frequency ({self.pw.freq_unit})"
    )
    self.fig.canvas.draw_idle()
```

`refresh()` should normally update inexpensive display state only. It should not automatically trigger a computationally expensive scientific analysis.

## 15. Export the GUI class

Add the class to:

```python
Pyriod/analyses/gui/__init__.py
```

For example:

```python
from .my_analysis import MyAnalysisGUI
```

and:

```python
__all__ = [
    # ...
    "MyAnalysisGUI",
]
```

Users can then import it with:

```python
from Pyriod.analyses.gui import MyAnalysisGUI
```

## 16. Add the analysis to PyriodGUI

No change to `PyriodGUI` is required for each new analysis class.

Once the class subclasses `AnalysisGUI`, it can be added dynamically:

```python
analysis_gui = gui.add_analysis(MyAnalysisGUI)
```

`PyriodGUI.add_analysis()` supplies the existing `Prewhitener`, connects log updates, provides the tab-removal callback, rebuilds the tabs, selects the newly created tab, and returns the analysis instance.

Only one instance of a particular analysis class can be open in a given `PyriodGUI` at a time.

The tab can later be removed with:

```python
gui.remove_analysis(analysis_gui)
```

## 17. Keep the scientific and GUI APIs separate

A useful rule is:

> `Pyriod.analyses` should know nothing about widgets, while `Pyriod.analyses.gui` should contain as little scientific calculation as possible.

For example, the existing spectral-window architecture is:

```text
Pyriod/analyses/spectral_window.py
    spectral_window()

Pyriod/analyses/gui/spectral_window.py
    SpectralWindowGUI
```

The numerical function can therefore be used independently:

```python
from Pyriod.analyses import spectral_window

frequency, amplitude = spectral_window(pw)
```

while the same calculation can be exposed interactively:

```python
from Pyriod.analyses.gui import SpectralWindowGUI

window_gui = gui.add_analysis(SpectralWindowGUI)
```

The bootstrap analysis follows the same pattern.

## 18. Testing a new analysis

At minimum, test the headless analysis independently of the GUI.

Useful tests include:

- expected output shape and type,
- invalid parameter handling,
- correct use of the `Prewhitener` inclusion mask,
- reproducibility when a random seed is supplied,
- progress-callback behavior,
- cooperative cancellation,
- and partial-result behavior after cancellation where applicable.

For the GUI, useful smoke tests include:

- construction from a `Prewhitener`,
- addition through `PyriodGUI.add_analysis()`,
- duplicate-tab rejection,
- result storage after a small calculation,
- logging of calculation failures,
- cancellation,
- removal through `remove_analysis()`,
- and safe repeated calls to `close()`.

Tests should use small calculations so the test suite remains fast.

## API reference

### AnalysisGUI

::: Pyriod.analyses.gui.AnalysisGUI

### PyriodGUI analysis management

The public methods used to manage optional analysis tabs are:

```python
gui.add_analysis(...)
gui.remove_analysis(...)
```

See [Using Analysis GUI Tabs](../analyses/gui_tabs.md) for the user-facing workflow.
