# Spectral Window

The spectral window describes frequency-domain structure introduced by the time sampling of a light curve. It is particularly useful for identifying aliases and other features in a periodogram that arise from gaps or irregular sampling rather than from intrinsic variability.

Pyriod provides both an interactive spectral-window panel and a programmatic `spectral_window()` function.

## Interactive use

Add the spectral-window interface to an existing `PyriodGUI` with:

```python
from Pyriod.analyses.gui import SpectralWindowGUI

window_gui = gui.add_analysis(SpectralWindowGUI)
```

A new **Spectral Window** tab is added to the Pyriod interface.

The spectral window is calculated automatically when the panel is first opened.

The calculation uses the observation times currently included in the associated `Prewhitener`. Points excluded by the Pyriod inclusion mask are not used.

## Analysis options

The panel provides two calculation options.

### Maximum frequency

**Max frequency** sets the upper frequency limit of the spectral-window calculation in the current frequency units of the `Prewhitener`.

The default value is the smaller of:

```python
pw.nyquist
```

and

```text
100
```

in the current frequency units.

### Oversampling

**Oversampling** controls the spacing of the calculated spectral-window frequencies relative to the natural frequency resolution of the time series.

For included observation times spanning a baseline \(T\), Pyriod determines the natural frequency resolution from the baseline and evaluates the spectral window with a spacing

\[
\Delta f = \frac{1}{T\,N_{\rm over}},
\]

with the appropriate conversion to the frequency units used by the `Prewhitener`.

Larger oversampling values therefore produce a more finely sampled spectral window but require more frequency evaluations.

The default oversampling factor is:

```text
10
```

## Recalculating the spectral window

Change the maximum frequency or oversampling value and press **Calculate** to recompute the spectral window.

The progress bar reports the number of frequency evaluations completed.

The calculation can be stopped with **Cancel**. If cancelled, the function retains and displays the portion of the spectral window that was calculated before cancellation.

The calculated arrays are available from the GUI object:

```python
window_gui.frequencies
window_gui.amplitudes
```

The frequencies are expressed in the current frequency units of the associated `Prewhitener`. The spectral-window amplitudes are normalized.

## Programmatic use

The same calculation can be performed without creating a GUI:

```python
from Pyriod.analyses import spectral_window

frequencies, amplitudes = spectral_window(
    pw,
    maxfreq=100,
    oversample=10,
)
```

The function uses:

- the observation times in `pw.lc`,
- the `include` mask,
- `pw.freq_unit`,
- and `pw.freq_conversion`.

At least two included observations are required.

Both `maxfreq` and `oversample` must be positive.

## Example

A simple spectral-window calculation and plot can be made with:

```python
import matplotlib.pyplot as plt
from Pyriod.analyses import spectral_window

frequency, amplitude = spectral_window(
    pw,
    maxfreq=100,
    oversample=10,
)

plt.plot(frequency, amplitude)
plt.xlabel(f"Frequency ({pw.freq_unit})")
plt.ylabel("Spectral window amplitude")
```

To record the calculation in the `Prewhitener` log:

```python
frequency, amplitude = spectral_window(
    pw,
    maxfreq=100,
    oversample=10,
    log=True,
)
```

## Changing the Pyriod data state

The spectral window depends only on the time sampling of the currently included observations.

Consequently, operations that change the inclusion mask can change the spectral window. For example:

```python
pw.mask_indices(indices)
```

or

```python
pw.clear_mask()
```

changes which observation times are used the next time the spectral window is calculated.

Changes to the fitted sinusoidal model do not by themselves change the spectral window unless they are accompanied by a change in the included observations.

If the `Prewhitener` has been modified since the spectral-window calculation was performed, press **Calculate** again to obtain a new result.

## Interpreting the result

A peak in an astronomical periodogram can be accompanied by aliases produced by the temporal sampling of the observations. Comparing the periodogram with the spectral window can help identify frequency spacings associated with that sampling.

The spectral window should therefore be interpreted as a diagnostic of the observing cadence, not as another periodogram of the stellar variability.

## Cancellation and progress callbacks

For custom programmatic workflows, `spectral_window()` supports optional callbacks:

```python
def progress(completed, total):
    print(completed, total)

frequency, amplitude = spectral_window(
    pw,
    maxfreq=100,
    oversample=10,
    progress_callback=progress,
)
```

A cancellation function can also be supplied:

```python
stop = False

def cancel_check():
    return stop

frequency, amplitude = spectral_window(
    pw,
    cancel_check=cancel_check,
)
```

The cancellation state is checked between frequency evaluations. If cancellation is requested, the function returns only the frequencies and amplitudes that have already been calculated.

These hooks are used by `SpectralWindowGUI` to provide its interactive progress bar and Cancel button.

## Closing the panel

The panel can be removed from the parent `PyriodGUI` with:

```python
gui.remove_analysis(window_gui)
```

or by pressing the **Close** button in the panel.

See [Using Analysis GUI Tabs](gui_tabs.md) for the general lifecycle of optional analysis interfaces.

## API reference

### Spectral-window calculation

::: Pyriod.analyses.spectral_window.spectral_window

### Spectral-window GUI

::: Pyriod.analyses.gui.SpectralWindowGUI
