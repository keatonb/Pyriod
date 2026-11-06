# Bootstrap Significance Testing

Pyriod provides bootstrap tools for estimating the significance of peaks in a periodogram. The analysis can be performed interactively through a `PyriodGUI` tab or programmatically using functions in `Pyriod.analyses`.

Bootstrap significance testing generates randomized light curves by resampling flux values with replacement while preserving the observation times. A periodogram is calculated for each realization, and the distribution of the highest periodogram peaks is used to estimate a false-alarm-probability (FAP) threshold.

## Interactive use

A bootstrap-significance panel can be added to an existing `PyriodGUI`:

```python
from Pyriod.analyses.gui import BootstrapSignificanceGUI

bootstrap_gui = gui.add_analysis(BootstrapSignificanceGUI)
```

The new **Bootstrap** tab operates on the same `Prewhitener` as the main Pyriod interface.

The GUI samples the **current residual light curve**. Therefore, the fitted model should normally be up to date before calculating bootstrap samples.

### Calculating bootstrap samples

The **Bootstrap samples** controls specify how the randomized periodograms are generated:

- **Runs** — number of bootstrap realizations.
- **Min frequency** — lower frequency limit. Enter `None` to use the full frequency range.
- **Max frequency** — upper frequency limit. Enter `None` to use the full frequency range.
- **Time limit** — optional limit on the estimated remaining calculation time, in seconds.
- **Random seed** — optional integer seed for reproducible bootstrap samples.

Press **Calculate samples** to begin the calculation.

A progress bar reports the number of completed realizations. The calculation can be stopped with the **Cancel** button. If a calculation is cancelled or stopped by the time-limit criterion, successfully completed samples are retained.

The generated peak amplitudes are stored in:

```python
bootstrap_gui.maxima
```

and the median periodogram amplitudes from the same realizations are stored in:

```python
bootstrap_gui.medians
```

### Calculating a significance threshold

After bootstrap samples have been generated, the **Significance threshold** controls can be used to calculate a FAP threshold.

Two definitions of peak height are available:

**Relative to median noise**

For each bootstrap realization, the highest periodogram amplitude is divided by the median periodogram amplitude:

\[
R = \frac{A_{\rm max}}{A_{\rm median}}.
\]

The reported threshold is therefore a multiplier relative to the characteristic periodogram noise level.

**Absolute amplitude**

The distribution of maximum periodogram amplitudes is used directly. The resulting threshold is expressed in the amplitude units of the associated `Prewhitener`.

The **FAP** field specifies the requested false-alarm probability. For example,

```text
FAP = 0.01
```

returns the 99th percentile of the bootstrap distribution.

Because the bootstrap samples are retained, the FAP or the choice between relative and absolute thresholds can be changed with **Update threshold** without recalculating the bootstrap periodograms.

The panel displays both a histogram of the bootstrap distribution and its empirical cumulative distribution function.

## Programmatic use

The bootstrap analysis can also be performed without the GUI.

### Generate bootstrap samples

```python
from Pyriod.analyses import bootstrap_periodogram_samples

maxima, medians = bootstrap_periodogram_samples(
    pw,
    nruns=1000,
    minfreq=None,
    maxfreq=None,
    residuals=True,
)
```

`bootstrap_periodogram_samples()` uses the frequency grid, frequency units, periodogram settings, amplitude conversion, and included-data mask of the supplied `Prewhitener`.

By default, samples are generated from the current residual light curve:

```python
residuals=True
```

To bootstrap the original light curve instead:

```python
maxima, medians = bootstrap_periodogram_samples(
    pw,
    nruns=1000,
    residuals=False,
)
```

A seeded NumPy random-number generator can be supplied for reproducible results:

```python
import numpy as np

rng = np.random.default_rng(42)

maxima, medians = bootstrap_periodogram_samples(
    pw,
    nruns=1000,
    rng=rng,
)
```

### Calculate a threshold from existing samples

Once the bootstrap samples have been generated, a threshold can be calculated without generating new periodograms:

```python
from Pyriod.analyses import bootstrap_threshold_from_samples

threshold, samples = bootstrap_threshold_from_samples(
    maxima,
    medians,
    fap=0.01,
    relative=True,
)
```

For a relative threshold,

```python
relative=True
```

uses:

```python
maxima / medians
```

as the bootstrap distribution.

For an absolute threshold:

```python
threshold, samples = bootstrap_threshold_from_samples(
    maxima,
    medians,
    fap=0.01,
    relative=False,
)
```

the maximum amplitudes themselves are used.

This separation is useful when several FAP levels need to be evaluated from the same computationally expensive set of bootstrap periodograms.

### Complete threshold calculation

`bootstrap_threshold()` provides a convenience function that generates the samples and calculates the threshold in one call:

```python
from Pyriod.analyses import bootstrap_threshold

results = bootstrap_threshold(
    pw,
    fap=0.01,
    oversample=10,
    relative=True,
    return_samples=True,
)
```

For this function, the number of bootstrap realizations is determined from

```python
nruns = int(oversample / fap)
```

so that the sampling density near the requested FAP level scales with the `oversample` parameter.

For example,

```python
fap = 0.01
oversample = 10
```

requests 1000 bootstrap realizations.

The returned dictionary contains:

```python
results["threshold"]
results["FAP"]
results["completed"]
results["nruns_requested"]
results["nruns_completed"]
```

and, when `return_samples=True`:

```python
results["maxima"]
results["medians"]
```

If the time-limit criterion stops the calculation before all requested realizations are completed, `completed` is `False` and no threshold is reported.

### Plotting bootstrap samples

Existing samples can be visualized directly:

```python
from Pyriod.analyses import plot_bootstrap

samples = maxima / medians

ax, ax_cdf = plot_bootstrap(
    samples,
    fap=0.01,
)
```

The histogram shows the bootstrap peak distribution and the secondary axis shows its empirical cumulative distribution function. The requested FAP threshold is marked on the distribution.

## Computational considerations

Bootstrap periodograms can be computationally expensive. The number of periodograms required should be chosen with the desired FAP in mind: a threshold in the extreme tail of the distribution cannot be estimated reliably from only a small number of realizations.

For interactive work, it can be useful to begin with a modest number of runs to verify the frequency limits and other settings before starting a larger calculation.

The GUI performs the bootstrap calculation outside the notebook's main event loop, allowing the interface and progress display to remain responsive.

## API reference

### Bootstrap analysis

::: Pyriod.analyses.bootstrap.bootstrap_lc

::: Pyriod.analyses.bootstrap.bootstrap_periodogram_samples

::: Pyriod.analyses.bootstrap.bootstrap_threshold_from_samples

::: Pyriod.analyses.bootstrap.bootstrap_threshold

::: Pyriod.analyses.bootstrap.plot_bootstrap

### Bootstrap GUI

::: Pyriod.analyses.gui.BootstrapSignificanceGUI
