# PyriodGUI

`PyriodGUI` provides the interactive notebook interface for working
with a [`Prewhitener`][Pyriod.core.Prewhitener] object.

It manages the plots, widgets, signal table, and user interactions,
while the underlying analysis state is stored by the `Prewhitener`. 

Note that the `PyriodGUI` interface is designed for interactive 
analysis through the GUI, and changes made directly to the 
`Prewhitener` will generally not be reflected automatically by the
GUI state.

::: Pyriod.gui.PyriodGUI
    options:
      members:
        - TimeSeries
        - Periodogram
        - Signals
        - Log
        - Pyriod
        - save_tsfig
        - save_perfig
        - fit_model
        - update_log
        - log
        - close
