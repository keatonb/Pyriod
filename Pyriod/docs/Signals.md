### The Signals widget

The Signals cell provides an interactive table with details of the model. The model is a sum of sinusoids, each parameterized as 
$amplitude*sin(2pi(frequency*time + phase)).$
Phase is displayed as the phase at time = 0.  


The table is powered by QGrid and allows you to directly edit the values in the table. ***Manual changes to the table not take effect unless you hit return after the change, or click on another entry in the table*** (if the cell still is highlighted in edit mode, the change has not been communicated to Pyriod).  


Buttons to add signals to the table are provided at top.  There is also a button that will delete all selected rows from the table.  ***Changes are not made to the model until you click "Refine fit."***  


You can add frequencies that maintain arithmetic relationships to other signal frequencies that are already in the fit by typing the arithmetic relationship in the frequency box. For example, to add a harmonic of signal f0, type "2*f0" in the frequency field and click "Add to solution." You can add, subtract, multiply or divide any combination of signal frequencies and constant values.  


Frequencies, amplitudes, and phases can also be individually held fixed to constant values by marking the checkboxes in the associated columns.  Don't forget to hit return after checking a box to communicate the change to Pyriod!  The "brute" checkbox forces Pyriod to explore a full range of possible phase values, which is useful in rare cases when a fit is behaving poorly because a signal fit is stuck in a local minimum.  


Below the table are buttons to save the frequency solution, or to load a previously saved solution, as well as to change the file to save/load.  


The "fit report" dropdown will display information about the most recent fit performed by lmfit.