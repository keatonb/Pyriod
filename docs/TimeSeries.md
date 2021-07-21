# The TimeSeries widget

The TimeSeries cell allows you to interact with your time series data and to view the current model fit. The data are displayed as blue dots and the model is a red line.

The toolbar includes "pan/zoom" and "zoom rect" tools for exploring the plot. With neither of these selected, you'll be able to click and drag a lasso selector around points in the time series data. Hit "delete" or "backspace" to exclude these points from the periodogram and model fit calculations. This can be useful for discarding outliers or particularly noisy data. The "Reset mask" button under the "options" tab will return all masked points to the data set.

Under the "options" tab, you can also select whether to display the original data or the current residuals (data - model), and there is an option to fold the data on any given frequency (which can also be selected from the current signal list).

Future developments include the ability to bin the data (especially phase-folded) 