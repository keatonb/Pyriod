import numpy as np

def make_all_iter(variables):
    """Return iterables of given variables.

    Parameters
    ----------
    variables : list or tuple
        Set of values to return as iterables if necessary.
        Each must have length 1 or length of first variable
    Returns
    -------
    tuple of iterable versions of input variables
    """
    # Wrap all single values or strings in lists
    variables = [[v] if (not hasattr(v, '__iter__')) or (type(v) == str)
                    else v for v in variables]
    # Get length of first variable
    nvals = len(variables[0])
    # check that all lengths are the same or 1
    if not all([len(l) in [nvals, 1] for l in variables]):
        raise ValueError("Arguments passed have inconsistent lengths.")
    else:
        variables = [[v[0] for i in range(nvals)] if (len(v) == 1)
                        else v for v in variables]
    return tuple(variables)

def _as_scalar_float(value):
    """Convert scalar-like, length-1 array-like, or Quantity-like value to float."""
    # Strip astropy Quantity if present
    if hasattr(value, "value"):
        value = value.value

    # Convert numpy arrays/lists/scalars to ndarray, then flatten
    arr = np.asarray(value).ravel()

    if len(arr) != 1:
        raise ValueError(f"Expected scalar or length-1 value, got shape {np.shape(value)}")

    return float(arr[0])