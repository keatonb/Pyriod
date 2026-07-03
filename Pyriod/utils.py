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
