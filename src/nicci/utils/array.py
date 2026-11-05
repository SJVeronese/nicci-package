import numpy as np


def extend(var, target_length):
    """Extend an array-like value to a target length by repeating its last entry.

    Parameters
    ----------
    var : scalar or array_like
        Value(s) to extend.
    target_length : int
        Desired output length.

    Returns
    -------
    ndarray
        Array of length `target_length`. If `var` is already at least
        that long, it is returned unchanged (as an array).
    """

    var = np.atleast_1d(var)
    if var.size < target_length:
        var = np.concatenate((var, np.repeat(var[-1], target_length - var.size)))
    return var


def insert_key(dictionary, key, value, target_key):
    """Insert a new key/value pair into a dict immediately after a given key.

    Parameters
    ----------
    dictionary : dict
        Input dictionary.
    key : hashable
        New key to insert.
    value : object
        Value associated with `key`.
    target_key : hashable
        Existing key after which `key` is inserted.

    Returns
    -------
    dict
        A new dictionary with `key`/`value` inserted right after
        `target_key`, preserving the order of all other entries.

    Raises
    ------
    ValueError
        If `target_key` is not in `dictionary`.
    """
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    # index of the target key
    target_idx = keys.index(target_key)

    # rebuild the key/value lists with the new entry inserted
    new_keys = keys[: target_idx + 1] + [key] + keys[target_idx + 1 :]
    new_values = values[: target_idx + 1] + [value] + values[target_idx + 1 :]

    return dict(zip(new_keys, new_values))
    

def normalise(arr, new_min, new_max):
    """Rescale an array to a new [min, max] range.

    Parameters
    ----------
    arr : array_like
        Input data.
    new_min, new_max : float
        Desired output range.

    Returns
    -------
    ndarray
        Rescaled array, the same shape as `arr`. If `arr` is constant,
        an array filled with `new_min` is returned.
    """

    arr = np.asarray(arr)
    old_min = np.nanmin(arr)
    old_max = np.nanmax(arr)

    # handle constant arrays
    if old_max == old_min:
        return np.full_like(arr, new_min)

    scale = (new_max - new_min) / (old_max - old_min)
    return (arr - old_min) * scale + new_min


def RMS(data, *, mode='mad', fluxrange='negative', if_empty='nan'):
    """Compute the root-mean-square (or median absolute deviation) noise.

    Parameters
    ----------
    data : array_like
        Input data, assumed to be noise centred on zero.
    mode : {'mad', 'std'}, default 'mad'
        Statistic to use: ``'std'`` for the standard deviation, or
        ``'mad'`` for the median absolute deviation rescaled to match the
        standard deviation of a Gaussian (factor 1.4826).
    fluxrange : {'negative', 'positive', 'all'}, default 'negative'
        Subset of the data to use: only negative values, only positive
        values, or all values.
    if_empty : {'nan', 'error'}, default 'nan'
        What to do if `fluxrange` filtering removes all data: return NaN,
        or raise a ValueError.

    Returns
    -------
    float
        The estimated noise, or NaN if no data remain and `if_empty` is
        ``'nan'``.
    """

    # ensure data is an array
    data = np.asarray(data)

    # the noise is assumed to be gaussian and centered on 0
    value = 0

    # flux filtering
    if fluxrange == 'positive':
        data = data[data > 0]
    elif fluxrange == 'negative':
        data = data[data < 0]

    if data.size == 0:
        if if_empty == 'error':
            raise ValueError("Filtering removed all data.")
        return np.nan

    if mode == "std":
        return np.sqrt(np.nanmean((data - value) ** 2))
    elif mode == "mad":
        return 1.4826 * np.nanmedian(np.abs(data - value))
    else:
        raise ValueError("mode must be 'std' or 'mad'.")