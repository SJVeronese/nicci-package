import json
import numpy as np


def to_json(obj):
    """Recursively convert NumPy types/arrays to native Python types for JSON export.

    Parameters
    ----------
    obj : object
        Value to convert: an ndarray, a NumPy scalar, a dict, a list or
        tuple, or any JSON-serializable value.

    Returns
    -------
    object
        `obj` with ndarrays converted to lists, NumPy scalars converted
        to Python scalars, and dicts/lists/tuples converted recursively.
        Other values are returned unchanged.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {key: to_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(value) for value in obj]
    return obj


def from_json(obj):
    """Recursively convert JSON-decoded values back to NumPy arrays where possible.

    Parameters
    ----------
    obj : object
        Value decoded from JSON: a dict, a list, or a scalar.

    Returns
    -------
    object
        `obj` with dicts converted recursively, numeric lists converted
        to `ndarray`, non-numeric lists converted recursively element by
        element, and other values returned unchanged.
    """
    if isinstance(obj, dict):
        return {key: from_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        # try to convert to a numpy array; falls back to a plain list
        # if the contents aren't numeric (e.g. list of strings)
        try:
            return np.array(obj, dtype=float)
        except (ValueError, TypeError):
            return [from_json(value) for value in obj]
    return obj


def load_json(path):
    """Load a JSON file, converting numeric lists back to NumPy arrays.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    object
        The decoded content, processed with `from_json`.
    """
    with open(path, 'r') as fp:
        data = json.load(fp)
    return from_json(data)


def save_json(data, path):
    """Save data to a JSON file, converting NumPy arrays/scalars to native types.

    Parameters
    ----------
    data : object
        Data to save, processed with `to_json` before serialization.
    path : str
        Output file path.

    Returns
    -------
    None
    """
    with open(path, 'w') as fp:
        json.dump(to_json(data), fp)