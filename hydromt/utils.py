"""Utility functions for hydromt that have no other home."""

import numpy as np
import xarray as xr


class _classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def partition_dictionaries(left, right):
    """Calculate a partitioning of the two dictionaries.

    given dictionaries A and B this function will the follwing partition:
    (A ∩ B, A - B, B - A)
    """
    common = {}
    left_less_right = {}
    right_less_left = {}
    key_union = set(left.keys()) | set(right.keys())

    for key in key_union:
        value_left = left.get(key, None)
        value_right = right.get(key, None)
        if isinstance(value_left, dict) and isinstance(value_right, dict):
            (
                common_children,
                unique_left_children,
                unique_right_children,
            ) = partition_dictionaries(value_left, value_right)
            common[key] = common_children
            if unique_left_children != unique_right_children:
                left_less_right[key] = unique_left_children
                right_less_left[key] = unique_right_children
        elif value_left == value_right:
            common[key] = value_left
        else:
            if value_left is not None:
                left_less_right[key] = value_left
            if value_right is not None:
                right_less_left[key] = value_right

    return common, left_less_right, right_less_left


def elevation2rgba(val, nodata=np.nan):
    """Convert elevation to rgb tuple."""
    val += 32768
    r = np.floor(val / 256).astype(np.uint8)
    g = np.floor(val % 256).astype(np.uint8)
    b = np.floor((val - np.floor(val)) * 256).astype(np.uint8)
    mask = np.isnan(val) if np.isnan(nodata) else val == nodata
    a = np.where(mask, 0, 255).astype(np.uint8)
    return np.stack((r, g, b, a), axis=2)


def rgba2elevation(rgba: np.ndarray, nodata=np.nan, dtype=np.float32):
    """Convert rgb tuple to elevation."""
    r, g, b, a = np.split(rgba, 4, axis=2)
    val = (r * 256 + g + b / 256) - 32768
    return np.where(a == 0, nodata, val).squeeze().astype(dtype)


def _dict_pprint(d):
    import json

    return json.dumps(d, indent=2)


def has_no_data(data) -> bool:
    """Check whether various data containers are empty."""
    if data is None:
        return True
    elif isinstance(data, xr.Dataset):
        return all([v.size == 0 for v in data.data_vars.values()])
    else:
        return len(data) == 0
