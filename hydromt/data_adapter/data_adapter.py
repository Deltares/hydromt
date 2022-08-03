# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import yaml
import glob
from string import Formatter
from typing import Tuple
from itertools import product


__all__ = [
    "DataAdapter",
]


def round_latlon(ds, decimals=5):
    x_dim = ds.raster.x_dim
    y_dim = ds.raster.y_dim
    ds[x_dim] = np.round(ds[x_dim], decimals=decimals)
    ds[y_dim] = np.round(ds[y_dim], decimals=decimals)
    return ds


def to_datetimeindex(ds):
    if ds.indexes["time"].dtype == "O":
        ds["time"] = ds.indexes["time"].to_datetimeindex()
    return ds


def remove_duplicates(ds):
    return ds.sel(time=~ds.get_index("time").duplicated())


PREPROCESSORS = {
    "round_latlon": round_latlon,
    "to_datetimeindex": to_datetimeindex,
    "remove_duplicates": remove_duplicates,
}


class DataAdapter(object, metaclass=ABCMeta):
    """General Interface to data source for HydroMT"""

    _DEFAULT_DRIVER = None  # placeholder
    _DRIVERS = {}

    def __init__(
        self,
        path,
        driver,
        crs=None,
        nodata=None,
        rename={},
        unit_mult={},
        unit_add={},
        meta={},
        placeholders={},
        **kwargs,
    ):
        # general arguments
        self.path = path
        # driver and driver keyword-arguments
        # check for non default driver based on extension
        if driver is None:
            driver = self._DRIVERS.get(
                str(path).split(".")[-1].lower(), self._DEFAULT_DRIVER
            )
        self.driver = driver
        self.kwargs = kwargs
        # data adapter arguments
        self.crs = crs
        self.nodata = nodata
        self.rename = rename
        self.unit_mult = unit_mult
        self.unit_add = unit_add
        # meta data
        self.meta = {k: v for k, v in meta.items() if v is not None}

    @property
    def data_type(self):
        return type(self).__name__.replace("Adapter", "")

    def summary(self):
        """Returns a dictionary summary of the data adapter."""
        return dict(
            path=self.path,
            data_type=self.data_type,
            driver=self.driver,
            **self.meta,
        )

    def to_dict(self):
        """Returns a dictionary view of the data source. Can be used to initialize
        the data adapter."""
        source = dict(data_type=self.data_type)
        for k, v in vars(self).items():
            if v is not None and (not isinstance(v, dict) or len(v) > 0):
                source.update({k: v})
        return source

    def __str__(self):
        return yaml.dump(self.to_dict())

    def __repr__(self):
        return self.__str__()

    def resolve_paths(self, time_tuple: Tuple = None, variables: list = None):
        """Resolve {year}, {month} and {variable} keywords
        in self.path based on 'time_tuple' and 'variables' arguments

        Parameters
        ----------
        time_tuple : tuple of str, optional
            Start and end data in string format understood by :py:func:`pandas.to_timedelta`, by default None
        variables : list of str, optional
            List of variable names, by default None

        Returns
        -------
        List:
            list of filenames matching the path pattern given date range and variables
        """
        known_keys = ["year", "month", "variable"]
        fns = []
        keys = []
        # rebuild path based on arguments and escape unknown keys
        path = ""
        for literal_text, key, fmt, _ in Formatter().parse(self.path):
            path += literal_text
            if key is None:
                continue
            key_str = "{" + f"{key}:{fmt}" + "}" if fmt else "{" + key + "}"
            # remove unused fields
            if key in ["year", "month"] and time_tuple is None:
                path += "*"
            elif key == "variable" and variables is None:
                path += "*"
            # escape unknown fields
            elif key is not None and key not in known_keys:
                path = path + "{" + key_str + "}"
            else:
                path = path + key_str
                keys.append(key)
        # resolve dates: month & year keys
        dates, vrs, postfix = [None], [None], ""
        if time_tuple is not None:
            dt = pd.to_timedelta(self.unit_add.get("time", 0), unit="s")
            trange = pd.to_datetime(list(time_tuple)) - dt
            freq, strf = ("m", "%Y-%m") if "month" in keys else ("a", "%Y")
            dates = pd.period_range(*trange, freq=freq)
            postfix += "; date range: " + " - ".join([t.strftime(strf) for t in trange])
        # resolve variables
        if variables is not None:
            mv_inv = {v: k for k, v in self.rename.items()}
            vrs = [mv_inv.get(var, var) for var in variables]
            postfix += f"; variables: {variables}"
        # get filenames with glob for all date / variable combinations
        # FIXME: glob won't work with other than local file systems; use fsspec instead
        for date, var in product(dates, vrs):
            fmt = {}
            if date is not None:
                fmt.update(year=date.year, month=date.month)
            if var is not None:
                fmt.update(variable=var)
            fns.extend(glob.glob(path.format(**fmt)))
        if len(fns) == 0:
            raise FileNotFoundError(f"No such file found: {path}{postfix}")
        return list(set(fns))  # return unique paths

    @abstractmethod
    def get_data(self, bbox, geom, buffer):
        """Return a view (lazy if possible) of the data with standardized field names.
        If bbox of mask are given, clip data to that extent"""
