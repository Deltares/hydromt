"""DataAdapter class."""
from __future__ import annotations

import logging
import os
import pathlib
import warnings
from abc import ABCMeta, abstractmethod
from itertools import product
from string import Formatter
from typing import Optional, Union

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pystac import Catalog as StacCatalog
from upath import UPath

from hydromt._typing import (
    Bbox,
    Data,
    ErrorHandleMethod,
    Geom,
    GeomBuffer,
    NoDataStrategy,
    StrPath,
    TimeRange,
    Variables,
)

logger = logging.getLogger(__name__)


__all__ = ["DataAdapter"]


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


def harmonise_dims(ds):
    """Harmonise lon-lat-time dimensions.

    Where needed:
        - lon: Convert longitude coordinates from 0-360 to -180-180
        - lat: Do N->S orientation instead of S->N
        - time: Convert to datetimeindex.

    Parameters
    ----------
    ds: xr.DataSet
        DataSet with dims to harmonise

    Returns
    -------
    ds: xr.DataSet
        DataSet with harmonised longitude-latitude-time dimensions
    """
    # Longitude
    x_dim = ds.raster.x_dim
    lons = ds[x_dim].values
    if np.any(lons > 180):
        ds[x_dim] = xr.Variable(x_dim, np.where(lons > 180, lons - 360, lons))
        ds = ds.sortby(x_dim)
    # Latitude
    y_dim = ds.raster.y_dim
    if np.diff(ds[y_dim].values)[0] > 0:
        ds = ds.reindex({y_dim: ds[y_dim][::-1]})
    # Final check for lat-lon
    assert (
        np.diff(ds[y_dim].values)[0] < 0
    ), "orientation not N->S after get_data preprocess set_lon_lat_axis"
    assert (
        np.diff(ds[x_dim].values)[0] > 0
    ), "orientation not W->E after get_data preprocess set_lon_lat_axis"
    # Time
    if ds.indexes["time"].dtype == "O":
        ds = to_datetimeindex(ds)

    return ds


PREPROCESSORS = {
    "round_latlon": round_latlon,
    "to_datetimeindex": to_datetimeindex,
    "remove_duplicates": remove_duplicates,
    "harmonise_dims": harmonise_dims,
}


class DataAdapter(object, metaclass=ABCMeta):

    """General Interface to data source for HydroMT."""

    _DEFAULT_DRIVER = None  # placeholder
    _DRIVERS = {}

    def __init__(
        self,
        path: StrPath,
        driver: Optional[str] = None,
        filesystem: Optional[str] = None,
        nodata: Optional[Union[dict, float, int]] = None,
        rename: Optional[dict] = None,
        unit_mult: Optional[dict] = None,
        unit_add: Optional[dict] = None,
        meta: Optional[dict] = None,
        attrs: Optional[dict] = None,
        driver_kwargs: Optional[dict] = None,
        storage_options: Optional[dict] = None,
        name: str = "",
        catalog_name: str = "",
        provider: Optional[str] = None,
        version: Optional[str] = None,
    ):
        """General Interface to data source for HydroMT.

        Parameters
        ----------
        path: str, Path
            Path to data source. If the dataset consists of multiple files, the path may
            contain {variable}, {year}, {month} placeholders as well as path
            search pattern using a '*' wildcard.
        driver: {'vector', 'netcdf', 'zarr'}, optional
            Driver to read files with,
            for 'vector' :py:func:`~hydromt.io.open_geodataset`,
            for 'netcdf' :py:func:`xarray.open_mfdataset`.
            By default the driver is inferred from the file extension and falls back to
            'vector' if unknown.
        filesystem: str, optional
            Filesystem where the data is stored (local, cloud, http etc.).
            If None (default) the filesystem is inferred from the path.
            See :py:func:`fsspec.registry.known_implementations` for all options.
        nodata: float, int, optional
            Missing value number. Only used if the data has no native missing value.
            Nodata values can be differentiated between variables using a dictionary.
        rename: dict, optional
            Mapping of native data source variable to output source variable name as
            required by hydroMT.
        unit_mult, unit_add: dict, optional
            Scaling multiplication and addition to change to map from the native
            data unit to the output data unit as required by hydroMT.
        meta: dict, optional
            Metadata information of dataset, prefably containing the following keys:
            {'source_version', 'source_url', 'source_license',
            'paper_ref', 'paper_doi', 'category'}
        placeholders: dict, optional
            Placeholders to expand yaml entry to multiple entries (name and path)
            based on placeholder values
        attrs: dict, optional
            Additional attributes relating to data variables. For instance unit
            or long name of the variable.
        driver_kwargs, dict, optional
            Additional key-word arguments passed to the driver.
        storage_options: dict, optional
            Additional key-word arguments passed to the fsspec FileSystem object.
        name, catalog_name: str, optional
            Name of the dataset and catalog, optional for now.

        """
        unit_mult = unit_mult or {}
        unit_add = unit_add or {}
        meta = meta or {}
        attrs = attrs or {}
        driver_kwargs = driver_kwargs or {}
        storage_options = storage_options or {}
        rename = rename or {}
        self.name = name
        self.catalog_name = catalog_name
        self.provider = provider
        self.version = str(version) if version is not None else None  # version as str
        # general arguments
        self.path = path
        # driver and driver keyword-arguments
        # check for non default driver based on extension
        if driver is None:
            driver = self._DRIVERS.get(
                str(path).split(".")[-1].lower(), self._DEFAULT_DRIVER
            )
        self.driver = driver
        if "storage_options" in driver_kwargs:
            warnings.warn(
                "storage_options should be provided as a separate argument, "
                "not as part of driver_kwargs",
                DeprecationWarning,
                stacklevel=2,
            )
            storage_options.update(driver_kwargs.pop("storage_options"))
        self.driver_kwargs = driver_kwargs
        self.storage_options = storage_options
        # get filesystem
        self.filesystem = filesystem
        self._fs = None  # placeholder for fsspec filesystem object
        # data adapter arguments
        self.nodata = nodata
        self.rename = rename
        self.unit_mult = unit_mult
        self.unit_add = unit_add
        # meta data
        self.meta = {k: v for k, v in meta.items() if v is not None}
        # variable attributes
        self.attrs = {k: v for k, v in attrs.items() if v is not None}
        # keep track of wether the data is used
        self._used = False

    @property
    def data_type(self):
        """Return the datatype of the addapter."""
        return type(self).__name__.replace("Adapter", "")

    def mark_as_used(self):
        """Mark the data adapter as used."""
        self._used = True

    def summary(self):
        """Return a dictionary summary of the data adapter."""
        return dict(
            path=self.path,
            data_type=self.data_type,
            driver=self.driver,
            **self.meta,
        )

    def to_dict(self):
        """Return a dictionary view of the data source.

        Can be used to initialize the data adapter.
        """
        source = dict(data_type=self.data_type)
        for k, v in vars(self).items():
            if k in ["name", "catalog_name"] or k.startswith("_"):
                continue  # do not add these identifiers
            if v is not None and (not isinstance(v, dict) or len(v) > 0):
                source.update({k: v})
        return source

    def __str__(self):
        """Return string representation of self in yaml."""
        return yaml.dump(self.to_dict())

    def __repr__(self):
        """Pretty print string representation of self."""
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        """Return True if self and other are equal."""
        if type(other) is type(self):
            return self.to_dict() == other.to_dict()
        else:
            return False

    @property
    def fs(self):
        """Return the filesystem object ."""
        if self._fs is not None:
            return self._fs
        fs = None
        if self.filesystem:  # use filesystem property
            protocols = {"local": "file"}
            protocol = protocols.get(self.filesystem, self.filesystem)
        else:  # infer filesystem from path
            upath = UPath(self.path)
            if hasattr(upath, "fs"):
                fs = upath.fs
            else:
                protocol = "file"
                if hasattr(upath, "protocol"):
                    protocol = upath.protocol
                elif not isinstance(upath, (pathlib.PosixPath, pathlib.WindowsPath)):
                    warnings.warn(
                        "No filesystem found for path, using default local filesystem",
                        stacklevel=2,
                    )
        if fs is None:
            fs = fsspec.filesystem(protocol, **self.storage_options)
        self._fs = fs
        return fs

    def _resolve_paths(
        self,
        time_tuple: Optional[TimeRange] = None,
        variables: Optional[Variables] = None,
        zoom_level: Optional[int] = 0,
    ):
        """Resolve {year}, {month} and {variable} keywords in self.path.

          Keywords are based on 'time_tuple' and 'variables'.

        Parameters
        ----------
        time_tuple : tuple of str, optional
            Start and end date in string format understood by
            :py:func:`pandas.to_timedelta`, by default None
        variables : list of str, optional
            List of variable names, by default None
        zoom_level : int
            Parsed zoom level to use, by default 0
            See :py:meth:`RasterDataAdapter._parse_zoom_level` for more info
        logger:
            The logger to use. If none is provided, the devault logger will be used.

        Returns
        -------
        fns: list of str
            list of filenames matching the path pattern given date range and variables
        """
        known_keys = ["year", "month", "zoom_level", "variable"]
        fns = []
        keys = []
        # rebuild path based on arguments and escape unknown keys
        if "{" in str(self.path):
            path = ""
            for literal_text, key, fmt, _ in Formatter().parse(str(self.path)):
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
        else:
            path = self.path

        # expand path based on keys for dates, variables and zoomlevel
        dates, vrs, postfix = [None], [None], ""
        fmt = {}
        if len(keys) > 0:
            # resolve dates: month & year keys
            if time_tuple is not None:
                dt = pd.to_timedelta(self.unit_add.get("time", 0), unit="s")
                t_range = pd.to_datetime(list(time_tuple)) - dt
                freq, strf = ("m", "%Y-%m") if "month" in keys else ("a", "%Y")
                dates = pd.period_range(*t_range, freq=freq)
                t_range_str = [t.strftime(strf) for t in t_range]
                postfix += "; date range: " + " - ".join(t_range_str)
            # resolve variables
            if variables is not None:
                variables = np.atleast_1d(variables).tolist()
                mv_inv = {v: k for k, v in self.rename.items()}
                vrs = [mv_inv.get(var, var) for var in variables]  # type: ignore
                postfix += f"; variables: {variables}"
            # update based on zoomlevel (size = 1)
            if "zoom_level" in keys:
                zoom_level = zoom_level or 0
                fmt.update(zoom_level=zoom_level)
            # update based on dates and variables  (size >= 1)
            for date, var in product(dates, vrs):
                if date is not None:
                    fmt.update(year=date.year, month=date.month)
                if var is not None:
                    fmt.update(variable=var)
                fns.append(str(path).format(**fmt))
        else:
            fns.append(path)

        protocol = str(path).split("://")[0] if "://" in str(path) else ""
        if protocol and self.filesystem is None:
            self.filesystem = protocol
        # For s3, anonymous connection still requires --no-sign-request profile to
        # read the data setting environment variable works
        # assume anonymous connection if storage_options is not set
        if protocol in ["s3"]:
            if not self.storage_options:
                self.storage_options = {"anon": True}
            if self.storage_options.get("anon", False):
                os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
            else:
                os.environ["AWS_NO_SIGN_REQUEST"] = "NO"
        # expand path with glob and check if files exists
        fns_out = []
        for fn in list(set(fns)):
            if "*" in str(fn):
                for fn0 in self.fs.glob(fn):
                    if protocol and not str(fn0).startswith(protocol):
                        fn0 = f"{protocol}://{fn0}"
                    fns_out.append(UPath(fn0, **self.storage_options))
            elif self.fs.exists(fn):
                fns_out.append(UPath(fn, **self.storage_options))

        if len(fns_out) == 0:
            raise FileNotFoundError(f"No such file found: {path}{postfix}")

        return fns_out

    @abstractmethod
    def get_data(
        self,
        bbox: Optional[Bbox],
        geom: Optional[Geom],
        buffer: Optional[GeomBuffer],
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ):
        """Return a view (lazy if possible) of the data with standardized field names.

        If bbox of mask are given, clip data to that extent.
        """

    @staticmethod
    def _single_var_as_array(
        ds: Data, single_var_as_array: bool, variable_name: Optional[str] = None
    ) -> Data:
        # return data array if single variable dataset
        dvars = list(ds.data_vars.keys())
        if single_var_as_array and len(dvars) == 1:
            da = ds[dvars[0]]
            if isinstance(variable_name, list) and len(variable_name) == 1:
                da.name = variable_name[0]
            elif isinstance(variable_name, str):
                da.name = variable_name
            return da
        else:
            return ds

    @abstractmethod
    def to_stac_catalog(
        self,
        on_error: ErrorHandleMethod = ErrorHandleMethod.COERCE,
    ) -> Optional[StacCatalog]:
        """Create a stac item from the data adapter to be added to a stac catalog."""
