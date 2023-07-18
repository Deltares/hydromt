"""DataAdapter class."""
from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from itertools import product
from string import Formatter

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from fsspec.implementations import local
from pyproj import CRS
from upath import UPath

from .. import _compat, gis_utils

logger = logging.getLogger(__name__)


__all__ = [
    "DataAdapter",
]

FILESYSTEMS = ["local"]
# Add filesystems from optional dependencies
if _compat.HAS_GCSFS:
    import gcsfs

    FILESYSTEMS.append("gcs")
if _compat.HAS_S3FS:
    import s3fs

    FILESYSTEMS.append("s3")


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
        path,
        driver=None,
        filesystem="local",
        nodata=None,
        rename={},
        unit_mult={},
        unit_add={},
        meta={},
        attrs={},
        driver_kwargs={},
        name="",  # optional for now
        catalog_name="",  # optional for now
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
        filesystem: {'local', 'gcs', 's3'}, optional
            Filesystem where the data is stored (local, cloud, http etc.).
            By default, local.
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
        name, catalog_name: str, optional
            Name of the dataset and catalog, optional for now.

        """
        self.name = name
        self.catalog_name = catalog_name
        # general arguments
        self.path = path
        # driver and driver keyword-arguments
        # check for non default driver based on extension
        if driver is None:
            driver = self._DRIVERS.get(
                str(path).split(".")[-1].lower(), self._DEFAULT_DRIVER
            )
        self.driver = driver
        self.filesystem = filesystem
        self.driver_kwargs = driver_kwargs

        # data adapter arguments
        self.nodata = nodata
        self.rename = rename
        self.unit_mult = unit_mult
        self.unit_add = unit_add
        # meta data
        self.meta = {k: v for k, v in meta.items() if v is not None}
        # variable attributes
        self.attrs = {k: v for k, v in attrs.items() if v is not None}

    @property
    def data_type(self):
        """Return the datatype of the addapter."""
        return type(self).__name__.replace("Adapter", "")

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
            if k in ["name", "catalog_name"]:
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

    def _parse_zoom_level(
        self,
        zoom_level: int | tuple = None,
        geom: gpd.GeoSeries = None,
        bbox: list = None,
        logger=logger,
    ) -> int:
        """Return nearest smaller zoom level.

        Based on zoom resolutions defined in data catalog.
        """
        # common pyproj crs axis units
        known_units = ["degree", "metre", "US survey foot"]
        if self.zoom_levels is None or len(self.zoom_levels) == 0:
            logger.warning("No zoom levels available, default to zero")
            return 0
        zls = list(self.zoom_levels.keys())
        if zoom_level is None:  # return first zoomlevel (assume these are ordered)
            return next(iter(zls))
        # parse zoom_level argument
        if (
            isinstance(zoom_level, tuple)
            and isinstance(zoom_level[0], (int, float))
            and isinstance(zoom_level[1], str)
            and len(zoom_level) == 2
        ):
            res, unit = zoom_level
            # covert 'meter' and foot to official pyproj units
            unit = {"meter": "metre", "foot": "US survey foot"}.get(unit, unit)
            if unit not in known_units:
                raise TypeError(
                    f"zoom_level unit {unit} not understood;"
                    f" should be one of {known_units}"
                )
        elif not isinstance(zoom_level, int):
            raise TypeError(
                f"zoom_level argument not understood: {zoom_level}; should be a float"
            )
        else:
            return zoom_level
        if self.crs:
            # convert res if different unit than crs
            crs = CRS.from_user_input(self.crs)
            crs_unit = crs.axis_info[0].unit_name
            if crs_unit != unit and crs_unit not in known_units:
                raise NotImplementedError(
                    f"no conversion available for {unit} to {crs_unit}"
                )
            if unit != crs_unit:
                lat = 0
                if bbox is not None:
                    lat = (bbox[1] + bbox[3]) / 2
                elif geom is not None:
                    lat = geom.to_crs(4326).centroid.y.item()
                conversions = {
                    "degree": np.hypot(*gis_utils.cellres(lat=lat)),
                    "US survey foot": 0.3048,
                }
                res = res * conversions.get(unit, 1) / conversions.get(crs_unit, 1)
        # find nearest smaller zoomlevel
        eps = 1e-5  # allow for rounding errors
        smaller = [x < (res + eps) for x in self.zoom_levels.values()]
        zl = zls[-1] if all(smaller) else zls[max(smaller.index(False) - 1, 0)]
        logger.info(f"Getting data for zoom_level {zl} based on res {zoom_level}")
        return zl

    def resolve_paths(
        self,
        time_tuple: tuple = None,
        variables: list = None,
        zoom_level: int | tuple = None,
        geom: gpd.GeoSeries = None,
        bbox: list = None,
        logger=logger,
        **kwargs,
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
        zoom_level : int | tuple, optional
            zoom level of dataset, can be provided as tuple of
            (<zoom resolution>, <unit>)
        **kwargs
            key-word arguments are passed to fsspec FileSystem objects. Arguments
            depend on protocal (local, gcs, s3...).
        geom:
            A geoSeries describing the geometries.
        bbox:
            A list of bounding boxes.
        logger:
            The logger to use. If none is provided, the devault logger will be used.

        Returns
        -------
        List:
            list of filenames matching the path pattern given date range and variables
        """
        known_keys = ["year", "month", "zoom_level", "variable"]
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
        # parse zoom level
        if "zoom_level" in keys:
            # returns the first zoom_level if zoom_level is None
            zoom_level = self._parse_zoom_level(
                zoom_level=zoom_level, bbox=bbox, geom=geom, logger=logger
            )

        # get filenames with glob for all date / variable combinations
        fs = self.get_filesystem(**kwargs)
        fmt = {}
        # update based on zoomlevel (size = 1)
        if zoom_level is not None:
            fmt.update(zoom_level=zoom_level)
        # update based on dates and variables  (size >= 1)
        for date, var in product(dates, vrs):
            if date is not None:
                fmt.update(year=date.year, month=date.month)
            if var is not None:
                fmt.update(variable=var)
            fns.extend(fs.glob(path.format(**fmt)))
        if len(fns) == 0:
            raise FileNotFoundError(f"No such file found: {path}{postfix}")

        # With some fs like gcfs or s3fs, the first part of the path is not returned
        # properly with glob
        if not str(UPath(fns[0])).startswith(str(UPath(path))[0:2]):
            # Assumes it's the first part of the path that is not
            # correctly parsed with gcsfs, s3fs etc.
            last_parent = UPath(path).parents[-1]
            # add the rest of the path
            fns = [last_parent.joinpath(*UPath(fn).parts[1:]) for fn in fns]
        return list(set(fns))  # return unique paths

    def get_filesystem(self, **kwargs):
        """Return an initialised filesystem object."""
        if self.filesystem == "local":
            fs = local.LocalFileSystem(**kwargs)
        elif self.filesystem == "gcs":
            if _compat.HAS_GCSFS:
                fs = gcsfs.GCSFileSystem(**kwargs)
            else:
                raise ModuleNotFoundError(
                    "The gcsfs library is required to read data from gcs"
                    + "(Google Cloud Storage). Please install."
                )
        elif self.filesystem == "s3":
            if _compat.HAS_S3FS:
                fs = s3fs.S3FileSystem(**kwargs)
            else:
                raise ModuleNotFoundError(
                    "The s3fs library is required to read data from s3"
                    + " (Amazon Web Storage). Please install."
                )
        else:
            raise ValueError(
                f"Unknown or unsupported filesystem {self.filesystem}."
                + f" Use one of {FILESYSTEMS}"
            )

        return fs

    @abstractmethod
    def get_data(self, bbox, geom, buffer):
        """Return a view (lazy if possible) of the data with standardized field names.

        If bbox of mask are given, clip data to that extent.
        """
