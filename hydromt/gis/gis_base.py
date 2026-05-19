"""Base for HydroMT GIS xarray extension."""

import datetime
import itertools
from typing import Any, Union

import cftime
import dask
import numpy as np
import pandas as pd
import xarray as xr
from pyproj.crs import CRS

__all__ = ["XGeoBase"]

GEO_MAP_COORD = "spatial_ref"
XDIMS = ("x", "longitude", "lon", "long")
YDIMS = ("y", "latitude", "lat")


class XGeoBase(object):
    """Base class for the GIS extensions for xarray."""

    def __init__(self, xarray_obj: Union[xr.DataArray, xr.Dataset]) -> None:
        """Initialize new object based on the xarray object provided."""
        self._obj = xarray_obj
        self._crs = None

    def _initialize_spatial_coord(self) -> None:
        """Small initializer for the spatial attributes."""
        # create new coordinate with attributes in which to save x_dim, y_dim and crs.
        # other spatial properties are always calculated on the fly to ensure
        # consistency with data
        if isinstance(self._obj, xr.Dataset) and GEO_MAP_COORD in self._obj.data_vars:
            self._obj = self._obj.set_coords(GEO_MAP_COORD)
        elif GEO_MAP_COORD not in self._obj.coords:
            self._obj.coords[GEO_MAP_COORD] = xr.Variable((), 0)
        if isinstance(self._obj.coords[GEO_MAP_COORD].data, dask.array.Array):
            self._obj[GEO_MAP_COORD].load()  # make sure spatial ref is not lazy

    @property
    def attrs(self) -> dict:
        """Return dictionary of spatial attributes."""
        self._initialize_spatial_coord()
        return self._obj.coords[GEO_MAP_COORD].attrs

    @attrs.setter
    def attrs(self, values: dict):
        """Set new spatial attributes."""
        self._initialize_spatial_coord()
        self._obj.coords[GEO_MAP_COORD].attrs = values

    def get_attr(self, key, placeholder=None) -> Any:
        """Return single spatial attribute."""
        return self.attrs.get(key, placeholder)

    @property
    def time_dim(self):
        """Time dimension name."""
        dim = self.get_attr("time_dim")
        # Try early return the timedim in attrs
        if dim and dim in self._obj.dims:
            coord = self._obj.coords.get(dim)
            if coord is not None and np.dtype(coord).type == np.datetime64:
                return dim

        # Not found, so try detect
        tdims = [
            d
            for d in self._obj.dims
            if d in self._obj.coords and self._is_datetime_coord(self._obj.coords[d])
        ]
        if len(tdims) == 0:
            self.attrs.update(time_dim=None)
            return None
        elif len(tdims) == 1:
            self.attrs.update(time_dim=tdims[0])
            return tdims[0]
        else:
            raise ValueError(
                f"Multiple time dimensions found: {tdims}. "
                "Set 'time_dim' attribute manually to resolve."
            )

    @property
    def crs(self) -> CRS:
        """Return horizontal Coordinate Reference System."""
        # return horizontal crs by default to avoid errors downstream
        # with reproject / rasterize etc.
        if self._crs is not None:
            crs = self._crs
        else:
            crs = self.set_crs()
        return crs

    def set_crs(self, input_crs=None, write_crs=True) -> CRS:
        """Set the Coordinate Reference System.

        Arguments
        ---------
        input_crs: int, dict, or str, optional
            Coordinate Reference System. Accepts EPSG codes (int or str)
            and proj (str or dict)
        write_crs: bool, optional
            If True (default), write CRS to attributes.
        """
        crs_names = ["crs_wkt", "crs", "epsg"]
        names = list(self._obj.coords.keys())
        if isinstance(self._obj, xr.Dataset):
            names = names + list(self._obj.data_vars.keys())
        # user defined
        if isinstance(input_crs, (int, str, dict)):
            input_crs = CRS.from_user_input(input_crs)
        # look in grid_mapping and data variable attributes
        elif input_crs is not None and not isinstance(input_crs, CRS):
            raise ValueError(f"Invalid CRS type: {type(input_crs)}")
        elif not isinstance(input_crs, CRS):
            crs = None
            for name in crs_names:
                # check default > GEO_MAP_COORDS attrs, then global attrs
                if name in self.attrs:
                    crs = self.attrs.get(name)
                    break
                if name in self._obj.attrs:
                    crs = self._obj.attrs.pop(name)
                    break
            if crs is None:  # check data var and coords attrs
                for var, name in itertools.product(names, crs_names):
                    if name in self._obj[var].attrs:
                        crs = self._obj[var].attrs.pop(name)
                        break
            if crs is not None:
                # avoid Warning 1: +init=epsg:XXXX syntax is deprecated
                if isinstance(crs, str):
                    crs = crs.removeprefix("+init=")
                try:
                    input_crs = CRS.from_user_input(crs)
                except RuntimeError:
                    pass  # continue to next name in crs_names
        if input_crs is not None:
            if write_crs:
                grid_map_attrs = input_crs.to_cf()
                crs_wkt = input_crs.to_wkt()
                grid_map_attrs["spatial_ref"] = crs_wkt
                grid_map_attrs["crs_wkt"] = crs_wkt
                self.attrs = grid_map_attrs
            self._crs = input_crs
            return input_crs

    @staticmethod
    def _is_datetime_coord(coord) -> bool:
        """Return True if coord is datetime64 or object of datetimes."""
        dtype = np.asarray(coord).dtype
        if np.issubdtype(dtype, np.datetime64):
            return True
        if np.issubdtype(dtype, np.object_):
            # Handle object arrays of datetime.datetime or np.datetime64
            values = np.asarray(coord)
            if values.size == 0:
                return False
            sample = values.ravel()[0]
            return isinstance(
                sample,
                (
                    datetime.datetime,
                    np.datetime64,
                    pd.Timestamp,
                    cftime.DatetimeGregorian,
                ),
            )
        return False
