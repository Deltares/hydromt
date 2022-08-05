import pytest
import sys, os
from .model_api import Model
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from os.path import join, dirname, basename, isfile, isdir

from typing import Tuple, Union, Optional

import logging
import os

__all__ = ["LumpedModel"]
logger = logging.getLogger(__name__)


class LumpedModel(Model):
    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = None,
        data_libs: List[str] = None,
        logger=logger,
    ):
        # Initialize with the Model class
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )

        # placeholders
        self._response_units = (
            xr.Dataset()
        )  # representation of all response units. Geometry defined as coordinate "geometry"

    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        super().read()
        self.read_response_units()
        # Other specifics to LumpedModel...

    def write(self):
        """Method to write the complete model schematization and configuration to file."""
        super().write()
        self.write_response_units()
        # Other specifics to LumpedModel...

    def read_response_units(self):
        if not self._write:
            # start fresh in read-only mode
            self._response_units = xr.Dataset()
        if isfile(
            join(self.root, "response_units", "response_units.nc")
        ):  # Change of file not implemented yet
            ds = xr.open_dataset(join(self.root, "response_units", "response_units.nc"))
        if isfile(join(self.root, "response_units", "response_units.geoJSON")):
            gdf = gpd.GeoDataFrame(
                join(self.root, "response_units", "response_units.geoJSON")
            )
            self._response_units = ds.assign_coords(
                geometry=(["index"], gdf["geometry"])
            )

    def write_response_units(self):
        """Write response_units at <root/?/> in xarray.Dataset and a GeoJSON of the geometry"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        elif not self._response_units:
            self.logger.warning("No response_units data to write - Exiting")
            return
        # filename
        self.logger.info(f"Write response_units to {self.root}")

        ds_out = self.response_units
        ds_out.drop("geometry").to_netcdf(
            join(self.root, "response_units", "response_units.nc")
        )
        gdf = gpd.GeoDataFrame(ds_out[["geometry"]].to_dataframe(), crs=self.crs)
        gdf.to_file(join(self.root, "response_units", "response_units.GeoJSON"))

    def set_response_units(
        self, data: Union[xr.DataArray, xr.Dataset], name: Optional[str] = None
    ):
        """Add data to response_units.

        All layers of repsonse_units must have identical spatial index.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new data to add to response_units
        name: str, optional
            Name of new data, this is used to overwrite the name of a DataArray
            or to select a variable from a Dataset.
        """
        if name is None:
            if isinstance(data, xr.DataArray) and data.name is not None:
                name = data.name
            elif not isinstance(data, xr.Dataset):
                raise ValueError("Setting a layer requires a name")
        elif name is not None and isinstance(data, xr.Dataset):
            data_vars = list(data.data_vars)
            if len(data_vars) == 1 and name not in data_vars:
                data = data.rename_vars({data_vars[0]: name})
            elif name not in data_vars:
                raise ValueError("Name not found in DataSet")
            else:
                data = data[[name]]
        if isinstance(data, xr.DataArray):
            data.name = name
            data = data.to_dataset()
        if np.all(
            len(self._response_units) == 0 and "geometry" in data.coords
        ):  # new data with a geometry
            self._response_units = data
        else:
            for dvar in data.data_vars.keys():
                if dvar in self._response_units:
                    if self._read:
                        self.logger.warning(f"Replacing data for: {dvar}")
                self._response_units[dvar] = data[dvar]

    @property
    def response_units(self):
        """xr.Dataset object with an object of shapely object of the Geometry"""
        if not self._response_units:
            if self._read:
                self.read_response_units()
        return self._response_units

    @property
    def shape(self):
        return self._response_units.coords["index"].shape
