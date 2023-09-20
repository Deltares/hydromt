# -*- coding: utf-8 -*-
"""HydroMT LumpedModel class definition."""

import logging
import os
from os.path import dirname, isdir, isfile, join
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box

from .model_api import Model

__all__ = ["LumpedModel"]
logger = logging.getLogger(__name__)


class LumpedMixin:
    _API = {"response_units": xr.Dataset}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._response_units = None  # xr.Dataset()

    @property
    def response_units(self) -> xr.Dataset:
        """Model response unit (lumped) data. Returns xr.Dataset geometry coordinate."""
        if self._response_units is None:
            self._response_units = xr.Dataset()
            if self._read:
                self.read_response_units()
        return self._response_units

    def set_response_units(
        self,
        data: Union[xr.DataArray, xr.Dataset, np.ndarray],
        name: Optional[str] = None,
    ) -> None:
        """Add data to response_units.

        All layers of response_units must have identical spatial index.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new data to add to response_units
        name: str, optional
            Name of new data, this is used to overwrite the name of a DataArray
            or to select a variable from a Dataset.
        """
        name_required = isinstance(data, np.ndarray) or (
            isinstance(data, xr.DataArray) and data.name is None
        )
        if name is None and name_required:
            raise ValueError(f"Unable to set {type(data).__name__} data without a name")
        if isinstance(data, np.ndarray) and "geometry" in self.response_units:
            # TODO: index name is hard coded. Using GeoDataset.index property once ready
            index = self.response_units["index"]
            if data.size != index.size and data.ndim == 1:
                raise ValueError(
                    "Size of data and number of response_units do not match"
                )
            data = xr.DataArray(dims=["index"], data=data)
        if isinstance(data, xr.DataArray):
            if name is not None:  # rename
                data.name = name
            data = data.to_dataset()
        elif not isinstance(data, xr.Dataset):
            raise ValueError(f"cannot set data of type {type(data).__name__}")
        for dvar in data.data_vars:
            if dvar in self.response_units:
                self.logger.warning(f"Replacing response_units variable: {dvar}")
            # TODO: check on index coordinate before merging
            self._response_units[dvar] = data[dvar]

    def read_response_units(
        self,
        fn: str = "response_units/response_units.nc",
        fn_geom: str = "response_units/response_units.geojson",
        **kwargs,
    ) -> None:
        """Read model response units from combined netcdf and geojson file.

        Files are read at <root>/<fn> and geojson file at <root>/<fn_geom>.
        The netcdf file contains the attribute data and the geojson file the geometry
        vector data. key-word arguments are passed to
        :py:meth:`~hydromt.models.Model.read_nc`

        Parameters
        ----------
        fn : str, optional
            netcdf filename relative to model root,
            by default 'response_units/response_units.nc'
        fn_geom : str, optional
            geojson filename relative to model root,
            by default 'response_units/response_units.geojson'
        **kwargs:
            Additional keyword arguments that are passed to the `read_nc`
            function.
        """
        self._assert_read_mode()
        ds = xr.merge(self.read_nc(fn, **kwargs).values())
        if isfile(join(self.root, fn_geom)):
            gdf = gpd.read_file(join(self.root, fn_geom))
            # TODO: index name is hard coded. Using GeoDataset.index property once ready
            ds = ds.assign_coords(geometry=(["index"], gdf["geometry"]))
            if gdf.crs is not None:  # parse crs
                ds = ds.rio.write_crs(gdf.crs)
        self.set_response_units(ds)

    def write_response_units(
        self,
        fn: str = "response_units/response_units.nc",
        fn_geom: str = "response_units/response_units.geojson",
        **kwargs,
    ):
        """Write model response units to combined netcdf and geojson files.

        Files are written at <root>/<fn> and at <root>/<fn_geom> respectively.
        The netcdf file contains the attribute data and the geojson file the geometry
        vector data. Key-word arguments are passed to
        :py:meth:`~hydromt.models.Model.write_nc`

        Parameters
        ----------
        fn : str, optional
            netcdf filename relative to model root,
            by default 'response_units/response_units.nc'
        fn_geom : str, optional
            geojson filename relative to model root,
            by default 'response_units/response_units.geojson'
        **kwargs:
            Additional keyword arguments that are passed to the `write_nc`
            function.
        """
        if len(self.response_units) == 0:
            self.logger.debug("No response_units data found, skip writing.")
            return
        self._assert_write_mode()
        # write geometry
        ds = self.response_units
        gdf = gpd.GeoDataFrame(geometry=ds["geometry"].values, crs=ds.rio.crs)
        if not isdir(dirname(join(self.root, fn_geom))):
            os.makedirs(dirname(join(self.root, fn_geom)))
        gdf.to_file(join(self.root, fn_geom), driver="GeoJSON")
        # write_nc requires dict - use dummy key
        nc_dict = {"response_units": ds.drop_vars("geometry")}
        self.write_nc(nc_dict, fn, **kwargs)


class LumpedModel(LumpedMixin, Model):

    """Model class Lumped Model for lumped models in HydroMT."""

    _CLI_ARGS = {"region": "setup_region"}
    _NAME = "lumped_model"

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = None,
        data_libs: List[str] = None,
        logger=logger,
    ):
        """Initialize a LumpedModel for lumped and semi-distributed models."""
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )

    def read(
        self,
        components: List = None,
    ) -> None:
        """Read the complete model schematization and configuration from model files.

        Parameters
        ----------
        components : List, optional
            List of model components to read, each should have an
            associated read_<component> method.
            By default ['config', 'maps', 'response_units', 'geoms', 'tables',
            'forcing', 'states', 'results']
        """
        if components is None:
            components = [
                "config",
                "response_units",
                "geoms",
                "tables",
                "forcing",
                "states",
                "results",
            ]
        super().read(components=components)

    def write(
        self,
        components: List = None,
    ) -> None:
        """Write the complete model schematization and configuration to model files.

        Parameters
        ----------
        components : List, optional
            List of model components to write, each should have an
            associated write_<component> method. By default ['config',
            'maps', 'response_units', 'geoms', 'tables', 'forcing', 'states']
        """
        if components is None:
            components = [
                "config",
                "response_units",
                "geoms",
                "tables",
                "forcing",
                "states",
            ]
        super().write(components=components)

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns the geometry of the model area of interest."""
        region = gpd.GeoDataFrame()
        if "region" in self.geoms:
            region = self.geoms["region"]
        elif len(self.response_units) > 0:
            ds = self.response_units
            gdf = gpd.GeoDataFrame(geometry=ds["geometry"].values, crs=ds.rio.crs)
            region = gpd.GeoDataFrame(geometry=[box(*gdf.total_bounds)], crs=gdf.crs)
        return region
