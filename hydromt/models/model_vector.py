# -*- coding: utf-8 -*-
"""HydroMT VectorModel class definition."""

import logging
import os
from os.path import dirname, isdir, isfile, join
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box

from .model_api import Model

__all__ = ["VectorModel"]
logger = logging.getLogger(__name__)


class VectorMixin:
    _API = {"vector": xr.Dataset}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._vector = None  # xr.Dataset()

    @property
    def vector(self) -> xr.Dataset:
        """Model vector (polygon) data. Returns xr.Dataset geometry coordinate."""
        if self._vector is None:
            self._vector = xr.Dataset()
            if self._read:
                self.read_vector()
        return self._vector

    def set_vector(
        self,
        data: Union[xr.DataArray, xr.Dataset, np.ndarray],
        name: Optional[str] = None,
    ) -> None:
        """Add data to vector.

        All layers of vector must have identical spatial index.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new data to add to vector
        name: str, optional
            Name of new data, this is used to overwrite the name of a DataArray
            or to select a variable from a Dataset.
        """
        name_required = isinstance(data, np.ndarray) or (
            isinstance(data, xr.DataArray) and data.name is None
        )
        if name is None and name_required:
            raise ValueError(f"Unable to set {type(data).__name__} data without a name")
        if isinstance(data, np.ndarray) and "geometry" in self.vector:
            # TODO: index name is hard coded. Using GeoDataset.index property once ready
            index = self.vector["index"]
            if data.size != index.size and data.ndim == 1:
                raise ValueError("Size of data and number of vector do not match")
            data = xr.DataArray(dims=["index"], data=data)
        if isinstance(data, xr.DataArray):
            if name is not None:  # rename
                data.name = name
            data = data.to_dataset()
        elif not isinstance(data, xr.Dataset):
            raise ValueError(f"cannot set data of type {type(data).__name__}")
        for dvar in data.data_vars:
            if dvar in self.vector:
                self.logger.warning(f"Replacing vector variable: {dvar}")
            # TODO: check on index coordinate before merging
            self._vector[dvar] = data[dvar]

    def read_vector(
        self,
        fn: str = "vector/vector.nc",
        fn_geom: str = "vector/vector.geojson",
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
            by default 'vector/vector.nc'
        fn_geom : str, optional
            geojson filename relative to model root,
            by default 'vector/vector.geojson'
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
        self.set_vector(ds)

    def write_vector(
        self,
        fn: str = "vector/vector.nc",
        fn_geom: str = "vector/vector.geojson",
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
            by default 'vector/vector.nc'
        fn_geom : str, optional
            geojson filename relative to model root,
            by default 'vector/vector.geojson'
        **kwargs:
            Additional keyword arguments that are passed to the `write_nc`
            function.
        """
        if len(self.vector) == 0:
            self.logger.debug("No vector data found, skip writing.")
            return
        self._assert_write_mode()
        # write geometry
        ds = self.vector
        gdf = gpd.GeoDataFrame(geometry=ds["geometry"].values, crs=ds.rio.crs)
        if not isdir(dirname(join(self.root, fn_geom))):
            os.makedirs(dirname(join(self.root, fn_geom)))
        gdf.to_file(join(self.root, fn_geom), driver="GeoJSON")
        # write_nc requires dict - use dummy key
        nc_dict = {"vector": ds.drop_vars("geometry")}
        self.write_nc(nc_dict, fn, **kwargs)


class VectorModel(VectorMixin, Model):

    """Model class Vector Model for vector (polygons) models in HydroMT."""

    _CLI_ARGS = {"region": "setup_region"}
    _NAME = "vector_model"

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = None,
        data_libs: List[str] = None,
        logger=logger,
    ):
        """Initialize a VectorModel for lumped and semi-distributed models."""
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
            By default ['config', 'maps', 'vector', 'geoms', 'tables',
            'forcing', 'states', 'results']
        """
        components = components or [
            "config",
            "vector",
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
            'maps', 'vector', 'geoms', 'tables', 'forcing', 'states']
        """
        components = components or [
            "config",
            "vector",
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
        elif len(self.vector) > 0:
            ds = self.vector
            gdf = gpd.GeoDataFrame(geometry=ds["geometry"].values, crs=ds.rio.crs)
            region = gpd.GeoDataFrame(geometry=[box(*gdf.total_bounds)], crs=gdf.crs)
        return region
