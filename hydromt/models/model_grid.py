# -*- coding: utf-8 -*-
"""HydroMT GridModel class definition"""

import logging
from os.path import isfile, join
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from pyproj import CRS
from shapely.geometry import box

from .. import workflows
from .model_api import Model

__all__ = ["GridModel"]
logger = logging.getLogger(__name__)


class GridMixin(object):
    # placeholders
    # xr.Dataset representation of all static parameter maps at the same resolution and bounds - renamed from staticmaps
    _API = {"grid": xr.Dataset}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._grid = xr.Dataset()

    @property
    def grid(self):
        """Model static gridded data. Returns xarray.Dataset.
        Previously called staticmaps."""
        if len(self._grid) == 0 and self._read:
            self.read_grid()
        return self._grid

    def set_grid(
        self,
        data: Union[xr.DataArray, xr.Dataset, np.ndarray],
        name: Optional[str] = None,
    ):
        """Add data to grid.

        All layers of grid must have identical spatial coordinates.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to grid
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray
            and ignored if data is a Dataset
        """
        # NOTE: variables in a dataset are not longer renamed as used to be the case in set_staticmaps
        name_required = isinstance(data, np.ndarray) or (
            isinstance(data, xr.DataArray) and data.name is None
        )
        if name is None and name_required:
            raise ValueError(f"Unable to set {type(data).__name__} data without a name")
        if isinstance(data, np.ndarray):
            if data.shape != self.grid.raster.shape:
                raise ValueError("Shape of data and grid maps do not match")
            data = xr.DataArray(dims=self.grid.raster.dims, data=data, name=name)
        if isinstance(data, xr.DataArray):
            if name is not None:  # rename
                data.name = name
            data = data.to_dataset()
        elif not isinstance(data, xr.Dataset):
            raise ValueError(f"cannot set data of type {type(data).__name__}")
        for dvar in data.data_vars:
            if dvar in self._grid:
                if self._read:
                    self.logger.warning(f"Replacing grid map: {dvar}")
            self._grid[dvar] = data[dvar]

    def read_grid(self, fn: str = "grid/grid.nc", **kwargs) -> None:
        """Read model grid data at <root>/<fn> and add to grid property

        key-word arguments are passed to :py:func:`xarray.open_dataset`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, by default 'grid/grid.nc'
        """
        self._assert_read_mode
        for ds in self._read_nc(fn, **kwargs).values():
            self.set_grid(ds)

    def write_grid(self, fn: str = "grid/grid.nc", **kwargs) -> None:
        """Write model grid data to netcdf file at <root>/<fn>

        key-word arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, by default 'grid/grid.nc'
        """
        if len(self._grid) == 0:
            self.logger.debug("No grid data found, skip writing.")
        else:
            self._assert_write_mode
            # _write_nc requires dict - use dummy 'grid' key
            self._write_nc({"grid": self._grid}, fn, **kwargs)


class GridModel(GridMixin, Model):
    """Model class Grid Model for gridded models in HydroMT"""

    # TODO: add here "res": "setup_region" or "res": "setup_grid" when generic method is available
    _CLI_ARGS = {"region": "setup_region"}
    _NAME = "grid_model"

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = None,
        data_libs: List[str] = None,
        logger=logger,
    ):
        """Initialize a GridModel for distributed models with a regular grid."""
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )

    def read(
        self,
        components: List = [
            "config",
            "grid",
            "geoms",
            "forcing",
            "states",
            "results",
        ],
    ) -> None:
        """Read the complete model schematization and configuration from model files.

        Parameters
        ----------
        components : List, optional
            List of model components to read, each should have an associated read_<component> method.
            By default ['config', 'maps', 'grid', 'geoms', 'forcing', 'states', 'results']
        """
        super().read(components=components)

    def write(
        self,
        components: List = ["config", "grid", "geoms", "forcing", "states"],
    ) -> None:
        """Write the complete model schematization and configuration to model files.

        Parameters
        ----------
        components : List, optional
            List of model components to write, each should have an associated write_<component> method.
            By default ['config', 'maps', 'grid', 'geoms', 'forcing', 'states']
        """
        super().write(components=components)

    # Properties for subclass GridModel
    @property
    def res(self) -> Tuple[float, float]:
        """Returns the resolution of the model grid."""
        if len(self.grid) > 0:
            return self.grid.raster.res

    @property
    def transform(self):
        """Returns spatial transform of the model grid."""
        if len(self.grid) > 0:
            return self.grid.raster.transform

    @property
    def crs(self) -> Union[CRS, None]:
        """Returns coordinate reference system embedded in the model grid."""
        if len(self.grid) > 0 and self.grid.raster.crs is not None:
            return CRS(self.grid.raster.crs)

    @property
    def bounds(self) -> List[float]:
        """Returns the bounding box of the model grid."""
        if len(self.grid) > 0:
            return self.grid.raster.bounds

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns the geometry of the model area of interest."""
        region = gpd.GeoDataFrame()
        if "region" in self.geoms:
            region = self.geoms["region"]
        elif len(self.grid) > 0:
            crs = self.crs
            if crs is not None and hasattr(crs, "to_epsg"):
                crs = crs.to_epsg()  # not all CRS have an EPSG code
            region = gpd.GeoDataFrame(geometry=[box(*self.bounds)], crs=crs)
        return region

    def set_crs(self, crs: CRS) -> None:
        """Set coordinate reference system of the model grid."""
        if len(self.grid) > 0:
            self.grid.raster.set_crs(crs)
