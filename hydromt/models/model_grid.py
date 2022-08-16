# -*- coding: utf-8 -*-
"""HydroMT GridModel class definition"""

from typing import Dict, List, Tuple, Union, Optional
import logging
from os.path import join, isfile
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS

from .model_api import Model
from .. import workflows

__all__ = ["GridModel", "GridMixin"]
logger = logging.getLogger(__name__)


class GridMixin(object):
    # placeholders
    # xr.Dataset representation of all static parameter maps at the same resolution and bounds - renamed from staticmaps
    _grid = xr.Dataset()

    @property
    def grid(self):
        """Model static maps. Returns xarray.Dataset.
        Previously called staticmaps."""
        if len(self._grid) == 0:
            if self._read:
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
        nc_dict = dict()
        if len(self._grid) > 0:
            # _write_nc requires dict - use dummy key
            nc_dict.update({"grid": self._grid})
        self._write_nc(nc_dict, fn, **kwargs)


class GridModel(Model, GridMixin):

    # TODO: add here "res": "setup_region" or "res": "setup_grid" when generic method is available
    _CLI_ARGS = {"region": "setup_region"}

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

    # GridModel specific methods
    # TODO rename to setup_grid_from_table
    def setup_fromtable(self, path_or_key: str, fn_map: str, out_vars: List, **kwargs):
        """This function creates additional grid layers based on a table reclassification

        Adds model layers defined in out_vars

        Args:
            path_or_key (str): Name of RasterDataset in DataCatalog file (.yml).
            mapping_fn (str): Path to a mapping csv file from RasterDataset to parameter values in out_vars.
            out_vars (List): List of parameters to keep.
            **kwargs: if the RasterDataset has multiple variables, select the correct variable
        """
        self.logger.info(f"Preparing {out_vars} parameter maps from raster.")

        # TODO - makefn_map flexible with the DataCatalog as well
        if not isfile(fn_map):
            self.logger.error(
                f"Mapping file not found: {fn_map}"
            )  # TODO ask diff between logger.error and RaiseValueError (will log.error stop the code?)
            return

        # read RasterDataset map to DataArray
        da = self.data_catalog.get_rasterdataset(
            path_or_key, geom=self.region, buffer=2, **kwargs
        )  # TODO - ask about why buffer is 2  #variables=["landuse"]

        if not isinstance(da, xr.DataArray):
            raise ValueError(
                "RasterData has multiple variables, please specify variable to select"
            )

        # process landuse
        ds_maps = workflows.grid_maptable(
            da=da,
            ds_like=self.grid,
            fn_map=fn_map,
            params=out_vars,
            logger=self.logger,
        )
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_maps.data_vars}
        self.set_grid(ds_maps.rename(rmdict))

    def setup_fromvector(
        self,
        key: str,
        col2raster: Optional[str] = "",
        rasterize_method: Optional[str] = "value",
    ):
        """Creates additional grid map based on a vector, located either in the data library or geoms.

        Adds grid maps model layers defined in key

        Args:
            key (str): value in either geoms or the data catalog to extract the vector
            col2raster (str, optional): name of column in the vector to use for rasterization. Defaults to "".
            rasterize_method (str, optional): Method to rasterize the vector ("value" or "fraction"). Defaults to "value".
        """

        self.logger.info(f"Preparing {key} parameter maps from vector.")

        # Vector sources can be from staticgeoms, data_catalog or fn
        if key in self._geoms.keys():
            gdf = self._geoms[key]
        elif key in self.data_catalog:
            gdf = self.data_catalog.get_geodataframe(
                key, geom=self.region, dst_crs=self.crs
            )  # TODO: I think this is updated if gets a fn: ask
        else:
            self.logger.warning(f"Source '{key}' not found in geoms nor data_catalog.")
            return

        if gdf.empty:
            raise ValueError(f"No shapes of {key} found within region, exiting.")
        else:
            ds = workflows.vector_to_grid(
                gdf=gdf,
                ds_like=self.grid,
                col_name=col2raster,
                method=rasterize_method,
                mask_name="mask",
                logger=self.logger,
            )
        self.set_grid(ds.rename(key))

    # Properties for subclass GridModel
    @property
    def res(self) -> Tuple[float, float]:
        """Returns the resolution of the model grid."""
        if len(self._grid) > 0:
            return self.grid.raster.res

    @property
    def transform(self):
        """Returns spatial transform of the model grid."""
        if len(self._grid) > 0:
            return self.grid.raster.transform

    @property
    def crs(self) -> Union[CRS, None]:
        """Returns coordinate reference system embedded in the model grid."""
        if len(self._grid) > 0:
            return CRS(self._grid.raster.crs)

    @property
    def bounds(self) -> List[float]:
        """Returns the bounding box of the model grid."""
        if len(self._grid) > 0:
            return self.grid.raster.bounds

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns the geometry of the model area of interest."""
        region = gpd.GeoDataFrame()
        if "region" in self.geoms:
            region = self.geoms["region"]
        elif len(self.grid) > 0:
            crs = self.grid.raster.crs
            if crs is None and hasattr(crs, "to_epsg"):
                crs = crs.to_epsg()  # not all CRS have an EPSG code
            region = gpd.GeoDataFrame(geometry=[box(*self.bounds)], crs=crs)
        return region

    def _test_model_api(self) -> List:
        """Test compliance with HydroMT GridModel API.

        Returns
        -------
        non_compliant: list
            List of model components that are non-compliant with the model API structure.
        """
        return super()._test_model_api({"grid": xr.Dataset})
