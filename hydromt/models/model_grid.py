import pytest
import sys, os
from os.path import join, dirname, basename, isfile, isdir
from .model_api import Model
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import box

# from hydromt import workflows, flw, io
from .. import config, log, workflows

from typing import List

import logging
import os

__all__ = ["GridModel"]
logger = logging.getLogger(__name__)


class GridModel(Model):
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
        self._grid = (
            xr.Dataset()
        )  # xr.Dataset representation of all static parameter maps at the same resolution and bounds - renamed from staticmaps

    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        super().read()
        self.read_grid()

    def write(self):
        """Method to write the complete model schematization and configuration to file."""
        super().write()
        self.write_grid()

    def read_grid(self):
        """Read grid at <root/?/> and parse to xarray Dataset - previously called read_staticmaps"""
        # to read gdal raster files use: hydromt.open_mfraster()
        # to read netcdf use: xarray.open_dataset()
        if not self._write:
            # start fresh in read-only mode
            self._grid = xr.Dataset()
        if isfile(
            join(self.root, "staticmaps", "staticmaps.nc")
        ):  # Change of file not implemented yet
            self._grid = xr.open_dataset(join(self.root, "staticmaps", "staticmaps.nc"))
        if isfile(join(self.root, "grid", "grid.nc")):
            self._grid = xr.open_dataset(join(self.root, "grid", "grid.nc"))

    def write_grid(self):
        """Write grid at <root/?/> in xarray.Dataset - previously write_staticmaps"""
        # to write to gdal raster files use: self.staticmaps.raster.to_mapstack()
        # to write to netcdf use: self.staticmaps.to_netcdf()
        if not self._write:
            raise IOError("Model opened in read-only mode")
        elif not self._grid:
            self.logger.warning("No grid maps to write - Exiting")
            return
        # filename
        fn_default = join(self.root, "grid", "grid.nc")
        self.logger.info(f"Write grid maps to {self.root}")

        ds_out = self.grid
        ds_out.to_netcdf(fn_default)

    def set_grid(
        self, data: Union[xr.DataArray, xr.Dataset], name: Optional[str] = None
    ):
        """Add data to grid.

        All layers of grid must have identical spatial coordinates.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to grid
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray
            or to select a variable from a Dataset.
        """
        if name is None:
            if isinstance(data, xr.DataArray) and data.name is not None:
                name = data.name
            elif not isinstance(data, xr.Dataset):
                raise ValueError("Setting a map requires a name")
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
        if len(self._grid) == 0:  # new data
            self._grid = data
        else:
            if isinstance(data, np.ndarray):
                if data.shape != self.shape:
                    raise ValueError("Shape of data and grid maps do not match")
                data = xr.DataArray(dims=self.dims, data=data, name=name).to_dataset()
            for dvar in data.data_vars.keys():
                if dvar in self._grid:
                    if self._read:
                        self.logger.warning(f"Replacing grid map: {dvar}")
                self._grid[dvar] = data[dvar]

    # GridModel specific methods
    def setup_fromtable(self, path_or_key: str, fn_map: str, out_vars: List, **kwargs):
        """This function creates additional staticmaps layers based on a table reclassification

        Adds model layers defined in out_vars

        Args:
            path_or_key (str): Name of RasterDataset in DataCatalog file (.yml).
            mapping_fn (str): Path to a mapping csv file from RasterDataset to parameter values in out_vars.
            out_vars (List): List of parameters to keep.
            **kwargs: if the RasterDataset has multiple variables, select the correct variable
        """
        self.logger.info(f"Preparing {out_vars} parameter maps from raster.")

        if not isfile(
            fn_map
        ):  # TODO - makefn_map flexible with the DataCatalog as well
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
    def grid(self):
        """xarray.Dataset representation of all static parameter maps - previously called staticmaps"""
        if len(self._grid) == 0:
            if self._read:
                self.read_grid()
        return self._grid

    @property
    def dims(self) -> tuple:
        """Returns spatial dimension names of grid."""
        return self.grid.raster.dims

    @property
    def coords(self) -> dict:
        """Returns coordinates of grid."""
        return self.grid.raster.coords

    @property
    def res(self) -> tuple:
        """Returns coordinates of grid."""
        return self.grid.raster.res

    @property
    def transform(self):
        """Returns spatial transform grid."""
        return self.grid.raster.transform

    @property
    def width(self):
        """Returns width of grid."""
        return self.grid.raster.width

    @property
    def height(self):
        """Returns height of grid."""
        return self.grid.raster.height

    @property
    def shape(self) -> tuple:
        """Returns shape of grid."""
        return self.grid.raster.shape

    @property
    def bounds(self) -> tuple:
        """Returns shape of grid."""
        return self.grid.raster.bounds

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns geometry of region of the model area of interest."""
        region = gpd.GeoDataFrame()
        if "region" in self.geoms:
            region = self.geoms["region"]
        elif len(self.grid) > 0:
            crs = self.grid.raster.crs
            if crs is None and crs.to_epsg() is not None:
                crs = crs.to_epsg()  # not all CRS have an EPSG code
            region = gpd.GeoDataFrame(geometry=[box(*self.bounds)], crs=crs)
        return region

    def test_model_grid(self):
        """Test compliance to model API instances.

        Returns
        -------
        non_compliant: list
            List of objects that are non-compliant with the model API structure.
        """
        non_compliant = []
        # grid
        if not isinstance(self.grid, xr.Dataset):
            non_compliant.append("grid")

        return non_compliant
