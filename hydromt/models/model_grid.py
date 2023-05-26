# -*- coding: utf-8 -*-
"""HydroMT GridModel class definition."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from shapely.geometry import box

from .. import gis_utils, raster, workflows
from .model_api import Model

__all__ = ["GridModel"]
logger = logging.getLogger(__name__)


class GridMixin(object):
    # placeholders
    # xr.Dataset representation of all static parameter maps at the same resolution and
    # bounds - renamed from staticmaps
    _API = {"grid": xr.Dataset}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._grid = xr.Dataset()

    # generic grid methods
    def setup_grid_from_constant(
        self,
        constant: Union[int, float],
        name: str,
        dtype: Optional[str] = "float32",
        nodata: Optional[Union[int, float]] = None,
        mask_name: Optional[str] = "mask",
    ) -> List[str]:
        """HYDROMT CORE METHOD: Adds a grid based on a constant value.

        Parameters
        ----------
        constant: int, float
            Constant value to fill grid with.
        name: str
            Name of grid.
        dtype: str, optional
            Data type of grid. By default 'float32'.
        nodata: int, float, optional
            Nodata value. By default infered from dtype.
        mask_name: str, optional
            Name of mask in self.grid to use for masking raster_fn. By default 'mask'.
            Use None to disable masking.

        Returns
        -------
        list
            Names of added model grid layer.
        """
        da = workflows.grid.grid_from_constant(
            grid_like=self.grid,
            constant=constant,
            name=name,
            dtype=dtype,
            nodata=nodata,
            mask_name=mask_name,
        )
        # Add to grid
        self.set_grid(da)

        return [name]

    def setup_grid_from_rasterdataset(
        self,
        raster_fn: Union[str, Path, xr.DataArray, xr.Dataset],
        variables: Optional[List] = None,
        fill_method: Optional[str] = None,
        reproject_method: Optional[Union[List, str]] = "nearest",
        mask_name: Optional[str] = "mask",
        rename: Optional[Dict] = dict(),
    ) -> List[str]:
        """HYDROMT CORE METHOD: Add data variable(s) from ``raster_fn`` to grid object.

        If raster is a dataset, all variables will be added unless ``variables`` list
        is specified.

        Adds model layers:

        * **raster.name** grid: data from raster_fn

        Parameters
        ----------
        raster_fn: str, Path, xr.DataArray, xr.Dataset
            Data catalog key, path to raster file or raster xarray data object.
            If a path to a raster file is provided it will be added
            to the data_catalog with its name based on the file basename without
            extension.
        variables: list, optional
            List of variables to add to grid from raster_fn. By default all.
        fill_method : str, optional
            If specified, fills nodata values using fill_nodata method.
            Available methods are {'linear', 'nearest', 'cubic', 'rio_idw'}.
        reproject_method: list, str, optional
            See rasterio.warp.reproject for existing methods, by default 'nearest'.
            Can provide a list corresponding to ``variables``.
        mask_name: str, optional
            Name of mask in self.grid to use for masking raster_fn. By default 'mask'.
            Use None to disable masking.
        rename: dict, optional
            Dictionary to rename variable names in raster_fn before adding to grid
            {'name_in_raster_fn': 'name_in_grid'}. By default empty.

        Returns
        -------
        list
            Names of added model map layers
        """
        self.logger.info(f"Preparing grid data from raster source {raster_fn}")
        # Read raster data and select variables
        ds = self.data_catalog.get_rasterdataset(
            raster_fn,
            geom=self.region,
            buffer=2,
            variables=variables,
            single_var_as_array=False,
        )
        # Data resampling
        ds_out = workflows.grid.grid_from_rasterdataset(
            grid_like=self.grid,
            ds=ds,
            variables=variables,
            fill_method=fill_method,
            reproject_method=reproject_method,
            mask_name=mask_name,
            rename=rename,
        )
        # Add to grid
        self.set_grid(ds_out)

        return list(ds_out.data_vars.keys())

    def setup_grid_from_raster_reclass(
        self,
        raster_fn: Union[str, Path, xr.DataArray],
        reclass_table_fn: Union[str, Path, pd.DataFrame],
        reclass_variables: List,
        variable: Optional[str] = None,
        fill_method: Optional[str] = None,
        reproject_method: Optional[Union[List, str]] = "nearest",
        mask_name: Optional[str] = "mask",
        rename: Optional[Dict] = dict(),
        **kwargs,
    ) -> List[str]:
        """Add data variables to grid object by reclassifying the data in ``raster_fn``.

        Reclassifications are based on ``reclass_table_fn``.

        Adds model layers:

        * **reclass_variables** grid: reclassified raster data

        Parameters
        ----------
        raster_fn: str, Path, xr.DataArray
            Data catalog key, path to raster file or raster xarray data object.
            Should be a DataArray. Else use `variable` argument for selection.
        reclass_table_fn: str, Path, pd.DataFrame
            Data catalog key, path to tabular data file or tabular pandas dataframe
            object for the reclassification table of `raster_fn`.
        reclass_variables: list
            List of reclass_variables from reclass_table_fn table to add to maps.
            Index column should match values in `raster_fn`.
        variable: str, optional
            Name of raster_fn dataset variable to use. This is only required when
            reading datasets with multiple variables.
            By default None.
        fill_method : str, optional
            If specified, fills nodata values in `raster_fn` using fill_nodata method
            before reclassifying. Available methods are
            {'linear', 'nearest', 'cubic', 'rio_idw'}.
        reproject_method: str, optional
            See rasterio.warp.reproject for existing methods, by default "nearest".
            Can provide a list corresponding to ``reclass_variables``.
        mask_name: str, optional
            Name of mask in self.grid to use for masking raster_fn. By default 'mask'.
            Use None to disable masking.
        rename: dict, optional
            Dictionary to rename variable names in reclass_variables before adding to
            grid {'name_in_reclass_table': 'name_in_grid'}. By default empty.

        **kwargs : dict
            Additional keyword arguments to be passed to `get_rasterdataset`

        Returns
        -------
        list
            Names of added model grid layers
        """
        self.logger.info(
            f"Preparing grid data by reclassifying the data in {raster_fn} based "
            f"on {reclass_table_fn}"
        )
        # Read raster data and remapping table
        da = self.data_catalog.get_rasterdataset(
            raster_fn, geom=self.region, buffer=2, variables=variable, **kwargs
        )
        if not isinstance(da, xr.DataArray):
            raise ValueError(
                f"raster_fn {raster_fn} should be a single variable. "
                "Please select one using the 'variable' argument"
            )
        df_vars = self.data_catalog.get_dataframe(
            reclass_table_fn, variables=reclass_variables
        )
        # Data resampling
        ds_vars = workflows.grid.grid_from_raster_reclass(
            grid_like=self.grid,
            da=da,
            reclass_table=df_vars,
            reclass_variables=reclass_variables,
            fill_method=fill_method,
            reproject_method=reproject_method,
            mask_name=mask_name,
            rename=rename,
        )
        # Add to maps
        self.set_grid(ds_vars.rename(rename))

        return list(ds_vars.data_vars.keys())

    def setup_grid_from_geodataframe(
        self,
        vector_fn: Union[str, Path, gpd.GeoDataFrame],
        variables: Optional[Union[List, str]] = None,
        nodata: Optional[Union[List, int, float]] = -1,
        rasterize_method: Optional[str] = "value",
        mask_name: Optional[str] = "mask",
        rename: Optional[Dict] = dict(),
        all_touched: Optional[bool] = True,
    ) -> List[str]:
        """Add data variables to grid object by rasterizing the data from ``vector_fn``.

        Several type of rasterization are possible:
            * "fraction": the fraction of the grid cell covered by the vector
                shape is returned.
            * "area": the area of the grid cell covered by the vector shape is returned.
            * "value": the value from the variables columns of vector_fn are used.
                If this is used, variables must be specified.

        Parameters
        ----------
        vector_fn : str, Path, gpd.GeoDataFrame
            Data catalog key, path to vector file or a vector geopandas object.
        variables : List, str, optional
            List of variables to add to grid from vector_fn. Required if
            rasterize_method is "value", by default None.
        nodata : List, int, float, optional
            No data value to use for rasterization, by default -1. If a list is
            provided, it should have the same length has variables.
        rasterize_method : str, optional
            Method to rasterize the vector data. Either {"value", "fraction", "area"}.
            If "value", the value from the variables columns in vector_fn are used
            directly in the raster. If "fraction", the fraction of the grid
            cell covered by the vector file is returned. If "area", the area of the
            grid cell covered by the vector file is returned.
        mask_name: str, optional
            Name of mask in self.grid to use for masking raster_fn. By default 'mask'.
            Use None to disable masking.
        rename: dict, optional
            Dictionary to rename variable names in variables before adding to grid
            {'name_in_variables': 'name_in_grid'}. To rename with method fraction or
            area use {'vector_fn': 'name_in_grid'}. By default empty.
        all_touched : bool, optional
            If True (default), all pixels touched by geometries will be burned in.
            If false, only pixels whose center is within the polygon or that are
            selected by Bresenham's line algorithm will be burned in.

        Returns
        -------
        list
            Names of added model grid layers
        """
        self.logger.info(f"Preparing grid data from vector '{vector_fn}'.")
        gdf = self.data_catalog.get_geodataframe(
            vector_fn, geom=self.region, dst_crs=self.crs
        )
        if gdf.empty:
            self.logger.warning(
                f"No shapes of {vector_fn} found within region,"
                " skipping setup_grid_from_vector."
            )
            return
        # Data resampling
        if vector_fn in rename.keys():
            # In case of choosing a new name with area or fraction method pass
            # the name directly
            rename = rename[vector_fn]
        ds = workflows.grid.grid_from_geodataframe(
            grid_like=self.grid,
            gdf=gdf,
            variables=variables,
            nodata=nodata,
            rasterize_method=rasterize_method,
            mask_name=mask_name,
            rename=rename,
            all_touched=all_touched,
        )
        # Add to grid
        self.set_grid(ds)

        return list(ds.data_vars.keys())

    @property
    def grid(self):
        """Model static gridded data."""
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
        # NOTE: variables in a dataset are not longer renamed as used to be the case in
        # set_staticmaps
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
        if len(self._grid) == 0:  # new data
            self._grid = data
        else:
            for dvar in data.data_vars:
                if dvar in self._grid:
                    if self._read:
                        self.logger.warning(f"Replacing grid map: {dvar}")
                self._grid[dvar] = data[dvar]

    def read_grid(self, fn: str = "grid/grid.nc", **kwargs) -> None:
        """Read model grid data at <root>/<fn> and add to grid property.

        key-word arguments are passed to :py:func:`xarray.open_dataset`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, by default 'grid/grid.nc'
        **kwargs : dict
            Additional keyword arguments to be passed to the `_read_nc` method.
        """
        self._assert_read_mode
        for ds in self._read_nc(fn, **kwargs).values():
            self.set_grid(ds)

    def write_grid(self, fn: str = "grid/grid.nc", **kwargs) -> None:
        """Write model grid data to netcdf file at <root>/<fn>.

        key-word arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, by default 'grid/grid.nc'
        **kwargs : dict
            Additional keyword arguments to be passed to the `_write_nc` method.
        """
        if len(self._grid) == 0:
            self.logger.debug("No grid data found, skip writing.")
        else:
            self._assert_write_mode
            # _write_nc requires dict - use dummy 'grid' key
            self._write_nc({"grid": self._grid}, fn, **kwargs)


class GridModel(GridMixin, Model):

    """Model class Grid Model for gridded models in HydroMT."""

    _CLI_ARGS = {"region": "setup_grid"}
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

    ## GENERIC SETUP METHODS
    def setup_grid(
        self,
        region: dict,
        res: Optional[float] = None,
        crs: int = None,
        hydrography_fn: Optional[str] = None,
        basin_index_fn: Optional[str] = None,
        add_mask: bool = True,
        align: bool = True,
    ) -> xr.DataArray:
        """HYDROMT CORE METHOD: Create a 2D regular grid or reads an existing grid.

        An 2D regular grid will be created from a geometry (geom_fn) or bbox. If an
        existing grid is given, then no new grid will be generated.

        Adds/Updates model layers (if add_mask):
        * **mask** grid mask: add grid mask to grid object

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, e.g.:
            * {'bbox': [xmin, ymin, xmax, ymax]}
            * {'geom': 'path/to/polygon_geometry'}
            * {'grid': 'path/to/grid_file'}
            * {'basin': [x, y]}

            Region must be of kind [grid, bbox, geom, basin, subbasin, interbasin].
        res: float
            Resolution used to generate 2D grid [unit of the CRS], required if region
            is not based on 'grid'.
        crs : EPSG code, int, str optional
            EPSG code of the model or "utm" to let hydromt find the closest projected
            crs. If None using the one from region.
        hydrography_fn : str
            Name of data source for hydrography data. Required if region is of kind
                'basin', 'subbasin' or 'interbasin'.

            * Required variables: ['flwdir'] and any other 'snapping' variable required
                to define the region.

            * Optional variables: ['basins'] if the `region` is based on a
                (sub)(inter)basins without a 'bounds' argument.

        basin_index_fn : str
            Name of data source with basin (bounding box) geometries associated with
            the 'basins' layer of `hydrography_fn`. Only required if the `region` is
            based on a (sub)(inter)basins without a 'bounds' argument.
        add_mask : bool, optional
            Add mask variable to grid object, by default True.
        align : bool, optional
            If True (default), align target transform to resolution.

        Returns
        -------
        grid : xr.DataArray
            Generated grid mask.
        """
        self.logger.info("Preparing 2D grid.")

        kind = next(iter(region))  # first key of region
        if kind in ["bbox", "geom", "basin", "subbasin", "interbasin"]:
            # Do not parse_region for grid as we want to allow for more (file) formats
            kind, region = workflows.parse_region(region, logger=self.logger)
        elif kind != "grid":
            raise ValueError(
                f"Region for grid must be of kind [grid, bbox, geom, basin, subbasin,"
                f" interbasin], kind {kind} not understood."
            )

        # Derive xcoords, ycoords and geom for the different kind options
        if kind in ["bbox", "geom"]:
            if not isinstance(res, (int, float)):
                raise ValueError("res argument required for kind 'bbox', 'geom'")
            if kind == "bbox":
                bbox = region["bbox"]
                geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
            elif kind == "geom":
                geom = region["geom"]
                if geom.crs is None:
                    raise ValueError('Model region "geom" has no CRS')
            if crs is not None:
                crs = gis_utils.parse_crs(crs, bbox=geom.total_bounds)
                geom = geom.to_crs(crs)
            # Generate grid based on res for region bbox
            # TODO add warning on res value if crs is projected or not?
            xmin, ymin, xmax, ymax = geom.total_bounds
            res = abs(res)
            if align:
                xmin = round(xmin / res) * res
                ymin = round(ymin / res) * res
                xmax = round(xmax / res) * res
                ymax = round(ymax / res) * res
            xcoords = np.linspace(
                xmin + res / 2,
                xmax - res / 2,
                num=round((xmax - xmin) / res),
                endpoint=True,
            )
            ycoords = np.flip(
                np.linspace(
                    ymin + res / 2,
                    ymax - res / 2,
                    num=round((ymax - ymin) / res),
                    endpoint=True,
                )
            )

        elif kind in ["basin", "subbasin", "interbasin"]:
            # retrieve global hydrography data (lazy!)
            ds_hyd = self.data_catalog.get_rasterdataset(hydrography_fn)
            if "bounds" not in region:
                region.update(basin_index=self.data_catalog[basin_index_fn])
            # get basin geometry
            geom, xy = workflows.get_basin_geometry(
                ds=ds_hyd,
                kind=kind,
                logger=self.logger,
                **region,
            )
            # get ds_hyd again but clipped to geom, one variable is enough
            da_hyd = self.data_catalog.get_rasterdataset(
                hydrography_fn, geom=geom, variables=["flwdir"]
            )
            if not isinstance(res, (int, float)):
                self.logger.info(
                    "res argument not defined, using resolution of "
                    f"hydrography_fn {da_hyd.raster.res}"
                )
                res = da_hyd.raster.res
            # Reproject da_hyd based on crs and grid and align, method is not important
            # only coords will be used
            # TODO add warning on res value if crs is projected or not?
            if res != da_hyd.raster.res and crs != da_hyd.raster.crs:
                da_hyd = da_hyd.raster.reproject(
                    dst_crs=crs,
                    dst_res=res,
                    align=align,
                )
            # Get xycoords, geom
            xcoords = da_hyd.raster.xcoords.values
            ycoords = da_hyd.raster.ycoords.values
            if geom.crs != da_hyd.raster.crs:
                crs = da_hyd.raster.crs
                geom = geom.to_crs(crs)
        elif kind == "grid":
            # Support more formats for grid input (netcdf, zarr, io.open_raster)
            fn = region[kind]
            if isinstance(fn, (xr.DataArray, xr.Dataset)):
                da_like = fn
            else:
                da_like = self.data_catalog.get_rasterdataset(fn)
            # Get xycoords, geom
            xcoords = da_like.raster.xcoords.values
            ycoords = da_like.raster.ycoords.values
            geom = da_like.raster.box
            if crs is not None or res is not None:
                self.logger.warning(
                    "For region kind 'grid', the gris crs/res are used and not"
                    f" user-defined crs {crs} or res {res}"
                )
            crs = da_like.raster.crs

        # Instantiate grid object
        coords = {"y": ycoords, "x": xcoords}
        # Generate grid using hydromt full method
        grid = raster.full(
            coords=coords,
            nodata=1,
            dtype=np.uint8,
            name="mask",
            attrs={},
            crs=geom.crs,
            lazy=False,
        )
        # Create geometry_mask with geom
        if add_mask:
            grid = grid.raster.geometry_mask(geom, all_touched=True)
            grid.name = "mask"
        # Remove mask variable mask from grid if not add_mask
        else:
            grid = grid.to_dataset()
            grid = grid.drop_vars("mask")

        # Add region and grid to model
        self.set_geoms(geom, "region")
        self.set_grid(grid)

        return grid

    ## I/O
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
            List of model components to read, each should have an associated
            read_<component> method. By default ['config', 'maps', 'grid',
            'geoms', 'forcing', 'states', 'results']
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
            List of model components to write, each should have an
            associated write_<component> method.
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
        if self.grid.raster.crs is not None:
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
