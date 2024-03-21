"""Grid Component."""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from pyproj import CRS
from shapely.geometry import box

from hydromt import hydromt_step
from hydromt._typing.error import NoDataStrategy, _exec_nodata_strat
from hydromt._typing.type_def import DeferedFileClose
from hydromt.components.base import ModelComponent
from hydromt.gis import raster
from hydromt.io.readers import read_nc
from hydromt.io.writers import write_nc
from hydromt.workflows.grid import (
    grid_from_constant,
    grid_from_geodataframe,
    grid_from_raster_reclass,
    grid_from_rasterdataset,
    rotated_grid,
)

if TYPE_CHECKING:
    from hydromt.models.model import Model

__all__ = ["GridComponent"]


class GridComponent(ModelComponent):
    """ModelComponent class for grid components.

    This class is used for setting, creating, writing, and reading regular grid data for a
    HydroMT model. The grid component data stored in the ``data`` property of this class is of the
    hydromt.gis.raster.RasterDataset type which is an extension of xarray.Dataset for regular grid.
    """

    def __init__(
        self,
        model: "Model",
        region: str = "region",
    ):
        """Initialize a GridComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        """
        super().__init__(model=model)
        self._region_component = region
        self._data: Optional[xr.Dataset] = None

    def set(
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
        self._initialize_grid()
        # NOTE: variables in a dataset are not longer renamed as used to be the case in
        # set_staticmaps
        name_required = isinstance(data, np.ndarray) or (
            isinstance(data, xr.DataArray) and data.name is None
        )
        if name is None and name_required:
            raise ValueError(f"Unable to set {type(data).__name__} data without a name")
        if isinstance(data, np.ndarray):
            if data.shape != self._data.raster.shape:
                raise ValueError("Shape of data and grid maps do not match")
            data = xr.DataArray(dims=self._data.raster.dims, data=data, name=name)
        elif isinstance(data, xr.DataArray):
            if name is not None:
                data.name = name
            data = data.to_dataset()
        elif not isinstance(data, xr.Dataset):
            raise ValueError(f"cannot set data of type {type(data).__name__}")

        if len(self._data) == 0:  # empty grid
            self._data = data
        else:
            for dvar in data.data_vars:
                if dvar in self._data:
                    if self._root.is_reading_mode():
                        self._logger.warning(f"Replacing grid map: {dvar}")
                self._data[dvar] = data[dvar]

    @hydromt_step
    def write(
        self,
        fn: str = "grid/grid.nc",
        gdal_compliant: bool = False,
        rename_dims: bool = False,
        force_sn: bool = False,
        **kwargs,
    ) -> Optional[DeferedFileClose]:
        """Write model grid data to netcdf file at <root>/<fn>.

        key-word arguments are passed to :py:meth:`~hydromt.models.Model.write_nc`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, by default 'grid/grid.nc'
        gdal_compliant : bool, optional
            If True, write grid data in a way that is compatible with GDAL,
            by default False
        rename_dims: bool, optional
            If True and gdal_compliant, rename x_dim and y_dim to standard names
            depending on the CRS (x/y for projected and lat/lon for geographic).
        force_sn: bool, optional
            If True and gdal_compliant, forces the dataset to have
            South -> North orientation.
        **kwargs : dict
            Additional keyword arguments to be passed to the `write_nc` method.
        """
        self._root._assert_write_mode()
        if len(self.data) == 0:
            _exec_nodata_strat(
                msg="No grid data found, skip writing.",
                strategy=NoDataStrategy.IGNORE,
                logger=self._logger,
            )
        else:
            # write_nc requires dict - use dummy 'grid' key
            write_nc(  # Can return DeferedFileClose object
                {"grid": self.data},
                fn,
                temp_data_dir=self._model._TMP_DATA_DIR,
                gdal_compliant=gdal_compliant,
                rename_dims=rename_dims,
                force_sn=force_sn,
                **kwargs,
            )
        return None

    @hydromt_step
    def read(
        self,
        fn: str = "grid/grid.nc",
        mask_and_scale: bool = False,
        **kwargs,
    ) -> None:
        """Read model grid data at <root>/<fn> and add to grid property.

        key-word arguments are passed to :py:meth:`~hydromt.models.Model.read_nc`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, by default 'grid/grid.nc'
        mask_and_scale : bool, optional
            If True, replace array values equal to _FillValue with NA and scale values
            according to the formula original_values * scale_factor + add_offset, where
            _FillValue, scale_factor and add_offset are taken from variable attributes
        (if they exist).
        **kwargs : dict
            Additional keyword arguments to be passed to the `read_nc` method.
        """
        self._root._assert_read_mode()
        self._initialize_grid(skip_read=True)

        # Load grid data in r+ mode to allow overwritting netcdf files
        if self._root.is_reading_mode() and self._root.is_writing_mode():
            kwargs["load"] = True
        loaded_nc_files = read_nc(
            fn,
            self._root,
            logger=self._logger,
            single_var_as_array=False,
            mask_and_scale=mask_and_scale,
            **kwargs,
        )
        for ds in loaded_nc_files.values():
            self.set(ds)

    @hydromt_step
    def create(
        self,
        *,
        region: Optional[dict] = None,
        res: Optional[float] = None,
        crs: Optional[int] = None,
        rotated: bool = False,
        hydrography_fn: Optional[str] = None,
        basin_index_fn: Optional[str] = None,
        add_mask: bool = True,
        align: bool = True,
        dec_origin: int = 0,
        dec_rotation: int = 3,
    ) -> None:
        """HYDROMT CORE METHOD: Create a 2D regular grid or reads an existing grid.

        A 2D regular grid will be created from a geometry (geom_fn) or bbox. If an
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
        rotated : bool, optional
            if True, a minimum rotated rectangular grid is fitted around the region,
            by default False. Only  applies if region is of kind 'bbox', 'geom'
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
        dec_origin : int, optional
            number of decimals to round the origin coordinates, by default 0
        dec_rotation : int, optional
            number of decimals to round the rotation angle, by default 3

        Returns
        -------
        grid : xr.DataArray
            Generated grid mask.
        """
        self._logger.info("Preparing 2D grid.")

        # Pass region creation information to model.region.
        if region is not None:
            self._model.region.create(
                region=region,
                basin_index_fn=basin_index_fn,
                hydrography_fn=hydrography_fn,
            )

        kind = self._model.region.kind
        geom = self._model.region.data

        # Derive xcoords, ycoords and geom for the different kind options
        if kind in ["bbox", "geom"]:
            if not res:
                raise ValueError("res argument required for kind 'bbox', 'geom'")
            # Generate grid based on res for region bbox
            # TODO add warning on res value if crs is projected or not?
            if not rotated:
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
            else:  # rotated
                geomu = geom.unary_union
                x0, y0, mmax, nmax, rot = rotated_grid(
                    geomu, res, dec_origin=dec_origin, dec_rotation=dec_rotation
                )
                transform = (
                    Affine.translation(x0, y0)
                    * Affine.rotation(rot)
                    * Affine.scale(res, res)
                )
        elif kind in ["basin", "subbasin", "interbasin"]:
            # get ds_hyd again but clipped to geom, one variable is enough
            da_hyd = self._data_catalog.get_rasterdataset(
                hydrography_fn, geom=geom, variables=["flwdir"]
            )
            if not res:
                self._logger.info(
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
        elif kind == "grid":
            xcoords = geom.raster.xcoords.values
            ycoords = geom.raster.ycoords.values
            if crs is not None or res is not None:
                self._logger.warning(
                    "For region kind 'grid', the grid crs/res are used and not"
                    f" user-defined crs {crs} or res {res}"
                )

        # Instantiate grid object
        # Generate grid using hydromt full method
        if not rotated:
            coords = {"y": ycoords, "x": xcoords}
            grid = raster.full(
                coords=coords,
                nodata=1,
                dtype=np.uint8,
                name="mask",
                attrs={},
                crs=geom.crs,
                lazy=False,
            )
        else:
            grid = raster.full_from_transform(
                transform,
                shape=(mmax, nmax),
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

        self.set(grid)

    @property
    def res(self) -> Optional[Tuple[float, float]]:
        """Returns the resolution of the model grid."""
        if len(self.data) > 0:
            return self.data.raster.res
        _exec_nodata_strat(
            msg="No grid data found for deriving resolution",
            strategy=NoDataStrategy.IGNORE,
            logger=self._logger,
        )
        return None

    @property
    def transform(self) -> Optional[Affine]:
        """Returns spatial transform of the model grid."""
        if len(self.data) > 0:
            return self.data.raster.transform
        _exec_nodata_strat(
            msg="No grid data found for deriving transform",
            strategy=NoDataStrategy.IGNORE,
            logger=self._logger,
        )
        return None

    @property
    def crs(self) -> Optional[CRS]:
        """Returns coordinate reference system embedded in the model grid."""
        if self.data.raster.crs is not None:
            return CRS(self.data.raster.crs)
        self._logger.warning("Grid data has no crs")
        return None

    @property
    def bounds(self) -> Optional[List[float]]:
        """Returns the bounding box of the model grid."""
        if len(self.data) > 0:
            return self.data.raster.bounds
        _exec_nodata_strat(
            msg="No grid data found for deriving bounds",
            strategy=NoDataStrategy.IGNORE,
            logger=self._logger,
        )
        return None

    @property
    def region(self) -> Optional[gpd.GeoDataFrame]:
        """Returns the geometry of the model area of interest."""
        if len(self.data) > 0:
            crs = self.crs
            if crs is not None and hasattr(crs, "to_epsg"):
                crs = crs.to_epsg()  # not all CRS have an EPSG code
            return gpd.GeoDataFrame(geometry=[box(*self.bounds)], crs=crs)
        _exec_nodata_strat(
            msg="No grid data found for deriving region",
            strategy=NoDataStrategy.IGNORE,
            logger=self._logger,
        )
        return None

    @property
    def data(self) -> xr.Dataset:
        """Model static gridded data as xarray.Dataset."""
        if self._data is None:
            self._initialize_grid()
        return self._data

    def _initialize_grid(self, skip_read: bool = False) -> None:
        """Initialize grid object."""
        if self._data is None:
            self._data = xr.Dataset()
            if self._root.is_reading_mode() and not skip_read:
                self.read()

    def set_crs(self, crs: CRS) -> None:
        """Set coordinate reference system of the model grid."""
        if len(self.data) > 0:
            self.data.raster.set_crs(crs)

    @hydromt_step
    def add_data_from_constant(
        self,
        constant: Union[int, float],
        name: str,
        dtype: Optional[str] = "float32",  # TODO: change dtype to np.dtype
        nodata: Optional[Union[int, float]] = None,
        mask_name: Optional[str] = "mask",
    ) -> List[str]:
        """HYDROMT CORE METHOD: Adds data to grid component based on a constant value.

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
        da = grid_from_constant(
            grid_like=self.data,
            constant=constant,
            name=name,
            dtype=dtype,
            nodata=nodata,
            mask_name=mask_name,
        )
        # Add to grid
        self.set(da)

        return [name]

    @hydromt_step
    def add_data_from_rasterdataset(
        self,
        raster_fn: Union[str, Path, xr.DataArray, xr.Dataset],
        variables: Optional[List] = None,
        fill_method: Optional[str] = None,
        reproject_method: Optional[Union[List, str]] = "nearest",
        mask_name: Optional[str] = "mask",
        rename: Optional[Dict] = None,
    ) -> List[str]:
        """HYDROMT CORE METHOD: Add data variable(s) from ``raster_fn`` to grid component.

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
        rename = rename or {}
        self._logger.info(f"Preparing grid data from raster source {raster_fn}")
        # Read raster data and select variables
        ds = self._data_catalog.get_rasterdataset(
            raster_fn,
            geom=self.region,
            buffer=2,
            variables=variables,
            single_var_as_array=False,
        )
        # Data resampling
        ds_out = grid_from_rasterdataset(
            grid_like=self.data,
            ds=ds,
            variables=variables,
            fill_method=fill_method,
            reproject_method=reproject_method,
            mask_name=mask_name,
            rename=rename,
        )
        # Add to grid
        self.set(ds_out)

        return list(ds_out.data_vars.keys())

    @hydromt_step
    def add_data_from_raster_reclass(
        self,
        raster_fn: Union[str, Path, xr.DataArray],
        reclass_table_fn: Union[str, Path, pd.DataFrame],
        reclass_variables: List,
        variable: Optional[str] = None,
        fill_method: Optional[str] = None,
        reproject_method: Optional[Union[List, str]] = "nearest",
        mask_name: Optional[str] = "mask",
        rename: Optional[Dict] = None,
        **kwargs,
    ) -> List[str]:
        """HYDROMT CORE METHOD: Add data variable(s) to grid component by reclassifying the data in ``raster_fn`` based on ``reclass_table_fn``.

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
        """  # noqa: E501
        rename = rename or dict()
        self._logger.info(
            f"Preparing grid data by reclassifying the data in {raster_fn} based "
            f"on {reclass_table_fn}"
        )
        # Read raster data and remapping table
        da = self._data_catalog.get_rasterdataset(
            raster_fn, geom=self.region, buffer=2, variables=variable, **kwargs
        )
        if not isinstance(da, xr.DataArray):
            raise ValueError(
                f"raster_fn {raster_fn} should be a single variable. "
                "Please select one using the 'variable' argument"
            )
        df_vars = self._data_catalog.get_dataframe(
            reclass_table_fn, variables=reclass_variables
        )
        # Data resampling
        ds_vars = grid_from_raster_reclass(
            grid_like=self.data,
            da=da,
            reclass_table=df_vars,
            reclass_variables=reclass_variables,
            fill_method=fill_method,
            reproject_method=reproject_method,
            mask_name=mask_name,
            rename=rename,
        )
        # Add to maps
        self.set(ds_vars)

        return list(ds_vars.data_vars.keys())

    @hydromt_step
    def add_data_from_geodataframe(
        self,
        vector_fn: Union[str, Path, gpd.GeoDataFrame],
        variables: Optional[Union[List, str]] = None,
        nodata: Optional[Union[List, int, float]] = -1,
        rasterize_method: Optional[str] = "value",
        mask_name: Optional[str] = "mask",
        rename: Optional[Dict] = None,
        all_touched: Optional[bool] = True,
    ) -> List[str]:
        """HYDROMT CORE METHOD: Add data variable(s) to grid component by rasterizing the data from ``vector_fn``.

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
        """  # noqa: E501
        rename = rename or dict()
        self._logger.info(f"Preparing grid data from vector '{vector_fn}'.")
        gdf = self._data_catalog.get_geodataframe(
            vector_fn, geom=self.region, dst_crs=self.crs
        )
        if gdf.empty:
            self._logger.warning(
                f"No shapes of {vector_fn} found within region,"
                " skipping setup_grid_from_vector."
            )
            return None
        # Data resampling
        if vector_fn in rename.keys():
            # In case of choosing a new name with area or fraction method pass
            # the name directly
            rename = rename[vector_fn]
        ds = grid_from_geodataframe(
            grid_like=self._data,
            gdf=gdf,
            variables=variables,
            nodata=nodata,
            rasterize_method=rasterize_method,
            mask_name=mask_name,
            rename=rename,
            all_touched=all_touched,
        )
        # Add to grid
        self.set(ds)

        return list(ds.data_vars.keys())
