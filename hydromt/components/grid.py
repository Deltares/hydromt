"""Grid Component."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from pyproj import CRS
from shapely.geometry import box

from hydromt import hydromt_step
from hydromt._typing.error import NoDataStrategy, _exec_nodata_strat
from hydromt._typing.type_def import DeferedFileClose, StrPath
from hydromt.components.base import ModelComponent
from hydromt.components.spatial import SpatialModelComponent
from hydromt.io.readers import read_nc
from hydromt.io.writers import write_nc
from hydromt.workflows.grid import (
    create_grid_from_region,
    grid_from_constant,
    grid_from_geodataframe,
    grid_from_raster_reclass,
    grid_from_rasterdataset,
)

if TYPE_CHECKING:
    from hydromt.models.model import Model

__all__ = ["GridComponent"]


class GridComponent(SpatialModelComponent):
    """ModelComponent class for grid components.

    This class is used for setting, creating, writing, and reading regular grid data for a
    HydroMT model. The grid component data stored in the ``data`` property of this class is of the
    hydromt.gis.raster.RasterDataset type which is an extension of xarray.Dataset for regular grid.
    """

    DEFAULT_FILENAME = "grid/grid.nc"

    def __init__(
        self,
        model: "Model",
        *,
        filename: Optional[str] = None,
        region_filename: Optional[StrPath] = None,
    ):
        # region_component referencing is not possible for grids. The region should be passed via create().
        super().__init__(model=model, region_component=None, filename=region_filename)
        self._data: Optional[xr.Dataset] = None
        self._filename: str = filename or self.__class__.DEFAULT_FILENAME

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
        assert self._data is not None
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
                    if self.root.is_reading_mode():
                        self.logger.warning(f"Replacing grid map: {dvar}")
                self._data[dvar] = data[dvar]

    @hydromt_step
    def write(
        self,
        filename: Optional[str] = None,
        *,
        gdal_compliant: bool = False,
        rename_dims: bool = False,
        force_sn: bool = False,
        region_options: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[DeferedFileClose]:
        """Write model grid data to netcdf file at <root>/<fn>.

        key-word arguments are passed to :py:meth:`~hydromt.models.Model.write_nc`

        Parameters
        ----------
        filename : str, optional
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
        region_options : dict, optional
            Options to pass to the write_region method.
            Can contain `filename`, `to_wgs84`, and anything that will be passed to `GeoDataFrame.to_file`.
            If `filename` is not provided, `SpatialModelComponent.DEFAULT_REGION_FILENAME` will be used.
        **kwargs : dict
            Additional keyword arguments to be passed to the `write_nc` method.
        """
        self.root._assert_write_mode()
        region_options = region_options or {}
        self.write_region(**region_options)

        if len(self.data) == 0:
            _exec_nodata_strat(
                msg="No grid data found, skip writing.",
                strategy=NoDataStrategy.IGNORE,
                logger=self.logger,
            )
            return None
        # write_nc requires dict - use dummy 'grid' key
        return write_nc(  # Can return DeferedFileClose object
            {"grid": self.data},
            filename or str(self._filename),
            gdal_compliant=gdal_compliant,
            rename_dims=rename_dims,
            logger=self.logger,
            force_sn=force_sn,
            **kwargs,
        )

    @hydromt_step
    def read(
        self,
        filename: Optional[str] = None,
        *,
        mask_and_scale: bool = False,
        **kwargs,
    ) -> None:
        """Read model grid data at <root>/<fn> and add to grid property.

        key-word arguments are passed to :py:meth:`~hydromt.models.Model.read_nc`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root, by default 'grid/grid.nc'
        mask_and_scale : bool, optional
            If True, replace array values equal to _FillValue with NA and scale values
            according to the formula original_values * scale_factor + add_offset, where
            _FillValue, scale_factor and add_offset are taken from variable attributes
        (if they exist).
        **kwargs : dict
            Additional keyword arguments to be passed to the `read_nc` method.
        """
        self.root._assert_read_mode()
        self._initialize_grid(skip_read=True)

        # Load grid data in r+ mode to allow overwriting netcdf files
        if self.root.is_reading_mode() and self.root.is_writing_mode():
            kwargs["load"] = True
        loaded_nc_files = read_nc(
            filename or self._filename,
            self.root.path,
            logger=self.logger,
            single_var_as_array=False,
            mask_and_scale=mask_and_scale,
            **kwargs,
        )
        for ds in loaded_nc_files.values():
            self.set(ds)

    @hydromt_step
    def create(
        self,
        region: Dict[str, Any],
        *,
        res: Optional[float] = None,
        crs: Optional[int] = None,
        rotated: bool = False,
        hydrography_fn: Optional[str] = None,
        basin_index_fn: Optional[str] = None,
        add_mask: bool = True,
        align: bool = True,
        dec_origin: int = 0,
        dec_rotation: int = 3,
    ) -> xr.DataArray:
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
        res: float, optional
            Resolution used to generate 2D grid [unit of the CRS], required if region
            is not based on 'grid'.
        crs : int, optional
            EPSG code of the model
        rotated : bool
            if True, a minimum rotated rectangular grid is fitted around the region,
            by default False. Only applies if region is of kind 'bbox', 'geom'
        hydrography_fn : str, optional
            Name of data source for hydrography data. Required if region is of kind
                'basin', 'subbasin' or 'interbasin'.

            * Required variables: ['flwdir'] and any other 'snapping' variable required
                to define the region.

            * Optional variables: ['basins'] if the `region` is based on a
                (sub)(inter)basins without a 'bounds' argument.

        basin_index_fn : str, optional
            Name of data source with basin (bounding box) geometries associated with
            the 'basins' layer of `hydrography_fn`. Only required if the `region` is
            based on a (sub)(inter)basins without a 'bounds' argument.
        add_mask : bool
            Add mask variable to grid object, by default True.
        align : bool
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
        self.logger.info("Preparing 2D grid.")

        grid = create_grid_from_region(
            region,
            logger=self.logger,
            res=res,
            crs=crs,
            rotated=rotated,
            hydrography_fn=hydrography_fn,
            basin_index_fn=basin_index_fn,
            add_mask=add_mask,
            align=align,
            dec_origin=dec_origin,
            dec_rotation=dec_rotation,
        )
        self.set(grid)
        return grid

    @property
    def res(self) -> Optional[Tuple[float, float]]:
        """Returns the resolution of the model grid."""
        if len(self.data) > 0:
            return self.data.raster.res
        _exec_nodata_strat(
            msg="No grid data found for deriving resolution",
            strategy=NoDataStrategy.IGNORE,
            logger=self.logger,
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
            logger=self.logger,
        )
        return None

    @property
    def crs(self) -> Optional[CRS]:
        """Returns coordinate reference system embedded in the model grid."""
        if self.data.raster is None:
            _exec_nodata_strat(
                msg="No grid data found for deriving crs",
                strategy=NoDataStrategy.IGNORE,
                logger=self.logger,
            )
            return None
        if self.data.raster.crs is None:
            _exec_nodata_strat(
                msg="No crs found in grid data",
                strategy=NoDataStrategy.IGNORE,
                logger=self.logger,
            )
            return None
        return CRS(self.data.raster.crs)

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Returns the bounding box of the model grid."""
        if len(self.data) > 0:
            return self.data.raster.bounds
        _exec_nodata_strat(
            msg="No grid data found for deriving bounds",
            strategy=NoDataStrategy.IGNORE,
            logger=self.logger,
        )
        return None

    @property
    def _region_data(self) -> Optional[gpd.GeoDataFrame]:
        """Returns the geometry of the model area of interest."""
        if len(self.data) > 0:
            crs: Optional[Union[int, CRS]] = self.crs
            if crs is not None and hasattr(crs, "to_epsg"):
                crs = crs.to_epsg()  # not all CRS have an EPSG code
            return gpd.GeoDataFrame(geometry=[box(*self.bounds)], crs=crs)
        _exec_nodata_strat(
            msg="No grid data found for deriving region",
            strategy=NoDataStrategy.IGNORE,
            logger=self.logger,
        )
        return None

    @property
    def data(self) -> xr.Dataset:
        """Model static gridded data as xarray.Dataset."""
        if self._data is None:
            self._initialize_grid()
        assert self._data is not None
        return self._data

    def _initialize_grid(self, skip_read: bool = False) -> None:
        """Initialize grid object."""
        if self._data is None:
            self._data = xr.Dataset()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

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
    ) -> Optional[List[str]]:
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
        self.logger.info(f"Preparing grid data from vector '{vector_fn}'.")
        gdf = self.data_catalog.get_geodataframe(
            vector_fn, geom=self.region, dst_crs=self.crs
        )
        if gdf is None or gdf.empty:
            _exec_nodata_strat(
                f"No shapes of {vector_fn} found within region, skipping {self.add_data_from_geodataframe.__name__}.",
                NoDataStrategy.IGNORE,
                self.logger,
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

    def test_equal(self, other: ModelComponent) -> Tuple[bool, Dict[str, str]]:
        """Test if two components are equal.

        Parameters
        ----------
        other : ModelComponent
            The component to compare against.

        Returns
        -------
        Tuple[bool, Dict[str, str]]
            True if the components are equal, and a dict with the associated errors per property checked.
        """
        eq, errors = super().test_equal(other)
        if not eq:
            return eq, errors
        other_grid = cast(GridComponent, other)
        try:
            xr.testing.assert_allclose(self.data, other_grid.data)
        except AssertionError as e:
            errors["data"] = str(e)

        return len(errors) == 0, errors
