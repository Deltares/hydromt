"""Grid ModelComponent."""
import logging
import weakref
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from shapely.geometry import box

from hydromt.data_catalog import DataCatalog

from .. import workflows

logger = logging.getLogger(__name__)


class GridModelComponent:
    """GridModelComponent class."""

    def __init__(self, model, logger):
        self._model_ref = weakref.ref(model)
        self._data = None
        self.logger = logger

    def set(
        self,
        data: Union[xr.DataArray, xr.Dataset, np.ndarray],
        name: Optional[str] = None,
        read: bool = True,
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
        if isinstance(data, xr.DataArray):
            if name is not None:  # rename
                data.name = name
            data = data.to_dataset()
        elif not isinstance(data, xr.Dataset):
            raise ValueError(f"cannot set data of type {type(data).__name__}")

        if len(self._data) == 0:  # empty grid
            self._data = data
        else:
            for dvar in data.data_vars:
                if dvar in self._data:
                    if read:
                        self.logger.warning(f"Replacing grid map: {dvar}")
                self._data[dvar] = data[dvar]

    def write(self):
        """Write grid."""
        pass

    def read(
        self, fn: str = "grid/grid.nc", read: bool = True, write: bool = False, **kwargs
    ) -> None:
        """Read model grid data at <root>/<fn> and add to grid property.

        key-word arguments are passed to :py:meth:`~hydromt.models.Model.read_nc`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, by default 'grid/grid.nc'
        **kwargs : dict
            Additional keyword arguments to be passed to the `read_nc` method.
        """
        self._assert_read_mode()
        self._initialize_grid(skip_read=True)

        # Load grid data in r+ mode to allow overwritting netcdf files
        if read and write:
            kwargs["load"] = True
        loaded_nc_files = self.read_nc(
            fn, single_var_as_array=False, **kwargs
        )  # TODO: decide where read_nc should be placed
        for ds in loaded_nc_files.values():
            self.set(ds)

    def create(self):
        """Create grid."""
        pass

    @property
    def model(self):
        """Returns model reference."""
        # Access the Model instance through the weak reference
        return self._model_ref()

    @property
    def data(self):
        """Returns data."""
        return self._data

    @property
    def res(self) -> Tuple[float, float]:
        """Returns the resolution of the model grid."""
        if len(self._data) > 0:
            return self._data.raster.res

    @property
    def transform(self):
        """Returns spatial transform of the model grid."""
        if len(self._data) > 0:
            return self._data.raster.transform

    @property
    def crs(self) -> Union[CRS, None]:
        """Returns coordinate reference system embedded in the model grid."""
        if self._data.raster.crs is not None:
            return CRS(self._data.raster.crs)

    @property
    def bounds(self) -> List[float]:
        """Returns the bounding box of the model grid."""
        if len(self._data) > 0:
            return self._data.raster.bounds

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns the geometry of the model area of interest."""
        region = gpd.GeoDataFrame()
        if len(self._data) > 0:
            crs = self.crs
            if crs is not None and hasattr(crs, "to_epsg"):
                crs = crs.to_epsg()  # not all CRS have an EPSG code
            region = gpd.GeoDataFrame(geometry=[box(*self.bounds)], crs=crs)
        return region

    @property
    def grid(self):
        """Model static gridded data as xarray.Dataset."""
        if self._data is None:
            self._initialize_grid()
        return self._data

    def _initialize_grid(self, skip_read: bool = False, read: bool = True) -> None:
        """Initialize grid object."""
        if self._data is None:
            self._data = xr.Dataset()
            if read and not skip_read:
                self.read()

    def set_crs(self, crs: CRS) -> None:
        """Set coordinate reference system of the model grid."""
        if len(self._data) > 0:
            self._data.raster.set_crs(crs)

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
            grid_like=self._data,
            constant=constant,
            name=name,
            dtype=dtype,
            nodata=nodata,
            mask_name=mask_name,
        )
        # Add to grid
        self.set(da)

        return [name]

    def setup_grid_from_rasterdataset(
        self,
        raster_fn: Union[str, Path, xr.DataArray, xr.Dataset],
        data_catalog: DataCatalog,
        variables: Optional[List] = None,
        fill_method: Optional[str] = None,
        reproject_method: Optional[Union[List, str]] = "nearest",
        mask_name: Optional[str] = "mask",
        rename: Optional[Dict] = None,
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
        rename = rename or {}
        self.logger.info(f"Preparing grid data from raster source {raster_fn}")
        # Read raster data and select variables
        ds = data_catalog.get_rasterdataset(
            raster_fn,
            geom=self.region,
            buffer=2,
            variables=variables,
            single_var_as_array=False,
        )
        # Data resampling
        ds_out = workflows.grid.grid_from_rasterdataset(
            grid_like=self._data,
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

    def setup_grid_from_raster_reclass(
        self,
        data_catalog: DataCatalog,
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
        """HYDROMT CORE METHOD: Add data variable(s) to grid object by reclassifying the data in ``raster_fn`` based on ``reclass_table_fn``.

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
        da = data_catalog.get_rasterdataset(
            raster_fn, geom=self.region, buffer=2, variables=variable, **kwargs
        )
        if not isinstance(da, xr.DataArray):
            raise ValueError(
                f"raster_fn {raster_fn} should be a single variable. "
                "Please select one using the 'variable' argument"
            )
        df_vars = data_catalog.get_dataframe(
            reclass_table_fn, variables=reclass_variables
        )
        # Data resampling
        ds_vars = workflows.grid.grid_from_raster_reclass(
            grid_like=self._data,
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

    def setup_grid_from_geodataframe(
        self,
        data_catalog: DataCatalog,
        vector_fn: Union[str, Path, gpd.GeoDataFrame],
        variables: Optional[Union[List, str]] = None,
        nodata: Optional[Union[List, int, float]] = -1,
        rasterize_method: Optional[str] = "value",
        mask_name: Optional[str] = "mask",
        rename: Optional[Dict] = None,
        all_touched: Optional[bool] = True,
    ) -> List[str]:
        """HYDROMT CORE METHOD: Add data variable(s) to grid object by rasterizing the data from ``vector_fn``.

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
        gdf = data_catalog.get_geodataframe(
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
