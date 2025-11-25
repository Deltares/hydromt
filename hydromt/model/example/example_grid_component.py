"""Example grid component module."""

import logging
from pathlib import Path
from typing import Any

import xarray as xr

from hydromt.model import Model
from hydromt.model.components import GridComponent
from hydromt.model.processes.grid import (
    create_grid_from_region,
    grid_from_rasterdataset,
)
from hydromt.model.steps import hydromt_step

__all__ = ["ExampleGridComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class ExampleGridComponent(GridComponent):
    """Example grid component.

    Inherits from the HydroMT-core GridComponent model-component.
    It only add two hydromt_steps for creating a grid and adding data to it.
    """

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "grid.nc",
        region_filename: str = "region.geojson",
    ):
        """Initialize a ExampleGridComponent.

        Parameters
        ----------
        model : Model
            HydroMT model instance
        filename : str
            The path to use for reading and writing of component data by default.
            By default "grid.nc".
        region_filename : str
            The path to use for reading and writing of the region data by default.
            By default "region.geojson".
        """
        super().__init__(
            model,
            filename=filename,
            region_component=None,
            region_filename=region_filename,
        )

    ## Create and add_data methods
    @hydromt_step
    def create_from_region(
        self,
        region: dict[str, Any],
        *,
        res: float | int | None = None,
        crs: int | None = None,
        region_crs: int = 4326,
        rotated: bool = False,
        hydrography_path: str | None = None,
        basin_index_path: str | None = None,
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
        res: float or int, optional
            Resolution used to generate 2D grid [unit of the CRS], required if region
            is not based on 'grid'.
        crs : int, optional
            EPSG code of the grid to create.
        region_crs : int, optional
            EPSG code of the region geometry, by default None. Only applies if region is
            of kind 'bbox'or if geom crs is not defined in the file itself.
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

        basin_index_path: str, optional
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
        logger.info("Preparing 2D grid.")

        # Check if this component's region is a reference to another component
        if self._region_component is not None:
            raise ValueError(
                "Region is a reference to another component. Cannot create grid."
            )

        grid = create_grid_from_region(
            region,
            data_catalog=self.data_catalog,
            res=res,
            crs=crs,
            region_crs=region_crs,
            rotated=rotated,
            hydrography_path=hydrography_path,
            basin_index_path=basin_index_path,
            add_mask=add_mask,
            align=align,
            dec_origin=dec_origin,
            dec_rotation=dec_rotation,
        )
        self.set(grid)
        return grid

    @hydromt_step
    def add_data_from_rasterdataset(
        self,
        raster_data: str | Path | xr.DataArray | xr.Dataset,
        variables: list[str] | None = None,
        fill_method: str | None = None,
        reproject_method: list[str] | str | None = "nearest",
        mask_name: str | None = "mask",
        rename: dict[str, str] | None = None,
    ) -> list[str]:
        """HYDROMT CORE METHOD: Add data variable(s) from ``raster_data`` to grid component.

        If raster is a dataset, all variables will be added unless ``variables`` list
        is specified.

        Adds model layers:

        * **raster.name** grid: data from raster_data

        Parameters
        ----------
        raster_data: str, Path, xr.DataArray, xr.Dataset
            Data catalog key, path to raster file or raster xarray data object.
            If a path to a raster file is provided it will be added
            to the data_catalog with its name based on the file basename without
            extension.
        variables: list, optional
            List of variables to add to grid from raster_data. By default all.
        fill_method : str, optional
            If specified, fills nodata values using fill_nodata method.
            Available methods are {'linear', 'nearest', 'cubic', 'rio_idw'}.
        reproject_method: list, str, optional
            See rasterio.warp.reproject for existing methods, by default 'nearest'.
            Can provide a list corresponding to ``variables``.
        mask_name: str, optional
            Name of mask in self.grid to use for masking raster_data. By default 'mask'.
            Use None to disable masking.
        rename: dict, optional
            Dictionary to rename variable names in raster_data before adding to grid
            {'name_in_raster_data': 'name_in_grid'}. By default empty.

        Returns
        -------
        list
            Names of added model map layers
        """
        rename = rename or {}
        logger.info(f"Preparing grid data from raster source {raster_data}")
        # Read raster data and select variables
        ds = self.data_catalog.get_rasterdataset(
            raster_data,
            geom=self.region,
            buffer=2,
            variables=variables,
            single_var_as_array=False,
        )
        assert ds is not None
        # Data resampling
        ds_out = grid_from_rasterdataset(
            grid_like=self._get_grid_data(),
            ds=ds,
            variables=variables,
            fill_method=fill_method,
            reproject_method=reproject_method,
            mask_name=mask_name,
            rename=rename,
        )
        # Add to grid
        self.set(ds_out)
