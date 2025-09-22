"""Grid Component."""

import logging
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, cast

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from pyproj import CRS
from shapely.geometry import box

from hydromt.io.readers import open_ncs
from hydromt.io.writers import write_nc
from hydromt.model.components.base import ModelComponent
from hydromt.model.components.spatial import SpatialModelComponent
from hydromt.model.steps import hydromt_step

if TYPE_CHECKING:
    from hydromt.model.model import Model

__all__ = ["GridComponent"]

logger = logging.getLogger(__name__)


class GridComponent(SpatialModelComponent):
    """ModelComponent class for grid components.

    This class is used for setting, creating, writing, and reading regular grid data for a
    HydroMT model. The grid component data stored in the ``data`` property of this class is of the
    hydromt.gis.raster.RasterDataset type which is an extension of xarray.Dataset for regular grid.
    """

    def __init__(
        self,
        model: "Model",
        *,
        filename: str = "grid/grid.nc",
        region_component: Optional[str] = None,
        region_filename: Optional[str] = "grid/grid_region.geojson",
    ):
        """
        Initialize a GridComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        filename: str
            The path to use for reading and writing of component data by default.
            By default "grid/grid.nc".
        region_component: str, optional
            The name of the region component to use as reference for this component's region.
            If provided, the region is not written to disk.
            If None, the region will be set to the grid extent. Note that the create
            method only works if the region_component is None. For add_data_from_*
            methods, the other region_component should be a reference to another
            grid component for correct reprojection.
        region_filename: Optional[str] = "grid/grid_region.geojson",
            The path to use for writing the region data to a file.
            By default "grid/grid_region.geojson". If None, the region is not written to disk.
        """
        # region_component referencing is not possible for grids. The region should be passed via create().
        super().__init__(
            model=model,
            region_component=region_component,
            region_filename=region_filename,
        )
        self._data: Optional[xr.Dataset] = None
        self._filename: str = filename

    def set(
        self,
        data: Union[xr.DataArray, xr.Dataset, np.ndarray],
        name: Optional[str] = None,
        mask: Optional[Union[str, xr.DataArray]] = None,
        force_sn: bool = False,
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
        mask: xr.DataArray, optional
            Name of the mask layer in the grid (self) or data, or directly the mask layer to use.
            Should be a DataArray where `.raster.nodata` is used to define the mask.
            If None or not present as a layer, no masking is applied.
        force_sn: bool, optional, default=False
            If True, the y-axis is oriented such that increasing y values go from South to North.
            If False, incoming data is used as is.
        """
        self._initialize_grid()
        assert self._data is not None

        name_required = isinstance(data, np.ndarray) or (
            isinstance(data, xr.DataArray) and data.name is None
        )
        if name is None and name_required:
            raise ValueError(f"Unable to set {type(data).__name__} data without a name")
        if isinstance(data, np.ndarray):
            if data.size == 0:
                raise ValueError("Cannot set grid data with empty array")
            if data.shape != self._data.raster.shape:
                raise ValueError("Shape of data and grid maps do not match")
            data = xr.DataArray(dims=self._data.raster.dims, data=data, name=name)

        if isinstance(data, xr.DataArray):
            if name is not None:
                data.name = name
            data = data.to_dataset()

        if not isinstance(data, xr.Dataset):
            raise ValueError(f"cannot set data of type {type(data).__name__}")

        if data.raster.res[1] < 0 and force_sn:
            data = data.raster.flipud()

        # Set the data per layer
        if len(self._data) == 0:  # empty grid
            self._data = data
        else:
            mask = self.get_mask_layer(mask, self.data, data)

            for dvar in data.data_vars:
                if dvar in self._data:
                    logger.warning(f"Replacing grid map: {dvar}")
                if mask is not None:
                    if data[dvar].dtype != np.bool:
                        data[dvar] = data[dvar].where(mask, data[dvar].raster.nodata)
                    else:
                        data[dvar] = data[dvar].where(mask, False)
                self._data[dvar] = data[dvar]

    @hydromt_step
    def write(
        self,
        filename: Optional[str] = None,
        *,
        gdal_compliant: bool = False,
        rename_dims: bool = False,
        force_sn: bool = False,
        **kwargs,
    ) -> None:
        """Write model grid data to netcdf file at <root>/<fn>.

        key-word arguments are passed to :py:meth:`~hydromt.model.Model.write_nc`

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
        **kwargs : dict
            Additional keyword arguments to be passed to the `write_nc` method.
        """
        self.root._assert_write_mode()

        if len(self.data) == 0:
            logger.info(
                f"{self.model.name}.{self.name_in_model}: No grid data found, skip writing."
            )
            return

        filename = filename or self._filename
        full_path = self.root.path / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"{self.model.name}.{self.name_in_model}: Writing grid data to {full_path}."
        )

        close_handle = write_nc(
            self.data,
            file_path=full_path,
            gdal_compliant=gdal_compliant,
            rename_dims=rename_dims,
            force_overwrite=self.root.mode.is_override_mode(),
            force_sn=force_sn,
            **kwargs,
        )
        if close_handle is not None:
            self._deferred_file_close_handles.append(close_handle)

    @hydromt_step
    def read(self, filename: Optional[str] = None, **kwargs) -> None:
        """Read model grid data at <root>/<fn> and add to grid property.

        key-word arguments are passed to :py:meth:`~hydromt.model.Model.open_nc`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root, by default 'grid/grid.nc'
        **kwargs : dict
            Additional keyword arguments to be passed to the `open_nc` method.
        """
        self.root._assert_read_mode()
        self._initialize_grid(skip_read=True)

        loaded_nc_files = open_ncs(
            filename or self._filename,
            self.root.path,
            **kwargs,
        )
        for ds in loaded_nc_files.values():
            self.set(ds)
            self._open_datasets.append(ds)

    @property
    def res(self) -> Optional[Tuple[float, float]]:
        """Returns the resolution of the model grid."""
        if len(self.data) > 0:
            return self.data.raster.res
        logger.warning("No grid data found for deriving resolution")
        return None

    @property
    def transform(self) -> Optional[Affine]:
        """Returns spatial transform of the model grid."""
        if len(self.data) > 0:
            return self.data.raster.transform
        logger.warning("No grid data found for deriving transform")
        return None

    @property
    def crs(self) -> Optional[CRS]:
        """Returns coordinate reference system embedded in the model grid."""
        if self.data.raster is None:
            logger.warning("No grid data found for deriving crs")
            return None
        if self.data.raster.crs is None:
            logger.warning("No crs found in grid data")
            return None
        return CRS(self.data.raster.crs)

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Returns the bounding box of the model grid."""
        if len(self.data) > 0:
            return self.data.raster.bounds
        logger.warning("No grid data found for deriving bounds")
        return None

    @property
    def _region_data(self) -> Optional[gpd.GeoDataFrame]:
        """Returns the geometry of the model area of interest."""
        if len(self.data) > 0:
            crs: Optional[Union[int, CRS]] = self.crs
            if crs is not None and hasattr(crs, "to_epsg"):
                crs = crs.to_epsg()  # not all CRS have an EPSG code
            return gpd.GeoDataFrame(geometry=[box(*self.bounds)], crs=crs)
        logger.warning("No grid data found for deriving region")
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

    def _get_grid_data(self) -> Union[xr.DataArray, xr.Dataset]:
        """Get grid data as xarray.DataArray from this component or the reference."""
        if self._region_component is not None:
            reference_component = self.model.get_component(self._region_component)
            if not isinstance(reference_component, GridComponent):
                raise ValueError(
                    f"Unable to find the referenced grid component: '{self._region_component}'."
                )
            if reference_component.data is None:
                raise ValueError(
                    f"Unable to get grid from the referenced region component: '{self._region_component}'."
                )
            return reference_component.data

        if self.data is None:
            raise ValueError("Unable to get grid data from this component.")
        return self.data

    @staticmethod
    def get_mask_layer(mask: str | xr.DataArray | None, *args) -> xr.DataArray | None:
        """Get the proper mask layer based on itself or a layer in a Dataset.

        Parameters
        ----------
        mask : str | xr.DataArray | None
            The mask itself or the name of the mask layer in another dataset.
        *args : list
            These have to be xarray Datasets in which the mask (as a string)
            can be present
        """
        if mask is None:
            return None
        if isinstance(mask, xr.DataArray):
            return mask != mask.raster.nodata
        if not isinstance(mask, str):
            raise ValueError(
                f"Unknown type for determining mask: {type(mask).__name__}"
            )
        for ds in args:
            if mask in ds:
                return ds[mask] != ds[mask].raster.nodata
        return None  # Nothin found
