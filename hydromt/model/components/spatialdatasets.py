"""Spatial Xarrays component."""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, cast

import xarray as xr
from geopandas import GeoDataFrame
from pandas import DataFrame
from xarray import DataArray, Dataset

from hydromt._typing.type_def import XArrayDict
from hydromt.io.readers import open_ncs
from hydromt.io.writers import write_nc
from hydromt.log import get_hydromt_logger
from hydromt.model.components.base import ModelComponent
from hydromt.model.components.spatial import SpatialModelComponent
from hydromt.model.steps import hydromt_step

if TYPE_CHECKING:
    from hydromt.model.model import Model

logger = get_hydromt_logger(__name__)


class SpatialDatasetsComponent(SpatialModelComponent):
    """
    A component to manage collection of geospatial xarray objects.

    It contains a dictionary of xarray DataArray or Dataset objects.
    Compared to ``DatasetsComponent`` this component has a region property.
    """

    def __init__(
        self,
        model: "Model",
        *,
        region_component: str,
        filename: str = "spatial_datasets/{name}.nc",
        region_filename: str = "spatial_datasets/spatial_datasets_region.geojson",
    ):
        """Initialize a SpatialDatasetsComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        region_component: str
            The name of the region component to use as reference for this component's
            region.
        filename: str
            The path to use for reading and writing of component data by default.
            by default "spatial_datasets/{name}.nc" ie one file per xarray object in the
            data dictionary.
        region_filename: str
            The path to use for writing the region data to a file. By default
            "spatial_datasets/spatial_datasets_region.geojson".
        """
        self._data: Optional[XArrayDict] = None
        self._filename: str = filename
        super().__init__(
            model=model,
            region_component=region_component,
            region_filename=region_filename,
        )

    @property
    def data(self) -> XArrayDict:
        """Model data in the form of xarray objects.

        Return dict of xarray.Dataset or xarray.DataArray objects
        """
        if self._data is None:
            self._initialize()

        assert self._data is not None
        return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize geoms."""
        if self._data is None:
            self._data = {}
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    @property
    def _region_data(self) -> Optional[GeoDataFrame]:
        raise AttributeError(
            "region cannot be found in spatialdatasets component."
            "Meaning that the region_component is not set or could not be found in"
            " the model."
        )

    # @hydromt_step
    def set(
        self,
        data: Union[Dataset, DataArray],
        name: Optional[str] = None,
        split_dataset: bool = False,
    ):
        """Add data to the xarray component.

        Arguments
        ---------
        data: xarray.Dataset or xarray.DataArray
            New xarray object to add
        name: str
            name of the xarray.
        """
        self._initialize()
        if isinstance(data, DataFrame):
            data = data.to_xarray()
        assert self._data is not None
        if split_dataset:
            if isinstance(data, Dataset):
                ds = {str(name): data[name] for name in data.data_vars}
            else:
                raise ValueError(
                    f"Can only split dataset for Datasets not for {type(data).__name__}"
                )
        elif name is not None:
            ds = {name: data}
        elif data.name is not None:
            ds = {str(data.name): data}
        else:
            raise ValueError(
                "Either name should be set, the data needs to have a name or split_dataset should be True"
            )

        ds = cast(XArrayDict, ds)

        for name, d in ds.items():
            if name in self._data:
                logger.warning(f"Replacing xarray: {name}")
            self._data[name] = d

    @hydromt_step
    def read(self, filename: Optional[str] = None, **kwargs) -> None:
        """Read model dataset files at <root>/<filename>.

        key-word arguments are passed to :py:func:`hydromt.io.readers.open_nc`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root. should contain a {name} placeholder
            which will be used to determine the names/keys of the datasets.
            if None, the path that was provided at init will be used.
        **kwargs:
            Additional keyword arguments that are passed to the
            `hydromt.io.readers.open_nc` function.
        """
        self.root._assert_read_mode()
        self._initialize(skip_read=True)
        kwargs = {**{"engine": "netcdf4"}, **kwargs}
        filename_template = filename or self._filename
        ncs = open_ncs(filename_template, root=self.root.path, **kwargs)
        for name, ds in ncs.items():
            self._open_datasets.append(ds)
            if len(ds.data_vars) == 1:
                (da,) = ds.data_vars.values()
            else:
                da = ds
            self.set(data=da, name=name)

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
        """Write dictionary of xarray.Dataset and/or xarray.DataArray to netcdf files.

        Possibility to update the xarray objects attributes to get GDAL compliant NetCDF
        files, using :py:meth:`~hydromt.raster.gdal_compliant`.
        The function will first try to directly write to file. In case of
        PermissionError, it will first write a temporary file and then move the file.

        key-word arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`

        Parameters
        ----------
        nc_dict: dict
            Dictionary of xarray.Dataset and/or xarray.DataArray to write
        fn: str
            filename relative to model root and should contain a {name} placeholder
        gdal_compliant: bool, optional
            If True, convert xarray.Dataset and/or xarray.DataArray to gdal compliant
            format using :py:meth:`~hydromt.raster.gdal_compliant`
        rename_dims: bool, optional
            If True, rename x_dim and y_dim to standard names depending on the CRS
            (x/y for projected and lat/lon for geographic). Only used if
            ``gdal_compliant`` is set to True. By default, False.
        force_sn: bool, optional
            If True, forces the dataset to have South -> North orientation. Only used
            if ``gdal_compliant`` is set to True. By default, False.
        **kwargs:
            Additional keyword arguments that are passed to the `to_netcdf`
            function.
        """
        self.root._assert_write_mode()

        if len(self.data) == 0:
            logger.info(
                f"{self.model.name}.{self.name_in_model}: No data found, skipping writing."
            )
            return

        kwargs.setdefault("engine", "netcdf4")
        filename = filename or self._filename

        for name, ds in self.data.items():
            file_path = self.root.path / filename.format(name=name)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"{self.model.name}.{self.name_in_model}: Writing spatial dataset to {file_path}."
            )

            close_handle = write_nc(
                ds,
                file_path=file_path,
                gdal_compliant=gdal_compliant,
                rename_dims=rename_dims,
                force_sn=force_sn,
                force_overwrite=self.root.mode.is_override_mode(),
                **kwargs,
            )
            if close_handle is not None:
                self._deferred_file_close_handles.append(close_handle)

    @hydromt_step
    def add_raster_data_from_rasterdataset(
        self,
        raster_filename: Union[str, Path, Dataset],
        variables: Optional[List] = None,
        fill_method: Optional[str] = None,
        name: Optional[str] = None,
        reproject_method: Optional[str] = None,
        split_dataset: bool = True,
        rename: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """HYDROMT CORE METHOD: Add data variable(s) from ``raster_filename`` to datasets component.

        If raster is a xarray dataset, all variables will be added unless ``variables``
        list is specified.

        Adds model layers:
        * **raster.name**: data from raster_filename

        Parameters
        ----------
        raster_filename: str, Path, xr.Dataset
            Data catalog key, path to raster file or raster xarray data object.
        variables: list, optional
            List of variables to add to datasets from raster_filename. By default all.
        fill_method : str, optional
            If specified, fills nodata values using fill_nodata method.
            Available methods are {'linear', 'nearest', 'cubic', 'rio_idw'}.
        name: str, optional
            Name of new dataset in self.data dictionary,
            only in case split_dataset=False.
        reproject_method: str, optional
            See rasterio.warp.reproject for existing methods, by default the data is
            not reprojected (None).
        split_dataset: bool, optional
            If data is a xarray.Dataset split it into several xarray.DataArrays.
        rename: dict, optional
            Dictionary to rename variable names in raster_filename before adding to the datasets
            {'name_in_raster_filename': 'name_in_dataset'}. By default empty.

        Returns
        -------
        list
            Names of added model map layers
        """
        rename = rename or {}
        logger.info(f"Preparing dataset data from raster source {raster_filename}")
        # Read raster data and select variables
        ds = self.data_catalog.get_rasterdataset(
            raster_filename,
            geom=self.region,
            buffer=2,
            variables=variables,
            single_var_as_array=False,
        )
        if isinstance(ds, DataArray):
            ds = ds.to_dataset()

        ds = cast(Dataset, ds)
        # Fill nodata
        if fill_method is not None:
            ds = ds.raster.interpolate_na(method=fill_method)
        # Reprojection
        if ds.rio.crs != self.model.crs and reproject_method is not None:
            ds = ds.raster.reproject(dst_crs=self.model.crs, method=reproject_method)
        self.set(ds.rename(rename), name=name, split_dataset=split_dataset)

        return list(ds.data_vars.keys())

    @hydromt_step
    def add_raster_data_from_raster_reclass(
        self,
        raster_filename: Union[str, Path, DataArray],
        reclass_table_filename: Union[str, Path, DataFrame],
        reclass_variables: List,
        variable: Optional[str] = None,
        fill_method: Optional[str] = None,
        reproject_method: Optional[str] = None,
        name: Optional[str] = None,
        split_dataset: bool = True,
        rename: Optional[Dict] = None,
        **kwargs,
    ) -> List[str]:
        r"""HYDROMT CORE METHOD: Add data variable(s) to datasets component by reclassifying the data in ``raster_filename`` based on ``reclass_table_filename``.

        This is done by reclassifying the data in
        ``raster_filename`` based on ``reclass_table_filename``.

        Adds model layers:

        * **reclass_variables**: reclassified raster data

        Parameters
        ----------
        raster_filename: str, Path, xr.DataArray
            Data catalog key, path to raster file or raster xarray data object.
            Should be a DataArray. Else use `variable` argument for selection.
        reclass_table_filename: str, Path, pd.DataFrame
            Data catalog key, path to tabular data file or tabular pandas dataframe
            object for the reclassification table of `raster_filename`.
        reclass_variables: list
            List of reclass_variables from reclass_table_filename table to add to the datasets. Index
            column should match values in `raster_filename`.
        variable: str, optional
            Name of raster dataset variable to use. This is only required when reading
            datasets with multiple variables. By default None.
        fill_method : str, optional
            If specified, fills nodata values in `raster_filename` using fill_nodata method
            before reclassifying. Available methods are {'linear', 'nearest',
            'cubic', 'rio_idw'}.
        reproject_method: str, optional
            See rasterio.warp.reproject for existing methods, by default the data is
            not reprojected (None).
        name: str, optional
            Name of new dataset variable, only in case split_dataset=False.
        split_dataset: bool, optional
            If data is a xarray.Dataset split it into several xarray.DataArrays.
        rename: dict, optional
            Dictionary to rename variable names in reclass_variables before adding to
            grid {'name_in_reclass_table': 'name_in_grid'}. By default empty.
        \**kwargs:
            Additional keyword arguments that are passed to the
            `data_catalog.get_rasterdataset` function.

        Returns
        -------
        list
            Names of added model map layers
        """  # noqa: E501
        rename = rename or {}
        logger.info(
            f"Preparing map data by reclassifying the data in {raster_filename} based"
            f" on {reclass_table_filename}"
        )
        # Read raster data and remapping table
        da = self.data_catalog.get_rasterdataset(
            raster_filename,
            geom=self.region,
            buffer=2,
            variables=variable,
            **kwargs,
        )
        if not isinstance(da, DataArray):
            raise ValueError(
                f"raster_filename {raster_filename} should be a single variable. "
                "Please select one using the 'variable' argument"
            )
        df_vars = self.data_catalog.get_dataframe(
            reclass_table_filename, variables=reclass_variables
        )
        # Fill nodata
        if fill_method is not None:
            da = da.raster.interpolate_na(method=fill_method)
        # Mapping function
        ds_vars = da.raster.reclassify(reclass_table=df_vars, method="exact")
        # Reprojection
        if ds_vars.rio.crs != self.model.crs and reproject_method is not None:
            ds_vars = ds_vars.raster.reproject(dst_crs=self.model.crs)
        self.set(ds_vars.rename(rename), name=name, split_dataset=split_dataset)

        return list(ds_vars.data_vars.keys())

    def test_equal(self, other: ModelComponent) -> Tuple[bool, Dict[str, str]]:
        """
        Test if two DatasetsComponents are equal.

        Parameters
        ----------
        other: ModelComponent
            The other ModelComponent to compare with.

        Returns
        -------
        tuple[bool, dict[str, str]]
            True if the components are equal, and a dict with the associated errors per property checked.
        """
        eq, errors = super().test_equal(other)
        if not eq:
            return eq, errors
        other_datasets = cast(SpatialDatasetsComponent, other)
        for name, ds in self.data.items():
            if name not in other_datasets.data:
                errors[name] = "Dataset not found in other component."
            try:
                xr.testing.assert_allclose(ds, other_datasets.data[name])
            except AssertionError as e:
                errors[name] = str(e)

        return len(errors) == 0, errors
