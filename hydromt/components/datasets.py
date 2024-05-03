"""Xarrays component."""

from pathlib import Path
from shutil import move
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Union,
    cast,
)

from pandas import DataFrame
from xarray import DataArray, Dataset

from hydromt._typing.type_def import XArrayDict
from hydromt.components.base import ModelComponent
from hydromt.hydromt_step import hydromt_step
from hydromt.io.readers import read_nc
from hydromt.io.writers import write_nc

if TYPE_CHECKING:
    from hydromt.models.model import Model

_DEFAULT_DATASET_FILENAME = "datasets/{name}.nc"


class DatasetsComponent(ModelComponent):
    """A component to manage collections of Xarray objects.

    It contains a dictionary of xarray DataArray or DataSet objects.
    """

    def __init__(self, model: "Model", filename: str = _DEFAULT_DATASET_FILENAME):
        """Initialize a Datasets.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        filename: str
            The path to use for reading and writing of component data by default.
            by default "datasets/{name}.nc" ie one file per dataset in the data dictionnary.
        """
        self._data: Optional[XArrayDict] = None
        self._filename = filename
        self._defered_file_closes = []
        super().__init__(model=model)

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
            self._data = dict()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

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
                self.logger.warning(f"Replacing xarray: {name}")
            self._data[name] = d

    @hydromt_step
    def read(
        self, filename: Optional[str] = None, single_var_as_array: bool = True, **kwargs
    ) -> None:
        r"""Read model dataset files at <root>/<filename>.

        key-word arguments are passed to :py:func:`hydromt.io.readers.read_nc`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root. should contain a {name} placeholder
            which will be used to determine the names/keys of the datasets.
            if None, the path that was provided at init will be used.
        **kwargs:
            Additional keyword arguments that are passed to the
            `hydromt.io.readers.read_nc` function.
        """
        self.root._assert_read_mode()
        self._initialize(skip_read=True)
        kwargs = {**{"engine": "netcdf4"}, **kwargs}
        filename_template = filename or self._filename
        ncs = read_nc(
            filename_template,
            root=self.root.path,
            single_var_as_array=single_var_as_array,
            **kwargs,
        )
        for name, ds in ncs.items():
            self.set(data=ds, name=name)

    @hydromt_step
    def write(
        self,
        filename: Optional[str] = None,
        gdal_compliant: bool = False,
        rename_dims: bool = False,
        force_sn: bool = False,
        **kwargs,
    ) -> None:
        """Write dictionary of xarray.Dataset and/or xarray.DataArray to netcdf files.

        Possibility to update the xarray objects attributes to get GDAL compliant NetCDF
        files, using :py:meth:`~hydromt.raster.gdal_compliant`.
        The function will first try to directly write to file. In case of
        PermissionError, it will first write a temporary file and add to the
        self._defered_file_closes attribute. Renaming and closing of netcdf filehandles
        will be done by calling the self._cleanup function.

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
            self.logger.debug("No data found, skiping writing.")
            return

        kwargs = {**{"engine": "netcdf4"}, **kwargs}
        write_nc(
            self.data,
            filename_template=filename or self._filename,
            root=self.root.path,
            gdal_compliant=gdal_compliant,
            logger=self.logger,
            rename_dims=rename_dims,
            force_sn=force_sn,
            **kwargs,
        )

    def _cleanup(self, forceful_overwrite=False, max_close_attempts=2) -> List[str]:
        """Try to close all defered file handles.

        Try to overwrite the destination file with the temporary one until either the
        maximum number of tries is reached or until it succeeds. The forced cleanup
        also attempts to close the original file handle, which could cause trouble
        if the user will try to read from the same file handle after this function
        is called.

        Parameters
        ----------
        forceful_overwrite: bool
            Attempt to force closing defered file handles before writing to them.
        max_close_attempts: int
            Number of times to try and overwrite the original file, before giving up.

        """
        failed_closes = []
        while len(self._defered_file_closes) > 0:
            close_handle = self._defered_file_closes.pop()
            if close_handle["close_attempts"] > max_close_attempts:
                # already tried to close this to many times so give up
                self.logger.error(
                    f"Max write attempts to file {close_handle['org_fn']}"
                    " exceeded. Skipping..."
                    f"Instead data was written to tmpfile: {close_handle['tmp_fn']}"
                )
                continue

            if forceful_overwrite:
                close_handle["ds"].close()
            try:
                move(close_handle["tmp_fn"], close_handle["org_fn"])
            except PermissionError:
                self.logger.error(
                    f"Could not write to destination file {close_handle['org_fn']} "
                    "because the following error was raised: {e}"
                )
                close_handle["close_attempts"] += 1
                self._defered_file_closes.append(close_handle)
                failed_closes.append((close_handle["org_fn"], close_handle["tmp_fn"]))

        return list(set(failed_closes))

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
            Name of new dataset in self.data dictionnary,
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
        self.logger.info(f"Preparing dataset data from raster source {raster_filename}")
        # Read raster data and select variables
        ds = self.data_catalog.get_rasterdataset(
            raster_filename,
            geom=self.model.region.data,
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
        self.logger.info(
            f"Preparing map data by reclassifying the data in {raster_filename} based"
            f" on {reclass_table_filename}"
        )
        # Read raster data and remapping table
        da = self.data_catalog.get_rasterdataset(
            raster_filename,
            geom=self.model.region.data,
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
