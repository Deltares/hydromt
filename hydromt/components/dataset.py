"""Xarrays component."""

from shutil import move
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Union,
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

_DEFAULT_DATASET_FILENAME = "{name}.nc"


class DatasetComponent(ModelComponent):
    """A component to manage colections of Xarray objects.

    It contains a dictionary of xarray DataArray or DataSet objects.
    """

    def __init__(self, model: "Model", filename: str = _DEFAULT_DATASET_FILENAME):
        """Initialize a GeomComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        filename: str
            The path to use for reading and writing of component data by default.
            by default "geoms/{name}.geojson" ie one file per geodataframe in the data dictionnary.
        """
        self._data: Optional[XArrayDict] = None
        self._filename = filename
        self._tmp_data_dir = None
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
            if self._root.is_reading_mode() and not skip_read:
                self.read()

    def set(
        self, data: Union[Dataset, DataArray], name: str, split_dataset: bool = True
    ):
        """Add data to the xrarray component.

        Arguments
        ---------
        data: xarray.Dataset or xarray.DataArray
            New xarray object to add
        name: str
            name of the xaray.
        """
        self._initialize()
        if isinstance(data, DataFrame):
            data = data.to_xarray()
        assert self._data is not None
        if split_dataset and isinstance(data, Dataset):
            ds = {str(name): data[name] for name in data.data_vars}
        else:
            ds = {name: data}

        for name, d in ds.items():
            if name in self._data:
                self._logger.warning(f"Replacing xarray: {name}")
            self._data[name] = d

    @hydromt_step
    def read(
        self, filename: Optional[str] = None, single_var_as_array: bool = True, **kwargs
    ) -> None:
        r"""Read model geometries files at <root>/<filename>.

        key-word arguments are passed to :py:func:`geopandas.read_file`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root. should contain a {name} placeholder
            which will be used to determine the names/keys of the geometries.
            if None, the path that was provided at init will be used.
        **kwargs:
            Additional keyword arguments that are passed to the
            `geopandas.read_file` function.
        """
        self._root._assert_read_mode()
        self._initialize(skip_read=True)
        kwargs = {**{"engine": "netcdf4"}, **kwargs}
        filename_template = filename or self._filename
        ncs = read_nc(
            filename_template,
            root=self._root.path,
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
        self._root._assert_write_mode()

        if len(self.data) == 0:
            self._logger.debug("No data found, skiping writing.")
            return

        kwargs = {**{"engine": "netcdf4"}, **kwargs}
        write_nc(
            self.data,
            filename_template=filename or self._filename,
            root=self._root.path,
            gdal_compliant=gdal_compliant,
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
                self._logger.error(
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
                self._logger.error(
                    f"Could not write to destination file {close_handle['org_fn']} "
                    "because the following error was raised: {e}"
                )
                close_handle["close_attempts"] += 1
                self._defered_file_closes.append(close_handle)
                failed_closes.append((close_handle["org_fn"], close_handle["tmp_fn"]))

        return list(set(failed_closes))
