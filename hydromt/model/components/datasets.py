"""Xarrays component."""

import logging
from typing import TYPE_CHECKING, Union, cast

import xarray as xr
from pandas import DataFrame
from xarray import DataArray, Dataset

from hydromt.io.readers import open_ncs
from hydromt.io.writers import write_nc
from hydromt.model.components.base import ModelComponent
from hydromt.model.steps import hydromt_step
from hydromt.typing.type_def import XArrayDict

if TYPE_CHECKING:
    from hydromt.model.model import Model

logger = logging.getLogger(__name__)


class DatasetsComponent(ModelComponent):
    """A component to manage collections of Xarray objects.

    It contains a dictionary of xarray DataArray or Dataset objects.
    """

    def __init__(
        self,
        model: "Model",
        filename: str = "datasets/{name}.nc",
    ):
        """Initialize a DatasetsComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        filename: str
            The path to use for reading and writing of component data by default.
            by default "datasets/{name}.nc" ie one file per dataset in the data
            dictionary.
        """
        self._data: XArrayDict | None = None
        self._filename: str = filename
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
            self._data = {}
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def set(
        self,
        data: Union[Dataset, DataArray],
        name: str | None = None,
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
                ds: dict[str, Union[Dataset, DataArray]] = {
                    str(name): data[name] for name in data.data_vars
                }
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

        for name, d in ds.items():
            if name in self._data:
                logger.warning(f"Replacing xarray: {name}")
            self._data[name] = d

    @hydromt_step
    def read(self, filename: str | None = None, **kwargs) -> None:
        """Read model dataset files at <root>/<filename>.

        key-word arguments are passed to :py:func:`hydromt.io.readers.open_ncs`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root. should contain a {name} placeholder
            which will be used to determine the names/keys of the datasets.
            if None, the path that was provided at init will be used.
        **kwargs:
            Additional keyword arguments that are passed to the
            `hydromt.io.readers.open_ncs` function.
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
        filename: str | None = None,
        *,
        gdal_compliant: bool = False,
        rename_dims: bool = False,
        force_sn: bool = False,
        **kwargs,
    ) -> None:
        """Write dictionary of xarray.Dataset and/or xarray.DataArray to netcdf files.

        Possibility to update the xarray objects attributes to get GDAL compliant NetCDF
        files, using :py:meth:`~hydromt.raster.gdal_compliant`.
        The function will first try to directly write to file.
        In case of PermissionError, it will first write a temporary file then move the file over.

        key-word arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`

        Parameters
        ----------
        nc_dict: dict
            Dictionary of xarray.Dataset and/or xarray.DataArray to write
        filename: str, optional
            filename relative to model root and should contain a {name} placeholder
            Can be a relative path.
        gdal_compliant: bool
            If True, convert xarray.Dataset and/or xarray.DataArray to gdal compliant
            format using :py:meth:`~hydromt.raster.gdal_compliant`
        rename_dims: bool
            If True, rename x_dim and y_dim to standard names depending on the CRS
            (x/y for projected and lat/lon for geographic). Only used if
            ``gdal_compliant`` is set to True. By default, False.
        force_sn: bool
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
                f"{self.model.name}.{self.name_in_model}: Writing datasets to {file_path}."
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

    def test_equal(self, other: ModelComponent) -> tuple[bool, dict[str, str]]:
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
        other_datasets = cast(DatasetsComponent, other)
        for name, ds in self.data.items():
            if name not in other_datasets.data:
                errors[name] = "Dataset not found in other component."
            try:
                xr.testing.assert_allclose(ds, other_datasets.data[name])
            except AssertionError as e:
                errors[name] = str(e)

        return len(errors) == 0, errors
