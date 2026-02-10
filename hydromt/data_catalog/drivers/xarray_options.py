"""Options for configuring xarray-based drivers in the data catalog."""

from os.path import splitext

from pydantic import Field

from hydromt.data_catalog.drivers.base_driver import DriverOptions
from hydromt.data_catalog.drivers.preprocessing import Preprocessor, get_preprocessor

_ZARR_EXT = {".zarr"}
_NETCDF_EXT = {".nc", ".netcdf"}


class XarrayDriverOptions(DriverOptions):
    """Options for configuring xarray-based drivers."""

    preprocess: str | None = Field(
        default=None,
        description="Name of preprocessor to apply before merging datasets. Available preprocessors include: round_latlon, to_datetimeindex, remove_duplicates, harmonise_dims. See their docstrings for details.",
    )
    ext_override: str | None = Field(
        default=None,
        description="Override the file extension check and try to read all files as the given extension. Useful when reading zarr files without the .zarr extension.",
    )

    def get_preprocessor(self) -> Preprocessor:
        """Get the preprocessor instance based on the configured preprocess string."""
        return get_preprocessor(self.preprocess)

    def get_reading_ext(self, uri: str) -> str:
        """Determine the file extension to use for reading, either from the override or from the URI."""
        return self.ext_override or splitext(uri)[-1]

    def get_io_format(self, uri: str) -> str:
        """Determine the xarray reading format based on the file extension or override.

        Parameters
        ----------
        uri : str
            The URI of the file to read, used to infer the format from the extension if no override is set.

        Returns
        -------
        format : str
            The xarray reading format, either 'zarr' or 'netcdf4'.
        """
        ext = self.get_reading_ext(uri)
        if ext in _ZARR_EXT:
            return "zarr"
        if ext in _NETCDF_EXT:
            return "netcdf4"
        raise ValueError(f"Unsupported xarray extension: {ext}")
