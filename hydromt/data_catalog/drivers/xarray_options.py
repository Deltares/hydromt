"""Options for configuring xarray-based drivers in the data catalog."""

import enum
import logging
from os.path import splitext

from pydantic import Field

from hydromt.data_catalog.drivers.base_driver import DriverOptions
from hydromt.data_catalog.drivers.preprocessing import Preprocessor, get_preprocessor

logger = logging.getLogger(__name__)


class XarrayIOFormat(enum.Enum):
    """Enumeration of supported xarray IO formats, each associated with a set of file extensions."""

    ZARR = "zarr"
    NETCDF4 = "netcdf4"

    @property
    def extensions(self) -> set[str]:
        """Get the set of file extensions associated with this IO format."""
        if self == XarrayIOFormat.ZARR:
            return {".zarr"}
        elif self == XarrayIOFormat.NETCDF4:
            return {".nc", ".netcdf"}
        return set()


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

    def get_io_format(self, uri: str) -> XarrayIOFormat | None:
        """Determine the xarray reading format based on the file extension or override.

        Parameters
        ----------
        uri : str
            The URI of the file to read, used to infer the format from the extension if no override is set.

        Returns
        -------
        format : XarrayIOFormat | None
            The xarray reading format, either 'zarr' or 'netcdf4'. Returns None if the extension is unsupported.
        """
        ext = self.get_reading_ext(uri)

        if ext in XarrayIOFormat.ZARR.extensions:
            return XarrayIOFormat.ZARR
        elif ext in XarrayIOFormat.NETCDF4.extensions:
            return XarrayIOFormat.NETCDF4
        return None

    def filter_uris_by_format(
        self, uris: list[str]
    ) -> tuple[list[str], XarrayIOFormat | None]:
        """Filter the list of URIs to only include those matching the specified format.

        Parameters
        ----------
        uris : list[str]
            List of URIs to filter.

        Returns
        -------
        filtered_uris : list[str]
            List of URIs that match the specified format.
        io_format : XarrayIOFormat
            The xarray reading format to filter by, either 'zarr' or 'netcdf4'.
        """
        io_format = self.get_io_format(uris[0])
        filtered_uris = []
        for _uri in uris:
            _format = self.get_io_format(_uri)
            if _format is None:
                logger.warning(
                    f"URI {_uri} has unsupported extension for format {io_format.value} and ext_override is not set, skipping..."
                )
            elif _format != io_format:
                logger.warning(
                    f"Reading {io_format.value} and {_uri} has a different format {_format.value}, skipping..."
                )
            else:
                filtered_uris.append(_uri)
        return filtered_uris, io_format
