"""Abstract DataSource class."""
from typing import Any

from hydromt.drivers.abstract_driver import AbstractDriver


class DataSource:
    """
    A DataSource is a parsed section of a DataCatalog.

    The DataSource, specific for a data type within HydroMT, is responsible for
    validating the input from the DataCatalog, to
    ensure the workflow fails as early as possible. A DataSource has information on
    the driver that the data should be read with, and is responsible for initializing
    this driver.
    """

    name: str
    version: str
    provider: str  # ?
    driver: AbstractDriver
    driver_kwargs: dict[str, Any]
    extent: dict[str, Any]  # ?
    meta: dict[str, Any]

    def set_driver(self, **driver_kwargs):
        """Set and init driver."""
        pass

    def set_adapters(self, attrs, rename, unit_add, unit_mult):
        """Set possible adapters."""
        pass
