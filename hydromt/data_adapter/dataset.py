"""Implementation for the dataset DataAdapter."""
import logging
from datetime import datetime
from pathlib import Path
from typing import List, NewType, Optional, Tuple

import xarray as xr

from .data_adapter import DataAdapter

logger = logging.getLogger(__name__)

DatasetSource = NewType("DatasetSource", str | Path)


class DatasetAdapter(DataAdapter):
    """DatasetAdapter for non-spatial n-dimensional data."""

    _DEFAULT_DRIVER = ""
    _DRIVERS = {
        "nc": "netcdf",
    }

    def __init__(
        self,
        path: str | Path,
        driver: Optional[str] = None,
        filesystem: Optional[str] = None,
        nodata: Optional[dict | float | int] = None,
        rename: Optional[dict] = None,
        unit_mult: Optional[dict] = None,
        unit_add: Optional[dict] = None,
        meta: Optional[dict] = None,
        attrs: Optional[dict] = None,
        driver_kwargs: Optional[dict] = None,
        storage_options: Optional[dict] = None,
        name: Optional[str] = "",
        catalog_name: Optional[str] = "",
        provider: Optional[str] = None,
        version: Optional[str] = None,
    ):
        """Docstring."""
        super().__init__(
            path,
            driver,
            filesystem,
            nodata,
            rename,
            unit_mult,
            unit_add,
            meta,
            attrs,
            driver_kwargs,
            storage_options,
            name,
            catalog_name,
            provider,
            version,
        )

    def to_file(
        self,
        data_root: str | Path,
        data_name: str,
        time_tuple: Optional[Tuple[str | datetime]] = None,
        variables: Optional[List[str]] = None,
        driver: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        """Docstring."""
        pass

    def get_data(
        self,
        variables: Optional[List[str]] = None,
        time_tuple: Optional[Tuple[str | datetime]] = None,
        single_var_as_array: Optional[bool] = True,
        logger: Optional[logging.Logger] = logger,
    ) -> xr.Dataset:
        """Docstring."""
        pass
