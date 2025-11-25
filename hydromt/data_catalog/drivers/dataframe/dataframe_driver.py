"""Generic driver for reading and writing DataFrames."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from hydromt.data_catalog.drivers import BaseDriver
from hydromt.error import NoDataStrategy
from hydromt.typing import (
    Variables,
)

logger = logging.getLogger(__name__)


class DataFrameDriver(BaseDriver, ABC):
    """Abstract Driver to read DataFrames."""

    supports_writing: ClassVar[bool] = False

    @abstractmethod
    def read(
        self,
        uris: list[str],
        *,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        variables: Variables | None = None,
    ) -> pd.DataFrame:
        """
        Read data from one or more URIs into a pandas DataFrame.

        This abstract method defines the interface for all DataFrame-based drivers.
        Subclasses should implement data loading logic appropriate for the format being read.

        Parameters
        ----------
        uris : list[str]
            List of URIs to read data from. The driver decides how to handle multiple files.
        handle_nodata : NoDataStrategy, optional
            Strategy to handle missing or empty data. Default is NoDataStrategy.RAISE.
        variables : Variables | None, optional
            List of variable names (columns) to select from the source data. Default is None.

        Returns
        -------
        pd.DataFrame
            The loaded DataFrame.

        """
        ...

    @abstractmethod
    def write(
        self,
        path: Path | str,
        data: pd.DataFrame,
        *,
        write_kwargs: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write a pandas DataFrame to a file.

        This abstract method defines the interface for writing DataFrames in all DataFrame-based drivers.
        Subclasses should implement logic appropriate for the target file format (e.g., CSV, Parquet, Excel).

        Parameters
        ----------
        path : Path | str
            Destination path or URI where the DataFrame should be written.
        data : pd.DataFrame
            The DataFrame to be written to disk.
        write_kwargs : dict[str, Any], optional
            Additional keyword arguments to pass to the underlying pandas write function
            (e.g., `to_csv`, `to_excel`, `to_parquet`). Default is None.

        Returns
        -------
        Path
            The path where the data was written.

        """
        ...
