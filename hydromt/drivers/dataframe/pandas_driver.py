"""Driver for DataFrames using the pandas library."""

from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from hydromt._typing import (
    NoDataStrategy,
    SourceMetadata,
    StrPath,
    TimeRange,
    Variables,
)
from hydromt._utils.unused_kwargs import warn_on_unused_kwargs
from hydromt.drivers.dataframe import DataFrameDriver

logger: Logger = getLogger(__name__)


class PandasDriver(DataFrameDriver):
    """Driver for DataFrames using the pandas library."""

    name = "pandas"
    supports_writing = True

    def read_data(
        self,
        uris: List[str],
        *,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        metadata: Optional[SourceMetadata] = None,
        logger: Logger = logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> pd.DataFrame:
        """Read in any compatible data source to a pandas `DataFrame`."""
        if len(uris) > 1:
            raise ValueError(
                "DataFrame: Reading multiple files with the "
                f"{self.__class__.__name__} driver is not supported."
            )
        warn_on_unused_kwargs(
            self.__class__.__name__,
            {"time_range": time_range, "metadata": metadata},
            logger,
        )
        uri = uris[0]
        extension: str = uri.split(".")[-1]
        logger.info(f"Reading using {self.name} driver from {uri}")
        if extension == "csv":
            variables = self._unify_variables_and_pandas_kwargs(
                uri, pd.read_csv, variables
            )
            return pd.read_csv(
                uri,
                usecols=variables,
                **self.options,
            )
        elif extension == "parquet":
            return pd.read_parquet(uri, columns=variables, **self.options)
        elif extension in ["xls", "xlsx"]:
            variables = self._unify_variables_and_pandas_kwargs(
                uri, pd.read_excel, variables
            )
            return pd.read_excel(
                uri,
                usecols=variables,
                engine="openpyxl",
                **self.options,
            )
        elif extension in ["fwf", "txt"]:
            warn_on_unused_kwargs(
                self.__class__.__name__, {"variables": variables}, logger
            )
            return pd.read_fwf(uri, **self.options)
        else:
            raise IOError(f"DataFrame: extension {extension} unknown.")

    def write(
        self,
        path: StrPath,
        df: pd.DataFrame,
        **kwargs,
    ) -> None:
        """
        Write out a DataFrame to file.

        Not all drivers should have a write function, so this method is not
        abstract.

        args:
        """
        if isinstance(path, str):
            extension: str = path.split(".")[-1]
        elif isinstance(path, Path):
            # suffix includes the '.' which we don't want
            extension: str = path.suffix[1:]
        else:
            raise ValueError(f"unknown pathlike: {path}")

        logger.info(f"Writing dataframe using {self.name} driver to {str(path)}")

        if extension == "csv":
            df.to_csv(path, **kwargs)
        elif extension == "parquet":
            df.to_parquet(path, **kwargs)
        elif extension in ["xlsx", "xls"]:
            df.to_excel(path, engine="openpyxl", **kwargs)
        else:
            raise ValueError(f"DataFrame: file extension {extension} is unknown.")

    def _unify_variables_and_pandas_kwargs(
        self,
        uri: str,
        read_method: Callable,
        variables: Optional[Variables],
    ):
        """Prevent clashes between arguments and hydromt query parameters."""
        # include index_col in variables
        if variables:
            if not isinstance(self.options.get("index_col", "str"), str):
                # if index_col is an index, get name of col
                new_options: Dict[str, Any] = self.options.copy()
                new_options.pop("index_col")
                df: pd.DataFrame = read_method(uri, **{"nrows": 1, **new_options})
                return variables + [df.columns[0]]

        return variables
