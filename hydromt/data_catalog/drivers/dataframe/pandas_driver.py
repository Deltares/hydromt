"""Driver for DataFrames using the pandas library."""

from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs
from hydromt.data_catalog.drivers.dataframe import DataFrameDriver
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.typing import (
    StrPath,
    TimeRange,
    Variables,
)
from hydromt.typing.metadata import SourceMetadata


class PandasDriver(DataFrameDriver):
    """
    Driver for DataFrames using the pandas library: ``pandas``.

    Supports reading and writing csv, excel (xls, xlsx), parquet, fixed width formatted
    files (fwf) using pandas.
    """

    name = "pandas"
    supports_writing = True

    def read(
        self,
        uris: list[str],
        *,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        kwargs_for_open: dict[str, Any] | None = None,
        variables: Variables | None = None,
        time_range: TimeRange | None = None,
        metadata: SourceMetadata | None = None,
    ) -> pd.DataFrame:
        """Read in any compatible data source to a pandas `DataFrame`."""
        _warn_on_unused_kwargs(
            self.__class__.__name__,
            {"time_range": time_range},
        )
        if len(uris) > 1:
            raise ValueError(
                "DataFrame: Reading multiple files with the "
                f"{self.__class__.__name__} driver is not supported."
            )
        elif len(uris) == 0:
            df = pd.DataFrame()
        else:
            uri = uris[0]
            extension: str = uri.split(".")[-1]
            kwargs_for_open = kwargs_for_open or {}
            kwargs = self.options.get_kwargs() | kwargs_for_open

            if extension == "csv":
                variables = self._unify_variables_and_pandas_kwargs(
                    uri, pd.read_csv, variables
                )
                df = pd.read_csv(
                    uri,
                    usecols=variables,
                    **kwargs,
                )
            elif extension == "parquet":
                df = pd.read_parquet(uri, columns=variables, **kwargs)
            elif extension in ["xls", "xlsx"]:
                variables = self._unify_variables_and_pandas_kwargs(
                    uri, pd.read_excel, variables
                )
                df = pd.read_excel(
                    uri,
                    usecols=variables,
                    engine="openpyxl",
                    **kwargs,
                )
            elif extension in ["fwf", "txt"]:
                _warn_on_unused_kwargs(
                    self.__class__.__name__, {"variables": variables}
                )
                df = pd.read_fwf(uri, **kwargs)
            else:
                raise IOError(f"DataFrame: extension {extension} unknown.")
        if df.index.size == 0:
            exec_nodata_strat(
                f"No data from driver {self}'.",
                strategy=handle_nodata,
            )
        return df

    def write(
        self,
        path: StrPath,
        df: pd.DataFrame,
        **kwargs,
    ) -> str:
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

        if extension == "csv":
            df.to_csv(path, **kwargs)
        elif extension == "parquet":
            df.to_parquet(path, **kwargs)
        elif extension in ["xlsx", "xls"]:
            df.to_excel(path, engine="openpyxl", **kwargs)
        else:
            raise ValueError(f"DataFrame: file extension {extension} is unknown.")

        return str(path)

    def _unify_variables_and_pandas_kwargs(
        self,
        uri: str,
        read_method: Callable,
        variables: Optional[Variables],
    ):
        """Prevent clashes between arguments and hydromt query parameters."""
        # include index_col in variables
        if variables:
            if hasattr(self.options, "index_col") and not isinstance(
                self.options.index_col, str
            ):
                # if index_col is an index, get name of col
                new_options: dict[str, Any] = self.options.get_kwargs()
                new_options.pop("index_col", None)
                df: pd.DataFrame = read_method(uri, **{"nrows": 1, **new_options})
                return variables + [df.columns[0]]

        return variables
