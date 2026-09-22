"""Driver for DataFrames using the pandas library."""

from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fsspec import AbstractFileSystem

from hydromt.data_catalog.drivers.base_driver import resolve_filesystem
from hydromt.data_catalog.drivers.dataframe import DataFrameDriver
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.readers import _read_table
from hydromt.typing import Variables


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
        variables: Variables | None = None,
    ) -> pd.DataFrame:
        """
        Read tabular data into a pandas DataFrame using the pandas library.

        Supports multiple file formats including CSV, Parquet, Excel (xls, xlsx),
        and fixed-width text files (fwf). Applies hydromt's `NoDataStrategy` if no
        records are found. Column selection can be controlled through the `variables`
        argument. Only a single file can be read per call.

        Parameters
        ----------
        uris : list[str]
            List containing a single URI to read from. Multiple files are not supported.
        handle_nodata : NoDataStrategy, optional
            Strategy to handle missing or empty data. Default is NoDataStrategy.RAISE.
        variables : Variables | None, optional
            List of columns to read from the file. Ignored if None.

        Returns
        -------
        pd.DataFrame
            The loaded DataFrame containing the data from the source file.

        Raises
        ------
        ValueError
            If multiple files are provided or the file extension is unsupported.
        IOError
            If the file extension is not recognized.
        """
        if len(uris) > 1:
            raise ValueError(
                "DataFrame: Reading multiple files with the "
                f"{self.__class__.__name__} driver is not supported."
            )
        uri = uris[0]
        extension: str = uri.split(".")[-1]

        # storage_options are handled once, here at the driver boundary.
        options = self.options.get_kwargs()
        fs = resolve_filesystem(self.filesystem, options)

        if extension == "csv":
            variables = self._unify_variables_and_pandas_kwargs(
                uri, "csv", variables, options, filesystem=fs
            )
            df = _read_table(uri, "csv", filesystem=fs, usecols=variables, **options)
        elif extension == "parquet":
            df = _read_table(
                uri, "parquet", filesystem=fs, columns=variables, **options
            )
        elif extension in ["xls", "xlsx"]:
            variables = self._unify_variables_and_pandas_kwargs(
                uri, extension, variables, options, filesystem=fs
            )
            df = _read_table(
                uri, extension, filesystem=fs, usecols=variables, **options
            )
        elif extension in ["fwf", "txt"]:
            df = _read_table(uri, extension, filesystem=fs, **options)
        else:
            raise IOError(f"DataFrame: extension {extension} unknown.")

        if df.empty:
            exec_nodata_strat(
                f"No data from {self.name} driver for file uris: {', '.join(uris)}.",
                strategy=handle_nodata,
            )
            return None  # handle_nodata == ignore
        return df

    def write(
        self,
        path: Path | str,
        data: pd.DataFrame,
        *,
        write_kwargs: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write a pandas DataFrame to disk using the pandas library.

        Supports writing to common tabular formats including CSV, Parquet, and Excel (xls, xlsx).
        The file format is automatically inferred from the file extension in the provided path.

        Parameters
        ----------
        path : Path | str
            Destination path where the DataFrame will be saved.
            The file extension determines the output format: `.csv`, `.parquet`, `.xls`, `.xlsx`.
        data : pd.DataFrame
            The DataFrame to be written.
        write_kwargs : dict[str, Any], optional
            Additional keyword arguments passed to the pandas write function (e.g., compression, index, sheet_name).
            Default is None.

        Returns
        -------
        Path
            The path where the DataFrame was written.

        Raises
        ------
        ValueError
            If the path type or file extension is unsupported.
        """
        if isinstance(path, str):
            path = Path(path)
            extension: str = path.suffix[1:]
        elif isinstance(path, Path):
            # suffix includes the '.' which we don't want
            extension: str = path.suffix[1:]
        else:
            raise ValueError(f"unknown pathlike: {path}")

        if extension == "csv":
            data.to_csv(path, **(write_kwargs or {}))
        elif extension == "parquet":
            data.to_parquet(path, **(write_kwargs or {}))
        elif extension in ["xlsx", "xls"]:
            data.to_excel(path, engine="openpyxl", **(write_kwargs or {}))
        else:
            raise ValueError(f"DataFrame: file extension {extension} is unknown.")

        return path

    def _unify_variables_and_pandas_kwargs(
        self,
        uri: str,
        fmt: str,
        variables: Optional[Variables],
        options: dict[str, Any],
        *,
        filesystem: AbstractFileSystem | None = None,
    ):
        """Prevent clashes between arguments and hydromt query parameters."""
        # include index_col in variables
        if variables:
            if hasattr(self.options, "index_col") and not isinstance(
                self.options.index_col, str
            ):
                # if index_col is an index, get name of col
                probe_options = {k: v for k, v in options.items() if k != "index_col"}
                df: pd.DataFrame = _read_table(
                    uri, fmt, filesystem=filesystem, **{"nrows": 1, **probe_options}
                )
                return variables + [df.columns[0]]

        return variables
