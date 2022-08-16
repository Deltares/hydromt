# -*- coding: utf-8 -*-
from os.path import join
import numpy as np
import pandas as pd
from .data_adapter import DataAdapter

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "DataFrameAdapter",
]


class DataFrameAdapter(DataAdapter):
    _DEFAULT_DRIVER = "csv"
    _DRIVERS = {"xlsx": "excel", "xls": "excel"}

    def __init__(
        self,
        path,
        driver=None,
        nodata=None,
        rename={},
        unit_mult={},
        unit_add={},
        meta={},
        placeholders={},
        **kwargs,
    ):
        """Initiates data adapter for 2D tabular data.

        This object contains all properties required to read supported files into
        a :py:func:`pandas.DataFrame`.
        In addition it keeps meta data to be able to reproduce which data is used.

        Parameters
        ----------
        path: str, Path
            Path to data source.
        driver: {'csv', 'xlsx', 'xls', 'fwf'}, optional
            Driver to read files with, for 'csv' :py:func:`~pandas.read_csv`,
            for {'xlsx', 'xls'} :py:func:`~pandas.read_excel`, and for 'fwf'
            :py:func:`~pandas.read_fwf`.
            By default the driver is inferred from the file extension and falls back to
            'csv' if unknown.
        nodata: (dictionary) float, int, optional
            Missing value number. Only used if the data has no native missing value.
            Multiple nodata values can be provided in a list and differentiated between
            dataframe columns using a dictionary with variable (column) keys. The nodata
            values are only applied to columns with numeric data.
        rename: dict, optional
            Mapping of native column names to output column names as
            required by hydroMT.
        unit_mult, unit_add: dict, optional
            Scaling multiplication and addition to change to map from the native data unit
            to the output data unit as required by hydroMT.
        meta: dict, optional
            Metadata information of dataframe, prefably containing the following keys:
            {'source_version', 'source_url', 'source_license', 'paper_ref', 'paper_doi', 'category'}
        **kwargs
            Additional key-word arguments passed to the driver.
        """
        super().__init__(
            path=path,
            driver=driver,
            nodata=nodata,
            rename=rename,
            unit_mult=unit_mult,
            unit_add=unit_add,
            meta=meta,
            placeholders=placeholders,
            **kwargs,
        )

    def to_file(
        self,
        data_root,
        data_name,
        driver=None,
        variables=None,
        time_tuple=None,
        logger=logger,
        **kwargs,
    ):
        """Save dataframe slice to file.

        Parameters
        ----------
        data_root : str, Path
            Path to output folder
        data_name : str
            Name of output file without extension.
        driver : str, optional
            Driver to write file, e.g.: 'csv', 'excel', by default None
        variables : list of str, optional
            Names of DataFrame columns to return. By default all columns
            are returned.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the DataFrame is returned.

        Returns
        -------
        fn_out: str
            Absolute path to output file
        driver: str
            Name of driver to read data with, see :py:func:`~hydromt.data_catalog.DataCatalog.get_geodataset`
        """
        kwargs.pop("bbox", None)
        try:
            obj = self.get_data(
                time_tuple=time_tuple, variables=variables, logger=logger
            )
        except IndexError as err:  # out of bounds for time
            logger.warning(str(err))
            return None, None

        if driver is None or driver == "csv":
            # always write netcdf
            driver = "csv"
            fn_out = join(data_root, f"{data_name}.csv")

            obj.to_csv(fn_out, **kwargs)
        elif driver == "excel":
            fn_out = join(data_root, f"{data_name}.xlsx")
            obj.to_excel(fn_out, **kwargs)
        else:
            raise ValueError(f"DataFrame: Driver {driver} unknown.")

        return fn_out, driver

    def get_data(
        self,
        variables=None,
        time_tuple=None,
        logger=logger,
        **kwargs,
    ):
        """Returns a DataFrame, optionally sliced by time and variables, based on the properties of this DataFrameAdapter.

        For a detailed description see: :py:func:`~hydromt.data_catalog.DataCatalog.get_dataframe`
        """

        kwargs = self.kwargs.copy()
        _ = self.resolve_paths()  # throw nice error if data not found

        # read and clip
        logger.info(f"DataFrame: Read {self.driver} data.")

        if self.driver in ["csv"]:
            df = pd.read_csv(self.path, **kwargs)
        elif self.driver in ["xls", "xlsx", "excel"]:
            df = pd.read_excel(self.path, engine="openpyxl", **kwargs)
        elif self.driver in ["fwf"]:
            df = pd.read_fwf(self.path, **kwargs)
        else:
            raise IOError(f"DataFrame: driver {self.driver} unknown.")

        # rename and select columns
        if self.rename:
            rename = {k: v for k, v in self.rename.items() if k in df.columns}
            df = df.rename(columns=rename)
        if variables is not None:
            if np.any([var not in df.columns for var in variables]):
                raise ValueError(f"DataFrame: Not all variables found: {variables}")
            df = df.loc[:, variables]

        # nodata and unit conversion for numeric data
        if df.index.size == 0:
            logger.warning(f"DataFrame: No data within spatial domain {self.path}.")
        else:
            # parse nodata values
            cols = df.select_dtypes([np.number]).columns
            if self.nodata is not None and len(cols) > 0:
                if not isinstance(self.nodata, dict):
                    nodata = {c: self.nodata for c in cols}
                else:
                    nodata = self.nodata
                for c in cols:
                    mv = nodata.get(c, None)
                    if mv is not None:
                        is_nodata = np.isin(df[c], np.atleast_1d(mv))
                        df[c] = np.where(is_nodata, np.nan, df[c])

            # unit conversion
            unit_names = list(self.unit_mult.keys()) + list(self.unit_add.keys())
            unit_names = [k for k in unit_names if k in df.columns]
            if len(unit_names) > 0:
                logger.debug(f"DataFrame: Convert units for {len(unit_names)} columns.")
            for name in list(set(unit_names)):  # unique
                m = self.unit_mult.get(name, 1)
                a = self.unit_add.get(name, 0)
                df[name] = df[name] * m + a

        # clip time slice
        if time_tuple is not None and np.dtype(df.index).type == np.datetime64:
            logger.debug(f"DataFrame: Slicing time dime {time_tuple}")
            df = df[df.index.slice_indexer(*time_tuple)]
            if df.size == 0:
                raise IndexError(f"DataFrame: Time slice out of range.")

        # set meta data
        df.attrs.update(self.meta)

        return df
