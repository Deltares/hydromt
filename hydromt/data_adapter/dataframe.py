"""Implementation for the Pandas Dataframe adapter."""
import logging
import os
from os.path import join

import numpy as np
import pandas as pd

from .data_adapter import DataAdapter

logger = logging.getLogger(__name__)

__all__ = [
    "DataFrameAdapter",
]


class DataFrameAdapter(DataAdapter):

    """DataAdapter implementation for Pandas Dataframes."""

    _DEFAULT_DRIVER = "csv"
    _DRIVERS = {"xlsx": "excel", "xls": "excel"}

    def __init__(
        self,
        path,
        driver=None,
        filesystem="local",
        crs=None,
        nodata=None,
        rename={},
        unit_mult={},
        unit_add={},
        meta={},
        attrs={},
        driver_kwargs={},
        name="",  # optional for now
        catalog_name="",  # optional for now
    ):
        """Initiate data adapter for 2D tabular data.

        This object contains all properties required to read supported files into
        a :py:func:`pandas.DataFrame`.
        In addition it keeps meta data to be able to reproduce which data is used.

        Parameters
        ----------
        path: str, Path
            Path to data source. If the dataset consists of multiple files, the path may
            contain {variable}, {year}, {month} placeholders as well as path
            search pattern using a '*' wildcard.
        driver: {'vector', 'vector_table'}, optional
            Driver to read files with, for 'vector' :py:func:`~geopandas.read_file`,
            for {'vector_table'} :py:func:`hydromt.io.open_vector_from_table`
            By default the driver is inferred from the file extension and falls back to
            'vector' if unknown.
        filesystem: {'local', 'gcs', 's3'}, optional
            Filesystem where the data is stored (local, cloud, http etc.).
            By default, local.
        crs: int, dict, or str, optional
            Coordinate Reference System. Accepts EPSG codes (int or str);
            proj (str or dict) or wkt (str). Only used if the data has no native CRS.
        nodata: float, int, optional
            Missing value number. Only used if the data has no native missing value.
            Nodata values can be differentiated between variables using a dictionary.
        rename: dict, optional
            Mapping of native data source variable to output source variable name as
            required by hydroMT.
        unit_mult, unit_add: dict, optional
            Scaling multiplication and addition to change to map from the native
            data unit to the output data unit as required by hydroMT.
        meta: dict, optional
            Metadata information of dataset, prefably containing the following keys:
            {'source_version', 'source_url', 'source_license',
            'paper_ref', 'paper_doi', 'category'}
        placeholders: dict, optional
            Placeholders to expand yaml entry to multiple entries (name and path)
            based on placeholder values
        attrs: dict, optional
            Additional attributes relating to data variables. For instance unit
            or long name of the variable.
        driver_kwargs, dict, optional
            Additional key-word arguments passed to the driver.
        name, catalog_name: str, optional
            Name of the dataset and catalog, optional for now.
        """
        super().__init__(
            path=path,
            driver=driver,
            filesystem=filesystem,
            nodata=nodata,
            rename=rename,
            unit_mult=unit_mult,
            unit_add=unit_add,
            meta=meta,
            attrs=attrs,
            driver_kwargs=driver_kwargs,
            name=name,
            catalog_name=catalog_name,
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
        """Save a dataframe slice to a file.

        Parameters
        ----------
        data_root : str or Path
            Path to the output folder.
        data_name : str
            Name of the output file without extension.
        driver : str, optional
            Driver to write the file, e.g., 'csv', 'excel'. If None,
            the default behavior is used.
        variables : list of str, optional
            Names of DataFrame columns to include in the output. By default,
            all columns are included.
        time_tuple : tuple of str or datetime, optional
            Start and end date of the period of interest. By default, the entire time
            period of the DataFrame is included.
        logger : Logger, optional
            Logger object to log warnings or messages. By default, the module
            logger is used.
        **kwargs : dict
            Additional keyword arguments to be passed to the file writing method.

        Returns
        -------
        fn_out : str
            Absolute path to the output file.
        driver : str
            Name of the driver used to read the data.
            See :py:func:`~hydromt.data_catalog.DataCatalog.get_geodataset`.


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
            # always write as CSV
            driver = "csv"
            fn_out = join(data_root, f"{data_name}.csv")
            obj.to_csv(fn_out, **kwargs)
        elif driver == "excel":
            fn_out = join(data_root, f"{data_name}.xlsx")
            obj.to_excel(fn_out, **kwargs)
        else:
            raise ValueError(f"DataFrame: Driver {driver} is unknown.")

        return fn_out, driver

    def get_data(
        self,
        variables=None,
        time_tuple=None,
        logger=logger,
    ):
        """Return a DataFrame.

        Returned data is optionally sliced by time and variables,
        based on the properties of this DataFrameAdapter. For a detailed
        description see: :py:func:`~hydromt.data_catalog.DataCatalog.get_dataframe`
        """
        # Extract storage_options from kwargs to instantiate fsspec object correctly
        so_kwargs = {}
        if "storage_options" in self.driver_kwargs:
            so_kwargs = self.driver_kwargs["storage_options"]
            # For s3, anonymous connection still requires --no-sign-request profile
            # to read the data setting environment variable works
            if "anon" in so_kwargs:
                os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
            else:
                os.environ["AWS_NO_SIGN_REQUEST"] = "NO"
        _ = self.resolve_paths(**so_kwargs)  # throw nice error if data not found

        kwargs = self.driver_kwargs.copy()

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
                raise IndexError("DataFrame: Time slice out of range.")

        # set meta data
        df.attrs.update(self.meta)

        # set column attributes
        for col in self.attrs:
            if col in df.columns:
                df[col].attrs.update(**self.attrs[col])

        return df
