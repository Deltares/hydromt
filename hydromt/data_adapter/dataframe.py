"""Implementation for the Pandas Dataframe adapter."""
import logging
import warnings
from datetime import datetime
from os.path import join
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem

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
        path: str,
        driver: Optional[str] = None,
        filesystem: Optional[str] = None,
        nodata: Optional[Union[dict, float, int]] = None,
        rename: Optional[dict] = None,
        unit_mult: Optional[dict] = None,
        unit_add: Optional[dict] = None,
        meta: Optional[dict] = None,
        attrs: Optional[dict] = None,
        driver_kwargs: Optional[dict] = None,
        storage_options: Optional[dict] = None,
        name: str = "",  # optional for now
        catalog_name: str = "",  # optional for now
        provider: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
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
        driver: {'csv', 'parquet', 'xlsx', 'xls', 'fwf'}, optional
            Driver to read files with, for 'csv' :py:func:`~pandas.read_csv`,
            for 'parquet' :py:func:`~pandas.read_parquet`, for {'xlsx', 'xls'}
            :py:func:`~pandas.read_excel`, and for 'fwf' :py:func:`~pandas.read_fwf`.
            By default the driver is inferred from the file extension and falls back to
            'csv' if unknown.
        filesystem: str, optional
            Filesystem where the data is stored (local, cloud, http etc.).
            If None (default) the filesystem is inferred from the path.
            See :py:func:`fsspec.registry.known_implementations` for all options.
        nodata: dict, float, int, optional
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
        extent: None
            Not used in this adapter. Only here for compatability with other adapters.
        driver_kwargs, dict, optional
            Additional key-word arguments passed to the driver.
        storage_options: dict, optional
            Additional key-word arguments passed to the fsspec FileSystem object.
        name, catalog_name: str, optional
            Name of the dataset and catalog, optional for now.
        """
        driver_kwargs = driver_kwargs or {}
        attrs = attrs or {}
        meta = meta or {}
        unit_add = unit_add or {}
        unit_mult = unit_mult or {}
        rename = rename or {}
        storage_options = storage_options or {}

        if kwargs:
            warnings.warn(
                "Passing additional keyword arguments to be used by the "
                "DataFrameAdapter driver is deprecated and will be removed "
                "in a future version. Please use 'driver_kwargs' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            driver_kwargs.update(kwargs)
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
            storage_options=storage_options,
            name=name,
            catalog_name=catalog_name,
            provider=provider,
            version=version,
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
            Driver to write the file, e.g., 'csv','parquet', 'excel'. If None,
            the default behavior is used.
        variables : list of str, optional
            Names of DataFrame columns to include in the output. By default,
            all columns are included.
        time_tuple : tuple of str or datetime, optional
            Start and end date of the period of interest. By default, the entire time
            period of the DataFrame is included.
        **kwargs : dict
            Additional keyword arguments to be passed to the file writing method.

        Returns
        -------
        fn_out : str
            Absolute path to the output file.
        driver : str
            Name of the driver used to read the data.
            See :py:func:`~hydromt.data_catalog.DataCatalog.get_geodataset`.
        kwargs: dict
            The additional keyword arguments that were passed in.


        """
        kwargs.pop("bbox", None)
        obj = self.get_data(time_tuple=time_tuple, variables=variables, logger=logger)

        read_kwargs = dict()
        if driver is None or driver == "csv":
            # always write as CSV
            driver = "csv"
            fn_out = join(data_root, f"{data_name}.csv")
            obj.to_csv(fn_out, **kwargs)
            read_kwargs["index_col"] = 0
        elif driver == "parquet":
            fn_out = join(data_root, f"{data_name}.parquet")
            obj.to_parquet(fn_out, **kwargs)
        elif driver == "excel":
            fn_out = join(data_root, f"{data_name}.xlsx")
            obj.to_excel(fn_out, **kwargs)
        else:
            raise ValueError(f"DataFrame: Driver {driver} is unknown.")

        return fn_out, driver, read_kwargs

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
        # load data
        fns = self._resolve_paths(variables)
        df = self._read_data(fns, logger=logger)
        self.mark_as_used()  # mark used
        # rename variables and parse nodata
        df = self._rename_vars(df)
        df = self._set_nodata(df)
        # slice data
        df = DataFrameAdapter._slice_data(df, variables, time_tuple, logger=logger)
        # uniformize data
        df = self._apply_unit_conversion(df, logger=logger)
        df = self._set_metadata(df)
        return df

    def _read_data(self, fns, logger=logger):
        if len(fns) > 1:
            raise ValueError(
                f"DataFrame: Reading multiple {self.driver} files is not supported."
            )
        kwargs = self.driver_kwargs.copy()
        path = fns[0]
        logger.info(f"Reading {self.name} {self.driver} data from {self.path}")
        if self.driver in ["csv"]:
            df = pd.read_csv(path, **kwargs)
        elif self.driver == "parquet":
            _ = kwargs.pop("index_col", None)
            df = pd.read_parquet(path, **kwargs)
        elif self.driver in ["xls", "xlsx", "excel"]:
            df = pd.read_excel(path, engine="openpyxl", **kwargs)
        elif self.driver in ["fwf"]:
            df = pd.read_fwf(path, **kwargs)
        else:
            raise IOError(f"DataFrame: driver {self.driver} unknown.")

        return df

    def _rename_vars(self, df):
        if self.rename:
            rename = {k: v for k, v in self.rename.items() if k in df.columns}
            df = df.rename(columns=rename)
        return df

    def _set_nodata(self, df):
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
        return df

    @staticmethod
    def _slice_data(df, variables=None, time_tuple=None, logger=logger):
        """Return a sliced DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            the dataframe to be sliced.
        variables : list of str, optional
            Names of DataFrame columns to include in the output. By default all columns
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.

        Returns
        -------
        pd.DataFrame
            Tabular data
        """
        if variables is not None:
            variables = np.atleast_1d(variables).tolist()
            if np.any([var not in df.columns for var in variables]):
                raise ValueError(f"DataFrame: Not all variables found: {variables}")
            df = df.loc[:, variables]

        if time_tuple is not None and np.dtype(df.index).type == np.datetime64:
            logger.debug(f"Slicing time dime {time_tuple}")
            df = df[df.index.slice_indexer(*time_tuple)]
            if df.size == 0:
                raise IndexError("DataFrame: Time slice out of range.")

        return df

    def _apply_unit_conversion(self, df, logger=logger):
        unit_names = list(self.unit_mult.keys()) + list(self.unit_add.keys())
        unit_names = [k for k in unit_names if k in df.columns]
        if len(unit_names) > 0:
            logger.debug(f"Convert units for {len(unit_names)} columns.")
        for name in list(set(unit_names)):  # unique
            m = self.unit_mult.get(name, 1)
            a = self.unit_add.get(name, 0)
            df[name] = df[name] * m + a
        return df

    def _set_metadata(self, df):
        df.attrs.update(self.meta)

        # set column attributes
        for col in self.attrs:
            if col in df.columns:
                df[col].attrs.update(**self.attrs[col])

        return df

    def to_stac_catalog(
        self,
        on_error: Literal["raise", "skip", "coerce"] = "coerce",
    ) -> Optional[StacCatalog]:
        """
        Convert a rasterdataset into a STAC Catalog representation.

        The collection will contain an asset for each of the associated files.


        Parameters
        ----------
        - on_error (str, optional): The error handling strategy.
          Options are: "raise" to raise an error on failure, "skip" to skip the
          dataset on failure, and "coerce" (default) to set default values on failure.

        Returns
        -------
        - Optional[StacCatalog]: The STAC Catalog representation of the dataset, or None
          if the dataset was skipped.
        """
        if on_error == "skip":
            logger.warning(
                f"Skipping {self.name} during stac conversion because"
                "because detecting temporal extent failed."
            )
            return
        elif on_error == "coerce":
            stac_catalog = StacCatalog(
                self.name,
                description=self.name,
            )
            stac_item = StacItem(
                self.name,
                geometry=None,
                bbox=[0, 0, 0, 0],
                properties=self.meta,
                datetime=datetime(1, 1, 1),
            )
            stac_asset = StacAsset(str(self.path))
            stac_item.add_asset("hydromt_path", stac_asset)

            stac_catalog.add_item(stac_item)
            return stac_catalog
        else:
            raise NotImplementedError(
                "DataframeAdapter does not support full stac conversion as it lacks"
                " spatio-temporal dimentions"
            )
