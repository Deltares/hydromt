"""Data adapter for DataFrames."""
from logging import Logger, getLogger
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from hydromt._typing import NoDataStrategy, TimeRange, Variables
from hydromt.data_adapter.data_adapter_base import DataAdapterBase

if TYPE_CHECKING:
    from hydromt.data_source import SourceMetadata


logger: Logger = getLogger(__name__)


class DataFrameAdapter(DataAdapterBase):
    """Data adapter for DataFrames."""

    def transform(
        self,
        df: pd.DataFrame,
        metadata: "SourceMetadata",
        *,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> pd.DataFrame:
        """Read transform data to HydroMT standards."""
        # rename variables and parse nodata
        df = self._rename_vars(df)
        df = self._set_nodata(df, metadata)
        # slice data
        df = DataFrameAdapter._slice_data(
            df,
            variables,
            time_range,
        )
        # uniformize data
        df = self._apply_unit_conversion(df)
        df = self._set_metadata(df, metadata)
        return df

    def _rename_vars(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.rename:
            rename = {k: v for k, v in self.rename.items() if k in df.columns}
            df = df.rename(columns=rename)
        return df

    def _set_nodata(self, df: pd.DataFrame, metadata: "SourceMetadata") -> pd.DataFrame:
        # parse nodata values
        cols = df.select_dtypes([np.number]).columns
        if metadata.nodata is not None and len(cols) > 0:
            if not isinstance(metadata.nodata, dict):
                nodata = {c: metadata.nodata for c in cols}
            else:
                nodata = metadata.nodata
            for c in cols:
                mv = nodata.get(c, None)
                if mv is not None:
                    is_nodata = np.isin(df[c], np.atleast_1d(mv))
                    df[c] = np.where(is_nodata, np.nan, df[c])
        return df

    @staticmethod
    def _slice_data(
        df: pd.DataFrame,
        variables: Optional[Variables] = None,
        time_tuple: Optional[TimeRange] = None,
    ) -> Optional[pd.DataFrame]:
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
            try:
                df = df[df.index.slice_indexer(*time_tuple)]
            except IndexError:
                df = pd.DataFrame()

        if len(df) == 0:
            return None
        else:
            return df

    def _apply_unit_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        unit_names = list(self.unit_mult.keys()) + list(self.unit_add.keys())
        unit_names = [k for k in unit_names if k in df.columns]
        if len(unit_names) > 0:
            logger.debug(f"Convert units for {len(unit_names)} columns.")
        for name in list(set(unit_names)):  # unique
            m = self.unit_mult.get(name, 1)
            a = self.unit_add.get(name, 0)
            df[name] = df[name] * m + a
        return df

    def _set_metadata(
        self, df: pd.DataFrame, metadata: "SourceMetadata"
    ) -> pd.DataFrame:
        df.attrs.update(metadata.model_dump(exclude={"attrs"}))

        # set column attributes
        for col in metadata.attrs:
            if col in df.columns:
                df[col].attrs.update(**metadata.attrs[col])

        return df
