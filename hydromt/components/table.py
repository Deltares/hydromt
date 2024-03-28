"""Table component."""

import glob
import os
from os.path import basename, dirname, join
from typing import (
    TYPE_CHECKING,
    Dict,
    Optional,
    Union,
)

import pandas as pd

from hydromt.components.base import ModelComponent
from hydromt.hydromt_step import hydromt_step
from hydromt.utils.constants import DEFAULT_TABLE_FILENAME

if TYPE_CHECKING:
    from hydromt.models.model import Model

__all__ = ["TableComponent"]


class TableComponent(ModelComponent):
    """TablesComponent contains data as a dictionnary of pandas.DataFrame.

    It is well suited to represent non-geospatial tabular model data.
    """

    def __init__(
        self,
        model: "Model",
    ):
        """Initialize a TableComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        """
        self._data: Optional[Dict[str, Union[pd.DataFrame, pd.Series]]] = None
        super().__init__(model=model)

    @property
    def data(self) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """Model tables."""
        if self._data is None:
            self._initialize_tables()

        assert self._data is not None
        return self._data

    def _initialize_tables(self, skip_read=False) -> None:
        """Initialize the model tables."""
        if self._data is None:
            self._data = dict()
            if self._root.is_reading_mode() and not skip_read:
                self.read()

    @hydromt_step
    def write(self, fn: str = DEFAULT_TABLE_FILENAME, **kwargs) -> None:
        """Write tables at <root>/tables."""
        self._root._assert_write_mode()
        if len(self.data) > 0:
            self._model.logger.info("Writing table files.")
            local_kwargs = {"index": False, "header": True, "sep": ","}
            local_kwargs.update(**kwargs)
            for name in self.data:
                fn_out = join(self._root.path, fn.format(name=name))
                os.makedirs(dirname(fn_out), exist_ok=True)
                self.data[name].to_csv(fn_out, **local_kwargs)
        else:
            self._model.logger.debug("No tables found, skip writing.")

    @hydromt_step
    def read(self, fn: str = DEFAULT_TABLE_FILENAME, **kwargs) -> None:
        """Read table files at <root>/tables and parse to dict of dataframes."""
        self._root._assert_read_mode()
        self._initialize_tables(skip_read=True)
        self._model.logger.info("Reading model table files.")
        fns = glob.glob(join(self._root.path, fn.format(name="*")))
        if len(fns) > 0:
            for fn in fns:
                name = basename(fn).split(".")[0]
                tbl = pd.read_csv(fn, **kwargs)
                self.set(tbl, name=name)

    def set(
        self,
        tables: Union[
            Union[pd.DataFrame, pd.Series], Dict[str, Union[pd.DataFrame, pd.Series]]
        ],
        name: Optional[str] = None,
    ) -> None:
        """Add (a) table(s) <pandas.DataFrame> to model.

        Parameters
        ----------
        tables : pandas.DataFrame, pandas.Series or dict
            Table(s) to add to model.
            Multiple tables can be added at once by passing a dict of tables.
        name : str, optional
            Name of table, by default None. Required when tables is not a dict.
        """
        self._initialize_tables()
        assert self._data is not None
        if not isinstance(tables, dict):
            if name is None:
                raise ValueError("name required when tables is not a dict")
            else:
                tables_to_add: Dict[str, Union[pd.DataFrame, pd.Series]] = {
                    name: tables
                }
        else:
            tables_to_add: Dict[str, Union[pd.DataFrame, pd.Series]] = tables

        for df_name, df in tables_to_add.items():
            if not (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)):
                raise ValueError(
                    "table type not recognized, should be pandas DataFrame or Series."
                )
            if df_name in self._data:
                if not self._root.is_writing_mode():
                    raise IOError(f"Cannot overwrite table {df_name} in read-only mode")
                elif self._root.is_reading_mode():
                    self._logger.warning(f"Overwriting table: {df_name}")

            self._data[str(df_name)] = df

    def get_tables_merged(self) -> pd.DataFrame:
        """Return all tables of a model merged into one dataframe."""
        # This is mostly used for convenience and testing.
        return pd.concat(
            [df.assign(table_origin=name) for name, df in self.data.items()], axis=0
        )
