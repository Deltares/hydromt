"""Table component."""

import glob
import os
from os.path import basename, dirname, join
from typing import (
    TYPE_CHECKING,
    Dict,
    Optional,
    Union,
    cast,
)

import pandas as pd

from hydromt._typing.type_def import PandasLike
from hydromt.components.base import ModelComponent

if TYPE_CHECKING:
    from hydromt.models.model import Model

__all__ = ["TableComponent"]


class TableComponent(ModelComponent):
    """Table Component."""

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
        self._data: Optional[Dict[str, PandasLike]] = None
        super().__init__(model=model)

    @property
    def data(self) -> Dict[str, PandasLike]:
        """Model tables."""
        if self._data is None:
            self._initialize_tables()
        if self._data is None:
            raise RuntimeError("Could not load data for table component")
        else:
            return self._data

    def _initialize_tables(self, skip_read=False) -> None:
        """Initialize the model tables."""
        if self._data is None:
            self._data = dict()
            if self._root.is_reading_mode() and not skip_read:
                self.read()

    def write(self, fn: str = "tables/{name}.csv", **kwargs) -> None:
        """Write tables at <root>/tables."""
        self._root._assert_write_mode()
        if self.data:
            self._model.logger.info("Writing table files.")
            local_kwargs = {"index": False, "header": True, "sep": ","}
            local_kwargs.update(**kwargs)
            for name in self.data:
                fn_out = join(self._root.path, fn.format(name=name))
                os.makedirs(dirname(fn_out), exist_ok=True)
                self.data[name].to_csv(fn_out, **local_kwargs)
        else:
            self._model.logger.debug("No tables found, skip writing.")

    def read(self, fn: str = "tables/{name}.csv", **kwargs) -> None:
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

    def set(self, tables: Union[PandasLike, Dict], name=None) -> None:
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
        self._data = cast(Dict[str, PandasLike], self._data)
        if not isinstance(tables, dict) and name is None:
            raise ValueError("name required when tables is not a dict")
        elif not isinstance(tables, dict):
            tables = {name: tables}
        for name, df in tables.items():
            if not (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)):
                raise ValueError(
                    "table type not recognized, should be pandas DataFrame or Series."
                )
            if name in self._data:
                if not self._root.is_writing_mode():
                    raise IOError(f"Cannot overwrite table {name} in read-only mode")
                elif self._root.is_reading_mode():
                    self._root.logger.warning(f"Overwriting table: {name}")

            self._data[name] = df

    def get_tables_merged(self) -> pd.DataFrame:
        """Return all tables of a model merged into one dataframe."""
        # This is mostly used for convenience and testing.
        return pd.concat(
            [df.assign(table_origin=name) for name, df in self.data.items()], axis=0
        )
