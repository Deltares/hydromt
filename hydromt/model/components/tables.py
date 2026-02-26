"""Tables component."""

import logging
from typing import TYPE_CHECKING, cast

import pandas as pd

from hydromt.model.components.base import ModelComponent
from hydromt.model.steps import hydromt_step
from hydromt.readers import _expand_wildcards_and_name_placeholder

if TYPE_CHECKING:
    from hydromt.model.model import Model

__all__ = ["TablesComponent"]

logger = logging.getLogger(__name__)


class TablesComponent(ModelComponent):
    """TablesComponent contains data as a dictionary of pandas.DataFrame.

    It is well suited to represent non-geospatial tabular model data.
    """

    def __init__(
        self,
        model: "Model",
        filename: str = "tables/{name}.csv",
    ):
        """Initialize a TablesComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        filename: str
            The default place that should be used for reading and writing unless the
            user overrides it. If a relative path is given it will be used as being
            relative to the model root. By default ``tables/{name}.csv`` for this
            component, and can be either relative or absolute.
        """
        self._data: dict[str, pd.DataFrame | pd.Series] = None
        self._filename: str = filename
        super().__init__(model=model)

    @property
    def data(self) -> dict[str, pd.DataFrame | pd.Series]:
        """Model tables."""
        if self._data is None:
            self._initialize_tables()

        assert self._data is not None
        return self._data

    def _initialize_tables(self, skip_read=False) -> None:
        """Initialize the model tables."""
        if self._data is None:
            self._data = dict()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    @hydromt_step
    def write(self, filename: str | None = None, **kwargs) -> None:
        """Write tables at provided or default file path if none is provided."""
        self.root._assert_write_mode()

        if len(self.data) == 0:
            logger.info(
                f"{self.model.name}.{self.name_in_model}: No tables found, skip writing."
            )
            return

        kwargs.setdefault("index", False)
        kwargs.setdefault("header", True)
        kwargs.setdefault("sep", ",")

        for name, value in self.data.items():
            _filename = (filename or self._filename).format(name=name)
            write_path = self.root.path / _filename
            write_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"{self.model.name}.{self.name_in_model}: Writing table to {write_path}."
            )
            value.to_csv(write_path, **kwargs)

    @hydromt_step
    def read(self, filename: str | None = None, **kwargs) -> None:
        """Read tables at provided or default file path if none is provided.

        Parameters
        ----------
        filename : str | None, optional
            Filename relative to model root. Should contain either a * wildcard character
            or a {name} placeholder to read multiple files into the data dictionary.
            All files matching the glob pattern defined by filename will be read.
            If a {name} placeholder is used, that name will be used as the key in the
            data dictionary.
            If no {name} placeholder is used, the filename without extension will be used
            as the key in the data dictionary.
            If None, the path that was provided at init will be used.
        **kwargs:
            Additional keyword arguments that are passed to the
            `pandas.read_csv` function.
        """
        self.root._assert_read_mode()
        self._initialize_tables(skip_read=True)
        logger.info("Reading model table files.")
        placeholder_filename = (filename or self._filename).replace("*", "{name}")
        for path, name in _expand_wildcards_and_name_placeholder(
            placeholder_filename, self.root.path
        ).items():
            logger.info(f"Reading table from {path} into model as '{name}'.")
            tbl = pd.read_csv(path, **kwargs)
            self.set(tbl, name=name)

    def set(
        self,
        tables: pd.DataFrame | pd.Series | dict[str, pd.DataFrame | pd.Series],
        name: str | None = None,
    ) -> None:
        """Add (a) table(s) <pandas.DataFrame> to model.

        Parameters
        ----------
        tables : pandas.DataFrame, pandas.Series or dict[str, pd.DataFrame | pd.Series]
            Table(s) to add to model.
            Multiple tables can be added at once by passing a dict of tables.
        name : str | None, optional
            Name of table, by default None. Required when tables is not a dict.
        """
        self._initialize_tables()
        assert self._data is not None
        # added here so we only have to declare types once
        tables_to_add: dict[str, pd.DataFrame | pd.Series]

        if not isinstance(tables, dict):
            if name is None:
                raise ValueError("name required when tables is not a dict")
            else:
                tables_to_add = {name: tables}
        else:
            tables_to_add = tables

        for df_name, df in tables_to_add.items():
            if not (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)):
                raise ValueError(
                    "table type not recognized, should be pandas DataFrame or Series."
                )
            if df_name in self._data:
                if not self.root.is_writing_mode():
                    raise IOError(f"Cannot overwrite table {df_name} in read-only mode")
                elif self.root.is_reading_mode():
                    logger.warning(f"Overwriting table: {df_name}")

            self._data[str(df_name)] = df

    def get_tables_merged(self) -> pd.DataFrame:
        """Return all tables of a model merged into one dataframe."""
        # This is mostly used for convenience and testing.
        return pd.concat(
            [df.assign(table_origin=name) for name, df in self.data.items()], axis=0
        )

    def test_equal(self, other: ModelComponent) -> tuple[bool, dict[str, str]]:
        """Test if two components are equal.

        Parameters
        ----------
        other : ModelComponent
            The component to compare against.

        Returns
        -------
        tuple[bool, dict[str, str]]
            True if the components are equal, and a dict with the associated errors per property checked.
        """
        eq, errors = super().test_equal(other)
        if not eq:
            return eq, errors
        other_tables = cast(TablesComponent, other)
        for name, df in self.data.items():
            if name not in other_tables.data:
                errors[name] = "Table not found in other component."
            elif not df.equals(other_tables.data[name]):
                errors[name] = "Table content is not equal."

        return len(errors) == 0, errors
