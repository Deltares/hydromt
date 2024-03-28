"""A component to write configuration files for simulations/kernels."""

from os import makedirs
from os.path import abspath, dirname, isabs, join, splitext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from hydromt.components.base import ModelComponent
from hydromt.io.path import make_config_paths_relative
from hydromt.io.readers import read_toml, read_yaml
from hydromt.io.writers import write_toml, write_yaml

if TYPE_CHECKING:
    from hydromt.models import Model


class KernelConfigComponent(ModelComponent):
    """A component to write configuration files for simulations/kernels."""

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
        self._data: Optional[Dict[str, Any]] = None
        super().__init__(model=model)

    @property
    def data(self) -> Dict[str, Any]:
        """Model tables."""
        if self._data is None:
            self._initialize_kernel_config()
        if self._data is None:
            raise RuntimeError("Could not load data for table component")
        else:
            return self._data

    def _initialize_kernel_config(self, skip_read=False) -> None:
        """Initialize the model tables."""
        if self._data is None:
            self._data = dict()
            if self._root.is_reading_mode() and not skip_read:
                self.read()

    def write(
        self,
        path: str = "kernel_config.yaml",
    ) -> None:
        """Write kernel config at <root>/{path}."""
        self._root._assert_write_mode()
        if self.data:
            write_path = join(self._root.path, path)
            self._model.logger.info(f"Writing kernel config to {write_path}.")
            makedirs(dirname(write_path), exist_ok=True)

            write_data = make_config_paths_relative(self.data, self._root.path)
            ext = splitext(path)[-1]
            if ext in [".yml", ".yaml"]:
                write_yaml(write_path, write_data)
            elif ext == ".toml":
                write_toml(write_path, write_data)
            else:
                raise ValueError(f"Unknown file extention: {ext}")

        else:
            self._model.logger.debug("No kernel config found, skip writing.")

    def read(self, path: str = "kernel_config.yml") -> None:
        """Read table files at <root>/tables and parse to dict of dataframes."""
        self._root._assert_read_mode()
        self._initialize_kernel_config(skip_read=True)

        if isabs(path):
            read_path = path
        else:
            read_path = join(self._root.path, path)

        self._model.logger.info(f"Reading kernel config file from {read_path}.")

        ext = splitext(path)[-1]
        if ext in [".yml", ".yaml"]:
            self._data = read_yaml(read_path)
        elif ext == ".toml":
            self._data = read_toml(read_path)
        else:
            raise ValueError(f"Unknown file extention: {ext}")

    def set(self, key: str, value: Any):
        """Update the config dictionary at key(s) with values.

        Parameters
        ----------
        key : str
            a string with '.' indicating a new level: 'key1.key2' will translate
            to {"key1":{"key2": value}}
        value: Any
            the value to set the config to

        Examples
        --------
        >> self.set({'a': 1, 'b': {'c': {'d': 2}}})
        >> self.data
            {'a': 1, 'b': {'c': {'d': 2}}}
        >> self.set('a', 99)
        >> {'a': 99, 'b': {'c': {'d': 2}}}

        >> self.set('b.d.e', 24)
        >> {'a': 99, 'b': {'c': {'d': 24}}}
        """
        self._initialize_kernel_config()
        parts = key.split(".")
        num_parts = len(parts)
        current = self.data
        for i, part in enumerate(parts):
            if part not in current:
                current[part] = {}
            if i < num_parts - 1:
                current = current[part]
            else:
                current[part] = value

    def get_config_value(self, key: str, abs_path: bool = False):
        """Get a config value at key(s).

        Parameters
        ----------
        args : tuple or string
            keys can given by multiple args: ('key1', 'key2')
            or a string with '.' indicating a new level: ('key1.key2')
        abs_path: bool, optional
            If True return the absolute path relative to the model root,
            by deafult False.
            NOTE: this assumes the config is located in model root!

        Returns
        -------
        value : any type
            dictionary value

        Examples
        --------
        >> self.set({'a': 1, 'b': {'c': {'d': 2}}})

        >> get_config_value('a')
        >> 1

        >> get_config_value('b.c.d')
        >> 2

        >> get_config_value('b.c')
        >> {'d': 2}

        """
        self._initialize_kernel_config()
        parts = key.split(".")
        num_parts = len(parts)
        current = self.data
        value = None
        for i, part in enumerate(parts):
            if i < num_parts - 1:
                current = current[part]
            else:
                value = current[part]
                break

        if abs_path and isinstance(value, (str, Path)):
            value = Path(value)
            if not isabs(value):
                value = Path(abspath(join(self._root.path, value)))

        return value
