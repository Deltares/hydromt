"""A component to write configuration files for simulations/kernels."""

from os import makedirs
from os.path import abspath, dirname, isabs, isfile, join, splitext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from hydromt.components.base import ModelComponent
from hydromt.hydromt_step import hydromt_step
from hydromt.io.path import make_config_paths_relative
from hydromt.io.readers import read_yaml
from hydromt.io.writers import write_toml, write_yaml

if TYPE_CHECKING:
    from hydromt.models import Model

DEFAULT_CONFIG_PATH = "config.yml"


class ConfigComponent(ModelComponent):
    """A component to write configuration files for simulations/kernels."""

    def __init__(self, model: "Model", *, config_fn: Optional[str] = None):
        """Initialize a ConfigComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        """
        self._data: Optional[Dict[str, Any]] = None
        self._config_fn: str = config_fn or DEFAULT_CONFIG_PATH

        super().__init__(model=model)

    @property
    def data(self) -> Dict[str, Any]:
        """Kernel config values."""
        if self._data is None:
            self._initialize()
        if self._data is None:
            raise RuntimeError("Could not load data for kernel config component")
        else:
            return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize the kernel configs."""
        if self._data is None:
            self._data = dict()
            if not skip_read:
                self.read()

    @hydromt_step
    def write(
        self,
        path: Optional[str] = None,
    ) -> None:
        """Write kernel config at <root>/{path}."""
        self._root._assert_write_mode()
        if self.data:
            if path is not None:
                p = join(self._root.path, path)
            else:
                p = join(self._root.path, self._config_fn)

            write_path = join(self._root.path, p)
            self._model.logger.info(f"Writing kernel config to {write_path}.")
            makedirs(dirname(write_path), exist_ok=True)

            write_data = make_config_paths_relative(self.data, self._root.path)
            ext = splitext(p)[-1]
            if ext in [".yml", ".yaml"]:
                write_yaml(write_path, write_data)
            elif ext == ".toml":
                write_toml(write_path, write_data)
            else:
                raise ValueError(f"Unknown file extention: {ext}")

        else:
            self._model.logger.debug("No kernel config found, skip writing.")

    @hydromt_step
    def read(self, path: Optional[str] = None) -> None:
        """Read kernel config at <root>/{path}."""
        self._initialize(skip_read=True)
        # if path is abs, join will just return path
        if path is not None:
            p = path
        else:
            p = self._config_fn
        read_path = join(self._root.path, p)
        if isfile(read_path):
            self._model.logger.info(f"Reading kernel config file from {read_path}.")
        else:
            self._model.logger.warning(
                f"No default kernel config was found at {read_path}. It wil be initialized as empty dictionary"
            )
            return

        ext = splitext(p)[-1]
        if ext in [".yml", ".yaml"]:
            self._data = read_yaml(read_path)
        else:
            raise ValueError(f"Unknown file extention: {ext}")

    @hydromt_step
    def set(self, data: Dict[str, Any]):
        """Set the config dictionary at key(s) with values.

        Parameters
        ----------
        data: Dict[str,Any]
            A dictionary with the values to be set. keys can be dotted like in
            :py:meth:`~hydromt.components.config.ConfigComponent.set_value`

        Examples
        --------
        >> self.set({'a': 1, 'b': {'c': {'d': 2}}})
        >> self.data
            {'a': 1, 'b': {'c': {'d': 2}}}
        >> self.set({'a.d.f.g': 1, 'b': {'c': {'d': 2}}})
        >> self.data
            {'a': {'d':{'f':{'g': 1}}}, 'b': {'c': {'d': 2}}}
        """
        for k, v in data.items():
            self.update(k, v)

    @hydromt_step
    def update(self, key: str, value: Any):
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
        >> self.set_value('a', 99)
        >> {'a': 99, 'b': {'c': {'d': 2}}}

        >> self.set_value('b.d.e', 24)
        >> {'a': 99, 'b': {'c': {'d': 24}}}
        """
        self._initialize()
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

    def get_value(self, key: str, abs_path: bool = False) -> Any:
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
        self._initialize()
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
