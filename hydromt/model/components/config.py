"""A component to write configuration files for model simulations/kernels."""

from logging import Logger, getLogger
from os import makedirs
from os.path import abspath, dirname, isabs, isfile, join, splitext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast

from hydromt._io.readers import _read_toml, _read_yaml
from hydromt._io.writers import _write_toml, _write_yaml
from hydromt._utils.path import _make_config_paths_relative
from hydromt.model.components.base import ModelComponent
from hydromt.model.steps import hydromt_step

if TYPE_CHECKING:
    from hydromt.model import Model

logger: Logger = getLogger(__name__)

_YAML_EXTS = [".yml", ".yaml"]
_TOML_EXT = ".toml"


class ConfigComponent(ModelComponent):
    """
    A component to manage configuration files for model simulations/settings.

    ``ConfigComponent`` data is stored as a dictionary and can be written to a file
    in yaml or toml format. The component can be used to store model settings
    and parameters that are used in the model simulations or in the model
    settings.
    """

    def __init__(
        self,
        model: "Model",
        *,
        filename: str = "config.yaml",
        default_template_filename: Optional[str] = None,
    ):
        """Initialize a ConfigComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        filename: str
            A path relative to the root where the configuration file will
            be read and written if user does not provide a path themselves.
            By default 'config.yml'
        default_template_filename: Optional[Path]
            A path to a template file that will be used as default in the ``create``
            method to initialize the configuration file if the user does not provide
            their own template file. This can be used by model plugins to provide a
            default configuration template. By default None.
        """
        self._data: Optional[Dict[str, Any]] = None
        self._filename: str = filename
        self._default_template_filename: Optional[str] = default_template_filename

        super().__init__(model=model)

    @property
    def data(self) -> Dict[str, Any]:
        """Model config values."""
        if self._data is None:
            self._initialize()
        if self._data is None:
            raise RuntimeError("Could not load data for model config component")
        else:
            return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize the model config."""
        if self._data is None:
            self._data = dict()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    @hydromt_step
    def write(
        self,
        path: Optional[str] = None,
    ) -> None:
        """Write model config at <root>/{path}."""
        self.root._assert_write_mode()
        if self.data:
            p = path or self._filename

            write_path = join(self.root.path, p)
            logger.info(f"Writing model config to {write_path}.")
            makedirs(dirname(write_path), exist_ok=True)

            write_data = _make_config_paths_relative(self.data, self.root.path)
            ext = splitext(p)[-1]
            if ext in _YAML_EXTS:
                _write_yaml(write_path, write_data)
            elif ext == _TOML_EXT:
                _write_toml(write_path, write_data)
            else:
                raise ValueError(f"Unknown file extension: {ext}")

        else:
            logger.debug("Model config has no data, skip writing.")

    @hydromt_step
    def read(self, path: Optional[str] = None) -> None:
        """Read model config at <root>/{path}."""
        self._initialize(skip_read=True)

        if (
            self._data is not None
            and len(self._data) == 0
            and self._default_template_filename
        ):
            p = path or self._default_template_filename
            read_path = join(self.root.path, self._default_template_filename)
        else:
            p = path or self._filename
            # if path is abs, join will just return path
            read_path = join(self.root.path, p)

        if isfile(read_path):
            logger.info(f"Reading model config file from {read_path}.")
        else:
            logger.warning(
                f"No default model config was found at {read_path}. "
                "It wil be initialized as empty dictionary"
            )
            return

        ext = splitext(p)[-1]
        # Always overwrite config when reading
        if ext in _YAML_EXTS:
            self._data = _read_yaml(read_path)
        elif ext == _TOML_EXT:
            self._data = _read_toml(read_path)
        else:
            raise ValueError(f"Unknown file extension: {ext}")

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
        ::

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
        current = cast(Dict[str, Any], self._data)
        for i, part in enumerate(parts):
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            if i < num_parts - 1:
                current = current[part]
            else:
                current[part] = value

    def get_value(self, key: str, fallback=None, abs_path: bool = False) -> Any:
        """Get a config value at key(s).

        Parameters
        ----------
        args : tuple or string
            key can given as a string with '.' indicating a new level: ('key1.key2')
        fallback: any, optional
            Fallback value if key not found in config, by default None.
        abs_path: bool, optional
            If True return the absolute path relative to the model root,
            by default False.
            NOTE: this assumes the config is located in model root!

        Returns
        -------
        value : any type
            dictionary value

        Examples
        --------
        >> self.data = {'a': 1, 'b': {'c': {'d': 2}}}

        >> get_value('a')
        >> 1

        >> get_value('b.c.d')
        >> 2

        >> get_value('b.c')
        >> {'d': 2}

        """
        parts = key.split(".")
        num_parts = len(parts)
        current = self.data  # reads config at first call
        value = fallback
        for i, part in enumerate(parts):
            if i < num_parts - 1:
                current = current.get(part, {})
            else:
                value = current.get(part, fallback)

        if abs_path and isinstance(value, (str, Path)):
            value = Path(value)
            if not isabs(value):
                value = Path(abspath(join(self.root.path, value)))

        return value

    @hydromt_step
    def create(
        self,
        template: Optional[Union[str, Path]] = None,
    ):
        """Create a new config file based on a template file.

        It the template is not provided, the default template will be used if available.
        Only yaml and toml files are supported.

        Parameters
        ----------
        template : str or Path, optional
            Path to a template config file, by default None

        Examples
        --------
        ::

            >> self.create()
            >> self.data
                {}

            >> self.create(template='path/to/template.yml')
            >> self.data
                {'a': 1, 'b': {'c': {'d': 2}}}
        """
        # Check if self.data is not empty
        if len(self.data) > 0:
            raise ValueError(
                "Model config already exists, cannot create new config."
                "Use ``update`` method to update the existing config."
            )

        if template is not None:
            if isinstance(template, str):
                template = Path(template)
            prefix = "user-defined"
        elif self._default_template_filename is not None:
            template = self._default_template_filename
            prefix = "default"
        else:
            raise FileNotFoundError("No template file was provided.")

        if not isfile(template):
            raise FileNotFoundError(f"Template file not found: {template}")

        template = Path(template)
        # Here directly overwrite config with template
        logger.info(f"Creating model config from {prefix} template: {template}")
        if template.suffix in _YAML_EXTS:
            self._data = _read_yaml(template)
        elif template.suffix == _TOML_EXT:
            self._data = _read_toml(template)
        else:
            raise ValueError(f"Unknown file extension: {template.suffix}")

    @hydromt_step
    def update(self, data: Dict[str, Any]):
        """Set the config dictionary at key(s) with values.

        Parameters
        ----------
        data: Dict[str,Any]
            A dictionary with the values to be set. keys can be dotted like in
            :py:meth:`~hydromt.model.components.config.ConfigComponent.set_value`

        Examples
        --------
        Setting data as a nested dictionary::


            >> self.update({'a': 1, 'b': {'c': {'d': 2}}})
            >> self.data
            {'a': 1, 'b': {'c': {'d': 2}}}

        Setting data using dotted notation::

            >> self.update({'a.d.f.g': 1, 'b': {'c': {'d': 2}}})
            >> self.data
            {'a': {'d':{'f':{'g': 1}}}, 'b': {'c': {'d': 2}}}

        """
        if len(data) > 0:
            logger.debug("Setting model config options.")
        for k, v in data.items():
            self.set(k, v)

    def test_equal(self, other: ModelComponent) -> Tuple[bool, Dict[str, str]]:
        """Test if two components are equal.

        Parameters
        ----------
        other : ModelComponent
            The component to compare against.

        Returns
        -------
        tuple[bool, Dict[str, str]]
            True if the components are equal, and a dict with the associated errors per property checked.
        """
        eq, errors = super().test_equal(other)
        if not eq:
            return eq, errors
        other_config = cast(ConfigComponent, other)

        # for once python does the recursion for us
        if self.data == other_config.data:
            return True, {}
        else:
            return False, {"config": "Configs are not equal"}
