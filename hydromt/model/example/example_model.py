"""Implement Wflow base model class."""

# Implement model class following model API
import logging

from hydromt.model import Model
from hydromt.model.components import ConfigComponent
from hydromt.model.example.example_grid_component import (
    ExampleGridComponent,
)

__all__ = ["ExampleModel"]
logger = logging.getLogger(f"hydromt.{__name__}")


class ExampleModel(Model):
    """Example Model Class for demo purposes.

    This model contains a config and a regular grid component.

    Parameters
    ----------
    root : str, optional
        Model root, by default None (current working directory)
    config_filename : str, optional
        A path relative to the root where the configuration file will
        be read and written if user does not provide a path themselves.
        By default "settings.toml"
    mode : {'r','r+','w'}, optional
        read/append/write mode, by default "w"
    data_libs : list[str] | str, optional
        List of data catalog configuration files, by default None
    **catalog_keys:
        Additional keyword arguments to be passed down to the DataCatalog.
    """

    name: str = "example_model"

    def __init__(
        self,
        root: str | None = None,
        config_filename: str | None = "settings.toml",
        mode: str = "w",
        data_libs: list[str] | str | None = None,
        **catalog_keys,
    ):
        """Initialize ExampleModel."""
        components = {
            "config": ConfigComponent(
                self,
                filename=str(config_filename),
            ),
            "grid": ExampleGridComponent(self),
        }

        super().__init__(
            root,
            components=components,
            mode=mode,
            region_component="grid",
            data_libs=data_libs,
            **catalog_keys,
        )

    ## Properties
    # Components
    @property
    def config(self) -> ConfigComponent:
        """Return the config component."""
        return self.components["config"]

    @property
    def grid(self) -> ExampleGridComponent:
        """Return the grid component."""
        return self.components["grid"]
