# -*- coding: utf-8 -*-
"""HydroMT NetworkModel class definition."""

import logging
from typing import List

import xarray as xr

from .model_api import Model

__all__ = ["NetworkModel"]
logger = logging.getLogger(__name__)


class NetworkModel(Model):

    """Implementation for the network models."""

    _NAME = "network_model"

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = None,
        data_libs: List[str] = None,
        logger=logger,
    ):
        """Initialize a NetworkModel for network models."""
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )

        # placeholders
        # TODO decide on data type
        self._network = xr.Dataset()  # xr.Dataset representation of all network data

    def read(
        self,
        components: List = None,
    ) -> None:
        """Read the complete model schematization and configuration from model files.

        Parameters
        ----------
        components : List, optional
            List of model components to read, each should have an associated
            read_<component> method. By default ['config', 'maps',
            'network', 'geoms', 'tables', 'forcing', 'states', 'results']
        """
        components = components or [
            "config",
            "network",
            "geoms",
            "tables",
            "forcing",
            "states",
            "results",
        ]
        super().read(components=components)

    def write(
        self,
        components: List = None,
    ) -> None:
        """Write the complete model schematization and configuration to model files.

        Parameters
        ----------
        components : List, optional
            List of model components to write, each should have an
            associated write_<component> method. By default ['config', 'maps',
            'network', 'geoms', 'tables', 'forcing', 'states']
        """
        components = components or [
            "config",
            "network",
            "geoms",
            "tables",
            "forcing",
            "states",
        ]
        super().write(components=components)

    # TODO: make NetworkMixin class with following properties/methods
    @property
    def network(self):  # noqa: D102
        raise NotImplementedError()

    def set_network(self):  # noqa: D102
        raise NotImplementedError()

    def read_network(self):  # noqa: D102
        raise NotImplementedError()

    def write_network(self):  # noqa: D102
        raise NotImplementedError()
