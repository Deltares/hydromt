import pytest
import sys, os
from .model_api import Model
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import box

from typing import Tuple, Union, Optional

import logging
import os

__all__ = ["LumpedModel"]
logger = logging.getLogger(__name__)


class LumpedModel(Model):
    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = None,
        data_libs: List[str] = None,
        logger=logger,
    ):
        # Initialize with the Model class
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )

        # placeholders
        self._response_units = (
            gpd.GeoDataFrame()
        )  # representation of all response units OR maybe should be a dictionary with property and units key?

    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        super().read()
        self.read_response_units()
        # Other specifics to LumpedModel...

    def write(self):
        """Method to write the complete model schematization and configuration to file."""
        super().write()
        self.write_response_units()
        # Other specifics to LumpedModel...

    @property
    def response_units(self):  # TODO: name to be agreed by all
        """xr.Dataset object (GeoDataSet) with a string object or tuple of the Geometry"""
        if not self._response_units:
            if self._read:
                self.read_response_units()
        return self._response_units


# Property - basin ID
# Response_unit: one geodataframe
# Property stored in another geodataframe
