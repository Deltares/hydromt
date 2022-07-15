import pytest
import sys, os
from .model_api import Model
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from hydromt import workflows, flw, io

from typing import List

import logging
import os

__all__ = ["GridModel"]
logger = logging.getLogger(__name__)


class GridModel(Model):
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

    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        super().read()

    def write(self):
        """Method to write the complete model schematization and configuration to file."""
        super().write()
    
    def setup_
