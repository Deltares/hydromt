import pytest
import sys, os
from .model_api import Model
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import box

from typing import List

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
    def response_units(self):  #TODO: name to be agreed by all
        """xr.Dataset object (GeoDataSet) with a string object or tuple of the Geometry """
        if not self._response_units:
            if self._read:
                self.read_response_units()
        return self._response_units   

#Property - basin ID

# object auxiliary? geoms and maps. to store 

#Response_unit: one geodataframe
#Property stored in another geodataframe
    


# TODO: possible additional objects or properties
# Having a time series of a Polygon - for now a single xarray.Dataset object. Could save a string object or tuple of the Geometry. BUT slow!
# Could also link the ID of the gdf with staticgeoms. Could also be a geodataframe of two dimensions #--> property: xarray dataset that should match with a staticgeoms and check that index are mathcing 

# In the future, make an issue to support polygon in the vector method. 



