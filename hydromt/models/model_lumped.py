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
        # Other specifics to LumpedModel...

    def write(self):
        """Method to write the complete model schematization and configuration to file."""
        super().write()
    
    def set_forcing_name():

    def space_interpolation():
        #Check if more than one points are provided
        #From 1D points to 2D -> using Thiessen polygons, others > check gdal
        #Options: area average over selected region
        #OPtions: picking new locations points (from list of points but also centroid, )
        #Options: leave it as 2D?


    def time_interpolation(): #For 1D value #multiple methods 
        #--> set fillna as option
        #Linear interpolation
        #bfill, ffill
        
    def resample_freq(): # based on time steps of forcing, resample 1D time series
    
    def zonal_stats(): #2D from a scalar, we should also define variable and store this in a DataArray
        #As arguments: 2D data, polygon shape, operation (mean, mode, max, min) --> return this
        #return mode value of polygon within a raster --> the most frequent value
        #retun mean 
    
    @property
    def area(self):
        #Check crs is in unit m
    
    def set_areas():
        #Store the other areas as staticgeoms with keyname






