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
    


#%% original fucntions
    def setup_lulcmaps(
        self,
        lulc_fn="globcover",
        lulc_mapping_fn=None,
        lulc_vars=[
            "landuse",
            "Kext",
            "N",
            "PathFrac",
            "RootingDepth",
            "Sl",
            "Swood",
            "WaterFrac",
        ],
    ):
        """
        This component derives several wflow maps are derived based on landuse-
        landcover (LULC) data. 
        
        Currently, ``lulc_fn`` can be set to the "vito", "globcover"
        or "corine", fo which lookup tables are constructed to convert lulc classses to
        model parameters based on literature. The data is remapped at its original
        resolution and then resampled to the model resolution using the average
        value, unless noted differently.
        
        Adds model layers:
        * **landuse** map: Landuse class [-]     
        * **Kext** map: Extinction coefficient in the canopy gap fraction equation [-]
        * **Sl** map: Specific leaf storage [mm]
        * **Swood** map: Fraction of wood in the vegetation/plant [-]
        * **RootingDepth** map: Length of vegetation roots [mm]
        * **PathFrac** map: The fraction of compacted or urban area per grid cell [-]
        * **WaterFrac** map: The fraction of open water per grid cell [-]
        * **N** map: Manning Roughness [-]
        Parameters
        ----------
        lulc_fn : {"globcover", "vito", "corine"}
            Name of data source in data_sources.yml file.
        lulc_mapping_fn : str
            Path to a mapping csv file from landuse in source name to parameter values in lulc_vars.
        lulc_vars : list
            List of landuse parameters to keep.\
            By default ["landuse","Kext","N","PathFrac","RootingDepth","Sl","Swood","WaterFrac"]
        """
        self.logger.info(f"Preparing LULC parameter maps.")
        if lulc_mapping_fn is None:
            fn_map = join(DATADIR, "lulc", f"{lulc_fn}_mapping.csv")
        else:
            fn_map = lulc_mapping_fn
        if not isfile(fn_map):
            self.logger.error(f"LULC mapping file not found: {fn_map}")
            return
        # read landuse map to DataArray
        da = self.data_catalog.get_rasterdataset(
            lulc_fn, geom=self.region, buffer=2, variables=["landuse"]
        )
        # process landuse
        ds_lulc_maps = workflows.landuse(
            da=da,
            ds_like=self.staticmaps,
            fn_map=fn_map,
            params=lulc_vars,
            logger=self.logger,
        )
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_lulc_maps.data_vars}
        self.set_staticmaps(ds_lulc_maps.rename(rmdict))