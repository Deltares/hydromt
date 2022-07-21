import pytest
import sys, os
from os.path import join, dirname, basename, isfile, isdir
from .model_api import Model
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import box
#from hydromt import workflows, flw, io
from .. import config, log, workflows

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
    
    def setup_fromtable(self, path_or_key, fn_map, out_vars, **kwargs):
        """This function creates additional staticmaps layers based on a table reclassification
        
        Adds model layers defined in out_vars

        Args:
            path_or_key (str): Name of RasterDataset in DataCatalog file (.yml).
            mapping_fn (str): Path to a mapping csv file from RasterDataset to parameter values in out_vars.
            out_vars (List): List of parameters to keep.
            **kwargs: if the RasterDataset has multiple variables, select the correct variable
        """
        self.logger.info(f"Preparing {out_vars} parameter maps from raster.")

        if not isfile(fn_map): #TODO - makefn_map flexible with the DataCatalog as well
            self.logger.error(f"Mapping file not found: {fn_map}") #TODO ask diff between logger.error and RaiseValueError (will log.error stop the code?)
            return
        
        # read RasterDataset map to DataArray
        da = self.data_catalog.get_rasterdataset(
            path_or_key, geom=self.region, buffer=2, **kwargs
        ) #TODO - ask about why buffer is 2  #variables=["landuse"]

        if not isinstance(da, xr.DataArray):
            raise ValueError("RasterData has multiple variables, please specify variable to select")
        
        # process landuse
        ds_maps = workflows.grid_maptable(
            da=da,
            ds_like=self.staticmaps,
            fn_map=fn_map,
            params=out_vars,
            logger=self.logger,
        )
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_maps.data_vars}
        self.set_staticmaps(ds_maps.rename(rmdict))

    
    def setup_fromvector(self, key, col2raster="", rasterize_method="value"):
        """Creates additional staticmaps based on a vector, located either in the data library or staticgeoms.

        Adds staticmaps model layers defined in key

        Args:
            key (str): value in either staticgeoms or the data catalog to extract the vector
            col2raster (str, optional): name of column in the vector to use for rasterization. Defaults to "".
            rasterize_method (str, optional): Method to rasterize the vector ("value" or "fraction"). Defaults to "value".
        """

        
        self.logger.info(f"Preparing {key} parameter maps from vector.")

        #Vector sources can be from staticgeoms, data_catalog or fn
        if key in self._staticgeoms.keys():
            gdf = self._staticgeoms[key]
        elif key in self.data_catalog:
            gdf = self.data_catalog.get_geodataframe(key, geom=self.region, dst_crs=self.crs) #TODO: I think this is updated if gets a fn: ask
        else: 
            self.logger.warning(f"Source '{key}' not found in staticgeoms nor data_catalog.")
            return
        
        if gdf.empty:
            self.logger.warning(f"No shapes of {key} found within region, setting to default value.") #TODO: check this

            ds = self.hydromaps["basins"].copy() * 0.0 #TODO : fix this  
            ds.attrs.update(_FillValue=0.0)
        else:
            ds = workflows.vector_to_grid(
                gdf=gdf,
                ds_like=self.staticmaps,
                col_name=col2raster,
                method=rasterize_method,
                mask_name="mask",
                logger=self.logger,
            )
        self.set_staticmaps(ds.rename(key))







   def setup_emission_vector(
        self,
        emission_fn,
        col2raster="",
        rasterize_method="value",
    ):
        """Setup emission map from vector data.
        Adds model layer:
        * **emission_fn** map: emission data map
        Parameters
        ----------
        emission_fn : {'GDP_world'...}
            Name of raster emission map source.
        col2raster : str
            Name of the column from the vector file to rasterize.
            Can be left empty if the selected method is set to "fraction".
        rasterize_method : str
            Method to rasterize the vector data. Either {"value", "fraction"}.
            If "value", the value from the col2raster is used directly in the raster.
            If "fraction", the fraction of the grid cell covered by the vector file is returned.
        """
        if emission_fn is None:
            self.logger.warning(
                "Source name set to None, skipping setup_emission_vector."
            )
            return
        if emission_fn not in self.data_catalog:
            self.logger.warning(
                f"Invalid source '{emission_fn}', skipping setup_emission_vector."
            )
            return

        self.logger.info(f"Preparing '{emission_fn}' map.")
        gdf_org = self.data_catalog.get_geodataframe(
            emission_fn, geom=self.basins, dst_crs=self.crs
        )
        if gdf_org.empty:
            self.logger.warning(
                f"No shapes of {emission_fn} found within region, setting to default value."
            )
            ds_emi = self.hydromaps["basins"].copy() * 0.0
            ds_emi.attrs.update(_FillValue=0.0)
        else:
            ds_emi = emissions.emission_vector(
                gdf=gdf_org,
                ds_like=self.staticmaps,
                col_name=col2raster,
                method=rasterize_method,
                mask_name="mask",
                logger=self.logger,
            )
        self.set_staticmaps(ds_emi.rename(emission_fn))