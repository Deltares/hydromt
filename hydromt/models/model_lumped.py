import pytest
import sys, os

from pytz import NonExistentTimeError
from .model_api import Model
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import box

from typing import List

import logging
import os

from ..data_adapter import DataCatalog
from .. import config, log, workflows, flw

__all__ = ["LumpedModel"]
logger = logging.getLogger(__name__)


def _make_ds_mask(ds, gdf, col_name='value'):
    """make mask for xarray.Dataset or rasterDataset, True for each point within the geometry.

    Parameters
    ----------
    ds : xr.Dataset
        Grid dataset to make mask for
    gdf : gpd.GeoDataFrame
        Geopandas Dataframe with geometries (polygons, multipolygons) to make a mask for
    col_name : str, optional
        Name of the column in gdf that is used in making the mask, by default 'value'

    Returns
    -------
    xarray.DataArray
        Boolean dataarray, with True for any pixel inside the geometries in gdf
    """    
    ds_basin_raster = ds.raster.rasterize(gdf,col_name=col_name)
    #basin_mask = ds_basin_raster.where(ds_basin_raster)
    ds_basin_mask = ds.where(ds_basin_raster).mask
    return ds_basin_mask
    

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

    def setup_response_unit_geom(
        self,
        region,
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
        split_regions = False,
        split_method = "us_area",
        **split_kwargs
        ):
        """
        This component sets up the region and response units.
        
        See Also
        --------
        hydromt.models.model_api.setup_region
        """
        region = self.setup_region(
                region,
                hydrography_fn=hydrography_fn,
                basin_index_fn=basin_index_fn,
            )
        if not split_regions:
            self.set_geoms(self.geoms['region'], name="response_unit")
        if split_regions:
            subbasins_gpd = self.setup_subbasins(
                split_method=split_method,
                hydrography_fn=hydrography_fn,
                **split_kwargs,
            )
            self.set_geoms(subbasins_gpd, name="response_unit")        
        return region
    
    def setup_subbasins(
        self,
        split_method: str,
        hydrography_fn="merit_hydro",
        **split_kwargs
        ):
        """Partition region into subbasins based on flow direction related methods.
        
        The methods are described here: https://deltares.github.io/pyflwdir/latest/basins.html

        Parameters
        ----------
        split_method : str
            one of  ["streamorder","us_area","pfafstetter","outlets"]
        hydrography_fn : str, optional
            name of hydrography dataset, by default "merit_hydro"

        Returns
        -------
        gpd.GeoDataFrame
            Geopandas Dataframe with geometries including outlet point coordinates.

        Raises
        ------
        ValueError
            Error if given split_method is not valid method (see above)
            
        See also
        --------
        https://deltares.github.io/pyflwdir/latest/basins.html
        """        
        ds = self.data_catalog.get_rasterdataset(hydrography_fn,geom=self.region)
        flwdir = flw.flwdir_from_da(ds["flwdir"], ftype="d8", mask=_make_ds_mask(ds, self.region, col_name='value'))
        
        split_method_lst = ["streamorder","us_area","pfafstetter","outlets"]
        if split_method not in split_method_lst:
            msg = f"Unknown split_method: {split_method}, select from {split_method_lst}."
            raise ValueError(msg)
        if split_method=="streamorder":
            subbas, idxs_out = flwdir.subbasins_streamorder(**split_kwargs)
        elif split_method=="us_area":
            subbas, idxs_out = flwdir.subbasins_area(split_kwargs['min_area'])
        elif split_method=="pfafstetter":
            subbas, idxs_out = flwdir.subbasins_pfafstetter(**split_kwargs)
        elif split_method=="outlets":
            args = {'xy':split_kwargs['xy']}
            if 'min_sto' in split_kwargs:
                args['streams']=flwdir.stream_order() >= split_kwargs['min_sto']
            subbas = flwdir.basins(**args)
            # TODO: make idxs_out the snapped coordinates!
            idxs_out = ds.raster.xy_to_idx(args['xy'][0],args['xy'][1])
            
        # make into raster
        da_out = xr.DataArray(
                data=subbas.astype('int32'),
                coords=ds.raster.coords,
                dims=ds.raster.dims,
            )
        da_out.raster.set_nodata(ds._FillValue)
        da_out.raster.set_crs(ds.raster.crs)
        # vecorize raster
        subbasins_gpd = da_out.raster.vectorize()
        # drop elements that are not part of the basin
        subbasins_gpd = subbasins_gpd[subbasins_gpd.value !=0]
        subbasins_gpd.plot(edgecolor='black')

        # set xy locations of basin outlets
        outl_xs , outl_ys = ds.raster.idx_to_xy(idxs_out)
        outlet_geom = gpd.points_from_xy(x=outl_xs,y=outl_ys)
        subbasins_gpd['outlet_geometry']=outlet_geom
        
        return subbasins_gpd
    
    def setup_downstream_links(
        self,
        outflow_gpd: gpd.GeoDataFrame,
        hydrography_fn="merit_hydro",
        ):
        """Link basins from upstream to downstream based on flow direction map

        Args:
            outflow_gpd (gpd.GeoDataFrame): point geometries of the subbasin outlets
            hydrography_fn (str, optional): Hydrography dataset, must include flwdir variable.
            Defaults to "merit_hydro".

        Returns:
            np.array: array of downstream link ids
        """        
        ds = self.data_catalog.get_rasterdataset(hydrography_fn,geom=self.region)
        flwdir = flw.flwdir_from_da(ds["flwdir"], ftype="d8", mask=_make_ds_mask(ds, self.region, col_name='value'))
        rasterized_map = ds.raster.rasterize(self._geoms['response_unit'],col_name='value')

        # TODO: make one function in flw.py
        dwn_basin = flwdir.downstream(rasterized_map.values.flatten()).reshape(ds.raster.shape)
        da_out = xr.DataArray(
                data=dwn_basin,
                coords=ds.raster.coords,
                dims=ds.raster.dims,
            )
        da_out.raster.set_nodata(0)
        da_out.raster.set_crs(ds.raster.crs)
        basins_downstream = da_out.raster.sample(outflow_gpd)#.values[::-1] # NOTE: check if this reversal is true in all cases
        self._geoms['response_unit']['down_id'] = basins_downstream
        return basins_downstream
        
        
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



