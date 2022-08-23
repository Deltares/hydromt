import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import logging
from typing import Tuple, Union, Optional
from .. import flw
import geopandas as gpd
import hydromt

logger = logging.getLogger(__name__)

__all__ = ["make_ds_mask", "hydrography_to_basins","create_response_unit_ds",
           "ru_geometry_to_gpd", "fracs", "ds_class_mode"]


def make_ds_mask(ds, gdf, col_name='value'):
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

def create_response_unit_ds(response_units_gdf):
    ds = xr.Dataset(
        #data_vars=dict(
        #    basin_id=(["index"],response_units['value']),
        #),
        coords=dict(
            index = (["index"],response_units_gdf['value']),
            geometry = (["index"], response_units_gdf['geometry']),
            outlet_geometry = (["index"], response_units_gdf['outlet_geometry']),
        ),
    )
    if response_units_gdf.crs is not None:  # parse crs
        ds = ds.rio.write_crs(response_units_gdf.crs)
    ds['index']=(["index"],response_units_gdf['value'])
    return ds

def fracs(ds,varname,dimname):
    """Return fraction(al cover) over dimension of variable"""
    fracs = ds[varname] / ds[varname].sum(dim=dimname)
    return fracs

def ds_class_mode(ds,varname,dimname):
    dout = xr.Dataset({
        varname+'_mode': (["index"]  , 
                 [ds[dimname].values[i] for i in ds[varname].argmax(dim=dimname).data])
    })
    return dout


def ru_geometry_to_gpd(ru_ds, index=None, geometry_name = 'geometry'):
    """extract geometry from response unit and store in geopandas.DataFrame
    select by index.
    """    
    if index is not None:
        ru_ds = ru_ds.sel(index=index)
    geometries = ru_ds[geometry_name].values
    indexes = ru_ds.index.values
    # with singular index, different setup fpr dataframe is needed
    if np.shape(geometries)==():
        unit_df = pd.DataFrame(data=dict(geometry=geometries.item(),value=indexes.item()),
                                index=[indexes.item()])
    else:
        unit_df = pd.DataFrame(data=dict(geometry=geometries,value=indexes),
                                index=indexes)
    unit_gpd = gpd.GeoDataFrame(unit_df, crs = ru_ds.rio.crs)
    return unit_gpd

def hydrography_to_basins(
        ds: Union[xr.DataArray, xr.Dataset],
        region: Union[gpd.GeoSeries, gpd.GeoDataFrame],
        split_method: str,
        **kwargs
        ):
        """Partition region into subbasins based on flow direction related methods from hydrography.
        
        The methods are described here: https://deltares.github.io/pyflwdir/latest/basins.html
        Parameters
        ----------
        split_method : str
            one of  ["streamorder","us_area","pfafstetter","outlets"]
        hydrography_fn : str, optional
            name of hydrography dataset, by default "merit_hydro"
        kwargs : dict or arguments to pass to respective functions depending on split_method, see below
            split_method = "us_area" (see function :py:meth:`~pyflwdir.subbasins_area`) 
            split_method = "pfafstetter" (see function :py:meth:`~pyflwdir.subbasins_pfafstetter`) 
            split_method = "streamorder" (see function :py:meth:`~pyflwdir.subbasins_streamorder`)    
            split_method = "outlets" (see function :py:meth:`~pyflwdir.basins`)  
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
        ds = ds.raster.clip_geom(region, mask=True)
        flwdir = flw.flwdir_from_da(ds["flwdir"], ftype="d8")
        
        split_method_lst = ["streamorder","us_area","pfafstetter","outlets"]
        if split_method not in split_method_lst:
            msg = f"Unknown split_method: {split_method}, select from {split_method_lst}."
            raise ValueError(msg)
        if split_method=="streamorder":
            subbas, idxs_out = flwdir.subbasins_streamorder(**kwargs)
        elif split_method=="us_area":
            subbas, idxs_out = flwdir.subbasins_area(**kwargs)
        elif split_method=="pfafstetter":
            subbas, idxs_out = flwdir.subbasins_pfafstetter(**kwargs)
        elif split_method=="outlets": #TODO: provide an example for this or more description in the function
            # args = {'xy':split_kwargs['xy']}
            if 'min_sto' in kwargs:
                kwargs['streams']=flwdir.stream_order() >= kwargs['min_sto']
            subbas = flwdir.basins(**kwargs)
            # TODO: make idxs_out the snapped coordinates!
            idxs_out = ds.raster.xy_to_idx(kwargs['xy'][0],kwargs['xy'][1])
            
        # make into raster
        da_out = xr.DataArray(
                data=subbas.astype('int32'),
                coords=ds.raster.coords,
                dims=ds.raster.dims,
            )
        da_out.raster.set_crs(ds.raster.crs)
        # vecorize raster
        subbasins_gpd = da_out.raster.vectorize()
        # drop elements that are not part of the basin
        subbasins_gpd = subbasins_gpd[subbasins_gpd.value !=0]
        # subbasins_gpd.plot(edgecolor='black')

        # set xy locations of basin outlets
        outl_xs , outl_ys = ds.raster.idx_to_xy(idxs_out)
        outlet_geom = gpd.points_from_xy(x=outl_xs,y=outl_ys)
        subbasins_gpd['outlet_geometry']=outlet_geom
        
        ruds = create_response_unit_ds(subbasins_gpd)
        
        return ruds