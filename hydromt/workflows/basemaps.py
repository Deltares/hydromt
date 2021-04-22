# -*- coding: utf-8 -*-

import os
from os.path import join, isdir, dirname, basename, isfile, abspath
import glob
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import logging
import geopandas as gpd
from pyflwdir import dem

logger = logging.getLogger(__name__)
from .. import flw, gis_utils

__all__ = ["hydrography", "topography"]


def hydrography(
    ds,
    res,
    xy=None,
    upscale_method="com2",
    flwdir_name="flwdir",
    uparea_name="uparea",
    basins_name="basins",
    strord_name="strord",
    channel_dir="up",
    ftype="infer",
    logger=logger,
    **kwargs,
):
    """Returns hydrography maps (see list below) and FlwdirRaster object based on 
    gridded flow directon and elevation data input. If the resolution is larger than the 
    source resolution, the flow direction data gets resampled and river length and slope 
    are based on subgrid flow paths.

    The output maps are:\
    - flwdir : flow direction\
    - basins : basin map\
    - uparea : upstream area [km2]\
    - strord : stream order\
    TODO:
    - elvadj : hydrologically adjusted elevation [m]

    Parameters
    ----------
    ds : xarray.DataArray
        Dataset containing gridded flow direction and elevation data.
    res : float
        output resolution
    xy : geopandas.GeoDataFrame, optional
        Subbasin pits. Only required when upscaling a subbasin.
    river_upa : float
        minimum upstream area threshold for the river map [km2]
    smooth_len : float
        average length over which the river slope is smoothed [km2]
    upscale_method : {'com2', 'com', 'eam', 'dmm'}
        Upscaling method for flow direction data, by default 'com2'.
    flwdir_name, elevtn_name, uparea_name : str, optional
        Name of flow direction, elevation and upstream area variables in ds
    channel_dir : {'up', 'down'}
        Define channel up or downstream from outlet, be default upstream

    Returns
    -------
    ds_out : xarray.DataArray
        Dataset containing gridded hydrography data
    flwdir_out : pyflwdir.FlwdirRaster
        Flow direction raster object.
    """
    # TODO add check if flwdir in ds, calculate if not
    flwdir = None
    basins = None
    outidx = None
    if not "mask" in ds.coords and xy is None:
        ds.coords["mask"] = xr.Variable(
            dims=ds.raster.dims, data=np.ones(ds.raster.shape, dtype=np.bool)
        )
    elif not "mask" in ds.coords:
        # NOTE if no subbasin mask is provided calculate it here
        logger.debug(f"Delineate {xy[0].size} subbasin(s).")
        flwdir = flw.flwdir_from_da(ds[flwdir_name], ftype=ftype)
        basins = flwdir.basins(xy=xy).astype(np.int32)
        ds.coords["mask"].data = basins != 0
        if not np.any(ds.coords["mask"]):
            raise ValueError("Delineating subbasins not successfull.")
    elif xy is not None:
        # NOTE: this mask is passed on from get_basin_geometry method
        logger.debug(f"Mask in dataset assumed to represent subbasins.")
    ncells = np.sum(ds["mask"].values)
    logger.debug(f"(Sub)basin at original resolution has {ncells} cells.")

    scale_ratio = int(np.round(res / ds.raster.res[0]))
    if scale_ratio > 1:  # upscale flwdir
        if flwdir is None:
            # NOTE initialize with mask is FALSE
            flwdir = flw.flwdir_from_da(ds[flwdir_name], ftype=ftype, mask=False)
        if xy is not None:
            logger.debug(f"Burn subbasin outlet in upstream area data.")
            if isinstance(xy, gpd.GeoDataFrame):
                assert xy.crs == ds.raster.crs
                xy = xy.geometry.x, xy.geometry.y
            idxs_pit = flwdir.index(*xy)
            flwdir.add_pits(idxs=idxs_pit)
            uparea = ds[uparea_name].values
            uparea.flat[idxs_pit] = uparea.max() + 1.0
            ds[uparea_name].data = uparea
        logger.info(
            f"Upscale flow direction data: {scale_ratio:d}x, {upscale_method} method."
        )
        da_flw, flwdir_out = flw.upscale_flwdir(
            ds,
            flwdir=flwdir,
            scale_ratio=scale_ratio,
            method=upscale_method,
            uparea_name=uparea_name,
            flwdir_name=flwdir_name,
            logger=logger,
        )
        da_flw.raster.set_crs(ds.raster.crs)
        # make sure x_out and y_out get saved
        ds_out = da_flw.to_dataset().reset_coords(["x_out", "y_out"])
        dims = ds_out.raster.dims
        # find pits within basin mask
        idxs_pit0 = flwdir_out.idxs_pit
        outlon = ds_out["x_out"].values.ravel()
        outlat = ds_out["y_out"].values.ravel()
        sel = {
            ds.raster.x_dim: xr.Variable("yx", outlon[idxs_pit0]),
            ds.raster.y_dim: xr.Variable("yx", outlat[idxs_pit0]),
        }
        outbas_pit = (
            ds.coords["mask"]
            .sel(
                sel,
                method="nearest",
            )
            .values
        )
        # derive basins
        if np.any(outbas_pit != 0):
            idxs_pit = idxs_pit0[outbas_pit != 0]
            basins = flwdir_out.basins(idxs=idxs_pit).astype(np.int32)
            ds_out.coords["mask"] = xr.Variable(
                dims=ds_out.raster.dims, data=basins != 0, attrs=dict(_FillValue=0)
            )
        else:
            # NOTE: this else statement seems wrong. instead an error is at place
            # ds_out.coords["mask"] = (
            #     ds["mask"]
            #     .astype(np.int8)
            #     .raster.reproject_like(da_flw, method="nearest")
            #     .astype(np.bool)
            # )
            # basins = ds_out["mask"].values.astype(np.int32)
            raise ValueError(
                "Unable to upscale the flow direction. "
                "Consider using a larger domain or higher spatial resolution. "
                "For subbasin models, consider a (higher) threshold to snap the outlet."
            )
        ds_out[basins_name] = xr.Variable(dims, basins, attrs=dict(_FillValue=0))
        # calculate upstream area using subgrid ucat cell areas
        outidx = np.where(
            ds_out["mask"], da_flw.coords["idx_out"].values, flwdir_out._mv
        )
        subare = flwdir.ucat_area(outidx, unit="km2")[1]
        uparea = flwdir_out.accuflux(subare)
        attrs = dict(_FillValue=-9999, unit="km2")
        ds_out[uparea_name] = xr.Variable(dims, uparea, attrs=attrs)
        # NOTE: subgrid cella area is currently not used in wflow
        ds_out["subare"] = xr.Variable(dims, subare, attrs=attrs)
        # initiate masked flow dir
        flwdir_out = flw.flwdir_from_da(
            ds_out[flwdir_name], ftype=flwdir.ftype, mask=True
        )
    else:
        # NO upscaling : source resolution equals target resolution
        # NOTE (re-)initialize with mask is TRUE
        ftype = flwdir.ftype if flwdir is not None and ftype == "infer" else ftype
        flwdir = flw.flwdir_from_da(ds[flwdir_name], ftype=ftype, mask=True)
        flwdir_out = flwdir
        ds_out = xr.DataArray(
            name=flwdir_name,
            data=flwdir_out.to_array(),
            coords=ds.raster.coords,
            dims=ds.raster.dims,
            attrs=dict(
                long_name=f"{ftype} flow direction",
                _FillValue=flwdir_out._core._mv,
            ),
        ).to_dataset()
        dims = ds_out.raster.dims
        ds_out.coords["mask"] = xr.Variable(
            dims=dims, data=flwdir_out.mask.reshape(flwdir_out.shape)
        )
        # copy data variables from source if available
        for dvar in [basins_name, uparea_name, strord_name]:
            if dvar in ds.data_vars:
                ds_out[dvar] = xr.where(
                    ds_out["mask"],
                    ds[dvar],
                    ds[dvar].dtype.type(ds[dvar].raster.nodata),
                )
                ds_out[dvar].attrs.update(ds[dvar].attrs)
        # basins
        if basins_name not in ds_out.data_vars:
            if basins is None:
                basins = flwdir_out.basins(idxs=flwdir_out.idxs_pit).astype(np.int32)
            ds_out[basins_name] = xr.Variable(dims, basins, attrs=dict(_FillValue=0))
        # upstream area
        if uparea_name not in ds_out.data_vars:
            uparea = flwdir_out.upstream_area("km2")  # km2
            attrs = dict(_FillValue=-9999, unit="km2")
            ds_out[uparea_name] = xr.Variable(dims, uparea, attrs=attrs)
        # cell area
        # NOTE: subgrid cella area is currently not used in wflow
        ys, xs = ds.raster.ycoords.values, ds.raster.xcoords.values
        subare = gis_utils.reggrid_area(ys, xs) / 1e6  # km2
        attrs = dict(_FillValue=-9999, unit="km2")
        ds_out["subare"] = xr.Variable(dims, subare, attrs=attrs)
    # logging
    xy_pit = flwdir_out.xy(flwdir_out.idxs_pit[:5])
    xy_pit_str = ", ".join([f"({x:.5f},{y:.5f})" for x, y in zip(*xy_pit)])
    # stream order
    if strord_name not in ds_out.data_vars:
        logger.debug(f"Derive stream order.")
        strord = flwdir_out.stream_order()
        ds_out[strord_name] = xr.Variable(dims, strord, attrs=dict(_FillValue=-1))

    # clip to basin extent
    ds_out = ds_out.raster.clip_mask(mask=ds_out[basins_name])
    ds_out.raster.set_crs(ds.raster.crs)
    logger.debug(
        f"Map shape: {ds_out.raster.shape}; active cells: {flwdir_out.ncells}."
    )
    logger.debug(f"Outlet coordinates (head): {xy_pit_str}.")
    if np.any(np.asarray(ds_out.raster.shape) == 1):
        raise ValueError(
            "The output extent should at consist of two cells on each axis. "
            "Consider using a larger domain or higher spatial resolution. "
            "For subbasin models, consider a (higher) threshold to snap the outlet."
        )
    return ds_out, flwdir_out


def topography(
    ds,
    ds_like,
    elevtn_name="elevtn",
    lndslp_name="lndslp",
    method="average",
    logger=logger,
):
    """Returns topography maps (see list below) at model resolution based on gridded 
    elevation data input. 

    The following topography maps are calculated:\
    - elevtn : average elevation [m]\
    - lndslp : average land surface slope [m/m]\
    
    Parameters
    ----------
    ds : xarray.DataArray
        Dataset containing gridded flow direction and elevation data.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    elevtn_name, lndslp_name : str, optional
        Name of elevation and land surface slope variables in ds

    Returns
    -------
    ds_out : xarray.DataArray
        Dataset containing gridded hydrography data
    flwdir_out : pyflwdir.FlwdirRaster
        Flow direction raster object.
    """
    if lndslp_name not in ds.data_vars:
        logger.debug(f"Slope map {lndslp_name} not found: derive from elevation map.")
        crs = ds[elevtn_name].raster.crs
        nodata = ds[elevtn_name].raster.nodata
        ds[lndslp_name] = xr.Variable(
            dims=ds.raster.dims,
            data=dem.slope(
                elevtn=ds[elevtn_name].values,
                nodata=nodata,
                latlon=crs is not None and crs.to_epsg() == 4326,
                transform=ds[elevtn_name].raster.transform,
            ),
        )
        ds[lndslp_name].raster.set_nodata(nodata)
    # clip or reproject if non-identical grids
    ds_out = ds[[elevtn_name, lndslp_name]].raster.reproject_like(ds_like, method)
    ds_out[elevtn_name].attrs.update(unit="m")
    ds_out[lndslp_name].attrs.update(unit="m.m-1")
    return ds_out
