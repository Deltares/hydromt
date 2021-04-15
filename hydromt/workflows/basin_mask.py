# -*- coding: utf-8 -*-
"""Scripts to derive (sub)basin geometries from pre-cooked basin index files, 
basin maps or flow direction maps.
"""

from os.path import join, isdir, dirname, basename, isfile
import glob
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import xarray as xr
import pandas as pd
import logging
import copy
from pyflwdir.regions import region_bounds
from pyflwdir import pyflwdir

# local
from ..flw import flwdir_from_da, basin_map

logger = logging.getLogger(__name__)

__all__ = ["get_basin_geometry"]


def basin_index_id(root, da_bas, basids):
    """Returns combined table with basin bbox for all basids.
    The bbox are :
    - read from basin_index.csv
    - read from basin_index/<pfafid>.csv
        where <pfafid> corresponds to the first digits of the basid and all pfafid
        values have the same length
    - infered from da_bas map
    """
    basids0 = copy.copy(basids)
    basids = np.atleast_1d(basids)
    basids = basids[basids > 0]
    if basids.size == 0:
        raise ValueError(f"Basin IDs {basids0} not valid.")
    if isfile(join(root, "basin_index.csv")):
        # read from basin_index.csv
        df = pd.read_csv(join(root, "basin_index.csv"), index_col="basid").loc[basids]
    elif len(glob.glob(join(root, "basin_index", "*.csv"))) > 0:
        # read from basin_index/<pfafid>.csv
        fns = glob.glob(join(root, "basin_index", "*.csv"))
        pfafn = len(basename(fns[0])) - 4
        pfafids = [str(basid)[:2] for basid in basids]
        df_lst = []
        for pfaf_id in np.unique(pfafids):
            fn_outlets = join(root, "basin_index", f"{pfaf_id}.csv")
            df_lst.append(pd.read_csv(fn_outlets, index_col="basid"))
        df = pd.concat(df_lst, axis=0).loc[basids]
    else:
        # infer from data
        bas_mask = np.isin(da_bas.values, basids)
        if not np.any(bas_mask):
            raise ValueError(f"Basin IDs {basids} not found in map.")
        basins = np.where(bas_mask, da_bas.values, 0)
        basids, bboxs, _ = region_bounds(basins, da_bas.rio.transform)
        columsn = ["xmin", "ymin", "xmax", "ymax"]
        df = pd.DataFrame(index=basids, columns=columsn, data=bboxs)
    if len(df) == 0:
        raise ValueError(f"Basin IDs {basids} not found in tables.")
    return df


def basin_index_xy(root, da_bas, xy):
    """"Returns pandas.DataFrame of basins at xy sampled from da_bas map."""
    sel = {
        da_bas.rio.x_dim: xr.IndexVariable("xy", np.atleast_1d(xy[0])),
        da_bas.rio.y_dim: xr.IndexVariable("xy", np.atleast_1d(xy[1])),
    }
    basids = np.unique(da_bas.sel(**sel, method="nearest").values)
    basids = np.atleast_1d(basids[basids > 0])
    if len(basids) == 0:
        raise ValueError(f"No basins found at xy {xy}.")
    return basin_index_id(root, da_bas, basids)


def basin_index_all(root, da_bas_clip, da_bas=None):
    """Returns pandas.DataFrame of basins that intersects with the bbox/geom"""
    basids0 = np.unique(da_bas_clip.values)
    basids0 = np.atleast_1d(basids0[basids0 > 0])
    if len(basids0) == 0:
        raise ValueError(f"No basins found within region.")
    da_bas = da_bas_clip if da_bas is None else da_bas
    return basin_index_id(root, da_bas, basids0)


def basin_index_outlet(root, da_bas_clip, da_flw_clip, da_bas=None, ftype="infer"):
    """Returns pandas.DataFrame of basins with their outlet inside the bbox"""
    ftype = pyflwdir._infer_ftype(da_flw_clip.values) if ftype == "infer" else ftype
    pit_values = pyflwdir.FTYPES[ftype]._pv
    basids0 = da_bas_clip.values[np.isin(da_flw_clip.values, pit_values)]
    basids0 = np.atleast_1d(basids0[basids0 > 0])
    if len(basids0) == 0:
        raise ValueError(f"No basin outlets found within region.")
    da_bas = da_bas_clip if da_bas is None else da_bas
    return basin_index_id(root, da_bas, basids0)


def clip_mask_basin(da_bas, df):
    """Returns a basin mask DataArray"""
    total_bbox = np.asarray(
        [df["xmin"].min(), df["ymin"].min(), df["xmax"].max(), df["ymax"].max()]
    )
    bas0 = da_bas.rio.clip_bbox(total_bbox)
    mask = np.isin(bas0.values, df.index.values)
    if not np.any(mask):
        raise ValueError(f"No basins with given IDs found in data.")
    mask = xr.DataArray(
        dims=bas0.dims,
        coords=bas0.coords,
        data=np.where(mask, bas0.values, 0).astype(np.int32),
        attrs=dict(_FillValue=0),
    )
    return mask


def clip_and_mask(ds, bbox=None, geom=None, buffer=0):
    # clip
    ds_clip = ds
    # NOTE: prioritize geom of bbox
    if geom is not None:
        ds_clip = ds.rio.clip_geom(geom=geom, buffer=buffer)
    elif bbox is not None:
        ds_clip = ds.rio.clip_bbox(bbox=bbox, buffer=buffer)
    if "mask" in ds_clip:
        # mask values outside geom
        mask = ds_clip.coords["mask"]
        for name in ds.data_vars:
            ds_clip[name] = ds_clip[name].where(mask, ds_clip[name].rio.nodata)
    return ds_clip


def get_basin_geometry(
    ds,
    kind,
    root="",
    bbox=None,
    geom=None,
    xy=None,
    basid=None,
    basins_name="basins",
    flwdir_name="flwdir",
    ftype="infer",
    logger=logger,
    buffer=1,
    **stream_kwargs,
):
    """Returns a geometry of the (sub)basin(s) described by region.

    NOTE: a basin_index with the ID ('basid') and bounding box
    ('xmin', 'ymin', 'xmax', 'ymax') of each basin can be set in one or more csv files
    in root/basin_index.csv or root/basin_index/*.csv

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing basin and flow direction variable
    kind : {"basin", "subbasin", "outlet"}
        kind of basin description
    bbox : array_like of float
        [xmin, ymin, xmax, ymax] coordinates of bounding box of area of interest
    geom : geopandas.GeoDataFrame
        polygon geometry describing area of interest
    xy : tuple of array_like of float
        x, y coordinates of (sub)basin outlet locations
    basid : int or array_like of int
        basin IDs, must match values in basin maps
    flwdir_name : str, optional
        Name of flow direction variable in source, by default "flwdir"
    basins_name : str, optional
        Name of flow direction variable in source, by default "basins"
    ftype : {'d8', 'ldd', 'nextxy'}, optional
        name of flow direction type, by default None; use input ftype.
    stream_kwargs : key-word arguments
        name of variable in ds and threshold value

    Returns
    -------
    geopandas.geoDataFrame
        geometry the (sub)basin(s)
    """
    kind_lst = ["basin", "outlet", "subbasin"]
    if kind not in kind_lst:
        msg = f"Unknown kind: {kind}, select from {kind_lst}."
        raise ValueError(msg)
    outlets = bool(stream_kwargs.pop("outlets", False))
    within = bool(stream_kwargs.pop("within", False))
    dvars = [flwdir_name] + [v for v in stream_kwargs]
    for name in dvars:
        if name not in ds.data_vars:
            raise ValueError(f"Dataset variable {name} not in ds.")

    if kind in ["basin", "outlet"] or (bbox is None and geom is None):
        if basins_name not in ds:
            logger.info(f'basin map "{basins_name}" missing, calculating on the fly.')
            flwdir = flwdir_from_da(ds[flwdir_name], ftype=ftype)
            ds[basins_name] = xr.Variable(ds.rio.dims, flwdir.basins())
            ds[basins_name].rio.set_nodata(0)
        # clip
        dvars = dvars + [basins_name]
        ds_clip = clip_and_mask(ds[dvars], bbox=bbox, geom=geom)
        if kind == "outlet":
            logger.debug(f"Getting basin bounds of intersecting outlets.")
            df_basins = basin_index_outlet(
                root,
                ds_clip[basins_name],
                ds_clip[flwdir_name],
                ds[basins_name],
                ftype=ftype,
            )
        else:
            if basid is not None:
                logger.debug(f"Getting bounds of given basin IDs.")
                df_basins = basin_index_id(root, ds_clip[basins_name], basid)
            elif xy is not None:
                logger.debug(f"Getting bounds of basins at point locations.")
                df_basins = basin_index_xy(root, ds_clip[basins_name], xy)
                if kind != "subbasin":
                    xy = None  # avoid using xy to set a pit in downstream methods
            else:
                logger.debug(f"Getting bounds of intersecting basins.")
                df_basins = basin_index_all(root, ds_clip[basins_name], ds[basins_name])
        da_mask = clip_mask_basin(ds[basins_name], df_basins)
        bbox = da_mask.rio.transform_bounds(4326)

    if kind == "subbasin":
        # clip with geom to yield region_mask
        if geom is None and bbox is not None:
            geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
        ds_clip = ds[dvars].rio.clip_geom(geom, buffer=buffer)
        region_mask = ds_clip["mask"]
        # warning for large domain
        if np.multiply(*ds_clip.rio.shape) > 12e3 ** 2:  # > 10x10 degree at 3 arcsec
            logger.warning(
                "Loading very large spatial domain to derive a subbasin. "
                "Provide an initial 'bbox' region if this takes too long."
            )
        # get stream map
        stream = None
        if len(stream_kwargs) > 0:
            logger.debug(
                f"Delineating subbasin(s). Outlets at stream thresholds {stream_kwargs}"
            )
            stream = np.full(ds_clip.rio.shape, True, dtype=np.bool)
            for name, value in stream_kwargs.items():
                stream = np.logical_and(stream, ds_clip[name].values >= value)
        elif outlets:
            logger.debug(
                f"Delineating subbasin(s). Outlets at river mouth (sink or sea)."
            )
        # get flwdir
        flwdir = flwdir_from_da(ds_clip[flwdir_name], ftype=ftype, mask=False)
        # delineate subbasin
        kwargs = dict(stream=stream, outlets=outlets)
        da_mask, xy = basin_map(ds_clip, flwdir, xy, **kwargs)
        if within:
            logger.debug(f"Mask headwater subbasins which are completely in region.")
            region_mask = flwdir.subbasin_mask_within_region(region_mask.values, stream)
        # check if mask extent is smaller than original bbox
        elif np.any(np.logical_and(~region_mask, da_mask != 0)):
            logger.warning("The subbasin does not include all upstream cells.")
            # return only most downstream contiguous area of each basin
            region_mask = flwdir.contiguous_area_within_region(
                region_mask.values, stream
            )
        da_mask.data = np.where(region_mask, da_mask.values, 0)
        if not np.any(da_mask):
            raise ValueError("No subbasin found with given criteria.")
        da_mask = da_mask.astype(np.int32).rio.clip_mask(da_mask)

    w, s, e, n = da_mask.rio.bounds
    ncells = np.sum(da_mask.values != 0)
    logger.info(f"basin bbox: [{w:.4f}, {s:.4f}, {e:.4f}, {n:.4f}] / size: {ncells}")

    # vectorize
    da_mask.rio.set_nodata(0)
    da_mask.rio.set_crs(ds.rio.crs)
    geom = da_mask.rio.vectorize()
    return geom, xy
