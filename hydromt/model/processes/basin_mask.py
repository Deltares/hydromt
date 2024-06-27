# -*- coding: utf-8 -*-
"""Scripts to derive (sub)basin geometries.

Based on pre-cooked basin index files, basin maps or flow direction maps.
"""

import logging

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box

from hydromt.data_catalog.sources.geodataframe import GeoDataFrameSource
from hydromt.gis.flw import basin_map, flwdir_from_da, outlet_map, stream_map

logger = logging.getLogger(__name__)

__all__ = ["get_basin_geometry"]


def get_basin_geometry(
    ds,
    basin_index=None,
    kind="basin",
    bounds=None,
    bbox=None,
    geom=None,
    xy=None,
    basid=None,
    outlets=False,
    basins_name="basins",
    flwdir_name="flwdir",
    ftype="infer",
    buffer=10,
    **stream_kwargs,
):
    """Return a geometry of the (sub)(inter)basin(s).

    This method derives a geometry of sub-, inter- or full basin based on an input
    dataset with flow-direction and optional basins ID raster data in combination
    with a matching basin geometry file containing the bounding boxes of each basin.

    Either ``bbox``, ``geom``, ``xy`` (or ``basid`` in case of ``kind='basin'``) must
    be provided.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing basin and flow direction variables
    basin_index: geopandas.GeoDataFrame or GeoDataFrameAdapter
        Dataframe with basin geomtries or bounding boxes with "basid" column
        corresponding to the ``ds[<basins_name>]`` map.
    kind : {"basin", "subbasin", "interbasin"}
        kind of basin description
    bounds: array_like of float, optional
        [xmin, ymin, xmax, ymax] coordinates of total bounding box, i.e. the data is
        clipped to this domain before futher processing.
    bbox : array_like of float, optional
        [xmin, ymin, xmax, ymax] coordinates to infer (sub)(inter)basin(s)
    geom : geopandas.GeoDataFrame, optional
        polygon geometry describing area of interest
    xy : tuple of array_like of float, optional
        x, y coordinates of (sub)basin outlet locations
    basid : int or array_like of int, optional
        basin IDs, must match values in basin maps
    outlets: bool, optional
        If True, include (sub)basins of outlets within domain only.
    flwdir_name : str, optional
        Name of flow direction variable in source, by default "flwdir"
    basins_name : str, optional
        Name of flow direction variable in source, by default "basins"
    ftype : {'d8', 'ldd', 'nextxy'}, optional
        name of flow direction type, by default None; use input ftype.
    stream_kwargs : key-word arguments
        name of variable in ds and threshold value
    buffer:
        The buffer to apply.
    logger:
        The logger to use.

    Returns
    -------
    basin_geom : geopandas.geoDataFrame
        geometry the (sub)basin(s)
    outlet_geom : geopandas.geoDataFrame
        geometry the outlet point location
    """
    kind_lst = ["basin", "subbasin", "interbasin"]
    if kind not in kind_lst:
        msg = f"Unknown kind: {kind}, select from {kind_lst}."
        raise ValueError(msg)
    # check variables
    dvars = [flwdir_name] + [v for v in stream_kwargs]
    for name in dvars:
        if name not in ds.data_vars:
            raise ValueError(f"Dataset variable {name} not in ds.")

    # for interbasins we can limit the domain based on either bbox / geom or bounds
    if kind == "interbasin" and bounds is None:
        if bbox is None and geom is None:
            raise ValueError('"kind=interbasin" requires either "bbox" or "geom"')
        bounds = bbox if bbox is not None else geom.total_bounds
    # initial clip based on bounds
    if bounds is not None:
        ds = ds.raster.clip_bbox(bounds, buffer=buffer)
    # convert bbox to geom
    if geom is None and bbox is not None:
        geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=ds.raster.crs)

    # check basin index
    # TODO understand pfafstetter codes
    gdf_bas = None
    if basin_index is not None:
        if isinstance(basin_index, GeoDataFrameSource):
            kwargs = dict(variables=["basid"])
            if geom is not None:
                kwargs.update(mask=geom)
            elif xy is not None:
                xy0 = np.atleast_1d(xy[0])
                xy1 = np.atleast_1d(xy[1])
                kwargs.update(
                    bbox=[
                        min(xy0) - 0.1,
                        min(xy1) - 0.1,
                        max(xy0) + 0.1,
                        max(xy1) + 0.1,
                    ]
                )
            gdf_bas = basin_index.read_data(**kwargs)
        elif isinstance(basin_index, gpd.GeoDataFrame):
            gdf_bas = basin_index
            if "basid" not in gdf_bas.columns:
                raise ValueError("Basin geometries does not have 'basid' column.")
        if gdf_bas.crs != ds.raster.crs:
            logger.warning("Basin geometries CRS does not match the input raster CRS.")
            gdf_bas = gdf_bas.to_crs(ds.raster.crs)

    ## BASINS
    if kind == "basin" or bounds is None:
        dvars = dvars + [basins_name]
        if basins_name not in ds:
            if gdf_bas is not None:
                gdf_bas = None
                logger.warning(
                    "Basin geometries ignored as no corresponding"
                    + " basin map is provided."
                )
            _check_size(ds)
            logger.info(f'basin map "{basins_name}" missing, calculating on the fly.')
            flwdir = flwdir_from_da(ds[flwdir_name], ftype=ftype)
            ds[basins_name] = xr.Variable(ds.raster.dims, flwdir.basins())
        elif (
            ds[basins_name].raster.nodata != 0
            and ds[basins_name].raster.nodata is not None
        ):
            ds[basins_name].where(ds[basins_name] != ds[basins_name].raster.nodata, 0)
        ds[basins_name].raster.set_nodata(0)
        # clip
        ds_clip = ds[dvars]
        if geom is not None:
            ds_clip = ds[dvars].raster.clip_geom(geom, buffer=buffer, mask=True)
        # get basin IDs
        if xy is not None:
            logger.debug("Getting basin IDs at point locations.")
            sel = {
                ds.raster.x_dim: xr.IndexVariable("xy", np.atleast_1d(xy[0])),
                ds.raster.y_dim: xr.IndexVariable("xy", np.atleast_1d(xy[1])),
            }
            basid = np.unique(ds_clip[basins_name].sel(**sel, method="nearest").values)
        elif basid is None:
            if stream_kwargs or outlets:
                if stream_kwargs:
                    stream = stream_map(ds_clip, **stream_kwargs)
                if outlets:
                    outmap = outlet_map(ds_clip[flwdir_name], ftype=ftype)
                    if stream_kwargs:
                        stream = stream.where(outmap, 0)
                    else:
                        stream = outmap
                ds_clip[basins_name] = ds_clip[basins_name].where(stream, 0)
            logger.debug("Getting IDs of intersecting basins.")
            basid = np.unique(ds_clip[basins_name].values)
        basid = np.atleast_1d(basid)
        basid = basid[basid > 0]
        if basid.size == 0:
            raise ValueError("No basins found with given criteria.")
        # clip ds to total basin
        if gdf_bas is not None:
            gdf_match = np.isin(gdf_bas["basid"], basid)
            gdf_bas = gdf_bas.loc[gdf_match]
            if gdf_bas.index.size > 0:
                if geom is not None:
                    xminbas, yminbas, xmaxbas, ymaxbas = gdf_bas.total_bounds
                    # Check that total_bounds is at least bigger
                    # than original geom bounds
                    xmingeom, ymingeom, xmaxgeom, ymaxgeom = geom.total_bounds
                    total_bounds = [
                        min(xminbas, xmingeom),
                        min(yminbas, ymingeom),
                        max(xmaxbas, xmaxgeom),
                        max(ymaxbas, ymaxgeom),
                    ]
                else:
                    total_bounds = gdf_bas.total_bounds
                ds = ds[dvars].raster.clip_bbox(total_bounds, buffer=0)
            elif np.any(gdf_match):
                logger.warning("No matching basin IDs found in basin geometries.")
        # get full basin mask and use this mask ds_basin incl flow direction raster
        _mask = np.isin(ds[basins_name], basid)
        if not np.any(_mask > 0):
            raise ValueError(f"No basins found with IDs: {basid}")
        ds = ds[dvars].raster.mask(_mask)
        ds = ds.raster.clip_mask(ds[basins_name])
        bas_mask = ds[basins_name]

    # INTER- & SUBBASINS
    xy_out = None
    if kind in ["subbasin", "interbasin"]:
        # get flow directions
        _check_size(ds)  # warning for large domain
        mask = False
        # if interbasin, set flwdir mask within geometry / bounding box
        if kind == "interbasin":  # we have checked before that geom is not None
            mask = ds.raster.geometry_mask(geom)
        elif basins_name in ds:  # set flwdir mask based on basin map
            mask = ds[basins_name] > 0
        flwdir = flwdir_from_da(ds[flwdir_name], ftype=ftype, mask=mask)
        # get area of interest (aoi) mask
        if geom is not None:
            aoi = ds.raster.geometry_mask(geom)
        else:
            aoi = xr.DataArray(
                coords=ds.raster.coords,
                dims=ds.raster.dims,
                data=np.full(ds.raster.shape, True, dtype=bool),
            )  # all True
        # Convert xy to tuple
        if xy is not None:
            xy = (np.atleast_1d(xy[0]), np.atleast_1d(xy[1]))
        # get stream mask. Always over entire domain
        # to include cells downstream of aoi!
        kwargs = dict()
        if stream_kwargs:
            stream = stream_map(ds, **stream_kwargs)
            if not np.any(stream):
                raise ValueError(f"No streams found with: {stream_kwargs}.")
            if not outlets and xy is None:  # get aoi outflow cells if none provided
                xy = flwdir.xy(flwdir.outflow_idxs(np.logical_and(stream, aoi).values))
            elif not outlets:
                kwargs.update(stream=stream.values)
        if outlets:
            outmap = aoi.where(outlet_map(ds[flwdir_name], ftype=flwdir.ftype), False)
            if stream_kwargs:
                outmap = outmap.where(stream_kwargs, False)
            idxs_out = np.where(outmap.values.ravel())[0]
            if not np.any(outmap):
                raise ValueError("No outlets found with with given criteria.")
            xy = outmap.raster.idx_to_xy(idxs_out)
        # get subbasin map
        bas_mask, xy_out = basin_map(ds, flwdir, xy, **kwargs)
        # is subbasin with bounds check if all upstream cells are included
        if kind == "subbasin" and bounds is not None:
            geom = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=ds.raster.crs)
            mask = ds.raster.geometry_mask(geom)
            if np.any(np.logical_and(mask == 0, bas_mask != 0)):
                logger.warning("The subbasin does not include all upstream cells.")

    if not np.any(bas_mask > 0):
        raise ValueError(f"No {kind} found with given criteria.")
    bas_mask = bas_mask.astype(np.int32)
    bas_mask.raster.set_crs(ds.raster.crs)
    bas_mask.raster.set_nodata(0)
    w, s, e, n = bas_mask.raster.clip_mask(bas_mask).raster.bounds
    logger.info(f"{kind} bbox: [{w:.4f}, {s:.4f}, {e:.4f}, {n:.4f}]")

    # vectorize basins and outlets
    basin_geom = bas_mask.raster.vectorize()
    outlet_geom = None
    if xy_out is not None:
        points = gpd.points_from_xy(*xy_out)
        outlet_geom = gpd.GeoDataFrame(geometry=points, crs=ds.raster.crs)

    return basin_geom, outlet_geom


def _check_size(ds, threshold=12e3**2):
    # warning for large domain
    if (
        np.multiply(*ds.raster.shape) > threshold
    ):  # 12e3 ** 2 > 10x10 degree at 3 arcsec
        logger.warning(
            "Loading very large spatial domain to derive a subbasin. "
            "Provide initial 'bounds' if this takes too long."
        )
