# -*- coding: utf-8 -*-
"""Scripts to derive (sub)basin geometries.

Based on pre-cooked basin index files, basin maps or flow direction maps.
"""

import logging
from typing import Any

import geopandas as gpd
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from shapely.geometry import box

from hydromt.data_catalog.sources.geodataframe import GeoDataFrameSource
from hydromt.gis.flw import basin_map, flwdir_from_da, outlet_map, stream_map

logger = logging.getLogger(__name__)

__all__ = ["get_basin_geometry"]


def get_basin_geometry(
    ds: xr.Dataset,
    basin_index: gpd.GeoDataFrame | GeoDataFrameSource | None = None,
    kind: str = "basin",
    bounds: list[float] | tuple[float] | None = None,
    bbox: list[float] | tuple[float] | None = None,
    geom: gpd.GeoDataFrame | None = None,
    xy: list[float] | tuple[float, float] | None = None,
    basid: int | list[int] | None = None,
    outlets: bool = False,
    basins_name: str = "basins",
    flwdir_name: str = "flwdir",
    ftype: str = "infer",
    buffer: int = 10,
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
    ds : xr.Dataset
        Dataset containing basin and flow direction variables
    basin_index: gpd.GeoDataFrame
        Dataframe with basin geomtries or bounding boxes with "basid" column
        corresponding to the ``ds[<basins_name>]`` map.
    kind : str
        Kind of basin description, choose from "basin", "subbasin" or "interbasin"
    bounds : list[float] | tuple[float], optional
        [xmin, ymin, xmax, ymax] coordinates of total bounding box, i.e. the data is
        clipped to this domain before futher processing. By default None
    bbox : list[float] | tuple[float], optional
        [xmin, ymin, xmax, ymax] coordinates to infer (sub)(inter)basin(s),
        by default None
    geom : gpd.GeoDataFrame, optional
        Polygon geometry describing area of interest
    xy : list[float] | tuple[float], optional
        x, y coordinates of (sub)basin outlet locations, by default None
    basid : int | list[int], optional
        Basin IDs, must match values in basin maps, by default None
    outlets : bool, optional
        If True, include (sub)basins of outlets within domain only. By default False
    basins_name : str, optional
        Name of flow direction variable in source, by default "basins"
    flwdir_name : str, optional
        Name of flow direction variable in source, by default "flwdir"
    ftype : {'d8', 'ldd', 'nextxy'}, optional
        Name of flow direction type, by default None; use input ftype.
    buffer : int, optional
        The buffer to apply, by default 10
    **stream_kwargs : dict
        Name of variable in ds and threshold value

    Returns
    -------
    tuple[gpd.GeoDataFrame]
        Geometry of the (sub)basin(s) and geometry of the outlet point location
    """
    __assert_kind_supported(kind)

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

    outlet_geom = None
    basin_mask = None

    # INTER- & SUBBASINS
    if kind in ["subbasin", "interbasin"]:
        xy = __convert_list_to_tuple(xy)
        basin_mask, outlet_geom = __get_basin_geometry_from_sub_or_interbasin(
            ds=ds,
            geom=geom,
            kind=kind,
            basins_name=basins_name,
            flwdir_name=flwdir_name,
            ftype=ftype,
            xy=xy,
            outlets=outlets,
            bounds=bounds,
            **stream_kwargs,
        )
    ## BASINS
    elif kind == "basin" or bounds is None:
        basin_mask = __get_basin_geometry_from_basin(
            ds=ds,
            geom=geom,
            basid=basid,
            basins_name=basins_name,
            flwdir_name=flwdir_name,
            ftype=ftype,
            xy=xy,
            outlets=outlets,
            buffer=buffer,
            basin_index=basin_index,
            **stream_kwargs,
        )

    if not np.any(basin_mask > 0):
        raise ValueError(f"No {kind} found with given criteria.")
    basin_geom = __transform_basin_mask_to_geom(ds, kind, basin_mask)
    return basin_geom, outlet_geom


def __convert_list_to_tuple(
    xy: list[float] | tuple[float, float] | None,
) -> tuple[float, float] | None:
    if xy is not None:
        xy = (np.atleast_1d(xy[0]), np.atleast_1d(xy[1]))
    return xy


def __transform_basin_mask_to_geom(ds, kind, basin_mask):
    basin_mask = basin_mask.astype(np.int32)
    basin_mask.raster.set_crs(ds.raster.crs)
    basin_mask.raster.set_nodata(0)
    w, s, e, n = basin_mask.raster.clip_mask(basin_mask).raster.bounds
    logger.info(f"{kind} bbox: [{w:.4f}, {s:.4f}, {e:.4f}, {n:.4f}]")

    # vectorize basins and outlets
    basin_geom = basin_mask.raster.vectorize()
    return basin_geom


def __clip_dataset_to_basin(
    *,
    ds: xr.Dataset,
    geom: gpd.GeoDataFrame | None,
    basid: ArrayLike,
    dvars: list[str],
    gdf_bas: gpd.GeoDataFrame,
):
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
    return ds


def __assert_kind_supported(kind: str):
    kind_lst = ["basin", "subbasin", "interbasin"]
    if kind not in kind_lst:
        msg = f"Unknown kind: {kind}, select from {kind_lst}."
        raise ValueError(msg)


def __assert_data_vars_exist_in_ds(ds: xr.Dataset, dvars: list[str]):
    for name in dvars:
        if name not in ds.data_vars:
            raise ValueError(f"Dataset variable {name} not in ds.")


def __check_size(ds, threshold=12e3**2):
    # warning for large domain
    if (
        np.multiply(*ds.raster.shape) > threshold
    ):  # 12e3 ** 2 > 10x10 degree at 3 arcsec
        logger.warning("Loading very large spatial domain to derive a subbasin.")


def __get_basin_index_gdf(
    *,
    ds_crs,
    basin_index: gpd.GeoDataFrame | GeoDataFrameSource | None,
    geom: gpd.GeoDataFrame | None,
    xy: list[float] | tuple[float, float] | None,
) -> gpd.GeoDataFrame | None:
    result = None
    if basin_index is not None:
        if isinstance(basin_index, GeoDataFrameSource):
            kwargs = {"variables": ["basid"]}
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
            result = basin_index.read_data(**kwargs)
        elif isinstance(basin_index, gpd.GeoDataFrame):
            result = basin_index
            if "basid" not in result.columns:
                raise ValueError("Basin geometries does not have 'basid' column.")
        if result.crs != ds_crs:
            logger.warning("Basin geometries CRS does not match the input raster CRS.")
            result = result.to_crs(ds_crs)
    return result


def __get_basin_geometry_from_sub_or_interbasin(
    *,
    ds: xr.Dataset,
    geom: gpd.GeoDataFrame | None,
    kind: str,
    basins_name: str,
    flwdir_name: str,
    ftype: str,
    xy: tuple[float, float] | None,
    outlets: bool,
    bounds: list[float] | tuple[float] | None,
    **stream_kwargs,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:
    # get flow directions
    __check_size(ds)  # warning for large domain
    mask = False
    # if interbasin, set flwdir mask within geometry / bounding box
    if kind == "interbasin":  # we have checked before that geom is not None
        mask = ds.raster.geometry_mask(geom)
    elif basins_name in ds:  # set flwdir mask based on basin map
        mask = ds[basins_name] > 0
    flwdir = flwdir_from_da(ds[flwdir_name], ftype=ftype, mask=mask)
    # get area of interest (aoi) mask
    aoi = __get_area_of_interest_mask(ds=ds, geom=geom)
    # get stream mask. Always over entire domain
    # to include cells downstream of aoi!
    kwargs: dict[str, Any] = {}
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
        idxs_out = np.nonzero(outmap.values.ravel())[0]
        if not np.any(outmap):
            raise ValueError("No outlets found with with given criteria.")
        xy = outmap.raster.idx_to_xy(idxs_out)
    # get subbasin map
    bas_mask, xy_out = basin_map(ds, flwdir, xy=xy, **kwargs)
    __warn_if_no_upstream_cells_for_subbasin(
        kind=kind, bounds=bounds, ds=ds, bas_mask=bas_mask
    )
    outlet_geom = __get_outlet_geom_from_xy(xy=xy_out, ds_crs=ds.raster.crs)

    return bas_mask, outlet_geom


def __get_area_of_interest_mask(*, ds: xr.Dataset, geom: gpd.GeoDataFrame | None):
    if geom is not None:
        return ds.raster.geometry_mask(geom)
    else:
        return xr.DataArray(
            coords=ds.raster.coords,
            dims=ds.raster.dims,
            data=np.full(ds.raster.shape, True, dtype=bool),
        )  # all True


def __warn_if_no_upstream_cells_for_subbasin(
    *,
    kind: str,
    bounds: list[float] | tuple[float] | None,
    ds: xr.Dataset,
    bas_mask: xr.DataArray,
):
    # is subbasin with bounds check if all upstream cells are included
    if kind == "subbasin" and bounds is not None:
        geom = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=ds.raster.crs)
        mask = ds.raster.geometry_mask(geom)
        if np.any(np.logical_and(mask == 0, bas_mask != 0)):
            logger.warning("The subbasin does not include all upstream cells.")


def __get_outlet_geom_from_xy(
    *, xy: tuple[ArrayLike, ArrayLike] | None, ds_crs
) -> gpd.GeoDataFrame | None:
    if xy is not None:
        return gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[0], xy[1]), crs=ds_crs)
    return None


def __get_basin_geometry_from_basin(
    *,
    basins_name: str,
    ds: xr.Dataset,
    geom: gpd.GeoDataFrame | None,
    xy: tuple[float, float] | None,
    basid: int | list[int] | None,
    outlets: bool,
    flwdir_name: str,
    ftype: str,
    buffer: int,
    basin_index: gpd.GeoDataFrame | GeoDataFrameSource | None = None,
    **stream_kwargs,
):
    dvars = [flwdir_name] + list(stream_kwargs.keys())
    __assert_data_vars_exist_in_ds(ds, dvars)
    dvars = dvars + [basins_name]

    gdf_bas = __get_basin_index_gdf(
        ds_crs=ds.raster.crs, basin_index=basin_index, geom=geom, xy=xy
    )

    if basins_name not in ds:
        gdf_bas = __ignore_basin_geometry(gdf_bas)
        __check_size(ds)
        logger.info(f'basin map "{basins_name}" missing, calculating on the fly.')
        flwdir = flwdir_from_da(ds[flwdir_name], ftype=ftype)
        ds[basins_name] = xr.Variable(ds.raster.dims, flwdir.basins())

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
        if stream_kwargs:
            stream = stream_map(ds_clip, **stream_kwargs)
            ds_clip[basins_name] = ds_clip[basins_name].where(stream, 0)
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
        ds = __clip_dataset_to_basin(
            ds=ds, geom=geom, basid=basid, dvars=dvars, gdf_bas=gdf_bas
        )
    # get full basin mask and use this mask ds_basin incl flow direction raster
    _mask = np.isin(ds[basins_name], basid)
    if not np.any(_mask > 0):
        raise ValueError(f"No basins found with IDs: {basid}")
    ds = ds[dvars].raster.mask(_mask)
    ds = ds.raster.clip_mask(ds[basins_name])
    return ds[basins_name]


def __ignore_basin_geometry(gdf_bas):
    if gdf_bas is not None:
        gdf_bas = None
        logger.warning(
            "Basin geometries ignored as no corresponding" + " basin map is provided."
        )

    return gdf_bas
