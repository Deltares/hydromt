# -*- coding: utf-8 -*-
"""Scripts to derive (sub)basin geometries from pre-cooked basin index files, 
basin maps or flow direction maps.
"""

from os.path import isdir, isfile
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from sklearn.neighbors import VALID_METRICS
import xarray as xr
import logging

# local
from ..io import open_raster
from ..flw import flwdir_from_da, basin_map, stream_map, outlet_map
from ..data_adapter import GeoDataFrameAdapter
from ..models import ENTRYPOINTS, model_plugins

logger = logging.getLogger(__name__)

__all__ = ["get_basin_geometry", "parse_region"]


def parse_region(region, logger=logger):
    """Checks and returns parsed region arguments.

    Parameters
    ----------
    region : dict
        Dictionary describing region of interest.

        For an exact clip of the region:

        * {'bbox': [xmin, ymin, xmax, ymax]}

        * {'geom': /path/to/polygon_geometry}

        For a region based of another models grid:

        * {'<model_name>': root}

        For a region based of the grid of a raster file:

        * {'grid': /path/to/raster}

        Entire basin can be defined based on an ID, one or multiple point location (x, y),
        or a region of interest (bounding box or geometry) for which the basin IDs are
        looked up. The basins withint the area of interest can be further filtered to
        only include basins with their outlet within the area of interest ('outlets': true)
        of stream threshold arguments (e.g.: 'uparea': 1000). Common use-cases include:

        * {'basin': ID}

        * {'basin': [ID1, ID2, ..]}

        * {'basin': [x, y]}

        * {'basin': [[x1, x2, ..], [y1, y2, ..]]}

        * {'basin': /path/to/point_geometry}

        * {'basin': [xmin, ymin, xmax, ymax]}

        * {'basin': [xmin, ymin, xmax, ymax], 'outlets': true}

        * {'basin': [xmin, ymin, xmax, ymax], '<variable>': threshold}

        Subbasins are defined by its outlet locations and include all area upstream
        from these points. The outlet locations can be passed as xy coordinate pairs,
        but also derived from the most downstream cell(s) within a area of interest
        defined by a bounding box or geometry, optionally refined by stream threshold arguments.

        The method can be speed up by providing an additional ``bounds`` argument which
        should contain all upstream cell. If cells upstream of the subbasin are not
        within the provide bounds a warning will be raised. Common use-cases include:

        * {'subbasin': [x, y], '<variable>': threshold}

        * {'subbasin': [[x1, x2, ..], [y1, y2, ..]], '<variable>': threshold, 'bounds': [xmin, ymin, xmax, ymax]}

        * {'subbasin': /path/to/point_geometry, '<variable>': threshold}

        * {'subbasin': [xmin, ymin, xmax, ymax], '<variable>': threshold}

        * {'subbasin': /path/to/polygon_geometry, '<variable>': threshold}

        Interbasins are similar to subbasins but are bounded by a bounding box or geometry
        and do not include all upstream area. Common use-cases include:

        * {'interbasin': [xmin, ymin, xmax, ymax], '<variable>': threshold}

        * {'interbasin': [xmin, ymin, xmax, ymax], 'xy': [x, y]}

        * {'interbasin': /path/to/polygon_geometry, 'outlets': true}

    Returns
    -------
    kind : {'basin', 'subbasin', 'interbasin', 'geom', 'bbox', 'grid'}
        region kind
    kwargs : dict
        parsed region json
    """
    kwargs = region.copy()
    # NOTE: the order is important to prioritize the arguments
    options = {
        "basin": ["basid", "geom", "bbox", "xy"],
        "subbasin": ["geom", "bbox", "xy"],
        "interbasin": ["geom", "bbox", "xy"],  # FIXME remove interbasin & xy combi?
        "outlet": ["geom", "bbox"],  # deprecated!
        "geom": ["geom"],
        "bbox": ["bbox"],
        "grid": ["RasterDataArray"],
    }
    kind = next(iter(kwargs))  # first key of region
    value0 = kwargs.pop(kind)
    if kind in ENTRYPOINTS:
        model_class = model_plugins.load(ENTRYPOINTS[kind], logger=logger)
        kwargs = dict(mod=model_class(root=value0, mode="r", logger=logger))
        kind = "model"
    elif kind == "grid":
        if isinstance(value0, (str, Path)) and isfile(value0):
            kwargs = dict(grid=open_raster(value0, **kwargs))
        elif isinstance(value0, (xr.Dataset, xr.DataArray)):
            kwargs = dict(grid=value0)
    elif kind not in options:
        k_lst = '", "'.join(list(options.keys()) + list(ENTRYPOINTS.keys()))
        raise ValueError(f'Region key "{kind}" not understood, select from "{k_lst}"')
    else:
        kwarg = _parse_region_value(value0)
        if len(kwarg) == 0 or next(iter(kwarg)) not in options[kind]:
            v_lst = '", "'.join(list(options[kind]))
            raise ValueError(
                f'Region value "{value0}" for kind={kind} not understood, '
                f'provide one of "{v_lst}"'
            )
        kwargs.update(kwarg)
    kwargs_str = dict()
    for k, v in kwargs.items():
        if isinstance(v, gpd.GeoDataFrame):
            v = f"GeoDataFrame {v.total_bounds} (crs = {v.crs})"
        elif isinstance(v, xr.DataArray):
            v = f"DataArray {v.raster.bounds} (crs = {v.raster.crs})"
        kwargs_str.update({k: v})
    logger.debug(f"Parsed region (kind={kind}): {str(kwargs_str)}")
    return kind, kwargs


def _parse_region_value(value):
    kwarg = {}
    if isinstance(value, np.ndarray):
        value = value.tolist()  # array to list
    if isinstance(value, list):
        if np.all([isinstance(p0, int) and abs(p0) > 180 for p0 in value]):  # all int
            kwarg = dict(basid=value)
        elif len(value) == 4:  # 4 floats
            kwarg = dict(bbox=value)
        elif len(value) == 2:  # 2 floats
            kwarg = dict(xy=value)
    elif isinstance(value, tuple) and len(value) == 2:  # tuple of x and y coords
        kwarg = dict(xy=value)
    elif isinstance(value, int):  # single int
        kwarg = dict(basid=value)
    elif isinstance(value, str) and isfile(value):
        kwarg = dict(geom=gpd.read_file(value))
    elif isinstance(value, gpd.GeoDataFrame):  # geometry
        kwarg = dict(geom=value)
    elif isinstance(value, str) and isdir(value):
        kwarg = dict(root=value)
    if "geom" in kwarg and np.all(kwarg["geom"].geometry.type == "Point"):
        xy = (
            kwarg["geom"].geometry.x.values,
            kwarg["geom"].geometry.y.values,
        )
        kwarg = dict(xy=xy)
    return kwarg


def _check_size(ds, logger=logger, threshold=12e3**2):
    # warning for large domain
    if (
        np.multiply(*ds.raster.shape) > threshold
    ):  # 12e3 ** 2 > 10x10 degree at 3 arcsec
        logger.warning(
            "Loading very large spatial domain to derive a subbasin. "
            "Provide initial 'bounds' if this takes too long."
        )


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
    logger=logger,
    buffer=10,
    **stream_kwargs,
):
    """Returns a geometry of the (sub)(inter)basin(s).

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

    Returns
    -------
    basin_geom : geopandas.geoDataFrame
        geometry the (sub)basin(s)
    outlet_geom : geopandas.geoDataFrame
        geometry the outlet point location
    """
    kind_lst = ["basin", "subbasin", "interbasin"]
    if kind == "outlet":
        outlets = True
        kind = "basin"
        logger.warning(
            'kind="outlets" has been deprecated, use outlets=True in combination with '
            ' kind="basin" or kind="interbasin" instead.',
            DeprecationWarning,
        )
    elif kind not in kind_lst:
        msg = f"Unknown kind: {kind}, select from {kind_lst}."
        raise ValueError(msg)
    if bool(stream_kwargs.pop("within", False)):
        logger.warning(
            '"within" stream argument has been deprecated.', DeprecationWarning
        )

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
        if isinstance(basin_index, GeoDataFrameAdapter):
            kwargs = dict(variables=["basid"])
            if geom is not None:
                kwargs.update(geom=geom)
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
            gdf_bas = basin_index.get_data(**kwargs)
        elif isinstance(basin_index, gpd.GeoDataFrame):
            gdf_bas = basin_index
            if "basid" not in gdf_bas.columns:
                raise ValueError("Basin geometries does not have 'basid' column.")
        if gdf_bas.crs != ds.raster.crs:
            logger.warn("Basin geometries CRS does not match the input raster CRS.")
            gdf_bas = gdf_bas.to_crs(ds.raster.crs)

    ## BASINS
    if kind == "basin" or bounds is None:
        dvars = dvars + [basins_name]
        if basins_name not in ds:
            if gdf_bas is not None:
                gdf_bas = None
                logger.warn(
                    "Basin geometries ignored as no corresponding basin map is provided."
                )
            _check_size(ds, logger)
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
            logger.debug(f"Getting basin IDs at point locations.")
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
            logger.debug(f"Getting IDs of intersecting basins.")
            basid = np.unique(ds_clip[basins_name].values)
        basid = np.atleast_1d(basid)
        basid = basid[basid > 0]
        if basid.size == 0:
            raise ValueError(f"No basins found with given criteria.")
        # clip ds to total basin
        if gdf_bas is not None:
            gdf_match = np.isin(gdf_bas["basid"], basid)
            gdf_bas = gdf_bas.loc[gdf_match]
            if gdf_bas.index.size > 0:
                if geom is not None:
                    xminbas, yminbas, xmaxbas, ymaxbas = gdf_bas.total_bounds
                    # Check that total_bounds is at least bigger than original geom bounds
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
                logger.warn("No matching basin IDs found in basin geometries.")
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
        _check_size(ds, logger)  # warning for large domain
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
            # stream = stream.where(aoi, False)
        else:
            aoi = xr.DataArray(
                coords=ds.raster.coords,
                dims=ds.raster.dims,
                data=np.full(ds.raster.shape, True, dtype=bool),
            )  # all True
        # get stream mask (always over entire domain to include cells downstream of aoi!)
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
                raise ValueError(f"No outlets found with with given criteria.")
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
