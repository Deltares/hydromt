"""Caching mechanisms used in HydroMT."""

import logging
import os
import shutil
from ast import literal_eval
from os.path import basename, dirname, isdir, isfile, join
from typing import Optional

import geopandas as gpd
import numpy as np
import requests
from affine import Affine
from pyproj import CRS

from hydromt._utils.uris import _is_valid_url
from hydromt.config import SETTINGS

logger = logging.getLogger(__name__)


__all__ = ["_copyfile", "_cache_vrt_tiles"]


def _copyfile(src, dst, chunk_size=1024):
    """Copy src file to dst. This method supports both online and local files."""
    if not isdir(dirname(dst)):
        os.makedirs(dirname(dst))
    if _is_valid_url(str(src)):
        with requests.get(src, stream=True) as r:
            if r.status_code != 200:
                raise ConnectionError(
                    f"Data download failed with status code {r.status_code}"
                )
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
    else:
        shutil.copyfile(src, dst)


def _cache_vrt_tiles(
    vrt_path: str,
    geom: Optional[gpd.GeoSeries] = None,
    cache_dir: str = SETTINGS.cache_root,
) -> str:
    """Cache vrt tiles that intersect with geom.

    Note that the vrt file must contain relative tile paths.

    Parameters
    ----------
    vrt_path: str, Path
        path to source vrt
    geom: geopandas.GeoSeries, optional
        geometry to intersect tiles with
    cache_dir: str, Path
        path of the root folder where

    Returns
    -------
    vrt_destination_path : str
        path to cached vrt
    """
    import xmltodict as xd

    # cache vrt file
    vrt_root = dirname(vrt_path)
    vrt_destination_path = join(cache_dir, basename(vrt_path))
    if not isfile(vrt_destination_path):
        _copyfile(vrt_path, vrt_destination_path)
    # read vrt file
    # TODO check if this is the optimal xml parser
    with open(vrt_destination_path, "r") as f:
        ds = xd.parse(f.read())["VRTDataset"]

    def intersects(source: dict, affine, bbox):
        """Check whether source interesects with bbox."""
        names = ["@xOff", "@yOff", "@xSize", "@ySize"]
        x0, y0, dx, dy = [float(source["DstRect"][k]) for k in names]
        xs, ys = affine * (np.array([x0, x0 + dx]), np.array([y0, y0 + dy]))
        return (
            max(xs) < bbox[0]
            or max(ys) < bbox[1]
            or min(xs) > bbox[2]
            or min(ys) > bbox[3]
        )

    # get vrt transform and crs
    transform = Affine.from_gdal(*literal_eval(ds["GeoTransform"]))
    srs = ds["SRS"]["#text"] if isinstance(ds["SRS"], dict) else ds["SRS"]
    crs = CRS.from_string(srs)
    # get geometry bbox in vrt crs
    if geom is not None:
        if crs != geom.crs:
            geom = geom.to_crs(crs)
        bbox = geom.total_bounds
    # support multiple type of sources in vrt
    sname = [k for k in ds["VRTRasterBand"].keys() if k.endswith("Source")][0]
    # loop through files in VRT and check if in bbox
    paths = []
    for source in ds["VRTRasterBand"][sname]:
        if geom is None or intersects(source, transform, bbox):
            path = source["SourceFilename"]["#text"]
            dst = os.path.join(cache_dir, path)
            src = os.path.join(vrt_root, path)
            if not isfile(dst):
                paths.append((src, dst))
    # TODO multi thread download
    logger.info(f"Downloading {len(paths)} tiles to {cache_dir}")
    for src, dst in paths:
        _copyfile(src, dst)
    return vrt_destination_path
