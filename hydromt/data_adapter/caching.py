from affine import Affine
from ast import literal_eval
import geopandas as gpd
from pathlib import Path
from pyproj import CRS
import numpy as np
import os
from os.path import isfile, dirname, isdir, join, basename
import requests
import shutil
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


HYDROMT_DATADIR = join(Path.home(), ".hydromt_data")


def _uri_validator(uri: str) -> bool:
    """Check if uri is valid"""
    try:
        result = urlparse(uri)
        return all([result.scheme, result.netloc])
    except:
        return False


def _copyfile(src, dst):
    """Copy src file to dst. This method supports both online and local files."""
    if not isdir(dirname(dst)):
        os.makedirs(dirname(dst))
    if _uri_validator(str(src)):
        with requests.get(src, stream=True) as r:
            if r.status_code != 200:
                raise ConnectionError(
                    f"Data download failed with status code {r.status_code}"
                )
            with open(dst, "wb") as f:
                shutil.copyfileobj(r.raw, f)
    else:
        shutil.copyfile(src, dst)


def cache_vrt_tiles(
    vrt_fn: str,
    geom: gpd.GeoSeries = None,
    cache_dir: str = HYDROMT_DATADIR,
    logger=logger,
) -> str:
    """Cache vrt tiles that intersect with geom.

    Note that the vrt file must contain relative tile paths.

    Parameters
    ----------
    vrt_fn: str, Path
        path to source vrt
    geom: geopandas.GeoSeries, optional
        geometry to intersect tiles with
    cache_dir: str, Path
        path of the root folder where

    Returns
    -------
    dst_vrt_fn: str
        path to cached vrt
    """
    import xmltodict as xd

    # cache vrt file
    vrt_root = dirname(vrt_fn)
    dst_vrt_fn = join(cache_dir, basename(vrt_fn))
    if not isfile(dst_vrt_fn):
        _copyfile(vrt_fn, dst_vrt_fn)
    # read vrt file
    # TODO check if this is the optimal xml parser
    with open(dst_vrt_fn, "r") as f:
        ds = xd.parse(f.read())["VRTDataset"]

    def intersects(source: dict, affine, bbox):
        """Check whether source interesects with bbox"""
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
    fns = []
    for source in ds["VRTRasterBand"][sname]:
        if geom is None or intersects(source, transform, bbox):
            fn = source["SourceFilename"]["#text"]
            dst = os.path.join(cache_dir, fn)
            src = os.path.join(vrt_root, fn)
            if not isfile(dst):
                fns.append((src, dst))
    # TODO multi thread download
    logger.info(f"Downloading {len(fns)} tiles to {cache_dir}")
    for src, dst in fns:
        _copyfile(src, dst)
    return dst_vrt_fn
