"""Caching mechanisms used in HydroMT."""

import logging
import os
import xml.etree.ElementTree as ET
from ast import literal_eval
from os.path import basename, dirname, isabs, isdir, isfile, join
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
from affine import Affine
from fsspec import AbstractFileSystem, url_to_fs
from pyproj import CRS

from hydromt._typing.type_def import StrPath
from hydromt._utils.uris import _strip_scheme
from hydromt.config import SETTINGS

logger = logging.getLogger(__name__)


__all__ = ["_copy_to_local", "_cache_vrt_tiles"]


def _copy_to_local(
    src: str, dst: Path, fs: Optional[AbstractFileSystem] = None, block_size: int = 1024
):
    """Copy files from source uri to local file.

    Parameters
    ----------
    src : str
        Source URI
    dst : Path
        Destination path
    fs : Optional[AbstractFileSystem], optional
        Fsspec filesystem. Will be inferred from src if not supplied.
    block_size: int, optional
        Block size of blocks sent over wire, by default 1024
    """
    if fs is None:
        fs: AbstractFileSystem = url_to_fs(src)[0]
    if not isdir(dirname(dst)):
        os.makedirs(dirname(dst), exists_ok=True)

    fs.get(src, str(dst), block_size=block_size)


def _overlaps(source: ET.Element, affine: Affine, bbox: List[float]) -> bool:
    """Check whether source overlaps with bbox.

    Parameters
    ----------
    source : ET.Element
        Source element in .vrt
    affine : Affine
        Tile affine
    bbox : List[float]
        requested bbox

    Returns
    -------
    bool
        whether the tile overlaps with the requested bbox.

    Raises
    ------
    ValueError
        If tile bbox of tile not available as 'DstRect' sub element.
    """
    names = ["xOff", "yOff", "xSize", "ySize"]
    dest_rectangle: Optional[ET.Element] = source.find("DstRect")
    if dest_rectangle is None:
        raise ValueError("Tile metadata missing.")
    x0, y0, dx, dy = [float(dest_rectangle.get(name)) for name in names]

    xs, ys = affine * (np.array([x0, x0 + dx]), np.array([y0, y0 + dy]))

    return (
        max(xs) > bbox[0]
        and max(ys) > bbox[1]
        and min(xs) < bbox[2]
        and min(ys) < bbox[3]
    )


def _cache_vrt_tiles(
    vrt_uri: str,
    fs: Optional[AbstractFileSystem] = None,
    geom: Optional[gpd.GeoSeries] = None,
    cache_dir: StrPath = SETTINGS.cache_root,
) -> str:
    """Cache vrt tiles that intersect with geom.

    Note that the vrt file must contain relative tile paths.

    Parameters
    ----------
    vrt_uri: str
        path to source vrt
    fs : Optional[AbstractFileSystem], optional
        Fsspec filesystem. Will be inferred from src if not supplied.
    geom: geopandas.GeoSeries, optional
        geometry to intersect tiles with
    cache_dir: str, Path
        path of the root folder where

    Returns
    -------
    vrt_destination_path : str
        path to cached vrt
    """
    if fs is None:
        fs: AbstractFileSystem = url_to_fs(vrt_uri)[0]

    # cache vrt file
    vrt_root = dirname(vrt_uri)
    vrt_destination_path = join(cache_dir, basename(vrt_uri))
    if not isfile(vrt_destination_path):
        _copy_to_local(vrt_uri, vrt_destination_path, fs)
    # read vrt file
    with open(vrt_destination_path, "r") as f:
        root = ET.fromstring(f.read())

    # get vrt transform and crs
    geotransform_el: Optional[ET.Element] = root.find("GeoTransform")
    if geotransform_el is None:
        raise ValueError(f"No GeoTransform found in: {vrt_uri}")
    transform = Affine.from_gdal(*literal_eval(geotransform_el.text.strip()))

    srs_el: Optional[ET.Element] = root.find("SRS")
    if srs_el is None:
        raise ValueError(f"No SRS info found at: {vrt_uri}")

    crs: CRS = CRS.from_string(srs_el.text)
    # get geometry bbox in vrt crs
    if geom is not None:
        if crs != geom.crs:
            geom = geom.to_crs(crs)
        bbox = geom.total_bounds
    # support multiple type of sources in vrt
    band_name: Optional[ET.Element] = root.find("VRTRasterBand")
    if band_name is None:
        raise ValueError(f"Could not find VRTRasterBand in: {vrt_uri}")
    try:
        source_name: str = next(
            filter(lambda k: k.tag.endswith("Source"), band_name.iter())
        ).tag
    except StopIteration:
        raise ValueError(f"No Source information found at: {vrt_uri}")

    # loop through files in VRT and check if in bbox
    paths = []
    for source in band_name.findall(source_name):
        if geom is None or _overlaps(source, transform, bbox):
            vrt_ref_el: ET.Element = source.find("SourceFilename")
            if vrt_ref_el is None:
                raise ValueError(f"Could not find Source File in vrt: {vrt_uri}.")
            vrt_ref: str = vrt_ref_el.text

            if isabs(vrt_ref):
                # not relative uri, probably virtual file system https://gdal.org/user/virtual_file_systems.html

                # Strip scheme from base uri and get path to directory with vrt file
                scheme, stripped = _strip_scheme(vrt_uri)
                # get base path vrs
                vrs_ref: Tuple[str, ...] = Path(vrt_ref).parts[2:]
                if scheme is None:
                    # local path
                    # get directory of vrt file without vrs
                    vrt_directory: str = str(Path(stripped).parent)
                    # Get relative uri that the vrt points to
                    relative_uri_ref: str = str(Path(*vrs_ref)).lstrip(vrt_directory)
                    # Set download uri to match vrt_uri
                    src_uri: str = str(Path(*vrs_ref))
                else:
                    # protocol, assume path separator is "/"
                    sep = "/"
                    vrt_directory: str = sep.join(stripped.split(sep)[:-1])
                    # Get relative uri that the vrt points to
                    relative_uri_ref: str = sep.join(vrs_ref).lstrip(vrt_directory)
                    # Set download uri to match vrt_uri
                    src_uri: str = scheme + sep.join(vrs_ref)

                # Set download dest to cache_dir and refer to it in the vrt
                dst: Path = cache_dir / relative_uri_ref
                vrt_ref_el.text = relative_uri_ref
            else:
                src_uri: str = os.path.join(vrt_root, vrt_ref)
                dst: Path = cache_dir / vrt_ref
            if not isfile(dst):  # Skip cached files
                paths.append((src_uri, dst))

    # Write new xml file
    ET.ElementTree(root).write(vrt_destination_path)
    # TODO multi thread download
    logger.info(f"Downloading {len(paths)} tiles to {cache_dir}")
    for src, dst in paths:
        _copy_to_local(src, dst, fs)
    return vrt_destination_path
