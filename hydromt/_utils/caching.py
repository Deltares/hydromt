"""Caching mechanisms used in HydroMT."""

import logging
import os
import xml.etree.ElementTree as ET
from ast import literal_eval
from os.path import basename, dirname, isdir, isfile, join
from pathlib import Path
from typing import List, Optional, cast

import geopandas as gpd
import numpy as np
from affine import Affine
from fsspec import AbstractFileSystem, url_to_fs
from pyproj import CRS

from hydromt._compat import HAS_GDAL
from hydromt._typing.type_def import StrPath
from hydromt._utils.uris import _strip_scheme, _strip_vsi
from hydromt.config import SETTINGS

if HAS_GDAL:
    from osgeo import gdal

    gdal.UseExceptions()

logger = logging.getLogger(__name__)

__all__ = ["_copy_to_local", "_cache_vrt_tiles"]


def _copy_to_local(src: str, dst: Path, fs: Optional[AbstractFileSystem] = None):
    """Copy files from source uri to local file.

    Parameters
    ----------
    src : str
        Source URI
    dst : Path
        Destination path
    fs : Optional[AbstractFileSystem], optional
        Fsspec filesystem. Will be inferred from src if not supplied.
    """
    if fs is None:
        fs = cast(AbstractFileSystem, url_to_fs(src)[0])
    if not isdir(dirname(dst)):
        os.makedirs(dirname(dst), exist_ok=True)

    fs.get(src, str(dst))


def _overlaps(source: ET.Element, affine: Affine, bbox: List[float]) -> bool:
    """Check whether source overlaps with bbox.

    Parameters
    ----------
    source : ET.Element
        Source element in .vrt
    affine : Affine
        Tile affine
    bbox : List[float]
        Requested bbox

    Returns
    -------
    bool
        Whether the tile overlaps with the requested bbox.

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
) -> Path:
    """Cache vrt tiles that intersect with geom.

    Note that the vrt file must contain relative tile paths.

    Parameters
    ----------
    vrt_uri: str
        Path to source vrt
    fs : Optional[AbstractFileSystem], optional
        Fsspec filesystem. Will be inferred from src if not supplied.
    geom: geopandas.GeoSeries, optional
        Geometry to intersect tiles with
    cache_dir: str, Path
        Path of the root folder where

    Returns
    -------
    vrt_destination_path : Path
        Path to cached vrt
    """
    if not HAS_GDAL:
        raise ImportError("Can't cache vrt's without GDAL installed.")

    # Get the filesystem type
    if fs is None:
        fs: AbstractFileSystem = url_to_fs(vrt_uri)[0]

    # cache vrt file
    vrt_root: str = fs._parent(vrt_uri)

    # Strip scheme from base uri and get path to directory with vrt file
    scheme, stripped = _strip_scheme(vrt_uri)

    # read vrt file
    root = ET.fromstring(fs.read_text(vrt_uri))

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
            vrt_ref_el: Optional[ET.Element] = source.find("SourceFilename")
            if vrt_ref_el is None:
                raise ValueError(f"Could not find Source File in vrt: {vrt_uri}.")
            vrt_ref: str = vrt_ref_el.text.replace("\\", os.sep)  # dewindowsify
            vrt_relative: int = int(vrt_ref_el.get("relativeToVRT"))  # 0 or 1

            if not vrt_relative:
                if vrt_ref.startswith("/vsi"):
                    # virtual file system https://gdal.org/user/virtual_file_systems.html
                    # also starts with /vsi on windows

                    # get base path vrs, can safely strip /vsi<type>/
                    _, vrt_ref = _strip_vsi(vrt_ref)

                if scheme is None:
                    # local path
                    # Get relative uri that the vrt points to
                    relative_uri_ref: Path = Path(
                        os.path.relpath(vrt_ref, dirname(stripped))
                    )
                    # Set download uri to match vrt_uri
                    src_uri: str = vrt_ref
                else:
                    # matching fs sep
                    sep = fs.sep
                    vrt_directory: str = sep.join(stripped.split(sep)[:-1])
                    # Get relative uri that the vrt points to
                    relative_uri_ref: str = vrt_ref.lstrip(vrt_directory)
                    # Set download uri to match vrt_uri
                    src_uri: str = scheme + vrt_ref

                # Set download dest to cache_dir and refer to it in the vrt
                dst: Path = cache_dir / relative_uri_ref
                vrt_ref_el.text = relative_uri_ref
            else:
                if scheme is None:
                    # local uri
                    src_uri: str = join(vrt_root, vrt_ref)
                else:
                    # fsspec uri
                    src_uri: str = fs.sep.join((vrt_root), vrt_ref)

                dst: Path = cache_dir / vrt_ref
            if not isfile(dst):  # Skip cached files
                paths.append((src_uri, dst))

    # Define the output vrt
    vrt_destination_path = Path(cache_dir, basename(vrt_uri))

    # Copy the new files
    logger.info(f"Downloading {len(paths)} tiles to {cache_dir}")
    for src, dst in paths:
        _copy_to_local(src, dst, fs)

    # Check for the newly cached files
    new = [item[1] for item in paths]
    if len(new) == 0:
        if vrt_destination_path.is_file():
            return vrt_destination_path
        else:
            return Path(vrt_uri)  # The original and try to work with that

    # Second check for existing vrt and add the files to list to build the new vrt
    if vrt_destination_path.is_file():
        with open(vrt_destination_path, "r") as f:
            root = ET.fromstring(f.read())
        cur = [
            Path(vrt_destination_path.parent, item.find("SourceFilename").text)
            for item in root.find("VRTRasterBand").findall(source_name)
        ]
        # This shouldnt be possible, but just to be sure.
        cur = [item for item in cur if item not in new]
        # Add the current files in the vrt to the list and unlink the current vrt
        new = new + cur
        os.unlink(vrt_destination_path)

    # Build the vrt with gdal
    out_ds = gdal.BuildVRT(
        destName=vrt_destination_path.as_posix(),
        srcDSOrSrcDSTab=new,
    )

    # Close and dereference the gdal dataset
    out_ds.Close()
    out_ds = None

    return vrt_destination_path
