"""Accessors to extend the functionality of xarray structures."""

from .merge import merge
from .raster import (
    XGeoBase,
    XRasterBase,
    full,
    full_from_transform,
    full_like,
    tile_window_xyz,
)
from .utils import (
    affine_to_coords,
    affine_to_meshgrid,
    axes_attrs,
    bbox_from_file_and_filters,
    cellarea,
    cellres,
    create_vrt,
    filter_gdf,
    meridian_offset,
    nearest,
    nearest_merge,
    parse_crs,
    parse_geom_bbox_buffer,
    read_info,
    reggrid_area,
    spread2d,
    to_geographic_bbox,
    utm_crs,
)
from .vector import GeoBase

__all__ = [
    "nearest_merge",
    "nearest",
    "filter_gdf",
    "bbox_from_file_and_filters",
    "parse_geom_bbox_buffer",
    "parse_crs",
    "to_geographic_bbox",
    "utm_crs",
    "meridian_offset",
    "axes_attrs",
    "affine_to_coords",
    "affine_to_meshgrid",
    "read_info",
    "spread2d",
    "reggrid_area",
    "cellarea",
    "cellres",
    "create_vrt",
    "full_like",
    "full_from_transform",
    "full",
    "tile_window_xyz",
    "XGeoBase",
    "XRasterBase",
    "GeoBase",
    "merge",
    "spread2d",
]
