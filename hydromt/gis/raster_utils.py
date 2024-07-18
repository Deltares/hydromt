#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GIS related raster functions."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import xarray as xr
from pyflwdir import gis_utils as gis
from rasterio.transform import Affine

__all__ = [
    "affine_to_coords",
    "affine_to_meshgrid",
    "cellarea",
    "cellres",
    "meridian_offset",
    "reggrid_area",
    "spread2d",
]

logger = logging.getLogger(__name__)

_R = 6371e3  # Radius of earth in m. Use 3956e3 for miles

# TRANSFORM


def affine_to_coords(transform, shape, x_dim="x", y_dim="y"):
    """Return a raster axis with pixel center coordinates based on the transform.

    Parameters
    ----------
    transform : affine transform
        Two dimensional affine transform for 2D linear mapping
    shape : tuple of int
        The height, width  of the raster.
    x_dim, y_dim: str
        The name of the x and y dimensions

    Returns
    -------
    x, y coordinate arrays : dict of tuple with dims and coords
    """
    if not isinstance(transform, Affine):
        transform = Affine(*transform)
    height, width = shape
    if np.isclose(transform.b, 0) and np.isclose(transform.d, 0):
        x_coords, _ = transform * (np.arange(width) + 0.5, np.zeros(width) + 0.5)
        _, y_coords = transform * (np.zeros(height) + 0.5, np.arange(height) + 0.5)
        coords = {
            y_dim: (y_dim, y_coords),
            x_dim: (x_dim, x_coords),
        }
    else:
        x_coords, y_coords = (
            transform
            * transform.translation(0.5, 0.5)
            * np.meshgrid(np.arange(width), np.arange(height))
        )
        coords = {
            "yc": ((y_dim, x_dim), y_coords),
            "xc": ((y_dim, x_dim), x_coords),
        }
    return coords


def affine_to_meshgrid(transform, shape):
    """Return a meshgrid of pixel center coordinates based on the transform.

    Parameters
    ----------
    transform : affine transform
        Two dimensional affine transform for 2D linear mapping
    shape : tuple of int
        The height, width  of the raster.

    Returns
    -------
    x_coords, y_coords: ndarray
        2D arrays of x and y coordinates
    """
    if not isinstance(transform, Affine):
        transform = Affine(*transform)
    height, width = shape
    x_coords, y_coords = (
        transform
        * transform.translation(0.5, 0.5)
        * np.meshgrid(np.arange(width), np.arange(height))
    )
    return x_coords, y_coords


def meridian_offset(ds, bbox=None):
    """Shift data along the x-axis of global datasets to avoid issues along the 180 meridian.

    Without a bbox the data is shifted to span 180W to 180E.
    With bbox the data is shifted to at least span the bbox west to east,
    also if the bbox crosses the 180 meridian.

    Note that this method is only applicable to data that spans 360 degrees longitude
    and is set in a global geographic CRS (WGS84).

    Parameters
    ----------
    ds: xarray.Dataset
        input dataset
    bbox: tuple of float
        bounding box (west, south, east, north) in degrees

    Returns
    -------
    ds: xarray.Dataset
        dataset with x dim re-arranged if needed
    """
    w, _, e, _ = ds.raster.bounds
    if (
        ds.raster.crs is None
        or ds.raster.crs.is_projected
        or not np.isclose(e - w, 360)  # grid should span 360 degrees!
    ):
        raise ValueError(
            "This method is only applicable to data that spans 360 degrees "
            "longitude and is set in a global geographic CRS"
        )
    x_name = ds.raster.x_dim
    lons = np.copy(ds[x_name].values)
    if bbox is not None:  # bbox west and east
        bbox_w, bbox_e = bbox[0], bbox[2]
    else:  # global west and east in case of no bbox
        bbox_w, bbox_e = -180, 180
    if bbox_w < w:  # shift lons east of x0 by 360 degrees west
        x0 = 180 if bbox_w >= -180 else 0
        lons = np.where(lons > max(bbox_e, x0), lons - 360, lons)
    elif bbox_e > e:  # shift lons west of x0 by 360 degrees east
        x0 = -180 if bbox_e <= 180 else 0
        lons = np.where(lons < min(bbox_w, x0), lons + 360, lons)
    else:
        return ds
    ds = ds.copy(deep=False)  # make sure not to overwrite original ds
    ds[x_name] = xr.Variable(ds[x_name].dims, lons)
    return ds.sortby(x_name)


## CELLAREAS
def reggrid_area(lats, lons):
    """Return the cell area [m2] for a regular grid based on its cell centres lat, lon."""  # noqa: E501
    xres = np.abs(np.mean(np.diff(lons)))
    yres = np.abs(np.mean(np.diff(lats)))
    area = np.ones((lats.size, lons.size), dtype=lats.dtype)
    return cellarea(lats, xres, yres)[:, None] * area


def cellarea(lat, xres=1.0, yres=1.0):
    """Return the area [m2] of cell based on its center latitude and resolution in degrees.

    Resolution is in measured degrees.
    """  # noqa: E501
    l1 = np.radians(lat - np.abs(yres) / 2.0)
    l2 = np.radians(lat + np.abs(yres) / 2.0)
    dx = np.radians(np.abs(xres))
    return _R**2 * dx * (np.sin(l2) - np.sin(l1))


def cellres(lat, xres=1.0, yres=1.0):
    """Return the cell (x, y) resolution [m].

    Based on cell center latitude and its resolution measured in degrees.
    """
    m1 = 111132.92  # latitude calculation term 1
    m2 = -559.82  # latitude calculation term 2
    m3 = 1.175  # latitude calculation term 3
    m4 = -0.0023  # latitude calculation term 4
    p1 = 111412.84  # longitude calculation term 1
    p2 = -93.5  # longitude calculation term 2
    p3 = 0.118  # longitude calculation term 3

    radlat = np.radians(lat)  # numpy cos work in radians!
    # Calculate the length of a degree of latitude and longitude in meters
    dy = (
        m1
        + (m2 * np.cos(2.0 * radlat))
        + (m3 * np.cos(4.0 * radlat))
        + (m4 * np.cos(6.0 * radlat))
    )
    dx = (
        (p1 * np.cos(radlat))
        + (p2 * np.cos(3.0 * radlat))
        + (p3 * np.cos(5.0 * radlat))
    )

    return dx * xres, dy * yres


## SPREAD


def spread2d(
    da_obs: xr.DataArray,
    da_mask: Optional[xr.DataArray] = None,
    da_friction: Optional[xr.DataArray] = None,
    nodata: Optional[float] = None,
) -> xr.Dataset:
    """Return values of `da_obs` spreaded to cells with `nodata` value within `da_mask`.

    powered by :py:meth:`pyflwdir.gis_utils.spread2d`.

    Parameters
    ----------
    da_obs : xarray.DataArray
        Input raster with observation values and background/nodata values which are
        filled by the spreading algorithm.
    da_mask :  xarray.DataArray, optional
        Mask of cells to fill with the spreading algorithm, by default None
    da_friction :  xarray.DataArray, optional
        Friction values used by the spreading algorithm to calcuate the friction
        distance, by default None
    nodata : float, optional
        Nodata or background value. Must be finite numeric value. If not given the
        raster nodata value is used.

    Returns
    -------
    ds_out: xarray.Dataset
        Dataset with spreaded source values, linear index of the source cell
        "source_idx" and friction distance to the source cell "source_dst".
    """
    nodata = da_obs.raster.nodata if nodata is None else nodata
    if nodata is None or np.isnan(nodata):
        raise ValueError(f'"nodata" must be a finite value, not {nodata}')
    msk, frc = None, None
    if da_mask is not None:
        assert da_obs.raster.identical_grid(da_mask)
        msk = da_mask.values
    if da_friction is not None:
        assert da_obs.raster.identical_grid(da_friction)
        frc = da_friction.values
    out, src, dst = gis.spread2d(
        obs=da_obs.values,
        msk=msk,
        frc=frc,
        nodata=nodata,
        latlon=da_obs.raster.crs.is_geographic,
        transform=da_obs.raster.transform,
    )
    # combine outputs and return as dataset
    dims = da_obs.raster.dims
    coords = da_obs.raster.coords
    name = da_obs.name if da_obs.name else "source_value"
    da_out = xr.DataArray(dims=dims, coords=coords, data=out, name=name)
    da_out.raster.attrs.update(**da_obs.attrs)  # keep attrs incl nodata and unit
    da_src = xr.DataArray(dims=dims, coords=coords, data=src, name="source_idx")
    da_src.raster.set_nodata(-1)
    da_dst = xr.DataArray(dims=dims, coords=coords, data=dst, name="source_dst")
    da_dst.raster.set_nodata(-1)
    da_dst.attrs.update(unit="m")
    ds_out = xr.merge([da_out, da_src, da_dst])
    ds_out.raster.set_crs(da_obs.raster.crs)
    return ds_out
