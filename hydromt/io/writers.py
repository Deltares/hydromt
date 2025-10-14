"""Implementations for all of the necessary IO writing for HydroMT."""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import numpy as np
import xarray as xr
from tomli_w import dump as dump_toml
from yaml import dump as dump_yaml

from hydromt.typing.deferred_file_close import DeferredFileClose

logger = logging.getLogger(__name__)

__all__ = [
    "write_nc",
    "write_region",
    "write_toml",
    "write_xy",
    "write_yaml",
]


def write_yaml(path: str | Path, data: dict[str, Any]):
    """Write a dictionary to a yaml formatted file."""
    with open(path, "w") as f:
        dump_yaml(data, f)


def write_toml(path: str | Path, data: dict[str, Any]):
    """Write a dictionary to a toml formatted file."""
    with open(path, "wb") as f:
        dump_toml(data, f)


def write_xy(path: str | Path, gdf, fmt="%.4f"):
    """Write geopandas.GeoDataFrame with Point geometries to point xy files.

    Parameters
    ----------
    path: str or Path
        Path to the output file.
    gdf: geopandas.GeoDataFrame
        GeoDataFrame to write to point file.
    fmt: fmt
        String formatting. By default "%.4f".
    """
    if not np.all(gdf.geometry.type == "Point"):
        raise ValueError("gdf should contain only Point geometries.")
    xy = np.stack((gdf.geometry.x.values, gdf.geometry.y.values)).T
    with open(path, "w") as f:
        np.savetxt(f, xy, fmt=fmt)


def write_nc(
    ds: xr.DataArray | xr.Dataset,
    file_path: Path,
    *,
    compress: bool = False,
    gdal_compliant: bool = False,
    rename_dims: bool = False,
    force_sn: bool = False,
    force_overwrite: bool = False,
    **kwargs,
) -> DeferredFileClose | None:
    """Write xarray.Dataset and/or xarray.DataArray to netcdf file.

    Possibility to update the xarray objects attributes to get GDAL compliant NetCDF
    files, using :py:meth:`~hydromt.raster.gdal_compliant`.
    The function will first try to directly write to file. In case of
    PermissionError, it will first write a temporary file and move the file over.

    key-word arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`

    Parameters
    ----------
    ds : xr.DataArray | xr.Dataset
        Dataset to be written to the drive
    file_path : Path
        Full path to the outgoing file
    compress : bool, optional
        Whether or not to compress the data, by default False
    gdal_compliant : bool, optional
        If True, convert xarray.Dataset and/or xarray.DataArray to gdal compliant
        format using :py:meth:`~hydromt.raster.gdal_compliant`, by default False
    rename_dims : bool, optional
        If True, rename x_dim and y_dim to standard names depending on the CRS
        (x/y for projected and lat/lon for geographic). Only used if
        ``gdal_compliant`` is set to True. By default False
    force_sn : bool, optional
        If True, forces the dataset to have South -> North orientation. Only used
        if ``gdal_compliant`` is set to True. By default False
    **kwargs : dict
        Additional keyword arguments that are passed to the `to_netcdf`
        function
    """
    # Check the typing
    if not isinstance(ds, (xr.Dataset, xr.DataArray)) or len(ds) == 0:
        logger.error(f"Dataset object of type {type(ds).__name__} not recognized")
        return None
    if isinstance(ds, xr.DataArray):
        if ds.name is None:
            ds.name = file_path.stem
        ds = ds.to_dataset()
    # Check whether the file already exists
    if file_path.exists() and not force_overwrite:
        raise IOError(f"File {file_path.as_posix()} already exists")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Focus on the encoding and set these for all dims, coords and data vars
    encoding = kwargs.pop("encoding", {})
    for var in set(ds.coords) | set(ds.data_vars):
        if var not in encoding:
            encoding[var] = {}

    # Remove the nodata val specifier for the dimensions, CF compliant that is
    for dim in ds.dims:
        ds[dim].attrs.pop("_FillValue", None)
        if dim in encoding:
            encoding[dim].update({"_FillValue": None})

    # Set compression if True
    if compress:
        for var in ds.data_vars:
            encoding[var].update({"zlib": True})
    kwargs["encoding"] = encoding

    # Make gdal compliant if True, only in case of a spatial dataset
    if gdal_compliant:
        y_old, x_old = ds.raster.dims
        ds = ds.raster.gdal_compliant(rename_dims=rename_dims, force_sn=force_sn)
        y_dim, x_dim = ds.raster.dims
        encoding[y_dim] = encoding.pop(y_old)
        encoding[x_dim] = encoding.pop(x_old)

    # Try to write the file
    try:
        ds.to_netcdf(file_path, **kwargs)
    except OSError:
        logger.debug(f"Could not write to file {file_path.as_posix()}, deferring write")

        unique_str = f"{file_path}_{uuid.uuid4()}"
        hash_str = hashlib.sha256(unique_str.encode()).hexdigest()[:8]
        temp_filepath = file_path.with_stem(f"{file_path.stem}_{hash_str}")
        ds.to_netcdf(temp_filepath, **kwargs)

        return DeferredFileClose(original_path=file_path, temp_path=temp_filepath)

    return None


def write_region(
    region: gpd.GeoDataFrame,
    file_path: Path,
    *,
    to_wgs84=False,
    **write_kwargs,
):
    """Write the model region to a file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    gdf = cast(gpd.GeoDataFrame, region.copy())

    if to_wgs84 and (
        write_kwargs.get("driver") == "GeoJSON"
        or file_path.suffix.lower() == ".geojson"
    ):
        gdf = gdf.to_crs(4326)

    gdf.to_file(file_path, **write_kwargs)
