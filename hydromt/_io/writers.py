"""Implementations for all of the necessary IO writing for HydroMT."""

import os
from logging import Logger, getLogger
from os import makedirs
from os.path import dirname, exists, isdir, join
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Union, cast

import geopandas as gpd
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from tomli_w import dump as dump_toml
from yaml import dump as dump_yaml

from hydromt._typing.type_def import DeferedFileClose, StrPath

logger: Logger = getLogger(__name__)


def _write_yaml(path: StrPath, data: Dict[str, Any]):
    """Write a dictionary to a yaml formatted file."""
    with open(path, "w") as f:
        dump_yaml(data, f)


def _write_toml(path: StrPath, data: Dict[str, Any]):
    """Write a dictionary to a toml formatted file."""
    with open(path, "wb") as f:
        dump_toml(data, f)


def _write_xy(path, gdf, fmt="%.4f"):
    """Write geopandas.GeoDataFrame with Point geometries to point xy files.

    Parameters
    ----------
    path: str
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


def _netcdf_writer(
    obj: Union[xr.Dataset, xr.DataArray],
    data_root: Union[str, Path],
    data_name: str,
    variables: Optional[List[str]] = None,
    encoder: str = "zlib",
) -> str:
    """Utiliy function for writing a xarray dataset/data array to a netcdf file.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        Dataset.
    data_root : str | Path
        root to write the data to.
    data_name : str
        filename to write to.
    variables : Optional[List[str]]
        list of dataset variables to write, by default None

    Returns
    -------
    write_path: str
        Absolute path to output file
    """
    dvars = [obj.name] if isinstance(obj, xr.DataArray) else obj.data_vars
    if variables is None:
        encoding = {k: {encoder: True} for k in dvars}
        write_path = join(data_root, f"{data_name}.nc")
        obj.to_netcdf(write_path, encoding=encoding)
    else:  # save per variable
        if not isdir(join(data_root, data_name)):
            os.makedirs(join(data_root, data_name))
        for var in dvars:
            write_path = join(data_root, data_name, f"{var}.nc")
            obj[var].to_netcdf(write_path, encoding={var: {encoder: True}})
        write_path = join(data_root, data_name, "{variable}.nc")
    return write_path


def _zarr_writer(
    obj: Union[xr.Dataset, xr.DataArray],
    data_root: Union[str, Path],
    data_name: str,
    **kwargs,
) -> str:
    """Utiliy function for writing a xarray dataset/data array to a netcdf file.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        Dataset.
    data_root : str | Path
        root to write the data to.
    data_name : str
        filename to write to.

    Returns
    -------
    write_path: str
        Absolute path to output file
    """
    write_path = join(data_root, f"{data_name}.zarr")
    if isinstance(obj, xr.DataArray):
        obj = obj.to_dataset()
    obj.to_zarr(write_path, **kwargs)
    return write_path


def _compute_nc(
    ds: xr.Dataset,
    filepath: Path,
    compute: bool = True,
    **kwargs,
):
    """Write and compute the dataset.

    Either compute directly or with a progressbar.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to be written.
    filepath : Path
        The full path to the outgoing file.
    compute : bool, optional
        Whether to compute the output directly, by default True
    """
    obj = ds.to_netcdf(
        filepath,
        compute=compute,
        **kwargs,
    )
    if compute:
        return

    with ProgressBar():
        obj.compute()
    obj = None


def _write_nc(
    ds: xr.DataArray | xr.Dataset,
    filepath: Path | str,
    *,
    compress: bool = False,
    gdal_compliant: bool = False,
    rename_dims: bool = False,
    force_sn: bool = False,
    force_overwrite: bool = False,
    **kwargs,
) -> Optional[DeferedFileClose]:
    """Write xarray.Dataset and/or xarray.DataArray to netcdf file.

    Possibility to update the xarray objects attributes to get GDAL compliant NetCDF
    files, using :py:meth:`~hydromt.raster.gdal_compliant`.
    The function will first try to directly write to file. In case of
    PermissionError, it will first write a temporary file and add to the
    self._defered_file_closes attribute. Renaming and closing of netcdf filehandles
    will be done by calling the self._cleanup function.

    key-word arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`

    Parameters
    ----------
    ds : xr.DataArray | xr.Dataset
        Dataset to be written to the drive
    filepath : Path | str
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
    # Force typing
    filepath = Path(filepath)
    # Check the typing
    if not isinstance(ds, (xr.Dataset, xr.DataArray)) or len(ds) == 0:
        logger.error(f"Dataset object of type {type(ds).__name__} not recognized")
        return
    if isinstance(ds, xr.DataArray):
        if ds.name is None:
            ds.name = filepath.stem
        ds = ds.to_dataset()
    logger.debug(f"Writing file {filepath.as_posix()}")
    # Check whether the file already exists
    if filepath.is_file() and not force_overwrite:
        raise IOError(f"File {filepath.as_posix()} already exists")
    if not filepath.parent.is_dir():
        filepath.parent.mkdir(parents=True)

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
        _compute_nc(ds, filepath=filepath, **kwargs)
    except PermissionError:
        logger.warning(f"Could not write to file {filepath.as_posix()}, defering write")
        temp_data_dir = TemporaryDirectory()

        temp_filepath = Path(temp_data_dir.name, filepath.name)
        _compute_nc(ds, filepath=temp_filepath, **kwargs)

        return DeferedFileClose(
            ds=ds,
            original_path=filepath,
            temp_path=temp_filepath,
            close_attempts=1,
        )

    return None


def _write_region(
    region: gpd.GeoDataFrame,
    *,
    filename: StrPath,
    root_path: StrPath,
    to_wgs84=False,
    **write_kwargs,
):
    """Write the model region to a file."""
    write_path = join(root_path, filename)

    base_name = dirname(write_path)
    if not exists(base_name):
        makedirs(base_name, exist_ok=True)

    logger.info(f"writing region data to {write_path}")
    gdf = cast(gpd.GeoDataFrame, region.copy())

    if to_wgs84 and (
        write_kwargs.get("driver") == "GeoJSON"
        or str(filename).lower().endswith(".geojson")
    ):
        gdf = gdf.to_crs(4326)

    gdf.to_file(write_path, **write_kwargs)
