"""Utility functions for Model and ModelComponent class."""
import glob
import os
from logging import Logger
from os.path import basename, dirname, isdir, join
from tempfile import TemporaryDirectory
from typing import Dict

import xarray as xr

from hydromt._typing.type_def import DeferedFileClose, StrPath, XArrayDict
from hydromt.gis.raster import GEO_MAP_COORD


def read_nc(
    fn: StrPath,
    root,
    logger: Logger,
    mask_and_scale: bool = False,
    single_var_as_array: bool = True,
    load: bool = False,
    **kwargs,
) -> Dict[str, xr.Dataset]:
    """Read netcdf files at <root>/<fn> and return as dict of xarray.Dataset.

    NOTE: Unless `single_var_as_array` is set to False a single-variable data source
    will be returned as :py:class:`xarray.DataArray` rather than
    :py:class:`xarray.Dataset`.
    key-word arguments are passed to :py:func:`xarray.open_dataset`.

    Parameters
    ----------
    fn : str
        filename relative to model root, may contain wildcards
    mask_and_scale : bool, optional
        If True, replace array values equal to _FillValue with NA and scale values
        according to the formula original_values * scale_factor + add_offset, where
        _FillValue, scale_factor and add_offset are taken from variable attributes
        (if they exist).
    single_var_as_array : bool, optional
        If True, return a DataArray if the dataset consists of a single variable.
        If False, always return a Dataset. By default True.
    load : bool, optional
        If True, the data is loaded into memory. By default False.
    **kwargs:
        Additional keyword arguments that are passed to the `xr.open_dataset`
        function.

    Returns
    -------
    Dict[str, xr.Dataset]
        dict of xarray.Dataset
    """
    ncs = dict()
    fns = glob.glob(join(root, fn))
    if "chunks" not in kwargs:  # read lazy by default
        kwargs.update(chunks="auto")
    for fn in fns:
        name = basename(fn).split(".")[0]
        logger.debug(f"Reading model file {name}.")
        # Load data to allow overwritting in r+ mode
        if load:
            ds = xr.open_dataset(fn, mask_and_scale=mask_and_scale, **kwargs).load()
            ds.close()
        else:
            ds = xr.open_dataset(fn, mask_and_scale=mask_and_scale, **kwargs)
        # set geo coord if present as coordinate of dataset
        if GEO_MAP_COORD in ds.data_vars:
            ds = ds.set_coords(GEO_MAP_COORD)
        # single-variable Dataset to DataArray
        if single_var_as_array and len(ds.data_vars) == 1:
            (ds,) = ds.data_vars.values()
        ncs.update({name: ds})
    return ncs


def write_nc(
    nc_dict: XArrayDict,
    fn: str,
    root,
    logger: Logger,
    temp_data_dir=None,
    gdal_compliant: bool = False,
    rename_dims: bool = False,
    force_sn: bool = False,
    **kwargs,
) -> DeferedFileClose | None:
    """Write dictionnary of xarray.Dataset and/or xarray.DataArray to netcdf files.

    Possibility to update the xarray objects attributes to get GDAL compliant NetCDF
    files, using :py:meth:`~hydromt.raster.gdal_compliant`.
    The function will first try to directly write to file. In case of
    PermissionError, it will first write a temporary file and add to the
    self._defered_file_closes attribute. Renaming and closing of netcdf filehandles
    will be done by calling the self._cleanup function.

    key-word arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`

    Parameters
    ----------
    nc_dict: dict
        Dictionary of xarray.Dataset and/or xarray.DataArray to write
    fn: str
        filename relative to model root and should contain a {name} placeholder
    gdal_compliant: bool, optional
        If True, convert xarray.Dataset and/or xarray.DataArray to gdal compliant
        format using :py:meth:`~hydromt.raster.gdal_compliant`
    rename_dims: bool, optional
        If True, rename x_dim and y_dim to standard names depending on the CRS
        (x/y for projected and lat/lon for geographic). Only used if
        ``gdal_compliant`` is set to True. By default, False.
    force_sn: bool, optional
        If True, forces the dataset to have South -> North orientation. Only used
        if ``gdal_compliant`` is set to True. By default, False.
    **kwargs:
        Additional keyword arguments that are passed to the `to_netcdf`
        function.
    """
    for name, ds in nc_dict.items():
        if not isinstance(ds, (xr.Dataset, xr.DataArray)) or len(ds) == 0:
            logger.error(f"{name} object of type {type(ds).__name__} not recognized")
            continue
        logger.debug(f"Writing file {fn.format(name=name)}")
        _fn = join(root, fn.format(name=name))
        if not isdir(dirname(_fn)):
            os.makedirs(dirname(_fn))
        if gdal_compliant:
            ds = ds.raster.gdal_compliant(rename_dims=rename_dims, force_sn=force_sn)
        try:
            ds.to_netcdf(_fn, **kwargs)
        except PermissionError:
            logger.warning(f"Could not write to file {_fn}, defering write")
            if temp_data_dir is None:
                temp_data_dir = TemporaryDirectory()

            tmp_fn = join(str(temp_data_dir), f"{_fn}.tmp")
            ds.to_netcdf(tmp_fn, **kwargs)

            return DeferedFileClose(
                ds=ds,
                org_fn=join(str(temp_data_dir), _fn),
                tmp_fn=tmp_fn,
                close_attempts=1,
            )
