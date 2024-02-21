"""Implementations for all of the necessary IO writing for HydroMT."""
import os
from os.path import dirname, isdir, join, splitext
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import xarray as xr
import yaml

from hydromt.io.path import make_config_paths_relative


def write_xy(fn, gdf, fmt="%.4f"):
    """Write geopandas.GeoDataFrame with Point geometries to point xy files.

    Parameters
    ----------
    fn: str
        Path to the output file.
    gdf: geopandas.GeoDataFrame
        GeoDataFrame to write to point file.
    fmt: fmt
        String formatting. By default "%.4f".
    """
    if not np.all(gdf.geometry.type == "Point"):
        raise ValueError("gdf should contain only Point geometries.")
    xy = np.stack((gdf.geometry.x.values, gdf.geometry.y.values)).T
    with open(fn, "w") as f:
        np.savetxt(f, xy, fmt=fmt)


def configwrite(config_fn: Union[str, Path], cfdict: dict, **kwargs) -> None:
    """Write configuration/workflow dictionary to file.

    Parameters
    ----------
    config_fn : Union[Path, str]
        Path to configuration file
    cfdict : dict
        Configuration dictionary. If the configuration contains headers,
        the first level keys are the section headers, the second level
        option-value pairs.
    encoding : str, optional
        File encoding, by default "utf-8"
    cf : ConfigParser, optional
        Alternative configuration parser, by default None
    noheader : bool, optional
        Set true for a single-level configuration dictionary with no headers,
        by default False
    **kwargs
        Additional keyword arguments that are passed to the `write_ini_config`
        function.
    """
    root = Path(dirname(config_fn))
    _cfdict = make_config_paths_relative(cfdict.copy(), root)
    ext = splitext(config_fn)[-1].strip()
    if ext in [".yaml", ".yml"]:
        with open(config_fn, "w") as f:
            yaml.dump(_cfdict, f, sort_keys=False)
    else:
        raise ValueError(
            f"Could not write to unknown extention: {ext} hydromt only supports yaml"
        )


def netcdf_writer(
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
    fn_out: str
        Absolute path to output file
    """
    dvars = [obj.name] if isinstance(obj, xr.DataArray) else obj.data_vars
    if variables is None:
        encoding = {k: {encoder: True} for k in dvars}
        fn_out = join(data_root, f"{data_name}.nc")
        obj.to_netcdf(fn_out, encoding=encoding)
    else:  # save per variable
        if not isdir(join(data_root, data_name)):
            os.makedirs(join(data_root, data_name))
        for var in dvars:
            fn_out = join(data_root, data_name, f"{var}.nc")
            obj[var].to_netcdf(fn_out, encoding={var: {encoder: True}})
        fn_out = join(data_root, data_name, "{variable}.nc")
    return fn_out


def zarr_writer(
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
    fn_out: str
        Absolute path to output file
    """
    fn_out = join(data_root, f"{data_name}.zarr")
    if isinstance(obj, xr.DataArray):
        obj = obj.to_dataset()
    obj.to_zarr(fn_out, **kwargs)
    return fn_out
