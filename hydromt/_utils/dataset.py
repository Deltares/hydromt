import logging
from typing import TYPE_CHECKING, Dict, Optional, cast

import numpy as np
import pandas as pd
import xarray as xr

from hydromt.typing.metadata import SourceMetadata
from hydromt.typing.type_def import Variables

if TYPE_CHECKING:
    from pandas._libs.tslibs.timedeltas import TimeDeltaUnitChoices


logger = logging.getLogger(__name__)

__all__ = [
    "_set_metadata",
    "_shift_dataset_time",
    "_rename_vars",
    "_test_equal_grid_data",
    "_single_var_as_array",
]


def _shift_dataset_time(
    dt: int,
    ds: Optional[xr.Dataset],
    time_unit: "TimeDeltaUnitChoices" = "s",
) -> Optional[xr.Dataset]:
    """Shifts time of a xarray dataset.

    Parameters
    ----------
    dt : int
        time delta to shift the time of the dataset
    ds : xr.Dataset
        xarray dataset
    logger : logging.Logger
        logger

    Returns
    -------
    xr.Dataset
        time shifted dataset
    """
    if ds is None:
        return None

    if (
        dt != 0
        and "time" in ds.dims
        and ds["time"].size > 1
        and np.issubdtype(ds["time"].dtype, np.datetime64)
    ):
        logger.debug(f"Shifting time labels with {dt} {time_unit}.")
        ds["time"] = ds["time"] + pd.to_timedelta(dt, unit=time_unit)
    elif dt != 0:
        logger.warning("Time shift not applied, time dimension not found.")
    return ds


def _set_metadata(
    ds: Optional[xr.Dataset], metadata: "SourceMetadata"
) -> Optional[xr.Dataset]:
    if ds is None:
        return None
    elif metadata.attrs:
        if isinstance(ds, xr.DataArray):
            name = cast(str, ds.name)
            ds.attrs.update(metadata.attrs[name])
        else:
            for k in metadata.attrs:
                ds[k].attrs.update(metadata.attrs[k])

    # exclude extent and attrs that are dict
    ds.attrs.update(
        metadata.model_dump(exclude_unset=True, exclude={"attrs", "extent"})
    )
    return ds


def _rename_vars(
    ds: Optional[xr.Dataset], rename: Dict[str, str]
) -> Optional[xr.Dataset]:
    if ds is None:
        return None
    rm = {k: v for k, v in rename.items() if k in ds}
    ds = ds.rename(rm)
    return ds


def _single_var_as_array(
    maybe_ds: Optional[xr.Dataset],
    single_var_as_array: bool,
    variable_name: Optional[Variables] = None,
) -> Optional[xr.Dataset]:
    if isinstance(maybe_ds, xr.DataArray):
        return maybe_ds
    if maybe_ds is None:
        return None
    else:
        ds = maybe_ds
    # return data array if single variable dataset
    dvars = list(ds.data_vars.keys())
    if single_var_as_array and len(dvars) == 1:
        da = ds[dvars[0]]
        if isinstance(variable_name, list) and len(variable_name) == 1:
            da.name = variable_name[0]
        elif isinstance(variable_name, str):
            da.name = variable_name
        return da
    else:
        return ds


def _enforce_dataset_type(maybe_ds: xr.Dataset | xr.DataArray) -> xr.Dataset:
    if isinstance(maybe_ds, xr.Dataset):
        return maybe_ds
    if isinstance(maybe_ds, xr.DataArray):
        return maybe_ds.to_dataset()
    raise TypeError(f"Unsupported type: {type(maybe_ds)}")


def _test_equal_grid_data(
    grid: xr.Dataset | xr.DataArray,
    other_grid: xr.Dataset | xr.DataArray,
    *,
    skip_crs: bool = False,
) -> tuple[bool, dict[str, str]]:
    """
    Test if two grid datasets are equal.

    Checks the CRS, dimensions, and data variables for equality.

    Parameters
    ----------
    grid : xr.Dataset | xr.DataArray
        The first grid dataset to compare.
    other_grid : xr.Dataset | xr.DataArray
        The second grid dataset to compare.
    skip_crs : bool, optional
        Whether to skip the CRS check. Useful for non-geospatial grids. Default is False.

    Returns
    -------
    tuple[bool, dict[str, str]]
        True if the grids are equal, and a dict with the associated errors per property
        checked.
    """
    grid = _enforce_dataset_type(grid)
    other_grid = _enforce_dataset_type(other_grid)

    errors: dict[str, str] = {}
    invalid_maps: dict[str, str] = {}
    invalid_coords: dict[str, str] = {}

    # Check if grid is empty
    if len(grid) == 0:
        if len(other_grid) == 0:
            return True, {}
        else:
            errors["grid"] = "first grid is empty, second is not"
            return False, errors

    # Check CRS and dims
    maps = grid.raster.vars

    if not skip_crs:
        if not np.all(grid.raster.crs == other_grid.raster.crs):
            errors["crs"] = "the two grids have different crs"

    # check on dims names and values
    for dim in other_grid.dims:
        try:
            xr.testing.assert_identical(other_grid[dim], grid[dim])
        except AssertionError:
            errors["dims"] = f"dim {dim} not identical"
        except KeyError:
            errors["dims"] = f"dim {dim} not in grid"

    # Check if new maps in other grid
    new_maps = []
    for name in other_grid.raster.vars:
        if name not in maps:
            new_maps.append(name)
    if len(new_maps) > 0:
        errors["Other grid has additional maps"] = f"{', '.join(new_maps)}"

    # Check per map (dtype, value, nodata)
    missing_maps = []
    for name in maps:
        map0 = grid[name].fillna(0)
        if name not in other_grid.data_vars:
            missing_maps.append(name)
            continue
        map1 = other_grid[name].fillna(0)

        # hilariously np.nan == np.nan returns False, hence the additional check
        equal_nodata = map0.raster.nodata == map1.raster.nodata
        if not equal_nodata and (
            np.isnan(map0.raster.nodata) and np.isnan(map1.raster.nodata)
        ):
            equal_nodata = True

        if (
            not np.allclose(map0, map1, atol=1e-3, rtol=1e-3)
            or map0.dtype != map1.dtype
            or not equal_nodata
        ):
            if len(map0.dims) > 2:  # 3 dim map
                map0 = map0[0, :, :]
                map1 = map1[0, :, :]
            # Check on dtypes
            err = (
                ""
                if map0.dtype == map1.dtype
                else f"{map1.dtype} instead of {map0.dtype}"
            )
            # Check on nodata
            err = (
                err
                if equal_nodata
                else f"nodata {map1.raster.nodata} instead of {map0.raster.nodata}; {err}"
            )
            not_close = ~np.equal(map0, map1)
            n_cells = int(np.sum(not_close))
            if n_cells > 0:
                diff = (map0.values - map1.values)[not_close].mean()
                err = f"mean diff ({n_cells:d} cells): {diff:.4f}; {err}"
            invalid_maps[name] = err
    for coord in grid.coords:
        if coord in other_grid.coords:
            try:
                xr.testing.assert_identical(other_grid[coord], grid[coord])
            except AssertionError:
                invalid_coords[coord] = "not identical"
            except KeyError:
                invalid_coords[coord] = "not in grid"

    if len(invalid_coords) > 0:
        errors[f"{len(invalid_coords)} invalid coords"] = invalid_coords

    if len(missing_maps) > 0:
        errors["Other grid is missing maps"] = f"{', '.join(missing_maps)}"

    if len(invalid_maps) > 0:
        errors[f"{len(invalid_maps)} invalid maps"] = invalid_maps

    return len(errors) == 0, errors
