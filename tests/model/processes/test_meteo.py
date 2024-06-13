"""Test hydromt.model.processes.meteo submodule."""

import numpy as np
import pytest
import xarray as xr

from hydromt.gis.raster import full_from_transform
from hydromt.model.processes.meteo import (
    pet,
    pet_debruin,
    pet_makkink,
    precip,
    press,
    temp,
)


def test_precip(data_catalog):
    p_precip = data_catalog.get_rasterdataset("era5", variables="precip")  # era hourly

    # We create a more refined grid
    p_transform = p_precip.raster.transform
    p_shape = (21, 18)
    grid = full_from_transform(transform=p_transform, shape=p_shape, crs=4326)

    # Testing with default values
    pout = precip(p_precip, grid)

    assert pout.raster.shape == grid.raster.shape  # shape
    xr.testing.assert_equal(pout.x, grid.x)

    # Testing with clim argument
    p_clim = data_catalog.get_rasterdataset("worldclim")
    # give it a nodata value in the datacatalog >> issue to
    # create for the data artifacts
    p_clim.raster.set_nodata(-999.0)

    pout_clim = precip(p_precip, grid, clim=p_clim)
    # the values have changed. Could check if the value itself is correct
    assert not pout_clim.equals(p_clim)

    # Testing with freq argument
    pout_freq = precip(p_precip, grid, freq="h")
    assert pout_freq.sizes["time"] == 313


def test_temp(data_catalog, demda):
    et_data = data_catalog.get_rasterdataset("era5_daily_zarr")
    t = et_data["temp"]
    with pytest.raises(ValueError, match='First temp dim should be "time", not hours'):
        temp(t.rename({"time": "hours"}), demda)

    t_resampled = temp(t, demda)
    assert isinstance(t_resampled, xr.DataArray)
    assert t_resampled.name == "temp"
    assert t_resampled.attrs.get("unit") == "degree C."


def test_press(data_catalog, demda):
    et_data = data_catalog.get_rasterdataset("era5_daily_zarr")
    p = et_data["press_msl"]
    with pytest.raises(ValueError, match='First press dim should be "time", not hours'):
        press(p.rename({"time": "hours"}), demda)

    p_resampled = press(p, demda)
    assert isinstance(p_resampled, xr.DataArray)
    assert p_resampled.name == "press"
    assert p_resampled.attrs.get("unit") == "hPa"


@pytest.mark.skip(
    reason="era5 in v1 artifact_data is missing variables needed for this test"
)
def test_pet(data_catalog):
    et_data = data_catalog.get_rasterdataset("era5_daily_zarr")
    dem = data_catalog.get_rasterdataset("era5_orography").squeeze("time", drop=True)

    peto = pet(et_data, et_data, dem, method="penman-monteith_tdew")

    assert peto.raster.shape == dem.raster.shape
    np.testing.assert_almost_equal(peto.mean(), 0.57746, decimal=4)


def test_pet_debruin(data_catalog):
    et_data = data_catalog.get_rasterdataset("era5_daily_zarr")
    pet = pet_debruin(
        et_data["temp"], et_data["press_msl"], et_data["kin"], et_data["kout"]
    )
    assert isinstance(pet, xr.DataArray)
    np.testing.assert_almost_equal(pet.mean(), 0.8540, decimal=4)


def test_pet_makkink(data_catalog):
    et_data = data_catalog.get_rasterdataset("era5_daily_zarr")
    pet = pet_makkink(et_data["temp"], et_data["press_msl"], et_data["kin"])
    assert isinstance(pet, xr.DataArray)
    np.testing.assert_almost_equal(pet.mean(), 0.7113, decimal=4)
