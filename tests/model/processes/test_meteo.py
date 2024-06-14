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
    press_correction,
    temp,
    temp_correction,
    wind,
)


@pytest.fixture()
def era5_data(data_catalog):
    return data_catalog.get_rasterdataset("era5_daily_zarr")


@pytest.fixture()
def era5_dem(data_catalog):
    return data_catalog.get_rasterdataset("era5_orography").squeeze("time", drop=True)


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


def test_temp(era5_data, era5_dem):
    t = era5_data["temp"]
    with pytest.raises(ValueError, match='First temp dim should be "time", not hours'):
        temp(t.rename({"time": "hours"}), era5_dem)

    t_resampled = temp(t, era5_dem)
    assert isinstance(t_resampled, xr.DataArray)
    assert t_resampled.name == "temp"
    assert t_resampled.attrs.get("unit") == "degree C."


def test_press(era5_data, era5_dem):
    p = era5_data["press_msl"]
    with pytest.raises(ValueError, match='First press dim should be "time", not hours'):
        press(p.rename({"time": "hours"}), era5_dem)

    p_resampled = press(p, era5_dem)
    assert isinstance(p_resampled, xr.DataArray)
    assert p_resampled.name == "press"
    assert p_resampled.attrs.get("unit") == "hPa"


def test_wind(era5_data, era5_dem):
    with pytest.raises(
        ValueError, match="Either wind or wind_u and wind_v variables must be supplied."
    ):
        wind(era5_dem)
    with pytest.raises(ValueError, match='First wind dim should be "time", not test'):
        wind(era5_dem, era5_data["u10"].rename({"time": "test"}))

    wind_out = wind(
        era5_dem, wind_u=era5_data["u10"], wind_v=era5_data["v10"], freq="h"
    )
    assert wind_out.name == "wind"
    assert wind_out.attrs.get("unit") == "m s-1"
    assert wind_out.sizes["time"] == 313


@pytest.mark.skip(
    reason="era5 in v1 artifact_data is missing variables needed for this test"
)
def test_pet(era5_data, era5_dem):
    peto = pet(era5_data, era5_data, era5_dem, method="penman-monteith_tdew")

    assert peto.raster.shape == era5_dem.raster.shape
    np.testing.assert_almost_equal(peto.mean(), 0.57746, decimal=4)


def test_press_correction(era5_dem):
    press_cor = press_correction(era5_dem)
    assert isinstance(press_cor, xr.DataArray)
    np.testing.assert_almost_equal(press_cor.mean(), 0.9219, decimal=4)


def test_temp_correction(era5_dem):
    temp_cor = temp_correction(era5_dem)
    assert isinstance(temp_cor, xr.DataArray)
    np.testing.assert_almost_equal(temp_cor.mean(), -4.5743, decimal=4)


def test_pet_debruin(data_catalog):
    era5_data = data_catalog.get_rasterdataset("era5_daily_zarr")
    pet = pet_debruin(
        era5_data["temp"], era5_data["press_msl"], era5_data["kin"], era5_data["kout"]
    )
    assert isinstance(pet, xr.DataArray)
    np.testing.assert_almost_equal(pet.mean(), 0.8540, decimal=4)


def test_pet_makkink(data_catalog):
    era5_data = data_catalog.get_rasterdataset("era5_daily_zarr")
    pet = pet_makkink(era5_data["temp"], era5_data["press_msl"], era5_data["kin"])
    assert isinstance(pet, xr.DataArray)
    np.testing.assert_almost_equal(pet.mean(), 0.7113, decimal=4)
