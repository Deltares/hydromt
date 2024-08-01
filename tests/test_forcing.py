"""Test hydromt.forcing submodule."""

import numpy as np
import pytest
import xarray as xr

import hydromt._compat as compat
from hydromt.raster import full_from_transform
from hydromt.workflows.forcing import pet, precip


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
    pout_freq = precip(p_precip, grid, freq="H")
    assert pout_freq.sizes["time"] == 313


@pytest.mark.skipif(not compat.HAS_PYET, reason="pyET not installed.")
def test_pet(data_catalog):
    et_data = data_catalog.get_rasterdataset("era5_daily_zarr")
    dem = data_catalog.get_rasterdataset("era5_orography").squeeze("time", drop=True)

    peto_tdew = pet(et_data, et_data, dem, method="penman-monteith_tdew")
    assert peto_tdew.raster.shape == dem.raster.shape
    np.testing.assert_almost_equal(peto_tdew.mean(), 0.57746, decimal=4)

    # Convert to relative humidity using simple approach
    # as relative humidity is not provided in ERA5 data
    # Lawrence (2005) dx.doi.org/10.1175/BAMS-86-2-225
    et_data["rh"] = 5 * et_data["temp_dew"] + 100 - 5 * et_data["temp"]
    peto_rh = pet(et_data, et_data, dem, method="penman-monteith_rh_simple")
    assert peto_rh.raster.shape == dem.raster.shape
    np.testing.assert_almost_equal(peto_rh.mean(), 0.51279, decimal=4)
