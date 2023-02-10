"""Test hydromt.forcing submodule"""

import pytest
from hydromt.data_catalog import DataCatalog
from hydromt.raster import full_from_transform
from hydromt.workflows.forcing import precip, pet
import numpy as np
import xarray as xr
import hydromt._compat as compat


def test_precip():
    cat = DataCatalog()
    p_precip = cat.get_rasterdataset("era5", variables="precip")  # era hourly

    # We create a more refined grid
    p_transform = p_precip.raster.transform
    p_shape = (21, 18)
    grid = full_from_transform(transform=p_transform, shape=p_shape, crs=4326)

    # Testing with default values
    pout = precip(p_precip, grid)

    assert pout.raster.shape == grid.raster.shape  # shape
    xr.testing.assert_equal(pout.x, grid.x)

    # Testing with clim argument
    p_clim = cat.get_rasterdataset("worldclim")
    # give it a nodata value in the datacatalog >> issue to create for the data artifacts
    p_clim.raster.set_nodata(-999.0)

    pout_clim = precip(p_precip, grid, clim=p_clim)
    # the values have changed. Could check if the value itself is correct
    assert pout_clim.values != p_clim.values

    # Testing with freq argument
    pout_freq = precip(p_precip, grid, freq="H")
    assert pout_freq.sizes["time"] == 313


@pytest.mark.skipif(not compat.HAS_PYET, reason="pyET not installed.")
def test_pet():
    cat = DataCatalog()
    et_data = cat.get_rasterdataset("era5_daily_zarr")
    dem = cat.get_rasterdataset("era5_orography").squeeze("time").drop("time")

    peto = pet(et_data, et_data, dem, method="penman-monteith_tdew")

    assert peto.raster.shape == dem.raster.shape
    np.testing.assert_almost_equal(peto.mean(), 0.57746, decimal=4)
