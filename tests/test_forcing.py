"""Test hydromt.forcing submodule"""

import pytest
from hydromt.data_catalog import DataCatalog
from hydromt.raster import full_from_transform
from hydromt.workflows.forcing import precip
import pandas as pd
import xarray as xr


def test_precip():
    cat = DataCatalog()
    p_precip = cat.get_rasterdataset("era5", variables="precip")  # era horuly

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
    p_clim.raster.set_nodata(
        -999.0
    )  # give it a nodata value in the datacatalog >> issue to create for the data artifacts

    pout_clim = precip(p_precip, grid, clim=p_clim)
    assert (
        pout_clim.values != p_clim.values
    )  # the values have changed. Could check if the value itself is correct

    # Testing with freq argument
    pout_freq = precip(p_precip, grid, freq="H")
    assert pout_freq.sizes["time"] == 313
