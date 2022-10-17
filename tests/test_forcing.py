# -*- coding: utf-8 -*-
"""Tests for the hydromt.workflows.forcing"""

import pytest
import logging

from hydromt.data_catalog import DataCatalog
from hydromt.raster import full_from_transform

from hydromt.workflows.forcing import pet

logger = logging.getLogger("test_forcing")


def test_pet():

    # get data
    cat = DataCatalog()
    dat = cat.get_rasterdataset("era5", variables=["temp", "press_msl", "kin", "kout"]) #era5 hourly
    dem = cat.get_rasterdataset("era5_orography", variables="elevtn") #era5 orography

    # resample to the ds_like -  don't have it but don't have one here so we will make a fixture --> put in conftest
    dem_transform = dem.raster.transform
    dat_transform = dat.raster.transform
    #dat_shape = dat.raster.shape
    #dem_shape = dem.raster.shape
    dem_shape = (21,18)
    dat_shape = (21,18)
    grid_dat = full_from_transform(transform = dat_transform, shape = dat_shape, crs = 4326)
    gri

    pet_out = pet(dat, dat["temp"], )

    assert pet_out.raster.shape == grid.raster.shape #shape
    assert pout.x == grid.x #test coordinates

    #test other arguments     clim=None,
    freq=None,
    reproj_method="nearest_index",
    resample_kwargs={},
    logger=logger,

    p_clim = cat.get_rasterdataset("worldclim")
    p_clim.raster.set_nodata(-999.0) #give it a nodata value in the datacatalog >> issue to create for the data artifacts

    pout_clim = precip(p_precip, grid, clim=p_clim)

    assert pout_clim.values != grid.raster.shape #shape


    #test if transform and shape stayed the same

    #test downscaling with correction (check values) or without correction (check values)
    #Test whether resampling temporally working and check on one cell for example