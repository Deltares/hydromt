"""Test hydromt.forcing submodule"""

import pytest
from hydromt.data_catalog import DataCatalog
from hydromt.raster import full_from_transform
from hydromt.workflows.forcing import pet
from hydromt_wflow import WflowModel


def test_pet():

    # get data
    cat = DataCatalog()
    dat = cat.get_rasterdataset("era5", variables=["temp", "press_msl", "kin", "kout"]) #era5 hourly
#    dem = cat.get_rasterdataset("era5_orography", variables="elevtn") #era5 orography

    # we create a more refined grid - make a fixture? --> put in conftest
#    dem_transform = dem.raster.transform
#    dat_transform = dat.raster.transform
#    dem_shape = (21,18)
#    dat_shape = (21,18)
#    grid_dem = full_from_transform(transform=dem_transform, shape=dem_shape, crs = 4326)
#    grid_dat = full_from_transform(transform=dat_transform, shape=dat_shape, crs = 4326)

    # resample input to model grid
    mod = WflowModel(root=r"wflow_piave_forcing", mode="r")
    da_wflow_dem = mod.staticmaps["wflow_dem"]
    dat_out = dat.raster.reproject_like(da_wflow_dem, method="nearest_index")

    assert mod.shape == dat_out.shape  # shape


