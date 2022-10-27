"""Test hydromt.forcing submodule"""

import pytest
from hydromt.data_catalog import DataCatalog
from hydromt.workflows.forcing import pet
from hydromt.workflows.forcing import pet_debruin
from hydromt_wflow import WflowModel
import pandas

def test_pet():

    # get data
    cat = DataCatalog()
    ds = cat.get_rasterdataset("era5", variables=["temp", "press_msl", "kin", "kout"]) #era5 daily
#    da_dem = cat.get_rasterdataset("era5_orography", variables="elevtn") #era5 orography

    # resample input to model grid
    mod = WflowModel(root=r"c:\Users\marth\Projects\HydroMT\Software\wflow\hydromt_wflow\examples\wflow_piave_subbasin", mode="r") # wflow_piave_forcing
    da_wflow_dem = mod.staticmaps["wflow_dem"]
    ds_out = ds.raster.reproject_like(da_wflow_dem, method="nearest_index")

    ## check the shape
    assert [da_wflow_dem.shape[0], da_wflow_dem.shape[1]] == [ds_out["temp"].shape[1], ds_out["temp"].shape[2]]
    assert [da_wflow_dem.shape[0], da_wflow_dem.shape[1]] == [ds_out["press_msl"].shape[1], ds_out["press_msl"].shape[2]]
    assert [da_wflow_dem.shape[0], da_wflow_dem.shape[1]] == [ds_out["kin"].shape[1], ds_out["kin"].shape[2]]
    assert [da_wflow_dem.shape[0], da_wflow_dem.shape[1]] == [ds_out["kout"].shape[1], ds_out["kout"].shape[2]]

    ## check method nearest_index


    # check if press corerction is true the correction is calculated
    da_temp = ds_out["temp"]
    pet_out_press = pet(ds_out, da_temp, da_wflow_dem, press_correction=True)
    pet_out = pet(ds_out, da_temp, da_wflow_dem)

    assert pet_out_press[1, 1, 1].values != pet_out[1, 1, 1].values

    # check debruin
    pet_out_debruin = pet_debruin(da_temp[1, 1, 1], ds_out["press_msl"][1, 1, 1], ds_out["kin"][1, 1, 1], ds_out["kout"][1, 1, 1])

    assert pet_out_debruin.round(6) == 0.766257

