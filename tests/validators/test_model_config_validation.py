"""Tests for the cli submodule."""


import pytest

from hydromt._validators.model_config import HydromtModelSetup, HydromtModelStep
from hydromt.models import GridModel, Model


def test_base_model_build():
    d = {
        "update": {
            "model_out": "./",
            "write": False,
            "forceful_overwrite": False,
            "opt": {},
        },
    }
    HydromtModelStep.from_dict(d, Model)


def test_base_model_unknown_fn():
    d = {
        "asdfasdf": {"some_arg": "some_val"},
    }

    with pytest.raises(ValueError, match="Model does not have function"):
        HydromtModelStep.from_dict(d, model=Model)


def test_base_model_build_unknown_params():
    model = Model
    d = {
        "build": {"asdfasdf": "whatever"},
    }

    with pytest.raises(ValueError, match="Unknown parameters for function build"):
        HydromtModelStep.from_dict(d, model=model)


def test_setup_config_validation():
    model = GridModel
    d = {
        "setup_config": {
            "header.settings": "value",
            "timers.end": "2010-02-15",
            "timers.start": "2010-02-05",
        },
    }
    HydromtModelStep.from_dict(d, model=model)


def test_setup_grid_from_constant_validation():
    model = GridModel
    d = {
        "setup_grid_from_constant": {
            "constant": 0.01,
            "name": "c1",
            "dtype": "float32",
            "nodata": -99.0,
        },
    }
    HydromtModelStep.from_dict(d, model=model)


def test_setup_grid_from_rasterdataset_validation():
    model = GridModel
    d = {
        "setup_grid_from_rasterdataset": {
            "raster_fn": "merit_hydro_1k",
            "variables": ["elevtn", "basins"],
            "reproject_method": ["average", "mode"],
        },
    }
    HydromtModelStep.from_dict(d, model=model)


def test_setup_grid_from_geodataframe_validation():
    model = GridModel
    d = {
        "setup_grid_from_geodataframe2": {
            "vector_fn": "hydro_lakes",
            "variables": ["waterbody_id", "Depth_avg"],
            "nodata": [-1, -999.0],
            "rasterize_method": "value",
            "rename": {"waterbody_id": "lake_id", "Detph_avg": "lake_depth"},
        },
    }
    HydromtModelStep.from_dict(d, model=model)


def test_setup_non_existing_grid_model_function():
    model = GridModel
    d = {
        "setup_config": {
            "starttime": "2009-04-01T00:00:00",
            "endtime": "2011-01-30T00:00:00",
            "timestepsecs": 86400,
            "input.path_forcing": "inmaps.nc",
        },
        "setup_precip_forcing": {"precip_fn": "era5"},
        "setup_grid_from_rasterdataset": {
            "raster_fn": "pet_nc",
            "fill_method": "nearest",
        },
    }
    with pytest.raises(ValueError, match="Model does not have function"):
        HydromtModelSetup.from_dict(d, model=model)


def test_setup_grid_from_raster_reclass_validation():
    model = GridModel
    d = {
        "setup_grid_from_raster_reclass2": {
            "raster_fn": "vito",
            "reclass_table_fn": "vito_reclass",
            "reclass_variables": ["manning"],
            "reproject_method": ["average"],
        },
    }
    HydromtModelStep.from_dict(d, model=model)


def test_write_validation_validation():
    model = GridModel
    d = {
        "write": {"components": ["config", "geoms", "grid"]},
    }
    HydromtModelStep.from_dict(d, model=model)
