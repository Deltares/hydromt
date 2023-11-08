"""Tests for the cli submodule."""


import pytest

from hydromt.models import GridModel
from hydromt.validators.model_config import BaseHydromtStep


def test_base_model_build():
    d = {
        "update": {
            "model_out": "./",
            "write": False,
            "forceful_overwrite": False,
            "opt": {},
        },
    }
    BaseHydromtStep.from_dict(d)


def test_base_model_unknown_fn():
    d = {
        "asdfasdf": {"some_arg": "some_val"},
    }

    with pytest.raises(ValueError, match="Model does not have function"):
        _ = BaseHydromtStep.from_dict(d)


def test_base_model_build_unknown_params():
    d = {
        "build": {"asdfasdf": "whatever"},
    }

    with pytest.raises(ValueError, match="Unknown parameters for function build"):
        _ = BaseHydromtStep.from_dict(d)


def test_setup_config_validation():
    d = {
        "setup_config": {
            "header.settings": "value",
            "timers.end": "2010-02-15",
            "timers.start": "2010-02-05",
        },
    }
    BaseHydromtStep.from_dict(d, model_clf=GridModel)


def test_setup_grid_from_constant_validation():
    d = {
        "setup_grid_from_constant": {
            "constant": 0.01,
            "name": "c1",
            "dtype": "float32",
            "nodata": -99.0,
        },
    }
    BaseHydromtStep.from_dict(d, model_clf=GridModel)


def test_setup_grid_from_rasterdataset_validation():
    d = {
        "setup_grid_from_rasterdataset": {
            "raster_fn": "merit_hydro_1k",
            "variables": ["elevtn", "basins"],
            "reproject_method": ["average", "mode"],
        },
    }
    BaseHydromtStep.from_dict(d, model_clf=GridModel)


def test_setup_grid_from_geodataframe_validation():
    d = {
        "setup_grid_from_geodataframe": {
            "vector_fn": "hydro_lakes",
            "variables": ["waterbody_id", "Depth_avg"],
            "nodata": [-1, -999.0],
            "rasterize_method": "value",
            "rename": {"waterbody_id": "lake_id", "Detph_avg": "lake_depth"},
        },
    }
    BaseHydromtStep.from_dict(d, model_clf=GridModel)


def test_setup_grid_from_raster_reclass_validation():
    d = {
        "setup_grid_from_raster_reclass": {
            "raster_fn": "vito",
            "reclass_table_fn": "vito_reclass",
            "reclass_variables": ["manning"],
            "reproject_method": ["average"],
        },
    }
    BaseHydromtStep.from_dict(d, model_clf=GridModel)


def test_write_validation_validation():
    d = {
        "write": {"components": ["config", "geoms", "grid"]},
    }
    BaseHydromtStep.from_dict(d, model_clf=GridModel)
