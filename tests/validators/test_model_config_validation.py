"""Tests for the cli submodule."""


from hydromt.validators.model_config import HydromtStep


def test_setup_config_validation():
    d = {
        "setup_config": {
            "header.settings": "value",
            "timers.end": "2010-02-15",
            "timers.start": "2010-02-05",
        },
    }
    HydromtStep.from_dict(d)


def test_setup_grid_from_constant_validation():
    d = {
        "setup_grid_from_constant": {
            "constant": 0.01,
            "name": "c1",
            "dtype": "float32",
            "nodata": -99.0,
        },
    }
    HydromtStep.from_dict(d)


def test_setup_grid_from_rasterdataset_validation():
    d = {
        "setup_grid_from_rasterdataset": {
            "raster_fn": "merit_hydro_1k",
            "variables": ["elevtn", "basins"],
            "reproject_method": ["average", "mode"],
        },
    }
    HydromtStep.from_dict(d)


def test_setup_grid_from_rasterdataset2_validation():
    d = {
        "setup_grid_from_rasterdataset2": {
            "raster_fn": "vito",
            "fill_method": "nearest",
            "reproject_method": "mode",
            "rename": {"vito": "landuse"},
        },
    }
    HydromtStep.from_dict(d)


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
    HydromtStep.from_dict(d)


def test_setup_grid_from_raster_reclass_validation():
    d = {
        "setup_grid_from_raster_reclass": {
            "raster_fn": "vito",
            "reclass_table_fn": "vito_reclass",
            "reclass_variables": ["manning"],
            "reproject_method": ["average"],
        },
    }
    HydromtStep.from_dict(d)


def test_setup_grid_from_geodataframe2_validation():
    d = {
        "setup_grid_from_geodataframe2": {
            "vector_fn": "hydro_lakes",
            "rasterize_method": "fraction",
            "rename": {"hydro_lakes": "water_frac"},
        },
    }
    HydromtStep.from_dict(d)


def test_write_validation_validation():
    d = {
        "write": {"components": ["config", "geoms", "grid"]},
    }
    HydromtStep.from_dict(d)
