# -*- coding: utf-8 -*-
"""Tests for the hydromt.models module of HydroMT"""

from os.path import isfile, join
import pytest
import xarray as xr
import numpy as np
import geopandas as gpd
from hydromt.models.model_api import _check_data, AuxmapsModel
from hydromt.models import Model, GridModel, LumpedModel
from hydromt import _has_xugrid


def test_check_data(demda):
    data_dict = _check_data(demda.copy(), "elevtn")
    assert isinstance(data_dict["elevtn"], xr.DataArray)
    assert data_dict["elevtn"].name == "elevtn"
    with pytest.raises(ValueError, match="Name required for DataArray"):
        _check_data(demda)
    demda.name = "dem"
    demds = demda.to_dataset()
    data_dict = _check_data(demds, "elevtn", False)
    assert isinstance(data_dict["elevtn"], xr.Dataset)
    data_dict = _check_data(demds, split_dataset=True)
    assert isinstance(data_dict["dem"], xr.DataArray)
    with pytest.raises(ValueError, match="Name required for Dataset"):
        _check_data(demds, split_dataset=False)
    with pytest.raises(ValueError, match='Data type "dict" not recognized'):
        _check_data({"wrong": "type"})


def test_model_api(grid_model):
    assert np.all(np.isin(["grid", "geoms"], list(grid_model.api.keys())))
    # add some wrong data
    grid_model._geoms.update({"wrong_geom": xr.Dataset()})
    grid_model._forcing.update({"test": gpd.GeoDataFrame()})
    non_compliant = grid_model._test_model_api()
    assert non_compliant == ["geoms.wrong_geom", "forcing.test"]


def test_run_log_method():
    model = Model()
    region = {"bbox": [12.05, 45.30, 12.85, 45.65]}
    model._run_log_method("setup_region", region)  # args
    assert "region" in model._geoms
    model._geoms = {}
    model._run_log_method("setup_region", region=region)  # kwargs
    assert "region" in model._geoms


def test_model(model, tmpdir):
    # Staticmaps -> moved from _test_model_api as it is deprecated
    model._API.update({"staticmaps": xr.Dataset})
    non_compliant = model._test_model_api()
    assert len(non_compliant) == 0, non_compliant
    # write model
    model.set_root(str(tmpdir), mode="w")
    model.write()
    # read model
    model1 = Model(str(tmpdir), mode="r")
    model1.read()
    # check if equal
    model._results = {}  # reset results for comparison
    equal, errors = model._test_equal(model1)
    assert equal, errors
    # region from staticmaps
    model._geoms.pop("region")
    assert np.all(model.region.total_bounds == model.staticmaps.raster.bounds)


@pytest.mark.filterwarnings("ignore:The setup_basemaps")
def test_model_build_update(tmpdir):
    model = Model(root=str(tmpdir), mode="w")
    # NOTE: _CLI_ARGS still pointing setup_basemaps for backwards comp
    model._CLI_ARGS.update({"region": "setup_region"})
    model._NAME = "testmodel"
    model.build(
        region={"bbox": [12.05, 45.30, 12.85, 45.65]},
        res=1,
        opt={"setup_basemaps": {}, "write_geoms": {}, "write_config": {}},
    )
    assert "region" in model._geoms
    assert isfile(join(model.root, "model.ini"))
    assert isfile(join(model.root, "hydromt.log"))
    # test update with specific write method
    model.update(
        opt={
            "setup_region": {},  # should be removed with warning
            "setup_basemaps": {},
            "write_geoms": {"fn": "geoms/{name}.gpkg", "driver": "GPKG"},
        }
    )
    assert isfile(join(model.root, "geoms", "region.gpkg"))
    with pytest.raises(
        ValueError, match='Model testmodel has no method "unknown_method"'
    ):
        model.update(opt={"unknown_method": {}})
    # read and update model
    model = Model(root=str(tmpdir), mode="r")
    model_out = str(tmpdir.join("update"))
    model.update(model_out=model_out, opt={})  # write only
    assert isfile(join(model_out, "model.ini"))


def test_setup_region(model, demda, tmpdir):
    # bbox
    model.setup_region({"bbox": [12.05, 45.30, 12.85, 45.65]})
    region = model._geoms.pop("region")
    # geom
    model.setup_region({"geom": region})
    gpd.testing.assert_geodataframe_equal(region, model.region)
    # grid
    model._geoms.pop("region")  # remove old region
    grid_fn = str(tmpdir.join("grid.tif"))
    demda.raster.to_raster(grid_fn)
    model.setup_region({"grid": grid_fn})
    assert np.all(demda.raster.bounds == model.region.total_bounds)
    # # TODO model once we have registered the Model class entrypoint
    # model._geoms.pop('region') # remove old region
    # root = str(tmpdir.join('root'))
    # model.set_root(root, mode='w')
    # model.write()
    # model.setup_region({'model': root})
    # basin
    model._geoms.pop("region")  # remove old region
    model.setup_region({"basin": [12.2, 45.833333333333329]})
    assert np.all(model.region["value"] == 210000039)  # basin id


def test_config(model, tmpdir):
    # config
    model.set_root(str(tmpdir))
    model.set_config("global.name", "test")
    assert "name" in model._config["global"]
    assert model.get_config("global.name") == "test"
    fn = str(tmpdir.join("test.file"))
    with open(fn, "w") as f:
        f.write("")
    model.set_config("global.file", "test.file")
    assert str(model.get_config("global.file")) == "test.file"
    assert str(model.get_config("global.file", abs_path=True)) == fn


def test_auxmapsmixin(auxmap_model, tmpdir):
    assert "auxmaps" in auxmap_model.api
    assert len(auxmap_model.auxmaps) == 1
    non_compliant = auxmap_model._test_model_api()
    assert len(non_compliant) == 0, non_compliant
    # write model
    auxmap_model.set_root(str(tmpdir), mode="w")
    auxmap_model.write(components=["config", "geoms", "auxmaps"])
    # read model
    model1 = AuxmapsModel(str(tmpdir), mode="r")
    model1.read(components=["config", "geoms", "auxmaps"])
    # check if equal
    equal, errors = auxmap_model._test_equal(model1)
    assert equal, errors


def test_gridmodel(grid_model, tmpdir):
    assert "grid" in grid_model.api
    non_compliant = grid_model._test_model_api()
    assert len(non_compliant) == 0, non_compliant
    # grid specific attributes
    assert np.all(grid_model.res == grid_model.grid.raster.res)
    assert np.all(grid_model.bounds == grid_model.grid.raster.bounds)
    assert np.all(grid_model.transform == grid_model.grid.raster.transform)
    # write model
    grid_model.set_root(str(tmpdir), mode="w")
    grid_model.write()
    # read model
    model1 = GridModel(str(tmpdir), mode="r")
    model1.read()
    # check if equal
    equal, errors = grid_model._test_equal(model1)
    assert equal, errors


def test_lumpedmodel(lumped_model, tmpdir):
    assert "response_units" in lumped_model.api
    non_compliant = lumped_model._test_model_api()
    assert len(non_compliant) == 0, non_compliant
    # write model
    lumped_model.set_root(str(tmpdir), mode="w")
    lumped_model.write()
    # read model
    model1 = LumpedModel(str(tmpdir), mode="r")
    model1.read()
    # check if equal
    equal, errors = lumped_model._test_equal(model1)
    assert equal, errors


def test_networkmodel(network_model, tmpdir):
    network_model.set_root(str(tmpdir), mode="r+")
    with pytest.raises(NotImplementedError):
        network_model.read(["network"])
    with pytest.raises(NotImplementedError):
        network_model.write(["network"])
    with pytest.raises(NotImplementedError):
        network_model.set_network()
    with pytest.raises(NotImplementedError):
        network_model.network


@pytest.mark.skipif(not _has_xugrid(), reason="Xugrid not installed.")
def test_meshmodel(mesh_model, tmpdir):
    from hydromt.models import MeshModel

    assert "mesh" in mesh_model.api
    non_compliant = mesh_model._test_model_api()
    assert len(non_compliant) == 0, non_compliant
    # write model
    mesh_model.set_root(str(tmpdir), mode="w")
    mesh_model.write()
    # read model
    model1 = MeshModel(str(tmpdir), mode="r")
    model1.read()
    # check if equal
    equal, errors = mesh_model._test_equal(model1)
    assert equal, errors
