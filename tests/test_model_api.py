# -*- coding: utf-8 -*-
"""Tests for the models.model_api of hydromt."""

import pytest
from unittest.mock import Mock, patch
import os
from os.path import join, isfile, isdir
import glob
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from affine import Affine

from pyflwdir import core_d8
import hydromt
from hydromt import raster
from .testclass import TestModel
from hydromt.models import MODELS
from hydromt.models.model_api import Model
from hydromt.cli.cli_utils import parse_config


def _rand_float(shape, dtype=np.float32):
    return np.random.rand(*shape).astype(dtype)


def _rand_d8(shape):
    d8 = core_d8._ds.ravel()
    return d8.flat[np.random.randint(d8.size, size=shape)]


def _rand_msk(shape):
    mks = np.array([0, 1, 2], dtype=np.int8)
    return mks[np.random.randint(mks.size, size=shape)]


_maps = {
    "elevtn": {"func": _rand_float, "nodata": -9999.0},
    "flwdir": {"func": _rand_d8, "nodata": core_d8._mv},
    "mask": {"func": _rand_msk, "nodata": -1},
    "basins": {"func": _rand_msk, "nodata": -1},
}
_forcing = {
    "precip": {"func": _rand_float, "nodata": -9999.0},
    "temp": {"func": _rand_float, "nodata": -9999.0},
}
_states = {
    "waterlevel": {"func": _rand_float, "nodata": -9999.0},
    "discharge": {"func": _rand_float, "nodata": -9999.0},
}
_results = {
    "waterlevel": {"func": _rand_float, "nodata": -9999.0},
    "discharge": {"func": _rand_float, "nodata": -9999.0},
}

_models = {
    "test": [
        "elevtn",
        "flwdir",
        "basins",
        "mask",
        "precip",
        "temp",
        "waterlevel",
        "discharge",
    ],
}

TESTMODELS = MODELS.copy()
TESTMODELS.update({"api": Model, "test": TestModel})
p = patch.multiple(TestModel, __abstractmethods__=set())


def _create_staticmaps(model):
    left, top = 3.0, -5.0
    xres, yres = 0.5, -0.5
    shape = (6, 10)
    transform = Affine(xres, 0.0, left, 0.0, yres, top)
    data_vars = {
        n: (_maps[n]["func"](shape), _maps[n]["nodata"])
        for n in _maps
        if n in _models[model]
    }
    ds = raster.RasterDataset.from_numpy(data_vars=data_vars, transform=transform)
    ds.raster.set_crs(4326)
    return ds


def _create_dynmaps(model, ds_like, forcing=True):
    shape = 10, ds_like.raster.height, ds_like.raster.width
    dims = ["time"]
    dims.extend(list(ds_like.raster.dims))
    if forcing:
        data_vars = {
            n: (dims, _forcing[n]["func"](shape))
            for n in _forcing
            if n in _models[model]
        }
    else:
        data_vars = {
            n: (dims, _results[n]["func"](shape))
            for n in _results
            if n in _models[model]
        }
    coords = dict(time=pd.date_range(start="2010/01/01", periods=10, freq="D"))
    coords.update(ds_like.raster.coords)
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    return ds


def _create_states(model, ds_like):
    shape = ds_like.raster.shape
    data_vars = {
        n: (list(ds_like.raster.dims), _states[n]["func"](shape))
        for n in _results
        if n in _models[model]
    }
    ds = xr.Dataset(data_vars=data_vars, coords=ds_like.raster.coords)
    return ds


def _create_model(model, tmpdir):
    root = str(tmpdir.join(model))
    mod = TESTMODELS.get(model)(root=root, mode="w")
    mv = {n: mod._MAPS[n] for n in _maps if n in _models[model]}

    # staticmaps
    staticmaps = _create_staticmaps(model).rename(mv)
    mod.set_staticmaps(staticmaps)
    testmap = staticmaps[mod._MAPS["elevtn"]].rename("test")
    mod.set_staticmaps(testmap.to_dataset(), name="test1")
    testval = testmap.values
    with pytest.raises(ValueError):
        mod.set_staticmaps(testval, name=None)
    mod.set_staticmaps(testval, name="test")

    # config
    mod.set_config("global.name", "test")
    mod.set_config("input.elevtn", f"{mod._MAPS['elevtn']}")

    # staticgeoms
    with pytest.raises(ValueError):
        mod.set_staticgeoms("region", mod.region)
    mod.set_staticgeoms(mod.region, "region")

    # forcing
    forcing = _create_dynmaps(model, mod.staticmaps)
    mod.set_forcing(forcing)
    testfor = forcing["precip"].rename("test")
    with pytest.raises(ValueError):
        mod.set_forcing(testfor.values)
    mod.set_forcing(testfor)
    testfor.name = None
    with pytest.raises(ValueError):
        mod.set_forcing(testfor)
    mod.set_forcing(testfor, name="test1")

    # states
    states = _create_states(model, mod.staticmaps)
    mod.set_states(states)
    teststat = states["waterlevel"].rename("test")
    with pytest.raises(ValueError):
        mod.set_states(teststat.values)
    mod.set_states(teststat)
    teststat.name = None
    with pytest.raises(ValueError):
        mod.set_states(teststat)
    mod.set_states(teststat, name="test1")

    # results
    results = _create_dynmaps(model, mod.staticmaps, forcing=False)
    mod.set_results(results)
    testres = results["waterlevel"].rename("test")
    with pytest.raises(ValueError):
        mod.set_results(testres.values)
    mod.set_results(testres)
    testres.name = None
    with pytest.raises(ValueError):
        mod.set_results(testres)
    mod.set_results(testres, name="test1")
    return mod


@pytest.mark.parametrize("model", list(_models.keys()))
# @patch("TestModel.__abstractmethods__", set())
def test_model(model, tmpdir):
    p.start()
    mod = _create_model(model, tmpdir)
    mod.write()
    mod1 = TESTMODELS.get(model)(root=mod.root, mode="r")
    mod1.read()
    for name in mod1.staticmaps.data_vars:
        assert np.all(mod1.staticmaps[name] == mod.staticmaps[name])
    assert mod.config == mod1.config, f"config mismatch"

    # test properties
    mod.crs
    mod.dims
    mod.coords
    mod.res
    mod.transform
    mod.width
    mod.height
    mod.shape
    mod.bounds
    mod.region

    # test api compliance
    non_compliant_list = mod.test_model_api()
    assert len(non_compliant_list) == 0

    mod._staticgeoms["nc_test"] = dict()
    mod._forcing["nc_test"] = xr.Dataset()
    mod._config = "test"
    mod._states["nc_test"] = dict()
    mod._results["nc_test"] = dict()
    non_compliant_list = mod.test_model_api()
    assert len(non_compliant_list) == 5
    assert "staticgeoms.nc_test" in non_compliant_list
    assert "forcing.nc_test" in non_compliant_list

    mod._staticmaps = dict()
    mod._staticgeoms = ["test"]
    mod._forcing = ["test"]
    mod._config = ["test"]
    mod._results = ["test"]
    mod._states = ["test"]
    non_compliant_list = mod.test_model_api()
    assert len(non_compliant_list) == 6
    assert "staticmaps" in non_compliant_list


@pytest.mark.parametrize("model", list(_models.keys()))
# @patch("TestModel.__abstractmethods__", set())
def test_model_method(model, tmpdir):
    # test build model
    root = str(tmpdir.join(model))
    opt = {
        "setup_config": {"global.name": "test"},
        "setup_basemaps": {"add_geom": True},
        "param": {"name": "A", "value": 10.0},
    }
    region = {"bbox": [0.0, -5.0, 3.0, 0.0]}
    mod = TESTMODELS.get(model)(root=root, mode="w")
    with pytest.raises(DeprecationWarning):
        mod.build(region=region, opt=opt)
    opt["setup_parameter"] = opt.pop("param")
    with pytest.raises(ValueError):
        mod.build(region=region, opt=opt)
    opt["setup_param"] = opt.pop("setup_parameter")
    mod.build(region=region, res=0.5, opt=opt, write=False)
    assert "dem" in mod.staticmaps
    assert "A" in mod.staticmaps

    # test update model
    mod1 = TESTMODELS.get(model)(root=root, mode="w")
    with pytest.raises(ValueError):
        # region is None, mod was not written
        mod1.update(opt=opt)
    mod.write()
    mod1 = TESTMODELS.get(model)(root=root, mode="r")
    with pytest.raises(ValueError):
        # model_out None with mode=r
        mod1.update(opt=opt)
    with pytest.raises(IOError):
        mod1.write_staticmaps()
    with pytest.raises(IOError):
        mod1.write_staticgeoms()
    with pytest.raises(IOError):
        mod1.write_config()
    opt = {
        "setup_config": {"global.name": "test"},
        "setup_basemaps": {"add_geom": True},
        "setup_param": {"name": "B", "value": 2.5},
    }
    root1 = root + "1"
    mod1.update(model_out=root1, opt=opt)
    assert "A" in mod1.staticmaps
    assert "B" in mod1.staticmaps
    assert "B" not in mod.staticmaps
    assert mod1.get_config("global.name") == "test"


_models.update({"api": [""]})


@pytest.mark.parametrize("model", list(_models.keys()))
@patch("hydromt.models.model_api.Model.__abstractmethods__", set())
def test_object(model, tmpdir):
    mod = TESTMODELS.get(model)()
    # empty model
    with pytest.raises(ValueError):
        mod.root
        mod.staticmaps
        mod.crs
        mod.transform
    with pytest.raises(ValueError):
        mod.set_root(r"./non_existent_folder", mode="w+")
    with pytest.raises(IOError):
        mod.set_root(r"./non_existent_folder", mode="r")
    assert hasattr(mod, "build")

    # abstract methods
    if model == "api":
        root = str(tmpdir.join(model))
        if not isdir(root):
            with pytest.raises(IOError):
                mod1 = TESTMODELS.get(model)(root=root, mode="r")
            os.mkdir(root)
        mod1 = TESTMODELS.get(model)(root=root, mode="r")
        with pytest.raises(NotImplementedError):
            mod1.staticmaps()
        with pytest.raises(NotImplementedError):
            mod1.staticgeoms()
        with pytest.raises(NotImplementedError):
            mod1.forcing()
        with pytest.raises(NotImplementedError):
            mod1.states()
        with pytest.raises(NotImplementedError):
            mod1.results()
        with pytest.raises(IOError):
            mod1.write_staticmaps()
        with pytest.raises(IOError):
            mod1.write_staticgeoms()
        with pytest.raises(IOError):
            mod1.write_forcing()
        with pytest.raises(IOError):
            mod1.write_states()
        mod1 = TESTMODELS.get(model)(root=root, mode="w")
        with pytest.raises(NotImplementedError):
            mod1.write_staticmaps()
        with pytest.raises(NotImplementedError):
            mod1.write_staticgeoms()
        with pytest.raises(NotImplementedError):
            mod1.write_forcing()
        with pytest.raises(NotImplementedError):
            mod1.write_states()


p.stop()
