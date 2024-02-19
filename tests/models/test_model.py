# -*- coding: utf-8 -*-
"""Tests for the hydromt.models module of HydroMT."""

from copy import deepcopy
from os.path import abspath, dirname, isfile, join

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from hydromt._compat import EntryPoint, EntryPoints
from hydromt.data_catalog import DataCatalog
from hydromt.models import (
    MODELS,
    GridModel,
    Model,
    ModelCatalog,
    VectorModel,
    plugins,
)
from hydromt.models.api import _check_data
from hydromt.models.components.grid import GridMixin
from hydromt.region import Region

DATADIR = join(dirname(abspath(__file__)), "..", "data")


class _DummyModel(GridModel, GridMixin):
    _API = {"asdf": "yeah"}


def test_api_attrs():
    dm = _DummyModel()
    assert hasattr(dm, "_NAME")
    assert hasattr(dm, "_API")
    assert "asdf" in dm.api
    assert dm.api["asdf"] == "yeah"
    assert "grid" in dm.api
    assert dm.api["grid"] == xr.Dataset


def test_plugins(mocker):
    ep_lst = EntryPoints(
        [
            EntryPoint(
                name="test_model",
                value="hydromt.models.model_api:Model",
                group="hydromt.models",
            )
        ]
    )
    mocker.patch("hydromt.models.plugins._discover", return_value=ep_lst)
    eps = plugins.get_plugin_eps()
    assert "test_model" in eps
    assert isinstance(eps["test_model"], EntryPoint)


def test_plugin_duplicates(mocker):
    ep_lst = plugins.get_general_eps().values()
    mocker.patch("hydromt.models.plugins._discover", return_value=ep_lst)
    eps = plugins.get_plugin_eps()
    assert len(eps) == 0


def test_load():
    with pytest.raises(ValueError, match="Model plugin type not recognized"):
        plugins.load(
            EntryPoint(
                name="error",
                value="hydromt.data_catalog:DataCatalog",
                group="hydromt.data_catalog",
            )
        )
    with pytest.raises(ImportError, match="Error while loading model plugin"):
        plugins.load(
            EntryPoint(
                name="error", value="hydromt.models:DataCatalog", group="hydromt.models"
            )
        )


def test_global_models(mocker):
    _MODELS = ModelCatalog()
    keys = list(plugins.LOCAL_EPS.keys())
    # set first local model as plugin for testing
    _MODELS._plugins.append(keys[0])
    assert isinstance(_MODELS[keys[0]], EntryPoint)
    assert issubclass(_MODELS.load(keys[0]), Model)
    assert keys[0] in _MODELS.__str__()
    assert all([k in _MODELS for k in keys])  # eps
    assert all([k in _MODELS.cls for k in keys])
    with pytest.raises(ValueError, match="Unknown model"):
        _MODELS["unknown"]


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


@pytest.mark.skip(reason="gridmodel not implemented yet")
def test_model_api(grid_model):
    assert np.all(np.isin(["grid", "geoms"], list(grid_model.api.keys())))
    # add some wrong data
    grid_model.geoms.update({"wrong_geom": xr.Dataset()})
    grid_model.forcing.update({"test": gpd.GeoDataFrame()})
    non_compliant = grid_model._test_model_api()
    assert non_compliant == ["geoms.wrong_geom", "forcing.test"]


def test_run_log_setup_region_args():
    model = Model()
    region = {"bbox": [12.05, 45.30, 12.85, 45.65]}
    model._run_log_method("setup_region", region)  # args
    assert isinstance(model.region, Region)


def test_run_log_setup_region_kwargs():
    model = Model()
    region = {"bbox": [12.05, 45.30, 12.85, 45.65]}
    model._geoms = {}
    model._run_log_method("setup_region", region=region)  # kwargs
    assert isinstance(model.region, Region)


@pytest.mark.skip(reason="Needs implementation of RasterDataSet.")
def test_write_data_catalog(tmpdir):
    model = Model(root=join(tmpdir, "model"), data_libs=["artifact_data"])
    sources = list(model.data_catalog.sources.keys())
    data_lib_fn = join(model.root, "hydromt_data.yml")
    # used_only=True -> no file written
    model.write_data_catalog()
    assert not isfile(data_lib_fn)
    # write with single source
    model.data_catalog.get_source(sources[0]).mark_as_used()
    model.write_data_catalog()
    assert list(DataCatalog(data_lib_fn).sources.keys()) == sources[:1]
    # write to different file
    data_lib_fn1 = join(tmpdir, "hydromt_data2.yml")
    model.write_data_catalog(data_lib_fn=data_lib_fn1)
    assert isfile(data_lib_fn1)
    # append source
    model1 = Model(root=model.root, data_libs=["artifact_data"], mode="r+")
    model1.data_catalog.get_source(sources[1]).mark_as_used()
    model1.write_data_catalog(append=False)
    assert list(DataCatalog(data_lib_fn).sources.keys()) == [sources[1]]
    model1.data_catalog.get_source(sources[0]).mark_as_used()
    model1.write_data_catalog(append=True)
    assert list(DataCatalog(data_lib_fn).sources.keys()) == sources[:2]
    # test writing table of datacatalog as csv
    model.write_data_catalog(used_only=False, save_csv=True)
    assert isfile(join(model.root, "hydromt_data.csv"))
    data_catalog_df = pd.read_csv(join(model.root, "hydromt_data.csv"))
    assert len(data_catalog_df) == len(sources)
    assert data_catalog_df.iloc[0, 0] == sources[0]
    assert data_catalog_df.iloc[-1, 0] == sources[-1]


@pytest.mark.skip(reason="Needs implementation of RasterDataSet.")
def test_model(model, tmpdir):
    # Staticmaps -> moved from _test_model_api as it is deprecated
    model._API.update({"staticmaps": xr.Dataset})
    with pytest.deprecated_call():
        non_compliant = model._test_model_api()
    assert len(non_compliant) == 0, non_compliant
    # write model
    model.set_root(str(tmpdir), mode="w")
    model.write()
    with pytest.raises(IOError, match="Model opened in write-only mode"):
        model.read()
    # read model
    model1 = Model(str(tmpdir), mode="r")
    with pytest.deprecated_call():
        model1.read()
    with pytest.raises(IOError, match="Model opened in read-only mode"):
        model1.write()
    # check if equal
    model._results = {}  # reset results for comparison
    with pytest.deprecated_call():
        equal, errors = model._test_equal(model1)
    assert equal, errors
    # read region from staticmaps
    model._geoms.pop("region")
    with pytest.deprecated_call():
        assert np.all(model.region.total_bounds == model.staticmaps.raster.bounds)


@pytest.mark.skip(reason="Needs implementation of RasterDataSet.")
def test_model_tables(model, df, tmpdir):
    # make a couple copies of the dfs for testing
    dfs = {str(i): df.copy() for i in range(5)}
    model.set_root(tmpdir, mode="r+")  # append mode
    clean_model = deepcopy(model)

    with pytest.raises(KeyError):
        model.tables[1]

    for i, d in dfs.items():
        model.set_tables(d, name=i)
        assert df.equals(model.tables[i])

    # now do the same but interating over the stables instead
    for i, d in model.tables.items():
        model.set_tables(d, name=i)
        assert df.equals(model.tables[i])

    assert list(model.tables.keys()) == list(map(str, range(5)))

    model.write_tables()
    clean_model.read_tables()

    model_merged = model.get_tables_merged().sort_values(["table_origin", "city"])
    clean_model_merged = clean_model.get_tables_merged().sort_values(
        ["table_origin", "city"]
    )
    assert np.all(
        np.equal(model_merged, clean_model_merged)
    ), f"model: {model_merged}\nclean_model: {clean_model_merged}"


def test_model_append(demda, df, tmpdir):
    # write a model
    demda.name = "dem"
    mod = GridModel(mode="w", root=str(tmpdir))
    mod.set_config("test.data", "dem")
    mod.set_grid(demda, name="dem")
    mod.set_maps(demda, name="dem")
    mod.set_forcing(demda, name="dem")
    mod.set_states(demda, name="dem")
    mod.set_geoms(demda.raster.box, name="dem")
    mod.set_tables(df, name="df")
    mod.write()
    # append to model and check if previous data is still there
    mod1 = GridModel(mode="r+", root=str(tmpdir))
    mod1.set_config("test1.data", "dem")
    assert mod1.get_config("test.data") == "dem"
    mod1.set_grid(demda, name="dem1")
    assert "dem" in mod1.grid
    mod1.set_maps(demda, name="dem1")
    assert "dem" in mod1.maps
    mod1.set_forcing(demda, name="dem1")
    assert "dem" in mod1.forcing
    mod1.set_forcing(df, name="df1", split_dataset=False)
    assert "df1" in mod1.forcing
    assert isinstance(mod1.forcing["df1"], xr.Dataset)
    mod1.set_states(demda, name="dem1")
    assert "dem" in mod1.states
    mod1.set_geoms(demda.raster.box, name="dem1")
    assert "dem" in mod1.geoms
    mod1.set_tables(df, name="df1")
    assert "df" in mod1.tables


@pytest.mark.skip(
    "sam hasn't figured out how to integrate write_geoms with new region yet"
)
def test_model_build_update(tmpdir, demda, obsda):
    bbox = [12.05, 45.30, 12.85, 45.65]
    # build model
    model = Model(root=str(tmpdir), mode="w")
    model._NAME = "testmodel"
    model.build(
        region={"bbox": bbox},
        opt={"write_geoms": {}, "write_config": {}},
    )
    assert isfile(join(model.root, "model.ini"))
    assert isfile(join(model.root, "hydromt.log"))
    # test update with specific write method
    model.update(
        opt={
            "setup_region": {},  # should be removed with warning
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

    # Now test update for a model with some data
    geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
    # Quick check that model can't be overwritten without w+
    with pytest.raises(
        IOError, match="Model dir already exists and cannot be overwritten: "
    ):
        model = Model(root=str(tmpdir), mode="w")
    # Build model with some data
    model = Model(root=str(tmpdir), mode="w+")
    model._CLI_ARGS.update({"region": "setup_region"})
    model._NAME = "testmodel"
    model.build(
        region={"bbox": bbox},
        opt={
            "setup_config": {"input": {"dem": "elevtn", "prec": "precip"}},
            "set_geoms": {"geom": geom, "name": "geom1"},
            "set_maps": {"data": demda, "name": "elevtn"},
            "set_forcing": {"data": obsda, "name": "precip"},
        },
    )
    # Now update the model
    model = Model(root=str(tmpdir), mode="r+")
    model.update(
        opt={
            "setup_config": {"input.dem2": "elevtn2", "input.temp": "temp"},
            "set_geoms": {"geom": geom, "name": "geom2"},
            "set_maps": {"data": demda, "name": "elevtn2"},
            "set_forcing": {"data": obsda, "name": "temp"},
            "set_forcing2": {"data": obsda * 0.2, "name": "precip"},
        }
    )
    assert len(model._defered_file_closes) == 0
    # Check that variables from build AND update are present
    assert "dem" in model.config["input"]
    assert "dem2" in model.config["input"]
    assert "geom1" in model.geoms
    assert "geom2" in model.geoms
    assert "elevtn" in model.maps
    assert "elevtn2" in model.maps
    assert "precip" in model.forcing
    assert "temp" in model.forcing


@pytest.mark.skip(reason="Needs implementation of RasterDataSet.")
def test_setup_region(model, demda, tmpdir):
    # bbox
    model.setup_region({"bbox": [12.05, 45.30, 12.85, 45.65]})
    region = model._geoms.pop("region")
    # geom
    model.setup_region({"geom": region})
    gpd.testing.assert_geodataframe_equal(region, model.region)
    # geom via data catalog
    fn_region = str(tmpdir.join("region.gpkg"))
    region.to_file(fn_region, driver="GPKG")
    model.data_catalog.from_dict(
        {
            "region": {
                "path": fn_region,
                "data_type": "GeoDataFrame",
                "driver": "vector",
            }
        }
    )
    model._geoms.pop("region")  # remove old region
    model.setup_region({"geom": "region"})
    gpd.testing.assert_geodataframe_equal(region, model.region)
    # grid
    model._geoms.pop("region")  # remove old region
    grid_fn = str(tmpdir.join("grid.tif"))
    demda.raster.to_raster(grid_fn)
    model.setup_region({"grid": grid_fn})
    assert np.all(demda.raster.bounds == model.region.total_bounds)
    # basin
    model._geoms.pop("region")  # remove old region
    model.setup_region({"basin": [12.2, 45.833333333333329]})
    assert np.all(model.region["value"] == 210000039)  # basin id


@pytest.mark.skip(reason="Needs decision on how geoms work independant of region.")
def test_model_write_geoms(tmpdir):
    model = Model(root=str(tmpdir), mode="w")
    bbox = box(*[4.221067, 51.949474, 4.471006, 52.073727])
    geom = gpd.GeoDataFrame(geometry=[bbox], crs=4326)
    geom.to_crs(epsg=28992, inplace=True)
    model.set_geoms(geom=geom, name="region")
    model.write_geoms(to_wgs84=True)
    region_geom = gpd.read_file(str(join(tmpdir, "geoms/region.geojson")))
    assert region_geom.crs.to_epsg() == 4326


@pytest.mark.skip(reason="Needs decision on how geoms work independant of region.")
def test_model_set_geoms(tmpdir):
    bbox = box(*[4.221067, 51.949474, 4.471006, 52.073727])
    geom = gpd.GeoDataFrame(geometry=[bbox], crs=4326)
    geom_28992 = geom.to_crs(epsg=28992)
    model = Model(root=str(tmpdir), mode="w")
    model.setup_region({"geom": geom_28992})  # set model crs based on epsg28992
    model.set_geoms(geom, "geom_wgs84")  # this should convert the geom crs to epsg28992
    assert model._geoms["geom_wgs84"].crs.to_epsg() == model.crs.to_epsg()


@pytest.mark.skip(reason="Needs implementation of RasterDataSet.")
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


@pytest.mark.skip(reason="Needs implementation of RasterDataSet.")
def test_maps_setup(tmpdir):
    dc_param_fn = join(DATADIR, "parameters_data.yml")
    mod = Model(data_libs=["artifact_data", dc_param_fn], mode="w")
    bbox = [11.80, 46.10, 12.10, 46.50]  # Piava river
    mod.setup_region({"bbox": bbox})
    mod.setup_config(**{"header": {"setting": "value"}})
    mod.setup_maps_from_rasterdataset(
        raster_fn="merit_hydro",
        name="hydrography",
        variables=["elevtn", "flwdir"],
        split_dataset=False,
    )
    mod.setup_maps_from_rasterdataset(raster_fn="vito", fill_method="nearest")
    mod.setup_maps_from_raster_reclass(
        raster_fn="vito",
        reclass_table_fn="vito_mapping",
        reclass_variables=["roughness_manning"],
        split_dataset=True,
    )

    assert len(mod.maps) == 3
    assert "roughness_manning" in mod.maps
    assert len(mod.maps["hydrography"].data_vars) == 2
    non_compliant = mod._test_model_api()
    assert len(non_compliant) == 0, non_compliant
    # write model
    mod.set_root(str(tmpdir), mode="w")
    mod.write(components=["config", "geoms", "maps"])


@pytest.mark.skip(reason="gridmodel not implemented yet")
def test_gridmodel(grid_model, tmpdir, demda):
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

    # try update
    grid_model.set_root(str(join(tmpdir, "update")), mode="w")
    grid_model.write()

    model1 = GridModel(str(join(tmpdir, "update")), mode="r+")
    model1.update(
        opt={
            "set_grid": {"data": demda, "name": "testdata"},
            "write_grid": {},
        }
    )
    assert "testdata" in model1.grid
    assert "elevtn" in model1.grid


@pytest.mark.skip(reason="Needs implementation of RasterDataSet.")
def test_setup_grid(tmpdir, demda):
    # Initialize model
    model = GridModel(
        root=join(tmpdir, "grid_model"),
        data_libs=["artifact_data"],
        mode="w",
    )
    # wrong region kind
    with pytest.raises(ValueError, match="Region for grid must be of kind"):
        model.setup_grid({"vector_model": "test_model"})
    # bbox
    bbox = [12.05, 45.30, 12.85, 45.65]
    with pytest.raises(
        ValueError, match="res argument required for kind 'bbox', 'geom'"
    ):
        model.setup_grid({"bbox": bbox})
    model.setup_grid(
        region={"bbox": bbox},
        res=0.05,
        add_mask=False,
        align=True,
    )
    assert "mask" not in model.grid
    assert model.crs.to_epsg() == 4326
    assert model.grid.raster.dims == ("y", "x")
    assert model.grid.raster.shape == (7, 16)
    assert np.all(np.round(model.grid.raster.bounds, 2) == bbox)
    grid = model.grid
    model._grid = xr.Dataset()  # remove old grid

    # geom
    region = model._geoms.pop("region")
    model.setup_grid(
        region={"geom": region},
        res=0.05,
        add_mask=False,
    )
    gpd.testing.assert_geodataframe_equal(region, model.region)
    xr.testing.assert_allclose(grid, model.grid)
    model._grid = xr.Dataset()  # remove old grid

    model.setup_grid(
        region={"geom": region},
        res=10000,
        crs="utm",
        add_mask=True,
    )
    assert "mask" in model.grid
    assert model.crs.to_epsg() == 32633
    assert model.grid.raster.res == (10000, -10000)
    model._grid = xr.Dataset()  # remove old grid

    # bbox rotated
    model.setup_grid(
        region={"bbox": [12.65, 45.50, 12.85, 45.60]},
        res=0.05,
        crs=4326,
        rotated=True,
        add_mask=True,
    )
    assert "xc" in model.grid.coords
    assert model.grid.raster.y_dim == "y"
    assert np.isclose(model.grid.raster.res[0], 0.05)
    model._grid = xr.Dataset()  # remove old grid

    # grid
    grid_fn = str(tmpdir.join("grid.tif"))
    demda.raster.to_raster(grid_fn)
    model.setup_grid({"grid": grid_fn})
    assert np.all(demda.raster.bounds == model.region.total_bounds)
    model._grid = xr.Dataset()  # remove old grid

    # basin
    model.setup_grid(
        region={"subbasin": [12.319, 46.320], "uparea": 50},
        res=1000,
        crs="utm",
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
    )
    assert not np.all(model.grid["mask"].values is True)
    assert model.grid.raster.shape == (47, 61)


@pytest.mark.skip(reason="gridmodel not implemented yet")
@pytest.mark.skip(reason="Needs implementation of RasterDataSet.")
def test_gridmodel_setup(tmpdir):
    # Initialize model
    dc_param_fn = join(DATADIR, "parameters_data.yml")
    mod = GridModel(
        root=join(tmpdir, "grid_model"),
        data_libs=["artifact_data", dc_param_fn],
        mode="w",
    )
    # Add region
    mod.setup_grid(
        {"subbasin": [12.319, 46.320], "uparea": 50},
        res=0.008333,
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
        add_mask=True,
    )
    # Add data with setup_* methods
    mod.setup_grid_from_constant(
        constant=0.01,
        name="c1",
        nodata=-99.0,
    )
    mod.setup_grid_from_constant(
        constant=2,
        name="c2",
        dtype=np.int8,
        nodata=-1,
    )
    mod.setup_grid_from_rasterdataset(
        raster_fn="merit_hydro",
        variables=["elevtn", "basins"],
        reproject_method=["average", "mode"],
        mask_name="mask",
    )
    mod.setup_grid_from_rasterdataset(
        raster_fn="vito",
        fill_method="nearest",
        reproject_method="mode",
        rename={"vito": "landuse"},
    )
    mod.setup_grid_from_raster_reclass(
        raster_fn="vito",
        fill_method="nearest",
        reclass_table_fn="vito_mapping",
        reclass_variables=["roughness_manning"],
        reproject_method=["average"],
    )
    mod.setup_grid_from_geodataframe(
        vector_fn="hydro_lakes",
        variables=["waterbody_id", "Depth_avg"],
        nodata=[-1, -999.0],
        rasterize_method="value",
        rename={"waterbody_id": "lake_id", "Depth_avg": "lake_depth"},
    )
    mod.setup_grid_from_geodataframe(
        vector_fn="hydro_lakes",
        rasterize_method="fraction",
        rename={"hydro_lakes": "water_frac"},
    )

    # Checks
    assert len(mod.grid) == 10
    for v in ["mask", "c1", "basins", "roughness_manning", "lake_depth", "water_frac"]:
        assert v in mod.grid
    assert mod.grid["lake_depth"].raster.nodata == -999.0
    assert mod.grid["roughness_manning"].raster.nodata == -999.0
    assert np.unique(mod.grid["c2"]).size == 2
    assert np.isin([-1, 2], np.unique(mod.grid["c2"])).all()

    non_compliant = mod._test_model_api()
    assert len(non_compliant) == 0, non_compliant

    # write model
    mod.set_root(str(tmpdir), mode="w")
    mod.write(components=["geoms", "grid"])


def test_vectormodel(vector_model, tmpdir):
    assert "vector" in vector_model.api
    non_compliant = vector_model._test_model_api()
    assert len(non_compliant) == 0, non_compliant
    # write model
    vector_model.set_root(str(tmpdir), mode="w")
    vector_model.write()
    # read model
    model1 = VectorModel(str(tmpdir), mode="r")
    model1.read()
    # check if equal
    equal, errors = vector_model._test_equal(model1)
    assert equal, errors


def test_vectormodel_vector(vector_model, tmpdir, geoda):
    # test set vector
    testds = vector_model.vector.copy()
    # np.ndarray
    with pytest.raises(ValueError, match="Unable to set"):
        vector_model.set_vector(data=testds["zs"].values)
    with pytest.raises(
        ValueError, match="set_vector with np.ndarray is only supported if data is 1D"
    ):
        vector_model.set_vector(data=testds["zs"].values, name="precip")
    # xr.DataArray
    vector_model.set_vector(data=testds["zs"], name="precip")
    # geodataframe
    gdf = testds.vector.geometry.to_frame("geometry")
    gdf["param1"] = np.random.rand(gdf.shape[0])
    gdf["param2"] = np.random.rand(gdf.shape[0])
    vector_model.set_vector(data=gdf)
    assert "precip" in vector_model.vector
    assert "param1" in vector_model.vector
    # geometry and update grid
    geoda_test = geoda.vector.update_geometry(geoda.vector.geometry.buffer(0.1))
    with pytest.raises(ValueError, match="Geometry of data and vector do not match"):
        vector_model.set_vector(data=geoda_test)
    param3 = vector_model.vector["param1"].sel(index=slice(0, 3)).drop_vars("geometry")
    with pytest.raises(ValueError, match="Index coordinate of data variable"):
        vector_model.set_vector(data=param3, name="param3")
    vector_model.set_vector(data=geoda, overwrite_geom=True, name="zs")
    assert "param1" not in vector_model.vector
    assert "zs" in vector_model.vector

    # test write vector
    vector_model.set_vector(data=gdf)
    vector_model.set_root(str(tmpdir), mode="w")
    # netcdf+geojson --> tested in test_vectormodel
    # netcdf only
    vector_model.write_vector(fn="vector/vector_full.nc", fn_geom=None)
    # geojson only
    # automatic split
    vector_model.write_vector(fn=None, fn_geom="vector/vector_split.geojson")
    assert isfile(join(vector_model.root, "vector", "vector_split.nc"))
    assert not isfile(join(vector_model.root, "vector", "vector_all.nc"))
    # geojson 1D data only
    vector_model._vector = vector_model._vector.drop_vars("zs").drop_vars("time")
    vector_model.write_vector(fn=None, fn_geom="vector/vector_all2.geojson")
    assert not isfile(join(vector_model.root, "vector", "vector_all2.nc"))

    # test read vector
    vector_model1 = VectorModel(str(tmpdir), mode="r")
    # netcdf only
    vector_model1.read_vector(fn="vector/vector_full.nc", fn_geom=None)
    vector0 = vector_model1.vector
    assert len(vector0["zs"].dims) == 2
    vector_model1._vector = None
    # geojson only
    # automatic split
    vector_model1.read_vector(
        fn="vector/vector_split.nc", fn_geom="vector/vector_split.geojson"
    )
    vector1 = vector_model1.vector
    assert len(vector1["zs"].dims) == 2
    vector_model1._vector = None
    # geojson 1D data only
    vector_model1.read_vector(fn=None, fn_geom="vector/vector_all2.geojson")
    vector3 = vector_model1.vector
    assert "zs" not in vector3


def test_networkmodel(network_model, tmpdir):
    network_model.set_root(str(tmpdir), mode="r+")
    with pytest.raises(NotImplementedError):
        network_model.read(["network"])
    with pytest.raises(NotImplementedError):
        network_model.write(["network"])
    with pytest.raises(NotImplementedError):
        network_model.set_network()
    with pytest.raises(NotImplementedError):
        _ = network_model.network


@pytest.mark.skip(reason="Needs implementation of RasterDataSet.")
def test_meshmodel(mesh_model, tmpdir):
    MeshModel = MODELS.load("mesh_model")
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


@pytest.mark.skip(reason="Needs implementation of RasterDataSet.")
def test_setup_mesh(tmpdir, griduda):
    MeshModel = MODELS.load("mesh_model")
    # Initialize model
    model = MeshModel(
        root=join(tmpdir, "mesh_model"),
        data_libs=["artifact_data"],
        mode="w",
    )
    # wrong region kind
    with pytest.raises(ValueError, match="Region for mesh must be of kind "):
        model.setup_mesh2d(
            region={"basin": [12.5, 45.5]},
            res=0.05,
        )
    # bbox
    bbox = [12.05, 45.30, 12.85, 45.65]
    with pytest.raises(
        ValueError, match="res argument required for kind 'bbox', 'geom'"
    ):
        model.setup_mesh2d({"bbox": bbox})
    model.setup_mesh2d(
        region={"bbox": bbox},
        res=0.05,
        crs=4326,
        grid_name="mesh2d",
    )
    assert "mesh2d" in model.mesh_names
    assert model.crs.to_epsg() == 4326
    assert np.all(np.round(model.region.total_bounds, 3) == bbox)
    assert model.mesh.ugrid.grid.n_node == 136
    model._mesh = None  # remove old mesh

    # geom
    region = model._geoms.pop("region")
    model.setup_mesh2d(
        region={"geom": region},
        res=10000,
        crs="utm",
        grid_name="mesh2d",
    )
    assert model.crs.to_epsg() == 32633
    assert model.mesh.ugrid.grid.n_node == 35
    model._mesh = None  # remove old mesh

    # mesh
    # create mesh file
    mesh_fn = str(tmpdir.join("mesh2d.nc"))
    gridda = griduda.ugrid.to_dataset()
    gridda = gridda.rio.write_crs(griduda.ugrid.grid.crs)
    gridda.to_netcdf(mesh_fn)

    model.setup_mesh2d(
        region={"mesh": mesh_fn},
        grid_name="mesh2d",
    )
    assert np.all(griduda.ugrid.total_bounds == model.region.total_bounds)
    assert model.mesh.ugrid.grid.n_node == 169
    model._mesh = None  # remove old mesh

    # mesh with bounds
    bounds = [12.095, 46.495, 12.10, 46.50]
    model.setup_mesh2d(
        {"mesh": mesh_fn, "bounds": bounds},
        grid_name="mesh1",
    )
    assert "mesh1" in model.mesh_names
    assert model.mesh.ugrid.grid.n_node == 49
    assert np.all(np.round(model.region.total_bounds, 3) == bounds)


@pytest.mark.skip(reason="Needs implementation of RasterDataSet.")
def test_meshmodel_setup(griduda, world):
    MeshModel = MODELS.load("mesh_model")
    dc_param_fn = join(DATADIR, "parameters_data.yml")
    mod = MeshModel(data_libs=["artifact_data", dc_param_fn])
    mod.setup_config(**{"header": {"setting": "value"}})
    region = {"geom": world[world.name == "Italy"]}
    mod.setup_mesh2d(region, res=10000, crs=3857, grid_name="mesh2d")
    _ = mod.region

    region = {"mesh": griduda}
    mod1 = MeshModel(data_libs=["artifact_data", dc_param_fn])
    mod1.setup_mesh2d(region, grid_name="mesh2d")
    mod1.setup_mesh2d_from_rasterdataset(
        "vito", grid_name="mesh2d", resampling_method="mode"
    )
    assert "vito" in mod1.mesh.data_vars
    mod1.setup_mesh2d_from_raster_reclass(
        raster_fn="vito",
        reclass_table_fn="vito_mapping",
        reclass_variables=["landuse", "roughness_manning"],
        resampling_method=["mode", "centroid"],
        grid_name="mesh2d",
    )
    assert "roughness_manning" in mod1.mesh.data_vars
    assert np.all(mod1.mesh["landuse"].values == mod1.mesh["vito"].values)
