# -*- coding: utf-8 -*-
"""Tests for the hydromt.models module of HydroMT."""

from os import listdir
from os.path import abspath, dirname, isfile, join
from pathlib import Path
from typing import Any, List, cast
from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest_mock import MockerFixture
from shapely.geometry import box

from hydromt.components.base import ModelComponent
from hydromt.components.config import ConfigComponent
from hydromt.components.grid import GridComponent
from hydromt.components.spatial import SpatialModelComponent
from hydromt.components.vector import VectorComponent
from hydromt.data_catalog import DataCatalog
from hydromt.models import Model
from hydromt.plugins import PLUGINS

DATADIR = join(dirname(abspath(__file__)), "..", "data")


def _patch_plugin_components(
    mocker: MockerFixture, *component_classes: type
) -> List[Mock]:
    """Set up PLUGINS with mocked classes.

    Returns a list of mocked instances of the classes.
    These will be the components of the model.
    """
    type_mocks = {}
    for c in component_classes:
        class_type_mock = mocker.Mock(return_value=mocker.Mock(spec_set=c))
        type_mocks[c.__name__] = class_type_mock
    mocker.patch("hydromt.models.model.PLUGINS", component_plugins=type_mocks)
    return [type_mocks[c.__name__].return_value for c in component_classes]


@pytest.mark.skip(reason="Needs implementation of new Model class with GridComponent.")
def test_api_attrs():
    # class _DummyModel(GridModel, GridMixin):
    # _API = {"asdf": "yeah"}
    # dm = _DummyModel()
    dm: Any = ...  # bypass ruff
    assert hasattr(dm, "name")
    assert hasattr(dm, "_API")
    assert "asdf" in dm.api
    assert dm.api["asdf"] == "yeah"
    assert "region" in dm.api
    assert dm.api["region"] == gpd.GeoDataFrame
    assert "grid" in dm.api
    assert dm.api["grid"] == xr.Dataset


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
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


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_model(model, tmpdir):
    # write model
    model.root.set(str(tmpdir), mode="w")
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
        equal, errors = model.test_equal(model1)
    assert equal, errors


@pytest.mark.skip(reason="Needs implementation of new Model class with GridComponent.")
def test_model_append(demda, df, tmpdir):
    # write a model
    GridModel: Any = ...  # bypass ruff
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


@pytest.mark.integration()
@pytest.mark.skip(reason="needs method/yaml validation")
def test_model_build_update(tmpdir, demda, obsda):
    bbox = [12.05, 45.30, 12.85, 45.65]
    # build model
    model = Model(root=str(tmpdir), mode="w")
    model.name = "model"
    model.build(
        region={"bbox": bbox},
        steps={
            "region.create": {},
            "region.write": {"components": {"geoms", "config"}},
        },
    )
    assert hasattr(model, "region")
    assert isfile(join(model.root.path, "model.yml"))
    assert isfile(join(model.root.path, "hydromt.log"))
    assert isfile(join(model.root.path, "region.geojson")), listdir(model.root.path)

    # read and update model
    model = Model(root=str(tmpdir), mode="r")
    model_out = str(tmpdir.join("update"))
    model.update(model_out=model_out, steps={})  # write only
    assert isfile(join(model_out, "model.yml"))


@pytest.mark.integration()
@pytest.mark.skip(reason="needs method/yaml validation")
def test_model_build_update_with_data(tmpdir, demda, obsda):
    # Build model with some data
    bbox = [12.05, 45.30, 12.85, 45.65]
    geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
    model = Model(root=str(tmpdir), mode="w+")
    model.name = "model"
    model.build(
        region={"bbox": bbox},
        steps={
            "setup_config": {"input": {"dem": "elevtn", "prec": "precip"}},
            "set_geoms": {"geom": geom, "name": "geom1"},
            "set_maps": {"data": demda, "name": "elevtn"},
            "set_forcing": {"data": obsda, "name": "precip"},
        },
    )
    # Now update the model
    model = Model(root=str(tmpdir), mode="r+")
    model.update(
        steps={
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


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_setup_region(model, demda, tmpdir):
    # bbox
    model.region.create({"bbox": [12.05, 45.30, 12.85, 45.65]})
    region = model._geoms.pop("region")
    # geom
    model.region.create({"geom": region})
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
    model.region.create({"geom": "region"})
    gpd.testing.assert_geodataframe_equal(region, model.region)
    # grid
    model._geoms.pop("region")  # remove old region
    grid_fn = str(tmpdir.join("grid.tif"))
    demda.raster.to_raster(grid_fn)
    model.region.create({"grid": grid_fn})
    assert np.all(demda.raster.bounds == model.region.total_bounds)
    # basin
    model._geoms.pop("region")  # remove old region
    model.region.create({"basin": [12.2, 45.833333333333329]})
    assert np.all(model.region["value"] == 210000039)  # basin id


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_maps_setup(tmpdir):
    dc_param_fn = join(DATADIR, "parameters_data.yml")
    mod = Model(data_libs=["artifact_data", dc_param_fn], mode="w")
    bbox = [11.80, 46.10, 12.10, 46.50]  # Piava river
    mod.region.create({"bbox": bbox})
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
    mod.root.set(str(tmpdir), mode="w")
    mod.write(components=["config", "geoms", "maps"])


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_gridmodel(tmpdir, demda):
    grid_model = Model(
        components={"grid": {"type": GridComponent.__name__}},
        root=str(tmpdir),
        mode="w",
    )
    # grid specific attributes
    assert np.all(grid_model.grid.res == grid_model.grid.data.raster.res)
    assert np.all(grid_model.grid.bounds == grid_model.grid.data.raster.bounds)
    assert np.all(grid_model.grid.transform == grid_model.grid.data.raster.transform)
    # write model
    grid_model.write()
    # read model
    model1 = Model(str(tmpdir), mode="r")
    model1.read()
    # check if equal
    equal, errors = grid_model.test_equal(model1)
    assert equal, errors

    # try update
    grid_model.root.set(str(join(tmpdir, "update")), mode="w")
    grid_model.write()

    model1 = Model(str(join(tmpdir, "update")), mode="r+")
    model1.grid.set(demda, name="testdata")
    model1.grid.write()
    assert "testdata" in model1.grid
    assert "elevtn" in model1.grid


@pytest.mark.skip(reason="Needs implementation of new Model class with GridComponent.")
def test_setup_grid(tmpdir, demda):
    # Initialize model
    GridModel: Any = ...  # bypass ruff
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
    assert np.all(demda.raster.bounds == model.region.bounds)
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


@pytest.mark.skip(reason="Needs implementation of new Model class with GridComponent.")
def test_gridmodel_setup(tmpdir):
    # Initialize model
    dc_param_fn = join(DATADIR, "parameters_data.yml")
    GridModel: Any = ...  # bypass ruff
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
    mod.root.set(str(tmpdir), mode="w")
    mod.write(components=["geoms", "grid"])


def test_vectormodel(vector_model, tmpdir, mocker: MockerFixture, geodf):
    # write model
    vector_model.root.set(str(tmpdir), mode="w")
    vector_model.write()
    # read model
    region_component = mocker.Mock(spec_set=SpatialModelComponent)
    region_component.test_equal.return_value = (True, {})
    region_component.region = geodf
    model1 = Model(
        root=str(tmpdir),
        mode="r",
        region_component="area",
        components={
            "area": region_component,
            "vector": {"type": VectorComponent.__name__, "region_component": "area"},
            "config": {"type": ConfigComponent.__name__},
        },
    )
    model1.read()
    equal, errors = vector_model.test_equal(model1)
    assert equal, errors


def test_vectormodel_vector(vector_model_no_defaults, tmpdir, geoda):
    # test set vector
    testds = vector_model_no_defaults.vector.data.copy()
    vector_component = cast(VectorComponent, vector_model_no_defaults.vector)
    # np.ndarray
    with pytest.raises(ValueError, match="Unable to set"):
        vector_component.set(testds["zs"].values)
    with pytest.raises(
        ValueError, match="set_vector with np.ndarray is only supported if data is 1D"
    ):
        vector_component.set(testds["zs"].values, name="precip")
    # xr.DataArray
    vector_component.set(testds["zs"], name="precip")
    # geodataframe
    gdf = testds.vector.geometry.to_frame("geometry")
    gdf["param1"] = np.random.rand(gdf.shape[0])
    gdf["param2"] = np.random.rand(gdf.shape[0])
    vector_component.set(gdf)
    assert "precip" in vector_model_no_defaults.vector.data
    assert "param1" in vector_model_no_defaults.vector.data
    # geometry and update grid
    crs = geoda.vector.crs
    geoda_test = geoda.vector.update_geometry(
        geoda.vector.geometry.to_crs(3857).buffer(0.1).to_crs(crs)
    )
    with pytest.raises(ValueError, match="Geometry of data and vector do not match"):
        vector_component.set(geoda_test)
    param3 = (
        vector_component.data["param1"].sel(index=slice(0, 3)).drop_vars("geometry")
    )
    with pytest.raises(ValueError, match="Index coordinate of data variable"):
        vector_component.set(param3, name="param3")
    vector_component.set(geoda, overwrite_geom=True, name="zs")
    assert "param1" not in vector_component.data
    assert "zs" in vector_component.data

    # test write vector
    vector_component.set(gdf)
    vector_model_no_defaults.root.set(str(tmpdir), mode="w")
    # netcdf+geojson --> tested in test_vectormodel
    # netcdf only
    vector_component.write(filename="vector/vector_full.nc", geometry_filename=None)
    # geojson only
    # automatic split
    vector_component.write(
        filename=None, geometry_filename="vector/vector_split.geojson"
    )
    assert isfile(join(vector_model_no_defaults.root.path, "vector", "vector_split.nc"))
    assert not isfile(
        join(vector_model_no_defaults.root.path, "vector", "vector_all.nc")
    )
    # geojson 1D data only
    vector_component._data = vector_component._data.drop_vars("zs").drop_vars("time")
    vector_component.write(
        filename=None, geometry_filename="vector/vector_all2.geojson"
    )
    assert not isfile(
        join(vector_model_no_defaults.root.path, "vector", "vector_all2.nc")
    )

    # test read vector
    vector_model1 = Model(root=str(tmpdir), mode="r")
    vector_model1.add_component("vector", VectorComponent(vector_model1))
    # netcdf only
    vector_model1.vector.read(filename="vector/vector_full.nc", geometry_filename=None)
    vector0 = vector_model1.vector.data
    assert len(vector0["zs"].dims) == 2
    vector_model1.vector._data = None
    # geojson only
    # automatic split
    vector_model1.vector.read(
        filename="vector/vector_split.nc",
        geometry_filename="vector/vector_split.geojson",
    )
    vector1 = vector_model1.vector.data
    assert len(vector1["zs"].dims) == 2
    vector_model1.vector._data = None
    # geojson 1D data only
    vector_model1.vector.read(
        filename=None, geometry_filename="vector/vector_all2.geojson"
    )
    vector3 = vector_model1.vector.data
    assert "zs" not in vector3


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_meshmodel(mesh_model, tmpdir):
    MeshModel = PLUGINS.model_plugins["mesh_model"]
    assert "mesh" in mesh_model.api
    non_compliant = mesh_model._test_model_api()
    assert len(non_compliant) == 0, non_compliant
    # write model
    mesh_model.root.set(str(tmpdir), mode="w")
    mesh_model.write()
    # read model
    model1 = MeshModel(str(tmpdir), mode="r")
    model1.read()
    # check if equal
    equal, errors = mesh_model.test_equal(model1)
    assert equal, errors


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_setup_mesh(tmpdir, griduda):
    MeshModel = PLUGINS.model_plugins["mesh_model"]
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


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_meshmodel_setup(griduda, world):
    MeshModel = PLUGINS.model_plugins["mesh_model"]
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
    ds_mesh2d = mod1.get_mesh("mesh2d", include_data=True)
    assert "vito" in ds_mesh2d
    assert "roughness_manning" in mod1.mesh.data_vars
    assert np.all(mod1.mesh["landuse"].values == mod1.mesh["vito"].values)


def test_initialize_with_region_component(mocker: MockerFixture):
    (region,) = _patch_plugin_components(mocker, SpatialModelComponent)
    m = Model(components={"region": {"type": SpatialModelComponent.__name__}})
    assert m.region is region.region


def test_initialize_model_with_grid_component():
    m = Model(components={"grid": {"type": GridComponent.__name__}})
    assert isinstance(m.grid, GridComponent)


def test_write_multiple_components(mocker: MockerFixture, tmpdir: Path):
    m = Model(
        root=str(tmpdir),
    )
    foo = mocker.Mock(spec_set=ModelComponent)
    bar = mocker.Mock(spec_set=ModelComponent)
    m.add_component("foo", foo)
    m.add_component("bar", bar)
    m.write()
    bar.write.assert_called_once()
    foo.write.assert_called_once()


def test_getattr_component(mocker: MockerFixture):
    foo = mocker.Mock(spec_set=ModelComponent)
    m = Model()
    m.add_component("foo", foo)
    assert m.foo is foo


def test_add_component_wrong_name(mocker: MockerFixture):
    m = Model()
    foo = mocker.Mock(spec_set=ModelComponent)
    with pytest.raises(
        ValueError, match="Component name foo foo is not a valid identifier."
    ):
        m.add_component("foo foo", foo)


def test_get_component_non_existent():
    m = Model()
    with pytest.raises(KeyError):
        m.get_component("foo")


def test_read_calls_components(mocker: MockerFixture):
    m = Model(mode="r")
    foo = mocker.Mock(spec_set=ModelComponent)
    m.add_component("foo", foo)
    m.read()
    foo.read.assert_called_once()


def test_build_two_components_writes_one(mocker: MockerFixture, tmpdir: Path):
    foo = mocker.Mock(spec_set=ModelComponent)
    foo.write.__ishydromtstep__ = True
    bar = mocker.Mock(spec_set=ModelComponent)
    m = Model(root=str(tmpdir))
    m.add_component("foo", foo)
    m.add_component("bar", bar)
    assert m.foo is foo

    # Specify to only write foo
    m.build(steps=[{"foo.write": {}}])

    bar.write.assert_not_called()  # Only foo will be written, so no total write
    foo.write.assert_called_once()  # foo was written, because it was specified in steps


def test_build_write_disabled_does_not_write(mocker: MockerFixture, tmpdir: Path):
    foo = mocker.Mock(spec_set=ModelComponent)
    m = Model(root=str(tmpdir))
    m.add_component("foo", foo)

    m.build(steps=[], write=False)

    foo.write.assert_not_called()


def test_build_non_existing_step(mocker: MockerFixture, tmpdir: Path):
    m = Model(root=str(tmpdir))

    with pytest.raises(KeyError):
        m.build(steps=[{"foo": {}}])


def test_add_component_duplicate_throws(mocker: MockerFixture):
    m = Model()
    foo = mocker.Mock(spec_set=ModelComponent)
    m.add_component("foo", foo)
    foo2 = mocker.Mock(spec_set=ModelComponent)

    with pytest.raises(ValueError, match="Component foo already exists in the model."):
        m.add_component("foo", foo2)


def test_update_empty_model_with_region_none_throws(
    tmpdir: Path, mocker: MockerFixture
):
    (foo,) = _patch_plugin_components(mocker, SpatialModelComponent)
    foo.region = None
    m = Model(
        root=str(tmpdir), components={"foo": {"type": SpatialModelComponent.__name__}}
    )
    with pytest.raises(
        ValueError, match="Model region not found, setup model using `build` first."
    ):
        m.update()


def test_update_in_read_mode_without_out_folder_throws(tmpdir: Path):
    m = Model(root=str(tmpdir), mode="r")
    with pytest.raises(
        ValueError,
        match='"model_out" directory required when updating in "read-only" mode.',
    ):
        m.update(model_out=None)


def test_update_in_read_mode_with_out_folder_sets_to_write_mode(
    tmpdir: Path, mocker: MockerFixture
):
    (region,) = _patch_plugin_components(mocker, SpatialModelComponent)
    m = Model(
        root=str(tmpdir),
        mode="r",
        components={"region": {"type": SpatialModelComponent.__name__}},
    )
    assert region.region is m.region

    m.update(model_out=str(tmpdir / "out"))

    assert m.root.is_writing_mode()
    assert not m.root.is_override_mode()
    region.read.assert_called_once()
    region.write.assert_called_once()
