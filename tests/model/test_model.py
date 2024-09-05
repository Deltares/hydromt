# -*- coding: utf-8 -*-
"""Tests for the hydromt.model module of HydroMT."""

from os import listdir, makedirs
from os.path import abspath, dirname, isdir, isfile, join
from pathlib import Path
from typing import List, cast
from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest_mock import MockerFixture

from hydromt.data_catalog import DataCatalog
from hydromt.model import Model
from hydromt.model.components.base import ModelComponent
from hydromt.model.components.config import ConfigComponent
from hydromt.model.components.geoms import GeomsComponent
from hydromt.model.components.grid import GridComponent
from hydromt.model.components.spatial import SpatialModelComponent
from hydromt.model.components.spatialdatasets import SpatialDatasetsComponent
from hydromt.model.components.tables import TablesComponent
from hydromt.model.components.vector import VectorComponent

DATADIR = join(dirname(abspath(__file__)), "..", "data")
DC_PARAM_PATH = join(DATADIR, "parameters_data.yml")


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
    mocker.patch("hydromt.model.model.PLUGINS", component_plugins=type_mocks)
    return [type_mocks[c.__name__].return_value for c in component_classes]


def test_from_dict_simple():
    d = {
        "modeltype": "model",
        "global": {
            "components": {
                "grid": {"type": "GridComponent"},
                "config": {"type": "ConfigComponent"},
            },
            "region_component": "grid",
        },
    }

    model = Model.from_dict(d)
    grid_comp = model.components["grid"]
    assert isinstance(grid_comp, GridComponent)
    config_comp = model.components["config"]
    assert isinstance(config_comp, ConfigComponent)
    assert model._region_component_name == "grid"


def test_from_dict_multiple_spatials():
    d = {
        "modeltype": "model",
        "global": {
            "components": {
                "grid1": {"type": "GridComponent"},
                "grid2": {"type": "GridComponent"},
                "forcing": {
                    "type": "SpatialDatasetsComponent",
                    "region_component": "grid1",
                },
            },
            "region_component": "grid2",
        },
    }

    model = Model.from_dict(d)
    grid_comp = model.components["grid1"]
    assert isinstance(grid_comp, GridComponent)
    forcing_component = model.components["forcing"]
    assert isinstance(forcing_component, SpatialDatasetsComponent)
    assert forcing_component._region_component == "grid1"
    assert model._region_component_name == "grid2"


def test_write_data_catalog_no_used(tmpdir):
    model = Model(root=join(tmpdir, "model"), data_libs=["artifact_data"])
    data_lib_path = join(model.root.path, "hydromt_data.yml")
    model.write_data_catalog()
    assert not isfile(data_lib_path)


def test_write_data_catalog_single_source(tmpdir):
    model = Model(root=join(tmpdir, "model"), data_libs=["artifact_data"])
    data_lib_path = join(model.root.path, "hydromt_data.yml")
    sources = list(model.data_catalog.sources.keys())
    model.data_catalog.get_source(sources[0])._mark_as_used()
    model.write_data_catalog()
    assert list(DataCatalog(data_lib_path).sources.keys()) == sources[:1]


def test_write_data_catalog_append(tmpdir):
    model = Model(root=join(tmpdir, "model"), data_libs=["artifact_data"])
    data_lib_path = join(model.root.path, "hydromt_data.yml")
    sources = list(model.data_catalog.sources.keys())
    model1 = Model(root=str(model.root.path), data_libs=["artifact_data"], mode="r+")
    model1.data_catalog.get_source(sources[1])._mark_as_used()
    model1.write_data_catalog(append=False)
    assert list(DataCatalog(data_lib_path).sources.keys()) == [sources[1]]
    model1.data_catalog.get_source(sources[0])._mark_as_used()
    model1.write_data_catalog(append=True)
    assert list(DataCatalog(data_lib_path).sources.keys()) == sources[:2]


def test_write_data_catalog_csv(tmpdir):
    model = Model(root=join(tmpdir, "model"), data_libs=["artifact_data"])
    sources = list(model.data_catalog.sources.keys())
    model.write_data_catalog(used_only=False, save_csv=True)
    assert isfile(join(model.root.path, "hydromt_data.csv"))
    data_catalog_df = pd.read_csv(join(model.root.path, "hydromt_data.csv"))
    assert len(data_catalog_df) == len(sources)
    assert data_catalog_df.iloc[0, 0] == sources[0]
    assert data_catalog_df.iloc[-1, 0] == sources[-1]


def test_model_mode_errors_reading_in_write_only(grid_model, tmpdir):
    # write model
    grid_model.root.set(str(tmpdir), mode="w")
    grid_model.write()
    with pytest.raises(IOError, match="Model opened in write-only mode"):
        grid_model.read()


def test_model_mode_errors_writing_in_read_only(grid_model):
    # read model
    grid_model.root.mode = "r"
    with pytest.raises(IOError, match="Model opened in read-only mode"):
        grid_model.write()


@pytest.mark.integration()
def test_grid_model_append(demda, df, tmpdir):
    demda.name = "dem"
    model = Model(mode="w", root=str(tmpdir))

    config_component = ConfigComponent(model)
    config_component.set("test.data", "dem")
    model.add_component("config", config_component)

    grid_component = GridComponent(model)
    grid_component.set(demda, name="dem")
    model.add_component("grid", grid_component)

    maps_component = SpatialDatasetsComponent(model, region_component="grid")
    maps_component.set(demda, name="maps")
    model.add_component("maps", maps_component)

    forcing_component = SpatialDatasetsComponent(model, region_component="grid")
    forcing_component.set(demda, name="forcing")
    model.add_component("forcing", forcing_component)

    tables_component = TablesComponent(model)
    tables_component.set(df, name="df")
    model.add_component("tables", tables_component)

    model.write()

    # append to model and check if previous data is still there
    demda.name = "dem"
    model2 = Model(mode="w", root=str(tmpdir))

    config_component = ConfigComponent(model2)
    config_component.set("test.data", "dem")
    model2.add_component("config", config_component)

    grid_component = GridComponent(model2)
    grid_component.set(demda, name="dem")
    model2.add_component("grid", grid_component)

    maps_component = SpatialDatasetsComponent(model2, region_component="grid")
    maps_component.set(demda, name="maps")
    model2.add_component("maps", maps_component)

    forcing_component = SpatialDatasetsComponent(model2, region_component="grid")
    forcing_component.set(demda, name="forcing")
    model2.add_component("forcing", forcing_component)

    tables_component = TablesComponent(model2)
    tables_component.set(df, name="df")
    model2.add_component("tables", tables_component)

    model2.config.set("test1.data", "dem")
    assert model2.config.get_value("test.data") == "dem"

    model2.grid.set(demda, name="dem1")
    assert "dem" in model2.grid.data

    model2.maps.set(demda, name="dem1")
    assert "maps" in model2.maps.data

    model2.forcing.set(demda, name="dem1")
    assert "forcing" in model2.forcing.data

    model2.forcing.set(df, name="df1", split_dataset=False)
    assert "df1" in model2.forcing.data
    assert isinstance(model2.forcing.data["df1"], xr.Dataset)

    model2.tables.set(df, name="df1")
    assert "df" in model2.tables.data


@pytest.mark.integration()
def test_model_build_update(tmpdir, demda, obsda):
    bbox = [12.05, 45.30, 12.85, 45.65]
    model = Model(
        root=str(tmpdir),
        mode="w",
        components={"grid": {"type": "GridComponent"}},
        region_component="grid",
    )
    geoms_component = GeomsComponent(model)
    model.add_component("geoms", geoms_component)
    model.build(
        steps=[
            {"grid.create_from_region": {"region": {"bbox": bbox}, "res": 0.01}},
            {
                "grid.add_data_from_constant": {
                    "constant": 0.01,
                    "name": "c1",
                    "nodata": -99.0,
                }
            },
        ]
    )
    assert isfile(join(model.root.path, "hydromt.log"))
    assert isdir(join(model.root.path, "grid")), listdir(model.root.path)
    assert isfile(join(model.root.path, "grid", "grid_region.geojson")), listdir(
        model.root.path
    )

    # read and update model
    model = Model(
        root=str(tmpdir),
        mode="r",
        components={"grid": {"type": "GridComponent"}},
        region_component="grid",
    )
    geoms_component = GeomsComponent(model)
    model.add_component("geoms", geoms_component)
    model_out = str(tmpdir.join("update"))
    model.update(model_out=model_out, steps=[])  # write only
    assert isdir(join(model_out, "grid")), listdir(model_out)
    assert isfile(join(model_out, "grid", "grid_region.geojson")), listdir(model_out)


@pytest.mark.integration()
def test_model_build_update_with_data(tmpdir, demda, obsda, monkeypatch):
    # users will not have a use for `set` in their yaml file because there is
    # nothing they will have access to then that they cat set it to
    # so we want to keep `SpatialDatasetsComponent.set` a non-hydromt-step
    # but for testing it's much easier, so we'll enable it
    # for the duration of this test with monkeypatch
    monkeypatch.setattr(
        SpatialDatasetsComponent.set, "__ishydromtstep__", True, raising=False
    )
    # Build model with some data
    bbox = [12.05, 45.30, 12.85, 45.65]
    model = Model(
        root=str(tmpdir),
        components={
            "grid": {"type": "GridComponent"},
            "maps": {"type": "SpatialDatasetsComponent", "region_component": "grid"},
            "forcing": {"type": "SpatialDatasetsComponent", "region_component": "grid"},
            "geoms": {"type": "GeomsComponent"},
        },
        region_component="grid",
        mode="w+",
    )
    model.build(
        steps=[
            {
                "grid.create_from_region": {
                    "region": {"bbox": bbox},
                    "res": 0.01,
                    "crs": 4326,
                }
            },
            {
                "maps.set": {
                    "data": demda,
                    "name": "elevtn",
                }
            },
            {
                "forcing.set": {
                    "data": obsda,
                    "name": "precip",
                }
            },
        ],
    )
    # Now update the model
    model = Model(
        root=str(tmpdir),
        components={
            "grid": {"type": "GridComponent"},
            "maps": {"type": "SpatialDatasetsComponent", "region_component": "grid"},
            "forcing": {"type": "SpatialDatasetsComponent", "region_component": "grid"},
            "forcing2": {
                "type": "SpatialDatasetsComponent",
                "region_component": "grid",
            },
            "geoms": {"type": "GeomsComponent"},
        },
        region_component="grid",
        mode="r+",
    )
    model.update(
        steps=[
            {"maps.set": {"data": demda, "name": "elevtn2"}},
            {"forcing.set": {"data": obsda, "name": "temp"}},
            {"forcing2.set": {"data": obsda * 0.2, "name": "precip"}},
        ]
    )
    assert len(model._defered_file_closes) == 0
    # Check that variables from build AND update are present
    assert "elevtn" in model.maps.data
    assert "elevtn2" in model.maps.data
    assert "precip" in model.forcing.data
    assert "temp" in model.forcing.data


def test_setup_region_geom(grid_model, bbox):
    # geom
    grid_model.grid.create_from_region({"geom": bbox}, res=0.001)
    gpd.testing.assert_geodataframe_equal(
        bbox, grid_model.region, check_less_precise=True
    )


def test_setup_region_geom_catalog(grid_model, bbox, tmpdir):
    # geom via data catalog
    region_path = str(tmpdir.join("region.gpkg"))
    bbox.to_file(region_path, driver="GPKG")
    grid_model.data_catalog.from_dict(
        {
            "region": {
                "uri": region_path,
                "data_type": "GeoDataFrame",
                "driver": "pyogrio",
            }
        }
    )
    grid_model.grid.create_from_region({"geom": "region"}, res=0.01)
    gpd.testing.assert_geodataframe_equal(
        bbox, grid_model.region, check_less_precise=True
    )


def test_setup_region_grid(grid_model, demda, tmpdir):
    # grid
    grid_path = str(tmpdir.join("grid.tif"))
    demda.raster.to_raster(grid_path)
    grid_model.grid.create_from_region({"grid": grid_path})
    assert np.all(demda.raster.bounds == grid_model.region.total_bounds)


@pytest.mark.skip(reason="needs fix with hydrography_path?")
def test_setup_region_basin(model):
    # basin
    model.grid.create_from_region({"basin": [12.2, 45.833333333333329]})
    assert np.all(model.region["value"] == 210000039)  # basin id


@pytest.mark.integration()
def test_maps_setup():
    mod = Model(
        data_libs=["artifact_data", DC_PARAM_PATH],
        components={"grid": {"type": "GridComponent"}},
        region_component="grid",
        mode="w",
    )
    bbox = [11.80, 46.10, 12.10, 46.50]  # Piava river

    mod.grid.create_from_region({"bbox": bbox}, res=0.1)
    config_component = ConfigComponent(mod)
    config_component.set("header.setting", "value")
    mod.add_component("config", config_component)

    maps_component = SpatialDatasetsComponent(mod, region_component="grid")
    mod.add_component("maps", maps_component)

    mod.maps.add_raster_data_from_rasterdataset(
        raster_filename="merit_hydro",
        name="hydrography",
        variables=["elevtn", "flwdir"],
        split_dataset=False,
    )
    mod.maps.add_raster_data_from_rasterdataset(
        raster_filename="vito", fill_method="nearest"
    )
    mod.maps.add_raster_data_from_raster_reclass(
        raster_filename="vito",
        reclass_table_filename="vito_mapping",
        reclass_variables=["roughness_manning"],
        split_dataset=True,
    )

    assert len(mod.maps.data) == 3
    assert "roughness_manning" in mod.maps.data
    assert len(mod.maps.data["hydrography"].data_vars) == 2


@pytest.mark.integration()
def test_gridmodel(demda, tmpdir):
    grid_model = Model(
        root=str(tmpdir),
        data_libs=["artifact_data", DC_PARAM_PATH],
        components={"grid": {"type": "GridComponent"}},
        region_component="grid",
        mode="w",
    )
    bbox = [12.05, 45.30, 12.85, 45.65]
    grid_model.grid.create_from_region(
        region={"bbox": bbox},
        res=0.05,
        crs=4326,
    )
    grid_model.grid.add_data_from_constant(
        constant=0.01,
        name="c1",
        nodata=-99.0,
    )
    # grid specific attributes
    assert np.all(grid_model.grid.res == grid_model.grid.data.raster.res)
    assert np.all(grid_model.grid.bounds == grid_model.grid.data.raster.bounds)
    assert np.all(grid_model.grid.transform == grid_model.grid.data.raster.transform)

    # write model
    grid_model.write()
    # read model
    model1 = Model(
        root=str(tmpdir),
        data_libs=["artifact_data", DC_PARAM_PATH],
        components={"grid": {"type": "GridComponent"}},
        region_component="grid",
        mode="r",
    )
    model1.read()
    # check if equal
    equal, errors = grid_model.test_equal(model1)
    assert equal, errors

    # try update
    grid_model.root.set(str(join(tmpdir, "update")), mode="w")
    grid_model.write()

    update_root = str(join(tmpdir, "update"))
    model1 = Model(
        root=update_root,
        data_libs=["artifact_data", DC_PARAM_PATH],
        components={"grid": {"type": "GridComponent"}},
        region_component="grid",
        mode="r+",
    )
    model1.grid.set(demda, name="testdata")
    model1.grid.write()
    assert "testdata" in model1.grid.data


def test_setup_grid_from_wrong_kind(grid_model):
    with pytest.raises(ValueError, match="Region for grid must be of kind"):
        grid_model.grid.create_from_region({"vector_model": "test_model"})


def test_setup_grid_from_bbox_aligned(grid_model):
    bbox = [12.05, 45.30, 12.85, 45.65]
    with pytest.raises(
        ValueError, match="res argument required for kind 'bbox', 'geom'"
    ):
        grid_model.grid.create_from_region({"bbox": bbox})

    grid_model.grid.create_from_region(
        region={"bbox": bbox},
        res=0.05,
        add_mask=False,
        align=True,
    )
    grid_model.grid.add_data_from_constant(
        constant=0.01,
        name="c1",
        nodata=-99.0,
    )
    assert "mask" not in grid_model.grid.data
    assert grid_model.crs.to_epsg() == 4326
    assert grid_model.grid.data.raster.dims == ("y", "x")
    assert grid_model.grid.data.raster.shape == (7, 16)
    assert np.all(np.round(grid_model.grid.data.raster.bounds, 2) == bbox)


def test_setup_grid_from_wrong_kind_no_mask(grid_model):
    bbox = [12.05, 45.30, 12.85, 45.65]
    grid_model_tmp = Model(
        data_libs=["artifact_data", DC_PARAM_PATH],
        components={"grid": {"type": "GridComponent"}},
        region_component="grid",
    )
    grid_model_tmp.grid.create_from_region(
        region={"bbox": bbox}, res=0.05, add_mask=False, align=True, crs=4326
    )
    grid_model_tmp.grid.add_data_from_constant(
        constant=0.01,
        name="c1",
        nodata=-99.0,
    )

    region = grid_model_tmp.region
    grid = grid_model_tmp.grid.data
    grid_model.grid.create_from_region(
        region={"geom": region},
        res=0.05,
        add_mask=False,
    )
    grid_model.grid.add_data_from_constant(
        constant=0.01,
        name="c1",
        nodata=-99.0,
    )
    gpd.testing.assert_geodataframe_equal(region, grid_model.region)
    xr.testing.assert_allclose(grid, grid_model.grid.data)


@pytest.mark.skip("utm is not a valid crs?")
def test_setup_grid_from_geodataframe(grid_model):
    bbox = [12.05, 45.30, 12.85, 45.65]
    grid_model_tmp = Model(
        data_libs=["artifact_data", DC_PARAM_PATH],
        components={"grid": {"type": "GridComponent"}},
        region_component="grid",
    )
    grid_model_tmp.grid.create_from_region(
        region={"bbox": bbox}, res=0.05, add_mask=False, align=True, crs=4326
    )
    grid_model_tmp.grid.add_data_from_constant(
        constant=0.01,
        name="c1",
        nodata=-99.0,
    )
    region = grid_model_tmp.region
    grid_model.grid.create_from_region(
        region={"geom": region},
        res=10000,
        crs="utm",
        add_mask=True,
    )
    assert "mask" in grid_model.grid.data
    assert grid_model.crs.to_epsg() == 32633
    assert grid_model.grid.data.raster.res == (10000, -10000)


def test_setup_grid_from_bbox_rotated(grid_model):
    grid_model.grid.create_from_region(
        region={"bbox": [12.65, 45.50, 12.85, 45.60]},
        res=0.05,
        crs=4326,
        rotated=True,
        add_mask=True,
    )
    assert "xc" in grid_model.grid.data.coords
    assert grid_model.grid.data.raster.y_dim == "y"
    assert np.isclose(grid_model.grid.data.raster.res[0], 0.05)


def test_setup_grid_from_grid(grid_model, demda):
    # grid
    grid_path = str(grid_model.root.path / "grid.tif")
    demda.raster.to_raster(grid_path)
    grid_model.grid.create_from_region({"grid": grid_path})
    assert np.all(demda.raster.bounds == grid_model.region.bounds)


def test_grid_model_subbasin(grid_model):
    grid_model.grid.create_from_region(
        {"subbasin": [12.319, 46.320], "uparea": 50},
        res=0.008333,
        hydrography_path="merit_hydro",
        basin_index_path="merit_hydro_index",
        add_mask=True,
    )
    assert not np.all(grid_model.grid.data["mask"].values is True)
    assert grid_model.grid.data.raster.shape == (50, 93)


def test_grid_model_constant(grid_model):
    bbox = [12.05, 45.30, 12.85, 45.65]
    grid_model.grid.create_from_region(
        region={"bbox": bbox},
        res=0.05,
        add_mask=False,
        align=True,
    )
    grid_model.grid.add_data_from_constant(
        constant=0.01,
        name="c1",
        nodata=-99.0,
    )
    assert "c1" in grid_model.grid.data


def test_grid_model_constant_dtype(grid_model):
    bbox = [12.05, 45.30, 12.85, 45.65]
    grid_model.grid.create_from_region(
        region={"bbox": bbox},
        res=0.05,
        add_mask=False,
        align=True,
    )
    grid_model.grid.add_data_from_constant(
        constant=2,
        name="c2",
        dtype=np.int8,
        nodata=-1,
    )
    assert np.unique(grid_model.grid.data["c2"]).size == 1
    assert np.isin([2], np.unique(grid_model.grid.data["c2"])).all()
    assert "c2" in grid_model.grid.data


def test_grid_model_raster_dataset_merit_hydro(grid_model):
    bbox = [12.05, 45.30, 12.85, 45.65]
    grid_model.grid.create_from_region(
        region={"bbox": bbox},
        res=0.05,
        add_mask=False,
        align=True,
    )
    grid_model.grid.add_data_from_rasterdataset(
        raster_data="merit_hydro",
        variables=["elevtn", "basins"],
        reproject_method=["average", "mode"],
        mask_name="mask",
    )
    assert "basins" in grid_model.grid.data


def test_grid_model_raster_dataset_vito(grid_model):
    bbox = [12.05, 45.30, 12.85, 45.65]
    grid_model.grid.create_from_region(
        region={"bbox": bbox},
        res=0.05,
        add_mask=False,
        align=True,
    )
    grid_model.grid.add_data_from_rasterdataset(
        raster_data="vito",
        fill_method="nearest",
        reproject_method="mode",
        rename={"vito": "landuse"},
    )
    assert "landuse" in grid_model.grid.data


def test_grid_model_raster_reclass(grid_model):
    bbox = [12.05, 45.30, 12.85, 45.65]
    grid_model.grid.create_from_region(
        region={"bbox": bbox},
        res=0.05,
        add_mask=False,
        align=True,
    )
    grid_model.grid.add_data_from_raster_reclass(
        raster_data="vito",
        fill_method="nearest",
        reclass_table_data="vito_mapping",
        reclass_variables=["roughness_manning"],
        reproject_method=["average"],
    )

    assert grid_model.grid.data["roughness_manning"].raster.nodata == -999.0
    assert "roughness_manning" in grid_model.grid.data


def test_grid_model_geodataframe_value(grid_model):
    bbox = [12.05, 45.30, 12.85, 45.65]
    grid_model.grid.create_from_region(
        region={"bbox": bbox},
        res=0.05,
        add_mask=False,
        align=True,
    )
    grid_model.grid.add_data_from_geodataframe(
        vector_data="hydro_lakes",
        variables=["waterbody_id", "Depth_avg"],
        nodata=[-1, -999.0],
        rasterize_method="value",
        rename={"waterbody_id": "lake_id", "Depth_avg": "lake_depth"},
    )
    assert grid_model.grid.data["lake_depth"].raster.nodata == -999.0
    assert "lake_depth" in grid_model.grid.data


def test_grid_model_geodataframe_fraction(grid_model):
    bbox = [12.05, 45.30, 12.85, 45.65]
    grid_model.grid.create_from_region(
        region={"bbox": bbox},
        res=0.05,
        add_mask=False,
        align=True,
    )
    grid_model.grid.add_data_from_geodataframe(
        vector_data="hydro_lakes",
        rasterize_method="fraction",
        rename={"hydro_lakes": "water_frac"},
    )

    assert "water_frac" in grid_model.grid.data


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


def test_empty_mesh_model(mesh_model):
    mesh_model.write()
    # read model
    mesh_model2 = Model(
        data_libs=["artifact_data"],
        components={"mesh": {"type": "MeshComponent"}},
        region_component="mesh",
        mode="r",
    )
    mesh_model2.read()
    # check if equal
    equal, errors = mesh_model.test_equal(mesh_model2)
    assert equal, errors


def test_setup_mesh_from_wrong_kind(mesh_model):
    with pytest.raises(
        ValueError, match="Unsupported region kind 'basin' found in grid creation."
    ):
        mesh_model.mesh.create_2d_from_region(
            region={"basin": [12.5, 45.5]},
            res=0.05,
        )


def test_setup_mesh_from_bbox(mesh_model):
    bbox = [12.05, 45.30, 12.85, 45.65]
    with pytest.raises(ValueError, match="res argument required for kind bbox"):
        mesh_model.mesh.create_2d_from_region({"bbox": bbox})

    mesh_model.mesh.create_2d_from_region(
        region={"bbox": bbox},
        res=0.05,
        crs=4326,
        grid_name="mesh2d",
    )
    # need to add some data to it before checks will work
    mesh_model.mesh.add_2d_data_from_rasterdataset(
        "vito", grid_name="mesh2d", resampling_method="mode"
    )

    assert "vito" in mesh_model.mesh.data
    assert mesh_model.crs.to_epsg() == 4326
    assert np.all(np.round(mesh_model.region.total_bounds, 3) == bbox)
    assert mesh_model.mesh.data.ugrid.grid.n_node == 136


def test_setup_mesh_from_geom(mesh_model, tmpdir):
    bbox = [12.05, 45.30, 12.85, 45.65]
    dummy_mesh_model = Model(
        root=str(tmpdir),
        data_libs=["artifact_data", DC_PARAM_PATH],
        components={"mesh": {"type": "MeshComponent"}},
        region_component="mesh",
    )
    dummy_mesh_model.mesh.create_2d_from_region(
        region={"bbox": bbox},
        res=0.05,
        crs=4326,
        grid_name="mesh2d",
    )
    # need to add some data to it before checks will work
    dummy_mesh_model.mesh.add_2d_data_from_rasterdataset(
        "vito", grid_name="mesh2d", resampling_method="mode"
    )
    region = dummy_mesh_model.mesh.region
    mesh_model.mesh.create_2d_from_region(
        region={"geom": region},
        res=10000,
        crs="utm",
        grid_name="mesh2d",
    )
    assert mesh_model.crs.to_epsg() == 32633
    assert mesh_model.mesh.data.ugrid.grid.n_node == 35


def test_setup_mesh_from_mesh(mesh_model, griduda):
    mesh_path = str(mesh_model.root.path / "mesh" / "mesh.nc")
    makedirs(mesh_model.root.path / "mesh", exist_ok=True)
    gridda = griduda.ugrid.to_dataset()
    gridda = gridda.rio.write_crs(griduda.ugrid.grid.crs)
    gridda.to_netcdf(mesh_path)

    mesh_model.mesh.create_2d_from_region(
        region={"mesh": mesh_path},
        grid_name="mesh2d",
    )
    # need to add some data to it before checks will work
    mesh_model.mesh.add_2d_data_from_rasterdataset(
        "vito", grid_name="mesh2d", resampling_method="mode"
    )

    assert np.all(griduda.ugrid.total_bounds == mesh_model.mesh.region.total_bounds)
    assert mesh_model.mesh.data.ugrid.grid.n_node == 169


def test_setup_mesh_from_mesh_with_bounds(mesh_model, griduda):
    mesh_path = str(mesh_model.root.path / "mesh" / "mesh.nc")
    makedirs(mesh_model.root.path / "mesh", exist_ok=True)
    gridda = griduda.ugrid.to_dataset()
    gridda = gridda.rio.write_crs(griduda.ugrid.grid.crs)
    gridda.to_netcdf(mesh_path)

    mesh_model.mesh.create_2d_from_region(
        region={"mesh": mesh_path},
        grid_name="mesh2d",
    )
    bounds = [12.095, 46.495, 12.10, 46.50]
    mesh_model.mesh.create_2d_from_region(
        {"mesh": mesh_path, "bounds": bounds},
        grid_name="mesh1",
    )
    # need to add some data to it before checks will work
    mesh_model.mesh.add_2d_data_from_rasterdataset(
        "vito", grid_name="mesh1", resampling_method="mode"
    )
    assert "vito" in mesh_model.mesh.data, mesh_model.mesh.data
    assert mesh_model.mesh.data.ugrid.grid.n_node == 49
    assert np.all(np.round(mesh_model.region.total_bounds, 3) == bounds)


@pytest.mark.skip("needs oracle")
def test_mesh_model_setup_grid(mesh_model, world):
    region = {"geom": world[world.name == "Italy"]}
    mesh_model.mesh.create_2d_from_region(
        region, res=10000, crs=3857, grid_name="mesh2d"
    )
    assert mesh_model.mesh.data.equals(region["geom"])


def test_mesh_model_setup_from_raster_dataset(mesh_model, griduda):
    region = {"mesh": griduda}
    mesh_model.mesh.create_2d_from_region(region, grid_name="mesh2d")
    mesh_model.mesh.add_2d_data_from_rasterdataset(
        "vito", grid_name="mesh2d", resampling_method="mode"
    )
    assert "vito" in mesh_model.mesh.data.data_vars


def test_mesh_model_setup_from_raster_reclass(mesh_model, griduda):
    region = {"mesh": griduda}
    mesh_model.mesh.create_2d_from_region(region, grid_name="mesh2d")
    mesh_model.mesh.add_2d_data_from_rasterdataset(
        "vito", grid_name="mesh2d", resampling_method="mode"
    )
    mesh_model.mesh.add_2d_data_from_raster_reclass(
        raster_filename="vito",
        reclass_table_filename="vito_mapping",
        reclass_variables=["landuse", "roughness_manning"],
        resampling_method=["mode", "centroid"],
        grid_name="mesh2d",
    )
    ds_mesh2d = mesh_model.mesh.get_mesh("mesh2d", include_data=True)
    assert "vito" in ds_mesh2d
    assert "roughness_manning" in mesh_model.mesh.data.data_vars
    assert np.all(
        mesh_model.mesh.data["landuse"].values == mesh_model.mesh.data["vito"].values
    )


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


def test_build_non_existing_step(tmpdir: Path):
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
