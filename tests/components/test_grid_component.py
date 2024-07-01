import logging
from os.path import abspath, dirname, join
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest_mock import MockerFixture

from hydromt.model.components.grid import GridComponent
from hydromt.model.model import Model
from hydromt.model.root import ModelRoot

DATADIR = join(dirname(dirname(abspath(__file__))), "data")


def test_set_dataset(mock_model, hydds):
    grid_component = GridComponent(model=mock_model)
    grid_component.root.is_reading_mode.return_value = False
    grid_component.set(data=hydds)
    assert len(grid_component.data) > 0
    assert isinstance(grid_component.data, xr.Dataset)


def test_set_dataarray(mock_model, hydds):
    grid_component = GridComponent(model=mock_model)
    data_array = hydds.to_array()
    grid_component.root.is_reading_mode.return_value = False
    grid_component.set(data=data_array, name="data_array")
    assert "data_array" in grid_component.data.data_vars.keys()
    assert len(grid_component.data.data_vars) == 1


def test_set_raise_errors(mock_model, hydds):
    grid_component = GridComponent(model=mock_model)
    grid_component.root.is_reading_mode.return_value = False
    # Test setting nameless data array
    data_array = hydds.to_array()
    with pytest.raises(
        ValueError,
        match=f"Unable to set {type(data_array).__name__} data without a name",
    ):
        grid_component.set(data=data_array)
    # Test setting np.ndarray of different shape
    grid_component.set(data=data_array, name="data_array")
    ndarray = np.random.rand(4, 5)
    with pytest.raises(ValueError, match="Shape of data and grid maps do not match"):
        grid_component.set(ndarray, name="ndarray")


def test_write(
    mock_model, tmpdir, caplog: pytest.LogCaptureFixture, mocker: MockerFixture
):
    grid_component = GridComponent(model=mock_model)
    grid_component.root.is_reading_mode.return_value = False
    # Test skipping writing when no grid data has been set
    caplog.set_level(logging.WARNING)
    grid_component.write()
    assert "No grid data found, skip writing" in caplog.text
    # Test raise IOerror when model is in read only mode
    mock_model.root = ModelRoot(tmpdir, mode="r")
    grid_component = GridComponent(model=mock_model)
    mocker.patch.object(GridComponent, "data", ["test"])
    with pytest.raises(IOError, match="Model opened in read-only mode"):
        grid_component.write()


def test_read(tmpdir, mock_model, hydds, mocker: MockerFixture):
    # Test for raising IOError when model is in writing mode
    grid_component = GridComponent(model=mock_model)
    mock_model.root = ModelRoot(path=tmpdir, mode="w")
    with pytest.raises(IOError, match="Model opened in write-only mode"):
        grid_component.read()
    mock_model.root = ModelRoot(path=tmpdir, mode="r+")
    grid_component = GridComponent(model=mock_model)
    mocker.patch("hydromt.model.components.grid.read_nc", return_value={"grid": hydds})
    grid_component.read()
    assert grid_component.data == hydds


def test_create_grid_from_bbox_rotated(mock_model):
    grid_component = GridComponent(model=mock_model)
    grid_component.root.is_reading_mode.return_value = False
    grid_component.create_from_region(
        region={"bbox": [12.65, 45.50, 12.85, 45.60]},
        res=0.05,
        crs=4326,
        region_crs=4326,
        rotated=True,
        add_mask=True,
    )
    assert "xc" in grid_component.data.coords
    assert grid_component.data.raster.y_dim == "y"
    assert np.isclose(grid_component.data.raster.res[0], 0.05)
    assert isinstance(grid_component.region, gpd.GeoDataFrame)


def test_create_grid_from_bbox(mock_model):
    grid_component = GridComponent(model=mock_model)
    grid_component.root.is_reading_mode.return_value = False
    bbox = [12.05, 45.30, 12.85, 45.65]
    grid_component.create_from_region(
        region={"bbox": bbox},
        res=0.05,
        add_mask=True,
        align=True,
    )
    assert grid_component.data.raster.dims == ("y", "x")
    assert grid_component.data.raster.shape == (7, 16)
    assert np.all(np.round(grid_component.data.raster.bounds, 2) == bbox)
    assert isinstance(grid_component.region, gpd.GeoDataFrame)


def test_create_raise_errors(mock_model):
    grid_component = GridComponent(mock_model)
    # Wrong region kind
    with pytest.raises(ValueError, match="Region for grid must be of kind"):
        grid_component.create_from_region(region={"vector_model": "test_model"})
    # bbox
    bbox = [12.05, 45.30, 12.85, 45.65]
    with pytest.raises(
        ValueError, match="res argument required for kind 'bbox', 'geom'"
    ):
        grid_component.create_from_region(region={"bbox": bbox})


@pytest.mark.integration()
def test_create_basin_grid(tmpdir):
    root = join(tmpdir, "grid_model")
    model = Model(
        root=root,
        mode="w",
        data_libs=["artifact_data"],
    )
    grid_component = GridComponent(model=model)
    grid_component.create_from_region(
        region={"subbasin": [12.319, 46.320], "uparea": 50},
        res=1000,
        crs="utm",
        hydrography_path="merit_hydro",
        basin_index_path="merit_hydro_index",
    )
    assert not np.all(grid_component.data["mask"].values is True)
    assert grid_component.data.raster.shape == (47, 61)


def test_properties(caplog: pytest.LogCaptureFixture, demda, mock_model):
    grid_component = GridComponent(mock_model)
    grid_component.root.is_reading_mode.return_value = False
    # Test properties on empty grid
    caplog.set_level(logging.WARNING)
    res = grid_component.res
    assert "No grid data found for deriving resolution" in caplog.text
    transform = grid_component.transform
    assert "No grid data found for deriving transform" in caplog.text
    crs = grid_component.crs
    assert "No grid data found for deriving resolution" in caplog.text
    bounds = grid_component.bounds
    assert "No grid data found for deriving bounds" in caplog.text
    region = grid_component.region
    assert "No grid data found for deriving region" in caplog.text
    assert all(
        props is None for props in [res, transform, crs, bounds, region]
    )  # Ruff complains if prop vars are unused

    grid_component._data = demda
    assert grid_component.res == demda.raster.res
    assert grid_component.transform == demda.raster.transform
    assert grid_component.crs == demda.raster.crs
    assert grid_component.bounds == demda.raster.bounds

    region = grid_component.region
    assert isinstance(region, gpd.GeoDataFrame)
    assert all(region.bounds == demda.raster.bounds)


def test_initialize_grid(mock_model, tmpdir):
    mock_model.root = ModelRoot(path=tmpdir, mode="r")
    grid_component = GridComponent(mock_model)
    grid_component.read = MagicMock()
    grid_component._initialize_grid()
    assert isinstance(grid_component._data, xr.Dataset)
    assert grid_component.read.called


def test_add_data_from_constant(mock_model, demda, mocker: MockerFixture):
    grid_component = GridComponent(mock_model)
    grid_component.root.is_reading_mode.return_value = False
    demda.name = "demda"
    mocker.patch("hydromt.model.components.grid.grid_from_constant", return_value=demda)
    name = grid_component.add_data_from_constant(constant=0.01, name="demda")
    assert name == ["demda"]
    assert grid_component.data == demda


def test_add_data_from_rasterdataset(
    demda, caplog: pytest.LogCaptureFixture, mock_model, mocker: MockerFixture
):
    caplog.set_level(logging.INFO)
    demda.name = "demda"
    demda = demda.to_dataset()
    mock_grid_from_rasterdataset = mocker.patch(
        "hydromt.model.components.grid.grid_from_rasterdataset"
    )
    grid_component = GridComponent(mock_model)
    grid_component.root.is_reading_mode.return_value = False
    mock_grid_from_rasterdataset.return_value = demda
    mock_get_rasterdataset = mocker.patch.object(
        grid_component.data_catalog, "get_rasterdataset"
    )
    mock_get_rasterdataset.return_value = demda
    raster_data = "your_raster_file.tif"
    result = grid_component.add_data_from_rasterdataset(
        raster_data=raster_data,
        variables=["variable1", "variable2"],
        fill_method="mean",
        reproject_method="nearest",
        mask_name="mask",
        rename={"old_var": "new_var"},
    )
    # Test logging
    assert f"Preparing grid data from raster source {raster_data}" in caplog.text
    # Test whether get_rasterdataset and grid_from_rasterdataset are called
    mock_get_rasterdataset.assert_called_once()
    mock_grid_from_rasterdataset.assert_called_once()
    # Test if grid_component.set() succeeded
    assert grid_component.data == demda
    # Test returned result from add_data_from_rasterdataset
    assert all([x in result for x in demda.data_vars.keys()])


def test_add_data_from_raster_reclass(
    caplog: pytest.LogCaptureFixture, demda, mock_model, mocker: MockerFixture
):
    grid_component = GridComponent(mock_model)
    grid_component.root.is_reading_mode.return_value = False
    caplog.set_level(logging.INFO)
    raster_data = "vito"
    reclass_table_data = "vito_mapping"
    demda.name = "name"
    grid_component.data_catalog.get_rasterdataset.return_value = demda
    grid_component.data_catalog.get_dataframe.return_value = pd.DataFrame()
    mock_grid_from_raster_reclass = mocker.patch(
        "hydromt.model.components.grid.grid_from_raster_reclass"
    )
    mock_grid_from_raster_reclass.return_value = demda.to_dataset()

    result = grid_component.add_data_from_raster_reclass(
        raster_data=raster_data,
        fill_method="nearest",
        reclass_table_data=reclass_table_data,
        reclass_variables=["roughness_manning"],
        reproject_method=["average"],
    )
    # Test logging
    assert (
        f"Preparing grid data by reclassifying the data in {raster_data} based "
        f"on {reclass_table_data}"
    ) in caplog.text
    mock_grid_from_raster_reclass.assert_called_once()
    assert grid_component.data == demda
    # Test returned result from add_data_from_rasterdataset
    assert all([x in result for x in demda.to_dataset().data_vars.keys()])

    grid_component.data_catalog.get_rasterdataset.return_value = demda.to_dataset()

    with pytest.raises(
        ValueError,
        match=f"raster_data {raster_data} should be a single variable. "
        "Please select one using the 'variable' argument",
    ):
        result = grid_component.add_data_from_raster_reclass(
            raster_data=raster_data,
            fill_method="nearest",
            reclass_table_data=reclass_table_data,
            reclass_variables=["roughness_manning"],
            reproject_method=["average"],
        )


def test_add_data_from_geodataframe(
    caplog: pytest.LogCaptureFixture, geodf, demda, mock_model, mocker: MockerFixture
):
    grid_component = GridComponent(mock_model)
    grid_component.root.is_reading_mode.return_value = False
    caplog.set_level(logging.INFO)
    demda.name = "name"
    grid_component.data_catalog.get_geodataframe.return_value = geodf
    mock_grid_from_geodataframe = mocker.patch(
        "hydromt.model.components.grid.grid_from_geodataframe"
    )
    mock_grid_from_geodataframe.return_value = demda.to_dataset()
    vector_data = "hydro_lakes"
    result = grid_component.add_data_from_geodataframe(
        vector_data=vector_data,
        variables=["waterbody_id", "Depth_avg"],
        nodata=[-1, -999.0],
        rasterize_method="value",
        rename={
            "waterbody_id": "lake_id",
            "Depth_avg": "lake_depth",
            vector_data: "hydrolakes",
        },
    )
    assert f"Preparing grid data from vector '{vector_data}'." in caplog.text
    mock_grid_from_geodataframe.assert_called_once()
    assert grid_component.data == demda
    assert all([x in result for x in demda.to_dataset().data_vars.keys()])
    grid_component.data_catalog.get_geodataframe.return_value = gpd.GeoDataFrame()
    caplog.set_level(logging.WARNING)
    result = grid_component.add_data_from_geodataframe(
        vector_data=vector_data,
        variables=["waterbody_id", "Depth_avg"],
        nodata=[-1, -999.0],
        rasterize_method="value",
        rename={
            "waterbody_id": "lake_id",
            "Depth_avg": "lake_depth",
            vector_data: "hydrolakes",
        },
    )
    assert (
        f"No shapes of {vector_data} found within region,"
        " skipping add_data_from_geodataframe."
    ) in caplog.text
    assert result is None


@pytest.mark.integration()
def test_grid_component_model(tmpdir):
    # Initialize model
    dc_param_path = join(DATADIR, "parameters_data.yml")
    root = join(tmpdir, "grid_model")
    model = Model(
        root=root,
        mode="w",
        data_libs=["artifact_data", dc_param_path],
    )
    grid_component = GridComponent(model=model)
    model.add_component(name="grid", component=grid_component)
    # Add region
    model.grid.create_from_region(
        region={"subbasin": [12.319, 46.320], "uparea": 50},
        res=0.008333,
        hydrography_path="merit_hydro",
        basin_index_path="merit_hydro_index",
        add_mask=True,
    )

    # Add data with add_data_from_* methods
    model.grid.add_data_from_constant(constant=0.01, name="c1", nodata=-99.0)
    model.grid.add_data_from_constant(constant=2, name="c2", nodata=-1, dtype=np.int8)
    model.grid.add_data_from_rasterdataset(
        raster_data="merit_hydro",
        variables=["elevtn", "basins"],
        reproject_method=["average", "mode"],
        mask_name="mask",
    )
    model.grid.add_data_from_rasterdataset(
        raster_data="vito",
        fill_method="nearest",
        reproject_method="mode",
        rename={"vito": "landuse"},
    )
    model.grid.add_data_from_raster_reclass(
        raster_data="vito",
        fill_method="nearest",
        reclass_table_data="vito_mapping",
        reclass_variables=["roughness_manning"],
        reproject_method=["average"],
    )
    model.grid.add_data_from_geodataframe(
        vector_data="hydro_lakes",
        variables=["waterbody_id", "Depth_avg"],
        nodata=[-1, -999.0],
        rasterize_method="value",
        rename={"waterbody_id": "lake_id", "Depth_avg": "lake_depth"},
    )
    model.grid.add_data_from_geodataframe(
        vector_data="hydro_lakes",
        rasterize_method="fraction",
        rename={"hydro_lakes": "water_frac"},
    )

    def grid_data_checks(mod):
        assert len(mod.grid.data) == 10
        for v in [
            "mask",
            "c1",
            "basins",
            "roughness_manning",
            "lake_depth",
            "water_frac",
        ]:
            assert v in mod.grid.data

        assert mod.grid.data["lake_depth"].raster.nodata == -999.0
        assert mod.grid.data["roughness_manning"].raster.nodata == -999.0

        assert np.unique(mod.grid.data["c2"]).size == 2
        assert np.isin([-1, 2], np.unique(mod.grid.data["c2"])).all()

    grid_data_checks(model)

    # write model
    model.write()

    # Read model
    written_model = Model(root=root, mode="r")
    grid_component = GridComponent(model=written_model)
    written_model.add_component(name="grid", component=grid_component)
    written_model.read()
    grid_data_checks(mod=written_model)
