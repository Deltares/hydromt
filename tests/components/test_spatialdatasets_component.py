from pathlib import Path

import pytest
import xarray as xr
from pytest_mock import MockerFixture

from hydromt.model import Model
from hydromt.model.components.grid import GridComponent
from hydromt.model.components.spatialdatasets import SpatialDatasetsComponent
from tests.conftest import DC_PARAM_PATH


def test_model_spatialdataset_key_error(tmp_path: Path):
    m = Model(root=tmp_path, mode="r+")
    component = SpatialDatasetsComponent(m, region_component="fake")
    m.add_component("test_spatialdataset", component)

    with pytest.raises(KeyError):
        component.data["1"]


def test_model_spatialdataset_sets_correctly(raster_ds, tmp_path: Path):
    m = Model(root=tmp_path, mode="r+")
    component = SpatialDatasetsComponent(m, region_component="fake")
    m.add_component("test_spatialdataset", component)

    component.set(data=raster_ds, name="climate")
    xr.testing.assert_equal(raster_ds, component.data["climate"])

    component.set(data=raster_ds["precipitation"])
    xr.testing.assert_equal(raster_ds["precipitation"], component.data["precipitation"])


def test_model_spatialdataset_reads_and_writes_correctly(raster_ds, tmp_path: Path):
    model = Model(root=tmp_path, mode="w+")
    component = SpatialDatasetsComponent(model, region_component="fake")
    model.add_component("test_spatialdataset", component)

    component.set(data=raster_ds, name="data")

    model.write()
    clean_model = Model(root=tmp_path, mode="r")
    clean_component = SpatialDatasetsComponent(clean_model, region_component="fake")
    clean_model.add_component("test_spatialdataset", clean_component)
    clean_model.read()

    # we'll know that these types will always be the same, which mypy doesn't know
    assert component.data["data"].equals(clean_component.data["data"])  # type: ignore


def test_model_read_spatialdataset(raster_ds, tmp_path: Path):
    write_path = tmp_path / "spatial_datasets" / "forcing.nc"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    raster_ds.to_netcdf(write_path, engine="netcdf4")

    model = Model(root=tmp_path, mode="r")
    dataset_component = SpatialDatasetsComponent(model, region_component="fake")
    model.add_component("forcing", dataset_component)

    component_data = dataset_component.data["forcing"]
    xr.testing.assert_equal(raster_ds, component_data)


def test_add_raster_data_from_rasterdataset(
    demda, tmp_path: Path, mocker: MockerFixture
):
    demda.name = "dem"
    # Create a model with a GridComponent and a SpatialDatasetsComponent
    model = Model(root=tmp_path)
    model.add_component("grid", GridComponent(model))
    # add data to the grid
    model.grid.set(data=demda, name="dem")

    # Add a spatial dataset from the grid
    model.add_component(
        "maps", SpatialDatasetsComponent(model, region_component="grid")
    )

    # Add the dem data to the spatial dataset
    mock_get_rasterdataset = mocker.patch.object(
        model.maps.data_catalog, "get_rasterdataset"
    )
    mock_get_rasterdataset.return_value = demda
    model.maps.add_raster_data_from_rasterdataset(
        raster_filename=demda,
        rename={"dem": "elevation"},
    )

    assert model.maps._region_component == "grid"
    assert "elevation" in model.maps.data
    xr.testing.assert_equal(model.maps.data["elevation"], demda)


def test_add_raster_data_from_raster_reclass(tmp_path: Path, demda, lulcda):
    model = Model(root=tmp_path, data_libs=[DC_PARAM_PATH], mode="w")

    # add a grid and spatial component
    model.add_component("grid", GridComponent(model))
    # add data to the grid
    model.grid.set(data=demda, name="dem")

    # Add a spatial dataset from the grid
    model.add_component(
        "maps", SpatialDatasetsComponent(model, region_component="grid")
    )

    model.maps.add_raster_data_from_raster_reclass(
        raster_filename=lulcda,
        reclass_table_filename="vito_mapping",
        reclass_variables=["roughness_manning"],
        split_dataset=True,
    )

    assert "roughness_manning" in model.maps.data
