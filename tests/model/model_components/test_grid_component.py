import logging
from os.path import join
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from hydromt.data_catalog import DataCatalog
from hydromt.models.components.grid import GridComponent
from hydromt.models.root import ModelRoot

logger = logging.getLogger(__name__)
logger.propagate = True


def test_set(hydds, tmp_dir, rioda):
    model_root = ModelRoot(path=tmp_dir)
    data_catalog = DataCatalog()
    grid_component = GridComponent(
        root=model_root, data_catalog=data_catalog, model_region=None
    )
    # Test setting xr.Dataset
    grid_component.set(data=hydds)
    assert len(grid_component.data) > 0
    assert isinstance(grid_component.data, xr.Dataset)
    # Test setting xr.DataArray
    data_array = hydds.to_array()
    grid_component.set(data=data_array, name="data_array")
    assert "data_array" in grid_component.data.data_vars.keys()
    assert len(grid_component.data.data_vars) == 3
    # Test setting nameless data array
    data_array.name = None
    with pytest.raises(
        ValueError,
        match=f"Unable to set {type(data_array).__name__} data without a name",
    ):
        grid_component.set(data=data_array)
    # Test setting np.ndarray of different shape
    ndarray = np.random.rand(4, 5)
    with pytest.raises(ValueError, match="Shape of data and grid maps do not match"):
        grid_component.set(ndarray, name="ndarray")


def test_write(tmp_dir, caplog):
    model_root = ModelRoot(path=tmp_dir)
    data_catalog = DataCatalog()
    grid_component = GridComponent(
        root=model_root, data_catalog=data_catalog, model_region=None, logger=logger
    )
    # Test skipping writing when no grid data has been set
    caplog.set_level(logging.WARNING)
    grid_component.write()
    assert "No grid data found, skip writing" in caplog.text
    # Test raise IOerror when model is in read only mode
    model_root = ModelRoot(tmp_dir, mode="r")
    grid_component = GridComponent(
        root=model_root, data_catalog=data_catalog, model_region=None
    )
    with patch.object(GridComponent, "data", ["test"]):
        with pytest.raises(IOError, match="Model opened in read-only mode"):
            grid_component.write()


def test_read(tmp_dir, hydds):
    # Test for raising IOError when model is in writing mode
    model_root = ModelRoot(path=tmp_dir, mode="w")
    data_catalog = DataCatalog()
    grid_component = GridComponent(
        root=model_root, data_catalog=data_catalog, model_region=None, logger=logger
    )
    with pytest.raises(IOError, match="Model opened in write-only mode"):
        grid_component.read()
    model_root = ModelRoot(path=tmp_dir, mode="r+")
    data_catalog = DataCatalog()
    grid_component = GridComponent(
        root=model_root, data_catalog=data_catalog, model_region=None, logger=logger
    )
    with patch("hydromt.models.components.grid.read_nc", return_value={"grid": hydds}):
        grid_component.read()
        assert grid_component.data == hydds


def test_create(tmp_dir, demda):
    model_root = ModelRoot(path=join(tmp_dir, "grid_model"))
    data_catalog = DataCatalog(data_libs=["artifact_data"])
    grid_component = GridComponent(
        root=model_root, data_catalog=data_catalog, model_region=None
    )
    # Wrong region kind
    with pytest.raises(ValueError, match="Region for grid must be of kind"):
        grid_component.create(region={"vector_model": "test_model"})
    # bbox
    bbox = [12.05, 45.30, 12.85, 45.65]
    with pytest.raises(
        ValueError, match="res argument required for kind 'bbox', 'geom'"
    ):
        grid_component.create({"bbox": bbox})
    grid_component.create(
        region={"bbox": bbox},
        res=0.05,
        add_mask=False,
        align=True,
    )

    assert "mask" not in grid_component.data
    # assert model.crs.to_epsg() == 4326
    assert grid_component.data.raster.dims == ("y", "x")
    assert grid_component.data.raster.shape == (7, 16)
    assert np.all(np.round(grid_component.data.raster.bounds, 2) == bbox)
    grid_component._data = xr.Dataset()  # remove old grid

    # bbox rotated
    grid_component.create(
        region={"bbox": [12.65, 45.50, 12.85, 45.60]},
        res=0.05,
        crs=4326,
        rotated=True,
        add_mask=True,
    )
    assert "xc" in grid_component.data.coords
    assert grid_component.data.raster.y_dim == "y"
    assert np.isclose(grid_component.data.raster.res[0], 0.05)
    grid_component._data = xr.Dataset()  # remove old grid

    # grid
    grid_fn = str(tmp_dir.join("grid.tif"))
    demda.raster.to_raster(grid_fn)
    grid_component.create({"grid": grid_fn})
    assert np.all(demda.raster.bounds == grid_component.region.total_bounds)
    grid_component._data = xr.Dataset()  # remove old grid

    # basin
    grid_component.create(
        region={"subbasin": [12.319, 46.320], "uparea": 50},
        res=1000,
        crs="utm",
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
    )
    assert not np.all(grid_component.data["mask"].values is True)
    assert grid_component.data.raster.shape == (47, 61)


def test_properties(caplog, tmp_dir, demda, grid_component):
    # Test properties on empty grid

    caplog.set_level(logging.WARNING)
    res = grid_component.res
    assert "No grid data found for deriving resolution" in caplog.text
    transform = grid_component.transform
    assert "No grid data found for deriving transform" in caplog.text
    crs = grid_component.crs
    assert "Grid data has no crs" in caplog.text
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


def test_initialize_grid(tmp_dir):
    grid_component = GridComponent(
        root=ModelRoot(path=tmp_dir, mode="r"), data_catalog=None, model_region=None
    )
    grid_component.read = MagicMock()
    grid_component._initialize_grid()
    assert isinstance(grid_component._data, xr.Dataset)
    assert grid_component.read.called


def test_set_crs(grid_component, demda):
    grid_component._data = demda
    grid_component.set_crs(crs=4326)
    assert grid_component.data.raster.crs == 4326


def test_add_data_from_constant(grid_component, demda):
    demda.name = "demda"
    # demda = demda.to_dataset()
    with patch("hydromt.models.components.grid.grid_from_constant", return_value=demda):
        name = grid_component.add_data_from_constant(constant=0.01, name="demda")
        assert name == ["demda"]
        assert grid_component.data == demda


@pytest.mark.skip(reason="Needs implementation of RasterDataSet.")
def test_add_data_from_rasterdataset(grid_component):
    pass


def test_add_data_from_raster_reclass(grid_component):
    pass
    grid_component.add_data_from_raster_reclass(
        raster_fn="vito",
        fill_method="nearest",
        reclass_table_fn="vito_mapping",
        reclass_variables=["roughness_manning"],
        reproject_method=["average"],
    )


def test_add_data_from_geodataframe(grid_component):
    pass
    grid_component.add_data_from_geodataframe(
        vector_fn="hydro_lakes",
        variables=["waterbody_id", "Depth_avg"],
        nodata=[-1, -999.0],
        rasterize_method="value",
        rename={"waterbody_id": "lake_id", "Depth_avg": "lake_depth"},
    )
    grid_component.add_data_from_geodataframe(
        vector_fn="hydro_lakes",
        rasterize_method="fraction",
        rename={"hydro_lakes": "water_frac"},
    )
