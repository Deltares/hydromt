# -*- coding: utf-8 -*-
"""Tests for the hydromt.workflows.basin_mask."""

import logging

import numpy as np
import pytest

import hydromt
from hydromt.gis import raster
from hydromt.workflows.basin_mask import (
    _check_size,
    get_basin_geometry,
)

logger = logging.getLogger(__name__)


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_basin(caplog):
    data_catalog = hydromt.DataCatalog("artifact_data", logger=logger)
    ds = data_catalog.get_rasterdataset("merit_hydro_1k")
    gdf_bas_index = data_catalog.get_geodataframe("merit_hydro_index")
    bas_index = data_catalog.get_source("merit_hydro_index")

    with pytest.raises(ValueError, match=r"No basins found"):
        gdf_bas, gdf_out = get_basin_geometry(
            ds,
            kind="basin",
            basid=0,  # basin ID should be > 0
        )

    gdf_bas, gdf_out = get_basin_geometry(
        ds.drop_vars("basins"),
        kind="basin",
        xy=[12.2051, 45.8331],
        buffer=1,
    )
    assert gdf_out is None
    assert gdf_bas.index.size == 1
    assert np.isclose(gdf_bas.area.sum(), 0.16847222)

    gdf_bas, gdf_out = get_basin_geometry(
        ds, kind="subbasin", basin_index=bas_index, xy=[12.2051, 45.8331], strord=4
    )
    assert gdf_bas.index.size == 1
    assert np.isclose(gdf_bas.area.sum(), 0.001875)
    assert np.isclose(gdf_out.geometry.x, 12.17916667)
    assert np.isclose(gdf_out.geometry.y, 45.8041666)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="subbasin",
        basin_index=bas_index,
        xy=[[12.2051, 12.9788], [45.8331, 45.6973]],
        strord=5,
    )
    assert gdf_bas.index.size == 2
    assert np.isclose(gdf_bas.area.sum(), 0.021389)
    assert np.isclose(gdf_out.geometry.x[1], 12.970833333333266)
    assert np.isclose(gdf_out.geometry.y[1], 45.69583333333334)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="subbasin",
        xy=[12.2051, 45.8331],
        strord=4,
        bounds=gdf_bas.total_bounds,
    )
    assert gdf_bas.index.size == 1
    assert np.isclose(gdf_bas.area.sum(), 0.001875)
    assert np.isclose(gdf_out.geometry.x, 12.179167)
    assert np.isclose(gdf_out.geometry.y, 45.804167)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="basin",
        basin_index=gdf_bas_index,
        bbox=[12.6, 45.5, 12.9, 45.7],
        buffer=1,
    )
    assert gdf_bas.index.size == 30
    assert np.isclose(gdf_bas.area.sum(), 1.033125)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="basin",
        basin_index=gdf_bas_index,
        bbox=[12.6, 45.5, 12.9, 45.7],
        buffer=1,
        strord=4,
    )
    assert gdf_bas.index.size == 4
    assert np.isclose(gdf_bas.area.sum(), 1.03104167)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="subbasin",
        basin_index=gdf_bas_index,
        bbox=[12.2, 46.2, 12.4, 46.3],
        strord=6,
    )
    assert gdf_bas.index.size == 1
    assert np.isclose(gdf_bas.area.sum(), 0.198055)
    assert np.isclose(gdf_out.geometry.x, 12.295833)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="interbasin",
        basin_index=gdf_bas_index,
        bbox=[12.2, 46.2, 12.4, 46.3],
        strord=6,
    )
    assert gdf_bas.index.size == 1
    assert np.isclose(gdf_bas.area.sum(), 0.0172222)
    assert np.isclose(gdf_out.geometry.x, 12.295833)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="interbasin",
        basin_index=gdf_bas_index,
        bbox=[12.8, 45.55, 12.9, 45.65],
        outlets=True,
    )
    assert gdf_bas.index.size == 13

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="basin",
        basin_index=gdf_bas_index,
        bbox=[12.8, 45.55, 12.9, 45.65],
        outlets=True,
    )
    assert gdf_bas.index.size == 13

    msg = (
        'kind="outlets" has been deprecated, use outlets=True in combination with'
        + ' kind="basin" or kind="interbasin" instead.'
    )
    with pytest.warns(DeprecationWarning, match=msg):
        gdf_bas, gdf_out = get_basin_geometry(ds, kind="outlet")

    with pytest.raises(ValueError, match="Unknown kind: watershed,"):
        gdf_bas, gdf_out = get_basin_geometry(ds, kind="watershed")

    with pytest.raises(ValueError, match="Dataset variable stream_kwargs not in ds"):
        gdf_bas, gdf_out = get_basin_geometry(
            ds, kind="basin", stream_kwargs={"within": True}
        )
    with pytest.raises(
        ValueError, match='"kind=interbasin" requires either "bbox" or "geom"'
    ):
        gdf_bas, gdf_out = get_basin_geometry(
            ds,
            kind="interbasin",
        )


def test_check_size(caplog):
    test_raster = raster.full_from_transform(
        transform=[0.5, 0.0, 3.0, 0.0, -0.5, -9.0],
        shape=(13000, 13000),
        nodata=-1,
        name="test",
        crs=4326,
        lazy=True,  # create lazy dask array instead of numpy array
    )
    _check_size(test_raster)
    assert (
        "Loading very large spatial domain to derive a subbasin. "
        "Provide initial 'bounds' if this takes too long." in caplog.text
    )
