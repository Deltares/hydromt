# -*- coding: utf-8 -*-
"""Tests for the hydromt.workflows.basin_mask."""

import logging

import geopandas as gpd
import numpy as np
import pytest

from hydromt.gis import raster
from hydromt.model.processes.basin_mask import (
    _check_size,
    get_basin_geometry,
)

logger = logging.getLogger(__name__)


def reproject_to_utm_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    utm = gdf.geometry.estimate_utm_crs()
    return gdf.to_crs(utm)


def test_no_basin(basin_files):
    _, ds, _, _ = basin_files
    with pytest.raises(ValueError, match=r"No basins found"):
        _ = get_basin_geometry(
            ds,
            kind="basin",
            basid=0,  # basin ID should be > 0
        )


def test_basin(basin_files):
    _, ds, _, _ = basin_files
    gdf_bas, gdf_out = get_basin_geometry(
        ds.drop_vars("basins"),
        kind="basin",
        xy=[12.2051, 45.8331],
        buffer=1,
    )
    assert gdf_out is None
    assert gdf_bas.index.size == 1
    gdf_bas = reproject_to_utm_crs(gdf_bas)
    assert np.isclose(gdf_bas.area.sum(), 1460079731.4550357)


def test_no_basid(basin_files):
    _, ds, gdf_bas, _ = basin_files
    gdf_bas_no_id = gdf_bas.drop(
        columns="basid",
    )
    with pytest.raises(
        ValueError, match="Basin geometries does not have 'basid' column."
    ):
        get_basin_geometry(
            ds.drop_vars("basins"),
            kind="basin",
            basin_index=gdf_bas_no_id,
            xy=[12.2051, 45.8331],
            buffer=1,
        )


def test_crs_mismatch(basin_files, caplog):
    _, ds, gdf_bas, _ = basin_files
    caplog.set_level(logging.WARNING)
    gdf_bas.to_crs(epsg=6875, inplace=True)
    gdf_bas, gdf_out = get_basin_geometry(
        ds.drop_vars("basins"),
        kind="basin",
        basin_index=gdf_bas,
        xy=[12.2051, 45.8331],
        buffer=1,
    )
    assert "Basin geometries CRS does not match the input raster CRS." in caplog.text
    assert gdf_bas.crs.to_epsg() == 4326


def test_subbasin(basin_files, caplog):
    _, ds, _, bas_index = basin_files
    gdf_bas, gdf_out = get_basin_geometry(
        ds, kind="subbasin", basin_index=bas_index, xy=[12.2051, 45.8331], strord=4
    )
    assert gdf_bas.index.size == 1
    gdf_bas = reproject_to_utm_crs(gdf_bas)
    assert np.isclose(gdf_bas.area.sum(), 16201389.58563961)
    assert np.isclose(gdf_out.geometry.x[0], 12.179166666666664)
    assert np.isclose(gdf_out.geometry.y[0], 45.80416666666667)


def test_subbasin_xy(basin_files):
    _, ds, _, bas_index = basin_files
    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="subbasin",
        basin_index=bas_index,
        xy=[[12.2051, 12.9788], [45.8331, 45.6973]],
        strord=5,
    )
    assert gdf_bas.index.size == 2
    gdf_bas = reproject_to_utm_crs(gdf_bas)
    gdf_out = reproject_to_utm_crs(gdf_out)
    assert np.isclose(gdf_bas.area.sum(), 184897906.5105253)
    assert np.isclose(gdf_out.geometry.x[0], 278668.97710990265)
    assert np.isclose(gdf_out.geometry.y[0], 5070673.469794823)


def test_basin_bbox(basin_files):
    _, ds, gdf_bas_index, _ = basin_files
    gdf_bas, _ = get_basin_geometry(
        ds,
        kind="basin",
        basin_index=gdf_bas_index,
        bbox=[12.6, 45.5, 12.9, 45.7],
        buffer=1,
    )
    assert gdf_bas.index.size == 30
    gdf_bas = reproject_to_utm_crs(gdf_bas)
    assert np.isclose(gdf_bas.area.sum(), 8889419826.002827)


def test_basin_stord(basin_files):
    _, ds, gdf_bas_index, _ = basin_files
    gdf_bas, _ = get_basin_geometry(
        ds,
        kind="basin",
        basin_index=gdf_bas_index,
        bbox=[12.6, 45.5, 12.9, 45.7],
        buffer=1,
        strord=4,
    )
    assert gdf_bas.index.size == 4
    gdf_bas = reproject_to_utm_crs(gdf_bas)
    assert np.isclose(gdf_bas.area.sum(), 8871340671.25457)


def test_subbasin_stord(basin_files):
    _, ds, gdf_bas_index, _ = basin_files
    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="subbasin",
        basin_index=gdf_bas_index,
        bbox=[12.2, 46.2, 12.4, 46.3],
        strord=6,
    )
    assert gdf_bas.index.size == 1
    gdf_bas = reproject_to_utm_crs(gdf_bas)
    assert np.isclose(gdf_bas.area.sum(), 1691220911.8689833)
    assert np.isclose(gdf_out.geometry.x[0], 12.29583333333333)


def test_subbasin_with_bounds(basin_files, caplog):
    _, ds, _, _ = basin_files
    caplog.set_level(logging.WARNING)
    gdf_bas, gdf_out = get_basin_geometry(
        ds, kind="subbasin", bounds=[12.6, 45.5, 12.9, 45.7]
    )
    assert "The subbasin does not include all upstream cells." in caplog.text


def test_interbasin(basin_files):
    _, ds, gdf_bas_index, _ = basin_files
    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="interbasin",
        basin_index=gdf_bas_index,
        bbox=[12.2, 46.2, 12.4, 46.3],
        strord=6,
    )
    assert gdf_bas.index.size == 1
    gdf_bas = reproject_to_utm_crs(gdf_bas)
    assert np.isclose(gdf_bas.area.sum(), 147653247.45211384)
    assert np.isclose(gdf_out.geometry.x[0], 12.295833333333329)


def test_interbasin_outlets(basin_files):
    _, ds, gdf_bas_index, _ = basin_files
    gdf_bas, _ = get_basin_geometry(
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


def test_rejects_unknown_kind(basin_files):
    _, ds, _, _ = basin_files
    with pytest.raises(ValueError, match="Unknown kind: watershed,"):
        _ = get_basin_geometry(ds, kind="watershed")


def test_basin_rejects_dataset_without_stream_kwargs(basin_files):
    _, ds, _, _ = basin_files
    with pytest.raises(ValueError, match="Dataset variable stream_kwargs not in ds"):
        _ = get_basin_geometry(ds, kind="basin", stream_kwargs={"within": True})


def test_interbasin_requires_bbox_or_geom(basin_files):
    _, ds, _, _ = basin_files
    with pytest.raises(
        ValueError, match='"kind=interbasin" requires either "bbox" or "geom"'
    ):
        _ = get_basin_geometry(
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
