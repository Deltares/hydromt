# -*- coding: utf-8 -*-
"""Tests for the hydromt.workflows.basin_mask."""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import box

from hydromt import DataCatalog
from hydromt.gis import raster_utils
from hydromt.model.processes.basin_mask import (
    _check_size,
    get_basin_geometry,
)


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


def test_crs_mismatch(basin_files, caplog: pytest.LogCaptureFixture):
    _, ds, gdf_bas, _ = basin_files
    caplog.set_level(logging.WARNING)
    gdf_bas.to_crs(epsg=6875, inplace=True)
    gdf_bas, _ = get_basin_geometry(
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


def test_subbasin_with_bounds(basin_files, caplog: pytest.LogCaptureFixture):
    _, ds, _, _ = basin_files
    caplog.set_level(logging.WARNING)
    _ = get_basin_geometry(ds, kind="subbasin", bounds=[12.6, 45.5, 12.9, 45.7])
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

    gdf_bas, _ = get_basin_geometry(
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
    test_raster = raster_utils.full_from_transform(
        transform=[0.5, 0.0, 3.0, 0.0, -0.5, -9.0],
        shape=(13000, 13000),
        nodata=-1,
        name="test",
        crs=4326,
        lazy=True,  # create lazy dask array instead of numpy array
    )
    _check_size(test_raster)
    assert "Loading very large spatial domain to derive a subbasin." in caplog.text


def test_basin_geom_non_4326_crs(basin_files):
    _, ds, gdf_bas_index, _ = basin_files
    geom_4326 = gpd.GeoDataFrame(
        geometry=[box(12.2, 46.2, 12.4, 46.3)],
        crs=4326,
    )
    geom_32633 = geom_4326.to_crs(32633)

    gdf_inter_4326, _ = get_basin_geometry(
        ds,
        kind="interbasin",
        basin_index=gdf_bas_index,
        geom=geom_4326,
        strord=6,
    )
    gdf_inter_32633, _ = get_basin_geometry(
        ds,
        kind="interbasin",
        basin_index=gdf_bas_index,
        geom=geom_32633,
        strord=6,
    )
    assert gdf_inter_32633.index.size == gdf_inter_4326.index.size == 1

    gdf_sub_4326, _ = get_basin_geometry(
        ds,
        kind="subbasin",
        basin_index=gdf_bas_index,
        geom=geom_4326,
        strord=6,
    )
    gdf_sub_32633, _ = get_basin_geometry(
        ds,
        kind="subbasin",
        basin_index=gdf_bas_index,
        geom=geom_32633,
        strord=6,
    )
    assert gdf_sub_32633.index.size == gdf_sub_4326.index.size == 1


def test_basin_index_source_projected_crs(basin_files, tmp_path: Path):
    _, ds, gdf_bas_index, _ = basin_files
    gdf_32633 = gdf_bas_index.to_crs(32633)
    geojson_path = tmp_path / "basins_32633.geojson"
    gdf_32633.to_file(geojson_path, driver="GeoJSON")

    cat = DataCatalog()
    cat.from_dict(
        {
            "basins_no_meta_crs": {
                "uri": str(geojson_path),
                "data_type": "GeoDataFrame",
                "driver": "pyogrio",
            },
            "basins_with_meta_crs": {
                "uri": str(geojson_path),
                "data_type": "GeoDataFrame",
                "driver": "pyogrio",
                "metadata": {"crs": 32633},
            },
        }
    )

    transformer = Transformer.from_crs(4326, 32633, always_xy=True)
    x_proj, y_proj = transformer.transform(12.2051, 45.8331)

    # Direct read_data with bbox when metadata.crs is set (#1563)
    src_meta = cat.get_source("basins_with_meta_crs")
    gdf_read = src_meta.read_data(
        bbox=[x_proj - 0.1, y_proj - 0.1, x_proj + 0.1, y_proj + 0.1]
    )
    assert gdf_read is not None
    assert len(gdf_read) > 0
    assert gdf_read.crs.to_epsg() == 32633

    # get_basin_geometry with projected ds, xy, and GeoDataFrameSource (#1563)
    ds_clip = ds.raster.clip_bbox([12.0, 45.6, 12.5, 46.1])
    ds_32633 = ds_clip.raster.reproject(dst_crs=32633)
    gdf_bas, gdf_out = get_basin_geometry(
        ds_32633,
        kind="subbasin",
        basin_index=cat.get_source("basins_no_meta_crs"),
        xy=[x_proj, y_proj],
        strord=4,
    )
    assert gdf_bas.index.size == 1
    assert gdf_bas.crs.to_epsg() == 32633
    assert gdf_out is not None
