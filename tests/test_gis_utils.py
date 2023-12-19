"""Test for hydromt.gu submodule."""

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from pyproj import CRS
from rasterio.transform import from_origin
from shapely import Polygon, box

from hydromt import gis_utils as gu
from hydromt.io import open_raster
from hydromt.raster import RasterDataArray, full_from_transform


def test_crs():
    bbox = [3, 51.5, 4, 52]  # NL
    assert gu.utm_crs(bbox).to_epsg() == 32631
    assert gu.parse_crs("utm", bbox).to_epsg() == 32631
    bbox1 = [-77.5, -12.2, -77.0, -12.0]
    assert gu.utm_crs(bbox1).to_epsg() == 32718
    _, _, xattrs, yattrs = gu.axes_attrs(gu.parse_crs(4326))
    assert xattrs["units"] == "degrees_east"
    assert yattrs["units"] == "degrees_north"
    _, _, xattrs, yattrs = gu.axes_attrs(gu.utm_crs(bbox1))
    assert xattrs["units"] == yattrs["units"] == "metre"


def test_affine_to_coords():
    # create grid with x from 0-360E
    transform = from_origin(0, 90, 1, 1)  # upper left corner
    shape = (180, 360)
    coords = gu.affine_to_coords(transform, shape)
    xs, ys = coords["x"][1], coords["y"][1]
    assert np.all(ys == 90 - np.arange(0.5, shape[0]))
    assert np.all(xs == np.arange(0.5, shape[1]))


def test_meridian_offset():
    # test global grids with different west origins
    for x0 in [-180, 0, -360, 1]:
        da = full_from_transform(
            transform=Affine.identity() * Affine.translation(x0, -90),
            shape=(180, 360),
            crs="epsg:4326",
        )
        # return W180-E180 grid if no bbox is provided
        da1 = gu.meridian_offset(da)
        assert da1.raster.bounds == (-180, -90, 180, 90)
        # make sure bbox is respected
        for bbox in [
            [-2, -2, 2, 2],  # bbox crossing 0
            [178, -2, 182, 2],  # bbox crossing 180
            [-10, -2, 190, 2],  # bbox crossing 0 and 180
            [-190, -2, 170, 2],  # bbox crossing -180
        ]:
            da2 = gu.meridian_offset(da, bbox=bbox)
            assert (
                da2.raster.bounds[0] <= bbox[0]
            ), f"{da2.raster.bounds[0]} <= {bbox[0]}"
            assert (
                da2.raster.bounds[2] >= bbox[2]
            ), f"{da2.raster.bounds[2]} >= {bbox[2]}"

    # test error
    with pytest.raises(ValueError, match="The method is only applicable to"):
        gu.meridian_offset(da.raster.clip_bbox([0, 0, 10, 10]))


def test_transform_rotation():
    # with rotation
    transform = Affine.rotation(30) * Affine.scale(1, 2)
    shape = (10, 5)
    coords = gu.affine_to_coords(transform, shape)
    xs, ys = coords["xc"][1], coords["yc"][1]
    assert xs.ndim == 2
    assert ys.ndim == 2
    da = full_from_transform(transform, shape, crs=4326)
    assert da.raster.x_dim == "x"
    assert da.raster.xcoords.ndim == 2
    assert np.allclose(transform, da.raster.transform)


def test_area_res():
    # surface area of earth should be approx 510.100.000 km2
    transform = from_origin(-180, 90, 1, 1)
    shape = (180, 360)
    da = full_from_transform(transform, shape, crs=4326)
    assert np.isclose(da.raster.area_grid().sum() / 1e6, 510064511.156224)
    assert gu.cellres(0) == (111319.458, 110574.2727)


def test_gdf(world):
    country = world.iloc[[0], :].to_crs(3857)
    assert np.all(gu.filter_gdf(world, country) == 0)
    idx0 = gu.filter_gdf(world, bbox=[3, 51.5, 4, 52])[0]
    assert world.iloc[idx0,]["iso_a3"] == "NLD"
    with pytest.raises(ValueError, match="Unknown geometry mask type"):
        gu.filter_gdf(world, geom=[3, 51.5, 4, 52])


def test_nearest(world, geodf):
    idx, _ = gu.nearest(geodf, geodf)
    assert np.all(idx == geodf.index)
    idx, dst = gu.nearest(geodf, world)
    assert np.all(dst == 0)
    assert np.all(world.loc[idx, "name"].values == geodf["country"].values)
    gdf0 = geodf.copy()
    gdf0["iso_a3"] = ""
    gdf1 = gu.nearest_merge(geodf, world.drop(idx), max_dist=1e6)
    assert np.all(gdf1.loc[gdf1["distance_right"] > 1e6, "index_right"] == -1)
    assert np.all(gdf1.loc[gdf1["distance_right"] > 1e6, "iso_a3"] != "")


def test_spread():
    transform = from_origin(-15, 10, 1, 1)
    shape = (20, 30)
    data = np.zeros(shape)
    data[10, 10] = 1  # lin index 310
    frc = np.ones(shape)
    msk = np.ones(shape, dtype=bool)
    da_obs = RasterDataArray.from_numpy(data, transform=transform, nodata=0, crs=4326)
    da_msk = RasterDataArray.from_numpy(msk, transform=transform, crs=4326)
    da_frc = RasterDataArray.from_numpy(frc, transform=transform, crs=4326)
    # only testing the wrapping of pyflwdir method, not the method itself
    ds_out = gu.spread2d(da_obs, da_friction=da_frc, da_mask=da_msk)
    assert np.all(ds_out["source_value"] == 1)
    assert np.all(ds_out["source_idx"] == 310)
    assert ds_out["source_dst"].values[10, 10] == 0
    with pytest.raises(ValueError, match='"nodata" must be a finite value'):
        gu.spread2d(da_obs, nodata=np.nan)


def test_create_vrt(tmpdir, rioda_large):
    # NOTE: this method does not work in debug mode because of os.subprocess
    path = str(tmpdir)
    rioda_large.raster.to_xyz_tiles(
        os.path.join(path, "dummy_xyz"),
        tile_size=256,
        zoom_levels=[0],
    )
    # test create_vrt
    vrt_fn = os.path.join(path, "dummy_xyz", "vrt", "zl0.vrt")
    files_path = os.path.join(path, "dummy_xyz", "*", "*", "*.tif")
    gu.create_vrt(vrt_fn, files_path=files_path)
    assert os.path.isfile(vrt_fn)
    assert isinstance(open_raster(vrt_fn).load(), xr.DataArray)  # try reading
    with pytest.raises(ValueError, match="Either 'files' or 'files_path' is required"):
        gu.create_vrt(vrt_fn)
    with pytest.raises(IOError, match="No files found at "):
        gu.create_vrt(vrt_fn, files_path=os.path.join(path, "dummy_xyz", "*.abc"))


class TestBBoxFromFileAndFilters:
    @pytest.fixture(scope="class")
    def vector_data_with_crs(self, geodf: gpd.GeoDataFrame, tmp_dir: Path) -> Path:
        example_data = geodf.set_crs(crs=CRS.from_user_input(4326))
        example_data.to_crs(crs=CRS.from_user_input(3857), inplace=True)
        path = tmp_dir / "test.fgb"
        example_data.to_file(path, engine="pyogrio")
        return path

    @pytest.fixture(scope="class")
    def vector_data_without_crs(self, geodf: gpd.GeoDataFrame, tmp_dir: Path) -> Path:
        path = tmp_dir / "test.geojson"
        geodf.to_file(path, engine="pyogrio")
        return path

    @pytest.fixture(scope="class")
    def gdf_mask_without_crs(self, world: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        return world[world["name"] == "Chile"]

    @pytest.fixture(scope="class")
    def gdf_bbox_with_crs(
        self, gdf_mask_without_crs: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        return gdf_mask_without_crs.set_crs(CRS.from_user_input(4326))

    @pytest.fixture(scope="class")
    def shapely_bbox(self, gdf_mask_without_crs: gpd.GeoDataFrame) -> Polygon:
        return box(*list(gdf_mask_without_crs.total_bounds))

    def test_gdf_bbox_crs_source_crs(
        self, gdf_bbox_with_crs: gpd.GeoDataFrame, vector_data_with_crs: Path
    ):
        bbox = gu.bbox_from_file_and_filters(
            vector_data_with_crs, bbox=gdf_bbox_with_crs
        )
        # assert converted to CRS of source data EPSG:3857
        assert all(map(lambda x: abs(x) > 180, bbox))

    def test_gdf_mask_no_crs_source_crs(
        self, gdf_mask_without_crs: gpd.GeoDataFrame, vector_data_with_crs: Path
    ):
        bbox = gu.bbox_from_file_and_filters(
            vector_data_with_crs, bbox=gdf_mask_without_crs
        )
        # assert converted to CRS of source data EPSG:3857
        assert all(map(lambda x: abs(x) > 180, bbox))

    def test_gdf_mask_crs_source_no_crs(
        self, gdf_mask_without_crs: gpd.GeoDataFrame, vector_data_without_crs: Path
    ):
        bbox = gu.bbox_from_file_and_filters(
            vector_data_without_crs, bbox=gdf_mask_without_crs
        )
        assert all(map(lambda x: abs(x) < 180, bbox))

    def test_gdf_mask_no_crs_source_no_crs(
        self, gdf_mask_without_crs: gpd.GeoDataFrame, vector_data_without_crs: Path
    ):
        bbox = gu.bbox_from_file_and_filters(
            vector_data_without_crs, bbox=gdf_mask_without_crs, crs=4326
        )
        assert all(map(lambda x: abs(x) < 180, bbox))

    def test_shapely_input(self, shapely_bbox: Polygon, vector_data_with_crs: Path):
        bbox = gu.bbox_from_file_and_filters(vector_data_with_crs, bbox=shapely_bbox)
        assert all(map(lambda x: abs(x) > 180, bbox))

    def test_does_not_filter(self, vector_data_with_crs: Path):
        bbox = gu.bbox_from_file_and_filters(vector_data_with_crs)
        assert bbox is None

    def test_raises_valueerror(
        self, vector_data_with_crs: Path, gdf_bbox_with_crs: gpd.GeoDataFrame
    ):
        with pytest.raises(
            ValueError,
            match="Both 'bbox' and 'mask' are provided. Please provide only one.",
        ):
            gu.bbox_from_file_and_filters(
                vector_data_with_crs, bbox=gdf_bbox_with_crs, mask=gdf_bbox_with_crs
            )
