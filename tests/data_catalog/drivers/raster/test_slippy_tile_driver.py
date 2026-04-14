"""Tests for the SlippyTile raster driver."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from PIL import Image
from shapely.geometry import box

from hydromt.data_catalog.drivers.raster.slippy_tile_driver import (
    _HAS_BOTO3,
    SlippyTileDriver,
    SlippyTileOptions,
    _download_missing_tiles,
    _download_tile,
    _get_zoom_level_for_resolution,
    _latlon_to_tile_indices,
    _latlon_to_webmercator,
    _num2xy,
    _png2elevation,
    _webmercator_to_latlon,
    _xy2num,
)
from hydromt.error import NoDataStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _elevation2terrarium(val: np.ndarray) -> np.ndarray:
    """Encode elevation values as terrarium RGB (inverse of _png2elevation)."""
    val = val + 32768.0
    r = np.floor(val / 256).astype(np.uint8)
    g = np.floor(val % 256).astype(np.uint8)
    b = np.floor((val - np.floor(val)) * 256).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _make_tile_png(tmp_path: Path, elevation: np.ndarray, mode: str = "RGB") -> str:
    """Write a raw RGB/RGBA PNG and return the path."""
    png_path = tmp_path / "tile.png"
    img = Image.fromarray(elevation.astype(np.uint8), mode=mode)
    img.save(str(png_path))
    return str(png_path)


def _make_tile_dir(
    root: Path,
    zoom: int,
    tile_x: int,
    tile_y: int,
    elevation: np.ndarray,
    encoder: str = "terrarium",
) -> Path:
    """Create a single PNG tile in the standard directory structure."""
    tile_dir = root / str(zoom) / str(tile_x)
    tile_dir.mkdir(parents=True, exist_ok=True)
    png_path = tile_dir / f"{tile_y}.png"

    if encoder == "terrarium":
        rgb = _elevation2terrarium(elevation)
        img = Image.fromarray(rgb, mode="RGB")
    elif encoder == "uint16":
        r = (elevation // 256).astype(np.uint8)
        g = (elevation % 256).astype(np.uint8)
        b = np.zeros_like(r)
        rgb = np.stack([r, g, b], axis=-1)
        img = Image.fromarray(rgb, mode="RGB")
    else:
        raise ValueError(f"Test helper does not support encoder: {encoder}")

    img.save(str(png_path))
    return png_path


def _make_mask_gdf(xmin: float, ymin: float, xmax: float, ymax: float, crs: int = 3857):
    """Create a GeoDataFrame bounding box for use as a mask."""
    return gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs=crs)


# ---------------------------------------------------------------------------
# Tile utility tests
# ---------------------------------------------------------------------------


class TestTileUtilities:
    """Tests for the tile-math helper functions."""

    def test_webmercator_to_latlon_origin(self):
        lat, lon = _webmercator_to_latlon(0.0, 0.0)
        assert abs(lat) < 1e-10
        assert abs(lon) < 1e-10

    def test_latlon_to_webmercator_origin(self):
        x, y = _latlon_to_webmercator(0.0, 0.0)
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6

    def test_roundtrip_webmercator(self):
        x_in, y_in = 500000.0, 6000000.0
        lat, lon = _webmercator_to_latlon(x_in, y_in)
        x_out, y_out = _latlon_to_webmercator(lat, lon)
        assert abs(x_in - x_out) < 0.01
        assert abs(y_in - y_out) < 0.01

    def test_latlon_to_tile_indices_zoom0(self):
        tx, ty = _latlon_to_tile_indices(0.0, 0.0, 0)
        assert tx == 0
        assert ty == 0

    def test_latlon_to_tile_indices_zoom1(self):
        tx, ty = _latlon_to_tile_indices(45.0, -90.0, 1)
        assert tx == 0
        assert ty == 0

    def test_xy2num_and_num2xy_roundtrip(self):
        zoom = 5
        x_in, y_in = 500000.0, 6000000.0
        tx, ty = _xy2num(x_in, y_in, zoom)
        x_out, y_out = _num2xy(tx, ty, zoom)
        dx = 20037508.34 * 2 / (2**zoom)
        assert abs(x_in - x_out) < dx
        assert abs(y_in - y_out) < dx

    def test_get_zoom_level_for_resolution(self):
        assert _get_zoom_level_for_resolution(100000) <= 2
        assert _get_zoom_level_for_resolution(10) >= 13
        assert _get_zoom_level_for_resolution(0.001) == 23

    def test_num2xy_zoom0(self):
        """Zoom 0 tile (0,0) should map to the top-left of the world."""
        x, y = _num2xy(0, 0, 0)
        assert x == pytest.approx(-20037508.34, rel=1e-4)
        assert y == pytest.approx(20037508.34, rel=1e-4)


# ---------------------------------------------------------------------------
# PNG encoding / decoding tests
# ---------------------------------------------------------------------------


class TestPng2Elevation:
    """Tests for PNG tile decoding."""

    def test_terrarium_roundtrip(self, tmp_path):
        elevation = np.array([[0.0, 100.5], [-50.0, 8848.0]])
        path = _make_tile_dir(tmp_path, 0, 0, 0, elevation, "terrarium")
        result = _png2elevation(str(path), encoder="terrarium")
        np.testing.assert_allclose(result, elevation, atol=1.0)

    def test_terrarium_nodata(self, tmp_path):
        elevation = np.array([[-32768.0, 0.0], [0.0, -32768.0]])
        path = _make_tile_dir(tmp_path, 0, 0, 0, elevation, "terrarium")
        result = _png2elevation(str(path), encoder="terrarium")
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 1])
        assert not np.isnan(result[0, 1])

    def test_terrarium16(self, tmp_path):
        """Terrarium16 uses only R and G channels (lower precision)."""
        elevation = np.array([[0.0, 100.0], [-50.0, 8000.0]])
        path = _make_tile_dir(tmp_path, 0, 0, 0, elevation, "terrarium")
        result = _png2elevation(str(path), encoder="terrarium16")
        # Lower precision — only 16-bit
        np.testing.assert_allclose(result, elevation, atol=2.0)

    def test_uint8(self, tmp_path):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[0, 0, 0] = 42
        rgb[0, 1, 0] = 200
        rgb[1, 0, 0] = 255  # nodata
        rgb[1, 1, 0] = 0
        path = _make_tile_png(tmp_path, rgb)
        result = _png2elevation(path, encoder="uint8")
        assert result[0, 0] == 42
        assert result[0, 1] == 200
        assert result[1, 0] == -1  # nodata
        assert result[1, 1] == 0

    def test_uint16_encoding(self, tmp_path):
        elevation = np.array([[0, 1000], [500, 60000]], dtype=int)
        path = _make_tile_dir(tmp_path, 0, 0, 0, elevation, "uint16")
        result = _png2elevation(str(path), encoder="uint16")
        np.testing.assert_array_equal(result, elevation)

    def test_uint16_nodata(self, tmp_path):
        elevation = np.array([[65535, 100], [200, 65535]], dtype=int)
        path = _make_tile_dir(tmp_path, 0, 0, 0, elevation, "uint16")
        result = _png2elevation(str(path), encoder="uint16")
        assert result[0, 0] == -1
        assert result[1, 1] == -1

    def test_uint24(self, tmp_path):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[0, 0] = [1, 0, 0]  # 65536
        rgb[0, 1] = [0, 1, 0]  # 256
        rgb[1, 0] = [255, 255, 255]  # nodata
        path = _make_tile_png(tmp_path, rgb)
        result = _png2elevation(path, encoder="uint24")
        assert result[0, 0] == 65536
        assert result[0, 1] == 256
        assert result[1, 0] == -1

    def test_uint32(self, tmp_path):
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba[0, 0] = [0, 0, 1, 0]  # 256
        rgba[1, 1] = [255, 255, 255, 255]  # nodata
        path = _make_tile_png(tmp_path, rgba, mode="RGBA")
        result = _png2elevation(path, encoder="uint32")
        assert result[0, 0] == 256
        assert result[1, 1] == -1

    def test_float8(self, tmp_path):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[0, 0, 0] = 127  # midpoint
        rgb[0, 1, 0] = 254  # max
        rgb[1, 0, 0] = 0  # nodata
        path = _make_tile_png(tmp_path, rgb)
        result = _png2elevation(path, encoder="float8", encoder_vmin=0, encoder_vmax=10)
        assert result[0, 0] == pytest.approx(5.0, abs=0.1)
        assert result[0, 1] == pytest.approx(10.0, abs=0.1)
        assert np.isnan(result[1, 0])

    def test_float16(self, tmp_path):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[0, 0] = [0, 0, 0]  # nodata (idx=0)
        rgb[0, 1] = [255, 254, 0]  # max (idx=65534)
        path = _make_tile_png(tmp_path, rgb)
        result = _png2elevation(
            path, encoder="float16", encoder_vmin=-100, encoder_vmax=100
        )
        assert np.isnan(result[0, 0])
        assert result[0, 1] == pytest.approx(100.0, abs=0.01)

    def test_float24(self, tmp_path):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[0, 0] = [0, 0, 0]  # nodata
        rgb[0, 1] = [1, 0, 0]  # idx = 65536
        path = _make_tile_png(tmp_path, rgb)
        result = _png2elevation(
            path, encoder="float24", encoder_vmin=0, encoder_vmax=100
        )
        assert np.isnan(result[0, 0])
        assert result[0, 1] > 0

    def test_float32(self, tmp_path):
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba[0, 0] = [0, 0, 0, 0]  # nodata
        rgba[0, 1] = [0, 0, 1, 0]  # idx = 256
        path = _make_tile_png(tmp_path, rgba, mode="RGBA")
        result = _png2elevation(
            path, encoder="float32", encoder_vmin=0, encoder_vmax=1000
        )
        assert np.isnan(result[0, 0])
        assert result[0, 1] > 0

    def test_unknown_encoder_raises(self, tmp_path):
        elevation = np.array([[0.0, 1.0], [2.0, 3.0]])
        path = _make_tile_dir(tmp_path, 0, 0, 0, elevation, "terrarium")
        with pytest.raises(ValueError, match="Unknown encoder"):
            _png2elevation(str(path), encoder="bogus")


# ---------------------------------------------------------------------------
# S3 download tests (mocked)
# ---------------------------------------------------------------------------


class TestS3Download:
    """Tests for S3 tile download functions with mocked boto3."""

    def test_download_tile_success(self, tmp_path):
        mock_client = MagicMock()
        result = _download_tile(
            mock_client, "bucket", "key/0/0/0.png", str(tmp_path / "0" / "0" / "0.png")
        )
        assert result is True
        mock_client.download_file.assert_called_once()

    def test_download_tile_failure(self, tmp_path):
        mock_client = MagicMock()
        mock_client.download_file.side_effect = Exception("network error")
        result = _download_tile(
            mock_client, "bucket", "key/0/0/0.png", str(tmp_path / "fail.png")
        )
        assert result is False

    @patch("hydromt.data_catalog.drivers.raster.slippy_tile_driver._HAS_BOTO3", False)
    def test_download_missing_tiles_no_boto3(self, tmp_path):
        """Without boto3, download should return 0."""
        result = _download_missing_tiles(
            str(tmp_path), "bucket", "key", "us-east-1", [(1, 0, 0)]
        )
        assert result == 0

    @patch("hydromt.data_catalog.drivers.raster.slippy_tile_driver._HAS_BOTO3", True)
    @patch(
        "hydromt.data_catalog.drivers.raster.slippy_tile_driver.UNSIGNED", "UNSIGNED"
    )
    @patch(
        "hydromt.data_catalog.drivers.raster.slippy_tile_driver.BotoConfig", MagicMock()
    )
    @patch("hydromt.data_catalog.drivers.raster.slippy_tile_driver.boto3")
    def test_download_missing_tiles_downloads(self, mock_boto3, tmp_path):
        """Missing tiles should be downloaded."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        result = _download_missing_tiles(
            str(tmp_path), "bucket", "tiles", "us-east-1", [(1, 0, 0)]
        )
        assert result == 1
        mock_client.download_file.assert_called_once()

    @patch("hydromt.data_catalog.drivers.raster.slippy_tile_driver._HAS_BOTO3", True)
    @patch(
        "hydromt.data_catalog.drivers.raster.slippy_tile_driver.UNSIGNED", "UNSIGNED"
    )
    @patch(
        "hydromt.data_catalog.drivers.raster.slippy_tile_driver.BotoConfig", MagicMock()
    )
    @patch("hydromt.data_catalog.drivers.raster.slippy_tile_driver.boto3")
    def test_download_skips_existing_tiles(self, mock_boto3, tmp_path):
        """Tiles that already exist locally should not be downloaded."""
        tile_path = tmp_path / "1" / "0"
        tile_path.mkdir(parents=True)
        (tile_path / "0.png").touch()

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        result = _download_missing_tiles(
            str(tmp_path), "bucket", "tiles", "us-east-1", [(1, 0, 0)]
        )
        assert result == 0
        mock_client.download_file.assert_not_called()


# ---------------------------------------------------------------------------
# SlippyTileDriver tests
# ---------------------------------------------------------------------------


class TestSlippyTileDriver:
    """Tests for the SlippyTileDriver read() method."""

    @pytest.fixture
    def tile_dir(self, tmp_path):
        """Create a small tile directory with 2x2 tiles at zoom 1."""
        npix = 4
        zoom = 1
        for tx in range(2):
            for ty in range(2):
                elev = np.full((npix, npix), float(tx * 10 + ty), dtype=float)
                _make_tile_dir(tmp_path, zoom, tx, ty, elev)
        return str(tmp_path)

    @pytest.fixture
    def mask_global(self):
        """Bounding box covering most of the world in EPSG:3857."""
        return _make_mask_gdf(-15000000, -15000000, 15000000, 15000000, crs=3857)

    def test_read_returns_dataset(self, tile_dir, mask_global):
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4))
        ds = driver.read([tile_dir], mask=mask_global, zoom=1)
        assert isinstance(ds, xr.Dataset)
        assert "elevation" in ds

    def test_read_custom_variable_name(self, tile_dir, mask_global):
        driver = SlippyTileDriver(
            options=SlippyTileOptions(tile_size=4, variable_name="bathymetry")
        )
        ds = driver.read([tile_dir], mask=mask_global, zoom=1)
        assert "bathymetry" in ds
        assert "elevation" not in ds

    def test_read_has_xy_coords(self, tile_dir, mask_global):
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4))
        ds = driver.read([tile_dir], mask=mask_global, zoom=1)
        assert "x" in ds.coords
        assert "y" in ds.coords

    def test_read_crs_is_3857(self, tile_dir, mask_global):
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4))
        ds = driver.read([tile_dir], mask=mask_global, zoom=1)
        assert ds.rio.crs.to_epsg() == 3857

    def test_read_output_shape(self, tile_dir, mask_global):
        """With 2x2 tiles of 4px each, output should be 8x8."""
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4))
        ds = driver.read([tile_dir], mask=mask_global, zoom=1)
        assert ds["elevation"].shape == (8, 8)

    def test_read_no_mask_raises(self, tile_dir):
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4))
        with pytest.raises(ValueError, match="requires a spatial mask"):
            driver.read([tile_dir])

    def test_read_no_tiles_raises(self, tmp_path, mask_global):
        from hydromt.error import NoDataException

        empty_dir = str(tmp_path / "empty")
        os.makedirs(os.path.join(empty_dir, "1", "0"), exist_ok=True)
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4, max_zoom=1))
        with pytest.raises(NoDataException, match="No tiles found"):
            driver.read([empty_dir], mask=mask_global, zoom=1)

    def test_read_no_tiles_ignore(self, tmp_path, mask_global):
        empty_dir = str(tmp_path / "empty")
        os.makedirs(os.path.join(empty_dir, "1", "0"), exist_ok=True)
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4, max_zoom=1))
        ds = driver.read(
            [empty_dir],
            mask=mask_global,
            zoom=1,
            handle_nodata=NoDataStrategy.IGNORE,
        )
        assert ds is None

    def test_read_zoom_as_tuple(self, tile_dir, mask_global):
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4, max_zoom=1))
        ds = driver.read([tile_dir], mask=mask_global, zoom=(50000, "metre"))
        assert isinstance(ds, xr.Dataset)

    def test_read_zoom_as_tuple_unknown_unit(self, tile_dir, mask_global):
        """Unknown zoom unit should warn but still work."""
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4, max_zoom=1))
        ds = driver.read([tile_dir], mask=mask_global, zoom=(50000, "feet"))
        assert isinstance(ds, xr.Dataset)

    def test_read_zoom_none_auto(self, tile_dir, mask_global):
        """Zoom=None should auto-detect from mask extent."""
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4, max_zoom=1))
        ds = driver.read([tile_dir], mask=mask_global)
        assert isinstance(ds, xr.Dataset)

    def test_read_zoom_as_string_int(self, tile_dir, mask_global):
        """Zoom passed as something castable to int."""
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4, max_zoom=1))
        ds = driver.read([tile_dir], mask=mask_global, zoom=1.0)
        assert isinstance(ds, xr.Dataset)

    def test_read_mask_in_4326(self, tile_dir):
        """Mask in EPSG:4326 should be reprojected internally."""
        mask_4326 = _make_mask_gdf(-170, -80, 170, 80, crs=4326)
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4))
        ds = driver.read([tile_dir], mask=mask_4326, zoom=1)
        assert isinstance(ds, xr.Dataset)

    def test_write_raises(self, tile_dir):
        driver = SlippyTileDriver()
        with pytest.raises(NotImplementedError):
            driver.write("output", xr.Dataset())

    def test_driver_name(self):
        assert SlippyTileDriver.name == "slippy_tile"

    def test_driver_does_not_support_writing(self):
        assert SlippyTileDriver.supports_writing is False


class TestDetectMaxZoom:
    def test_detects_zoom_levels(self, tmp_path):
        for z in [0, 3, 5]:
            (tmp_path / str(z)).mkdir()
        assert SlippyTileDriver._detect_max_zoom(str(tmp_path)) == 5

    def test_empty_directory(self, tmp_path):
        assert SlippyTileDriver._detect_max_zoom(str(tmp_path)) == 0

    def test_nonexistent_directory(self, tmp_path):
        assert SlippyTileDriver._detect_max_zoom(str(tmp_path / "nope")) == 0

    def test_ignores_non_numeric_dirs(self, tmp_path):
        (tmp_path / "3").mkdir()
        (tmp_path / "metadata").mkdir()
        (tmp_path / "readme.txt").touch()
        assert SlippyTileDriver._detect_max_zoom(str(tmp_path)) == 3


class TestResolveZoom:
    def test_none_derives_from_extent(self):
        result = SlippyTileDriver._resolve_zoom(None, 0, 30000, 15, 256)
        assert 0 < result <= 15

    def test_explicit_int(self):
        assert SlippyTileDriver._resolve_zoom(5, 0, 30000, 15, 256) == 5

    def test_clamped_to_max_zoom(self):
        assert SlippyTileDriver._resolve_zoom(20, 0, 30000, 10, 256) == 10

    def test_clamped_to_zero(self):
        assert SlippyTileDriver._resolve_zoom(-1, 0, 30000, 10, 256) == 0

    def test_tuple_resolution(self):
        result = SlippyTileDriver._resolve_zoom((100, "metre"), 0, 30000, 15, 256)
        assert 0 < result <= 15

    def test_tuple_unknown_unit(self):
        """Unknown unit should still resolve (treats value as metres)."""
        result = SlippyTileDriver._resolve_zoom((100, "feet"), 0, 30000, 15, 256)
        assert 0 < result <= 15


class TestSlippyTileOptions:
    def test_defaults(self):
        opts = SlippyTileOptions()
        assert opts.encoder == "terrarium"
        assert opts.tile_size == 256
        assert opts.variable_name == "elevation"
        assert opts.max_zoom is None
        assert opts.s3_bucket is None

    def test_custom_values(self):
        opts = SlippyTileOptions(
            encoder="float16",
            encoder_vmin=-100,
            encoder_vmax=9000,
            max_zoom=12,
            variable_name="depth",
            tile_size=512,
            s3_bucket="my-bucket",
            s3_key="tiles/data",
            s3_region="us-east-1",
        )
        assert opts.encoder == "float16"
        assert opts.encoder_vmin == -100
        assert opts.encoder_vmax == 9000
        assert opts.max_zoom == 12
        assert opts.variable_name == "depth"
        assert opts.tile_size == 512
        assert opts.s3_bucket == "my-bucket"


# ---------------------------------------------------------------------------
# Integration tests against a real S3 bucket
# ---------------------------------------------------------------------------


# Path to the checked-in GEBCO 2024 catalog used for the integration tests.
# The catalog points at the public Deltares S3 bucket; tiles already cached
# under the catalog directory are reused, anything missing is downloaded.
# See tests/data/gebco_2024/data_catalog.yml.
_GEBCO_2024_CATALOG_PATH = (
    Path(__file__).parents[3] / "data" / "gebco_2024" / "data_catalog.yml"
)

# Bounding box over the southern North Sea / Wadden region (lon, lat, WGS84).
# Tiles for this bbox are checked in alongside the catalog, so the test
# covers the cache-hit path and runs offline.
_NORTH_SEA_BBOX = (5.0, 52.0, 6.0, 56.0)

# Bounding box off the US south-east coast (Atlantic, near Florida/Georgia).
# Tiles for this bbox are NOT checked in, so the test exercises the actual
# S3 download path. Downloads are redirected to a temp directory so the
# repository stays clean.
_FLORIDA_BBOX = (-80.0, 30.0, -78.0, 32.0)


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_BOTO3, reason="boto3 required for S3 integration")
class TestSlippyTileDriverS3Integration:
    """End-to-end tests against the public deltares-ddb GEBCO 2024 tile set.

    These tests exercise the full DataCatalog + SlippyTileDriver path
    described in the module docstring. They require network access and
    AWS-unsigned read access to
    ``s3://deltares-ddb/data/bathymetry/gebco_2024/``.
    """

    def test_load_catalog(self):
        """The checked-in catalog YAML loads and exposes the expected source."""
        from hydromt.data_catalog import DataCatalog

        cat = DataCatalog(data_libs=[str(_GEBCO_2024_CATALOG_PATH)])
        assert "gebco_2024" in cat.sources

    def test_get_rasterdataset_cached_north_sea(self):
        """Reading a bbox whose tiles are already cached should not hit S3.

        Tiles for the North-Sea bbox are committed under
        ``tests/data/gebco_2024/`` so this test runs without network access.
        """
        from hydromt.data_catalog import DataCatalog

        cat = DataCatalog(data_libs=[str(_GEBCO_2024_CATALOG_PATH)])
        ds = cat.get_rasterdataset(
            "gebco_2024",
            bbox=_NORTH_SEA_BBOX,
            zoom=(2000, "metre"),
        )

        assert isinstance(ds, (xr.Dataset, xr.DataArray))
        assert ds.rio.crs.to_epsg() == 3857
        values = ds.values if isinstance(ds, xr.DataArray) else ds["elevation"].values
        finite = np.isfinite(values)
        assert finite.any(), "Expected at least some finite elevation values"
        assert np.nanmin(values) < 0, "Expected negative bathymetry in the North Sea"
        assert np.nanmax(np.abs(values[finite])) < 5000

    def test_get_rasterdataset_downloads_florida(self, tmp_path):
        """Reading an uncached bbox should download tiles from S3.

        The catalog YAML is copied to ``tmp_path`` so its resolved root sits
        in the temp directory and the downloaded tiles vanish at the end of
        the test, keeping the repo clean.
        """
        import shutil

        from hydromt.data_catalog import DataCatalog

        local_catalog = tmp_path / "data_catalog.yml"
        shutil.copy(_GEBCO_2024_CATALOG_PATH, local_catalog)

        cat = DataCatalog(data_libs=[str(local_catalog)])
        ds = cat.get_rasterdataset(
            "gebco_2024",
            bbox=_FLORIDA_BBOX,
            zoom=(2000, "metre"),
        )

        assert isinstance(ds, (xr.Dataset, xr.DataArray))
        assert ds.rio.crs.to_epsg() == 3857
        values = ds.values if isinstance(ds, xr.DataArray) else ds["elevation"].values
        finite = np.isfinite(values)
        assert finite.any(), "Expected at least some finite elevation values"
        # The Florida bbox is mostly open Atlantic, so we expect deep
        # negative bathymetry to dominate.
        assert np.nanmin(values) < -100, "Expected deep ocean bathymetry off Florida"
        assert np.nanmax(np.abs(values[finite])) < 10000

        # Confirm tiles were actually downloaded to the temp catalog dir.
        downloaded = list(tmp_path.rglob("*.png"))
        assert downloaded, "Expected at least one tile downloaded to tmp_path"
