"""Tests for the SlippyTile raster driver."""

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from PIL import Image
from shapely.geometry import box

from hydromt.data_catalog.drivers.raster.slippy_tile_driver import (
    SlippyTileDriver,
    SlippyTileOptions,
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
        # Zoom 0: entire world is one tile (0, 0)
        tx, ty = _latlon_to_tile_indices(0.0, 0.0, 0)
        assert tx == 0
        assert ty == 0

    def test_latlon_to_tile_indices_zoom1(self):
        # Zoom 1: 2x2 tiles. (0,0) is top-left
        tx, ty = _latlon_to_tile_indices(45.0, -90.0, 1)
        assert tx == 0
        assert ty == 0

    def test_xy2num_and_num2xy_roundtrip(self):
        zoom = 5
        x_in, y_in = 500000.0, 6000000.0
        tx, ty = _xy2num(x_in, y_in, zoom)
        x_out, y_out = _num2xy(tx, ty, zoom)
        # Should be at the upper-left corner of the tile
        # The roundtrip won't be exact but the tile should contain the point
        dx = 20037508.34 * 2 / (2**zoom)
        assert abs(x_in - x_out) < dx
        assert abs(y_in - y_out) < dx

    def test_get_zoom_level_for_resolution(self):
        # Very coarse resolution -> low zoom
        assert _get_zoom_level_for_resolution(100000) <= 2
        # Fine resolution -> high zoom
        assert _get_zoom_level_for_resolution(10) >= 13
        # Extremely fine -> max zoom
        assert _get_zoom_level_for_resolution(0.001) == 23


# ---------------------------------------------------------------------------
# PNG encoding / decoding tests
# ---------------------------------------------------------------------------


class TestPng2Elevation:
    """Tests for PNG tile decoding."""

    def test_terrarium_roundtrip(self, tmp_path):
        """Encode then decode: values should survive the roundtrip."""
        elevation = np.array([[0.0, 100.5], [-50.0, 8848.0]])
        path = _make_tile_dir(tmp_path, 0, 0, 0, elevation, "terrarium")
        result = _png2elevation(str(path), encoder="terrarium")
        np.testing.assert_allclose(result, elevation, atol=1.0)

    def test_terrarium_nodata(self, tmp_path):
        """Values below -32767 should become NaN."""
        elevation = np.array([[-32768.0, 0.0], [0.0, -32768.0]])
        path = _make_tile_dir(tmp_path, 0, 0, 0, elevation, "terrarium")
        result = _png2elevation(str(path), encoder="terrarium")
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 1])
        assert not np.isnan(result[0, 1])

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

    def test_unknown_encoder_raises(self, tmp_path):
        elevation = np.array([[0.0, 1.0], [2.0, 3.0]])
        path = _make_tile_dir(tmp_path, 0, 0, 0, elevation, "terrarium")
        with pytest.raises(ValueError, match="Unknown encoder"):
            _png2elevation(str(path), encoder="bogus")


# ---------------------------------------------------------------------------
# SlippyTileDriver tests
# ---------------------------------------------------------------------------


class TestSlippyTileDriver:
    """Tests for the SlippyTileDriver read() method."""

    @pytest.fixture
    def tile_dir(self, tmp_path):
        """Create a small tile directory with 2x2 tiles at zoom 1."""
        npix = 4  # small tiles for testing
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

    def test_read_no_mask_raises(self, tile_dir):
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4))
        with pytest.raises(ValueError, match="requires a spatial mask"):
            driver.read([tile_dir])

    def test_read_no_tiles_raises(self, tmp_path, mask_global):
        """Empty tile directory with RAISE strategy."""
        empty_dir = str(tmp_path / "empty")
        os.makedirs(os.path.join(empty_dir, "1", "0"), exist_ok=True)
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4, max_zoom=1))
        with pytest.raises(IOError, match="No tiles found"):
            driver.read([empty_dir], mask=mask_global, zoom=1)

    def test_read_no_tiles_ignore(self, tmp_path, mask_global):
        """Empty tile directory with IGNORE strategy should warn, not raise."""
        empty_dir = str(tmp_path / "empty")
        os.makedirs(os.path.join(empty_dir, "1", "0"), exist_ok=True)
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4, max_zoom=1))
        ds = driver.read(
            [empty_dir],
            mask=mask_global,
            zoom=1,
            handle_nodata=NoDataStrategy.IGNORE,
        )
        assert isinstance(ds, xr.Dataset)

    def test_read_zoom_as_tuple(self, tile_dir, mask_global):
        driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4, max_zoom=1))
        # Pass zoom as (resolution_in_metres, unit) — pick a value that resolves to zoom 1
        ds = driver.read([tile_dir], mask=mask_global, zoom=(50000, "metre"))
        assert isinstance(ds, xr.Dataset)

    def test_write_raises(self, tile_dir):
        driver = SlippyTileDriver()
        with pytest.raises(NotImplementedError):
            driver.write("output", xr.Dataset())


class TestDetectMaxZoom:
    def test_detects_zoom_levels(self, tmp_path):
        for z in [0, 3, 5]:
            (tmp_path / str(z)).mkdir()
        result = SlippyTileDriver._detect_max_zoom(str(tmp_path))
        assert result == 5

    def test_empty_directory(self, tmp_path):
        result = SlippyTileDriver._detect_max_zoom(str(tmp_path))
        assert result == 0

    def test_nonexistent_directory(self, tmp_path):
        result = SlippyTileDriver._detect_max_zoom(str(tmp_path / "nope"))
        assert result == 0

    def test_ignores_non_numeric_dirs(self, tmp_path):
        (tmp_path / "3").mkdir()
        (tmp_path / "metadata").mkdir()
        (tmp_path / "readme.txt").touch()
        result = SlippyTileDriver._detect_max_zoom(str(tmp_path))
        assert result == 3


class TestResolveZoom:
    def test_none_derives_from_extent(self):
        # ~30km extent -> should pick a reasonable zoom
        result = SlippyTileDriver._resolve_zoom(None, 0, 30000, 15, 256)
        assert 0 < result <= 15

    def test_explicit_int(self):
        result = SlippyTileDriver._resolve_zoom(5, 0, 30000, 15, 256)
        assert result == 5

    def test_clamped_to_max_zoom(self):
        result = SlippyTileDriver._resolve_zoom(20, 0, 30000, 10, 256)
        assert result == 10

    def test_clamped_to_zero(self):
        result = SlippyTileDriver._resolve_zoom(-1, 0, 30000, 10, 256)
        assert result == 0

    def test_tuple_resolution(self):
        result = SlippyTileDriver._resolve_zoom((100, "metre"), 0, 30000, 15, 256)
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
