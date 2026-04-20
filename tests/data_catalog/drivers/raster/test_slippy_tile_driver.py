"""Tests for the SlippyTile raster driver."""

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from PIL import Image
from shapely.geometry import box

from hydromt._compat import HAS_BOTO3
from hydromt.data_catalog import DataCatalog
from hydromt.data_catalog.drivers.raster.slippy_tile_driver import (
    SlippyTileDriver,
    SlippyTileOptions,
    _download_missing_tiles,
    _download_tile,
    _num2xy,
    _png2value,
    _xy2num,
)
from hydromt.error import NoDataException, NoDataStrategy

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _elevation2terrarium(val: np.ndarray) -> np.ndarray:
    """Encode elevation values as terrarium RGB (inverse of _png2value)."""
    val = val + 32768.0
    r = np.floor(val / 256).astype(np.uint8)
    g = np.floor(val % 256).astype(np.uint8)
    b = np.floor((val - np.floor(val)) * 256).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _make_tile_png(tmp_path: Path, rgb: np.ndarray, mode: str = "RGB") -> str:
    """Write a raw RGB/RGBA PNG to ``tmp_path/tile.png`` and return its path."""
    png_path = tmp_path / "tile.png"
    Image.fromarray(rgb.astype(np.uint8), mode=mode).save(str(png_path))
    return str(png_path)


def _make_tile_dir(root: Path, zoom: int, tx: int, ty: int, elevation: np.ndarray):
    """Create a single terrarium-encoded tile at ``root/zoom/tx/ty.png``."""
    tile_dir = root / str(zoom) / str(tx)
    tile_dir.mkdir(parents=True, exist_ok=True)
    png_path = tile_dir / f"{ty}.png"
    Image.fromarray(_elevation2terrarium(elevation), mode="RGB").save(str(png_path))
    return png_path


def _make_mask_gdf(xmin, ymin, xmax, ymax, crs=3857):
    return gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs=crs)


# ---------------------------------------------------------------------------
# Tile-math sanity check
# ---------------------------------------------------------------------------


def test_xy2num_num2xy_roundtrip():
    """Xy <-> tile index roundtrip exercises the full mercator + tile math path."""
    zoom = 5
    x_in, y_in = 500000.0, 6000000.0
    tx, ty = _xy2num(x_in, y_in, zoom)
    x_out, y_out = _num2xy(tx, ty, zoom)
    # Tile size at zoom 5 in metres
    dx = 20037508.34 * 2 / (2**zoom)
    assert abs(x_in - x_out) < dx
    assert abs(y_in - y_out) < dx


# ---------------------------------------------------------------------------
# PNG decoders — one test per encoder family
# ---------------------------------------------------------------------------


def test_terrarium_roundtrip(tmp_path):
    elevation = np.array([[0.0, 100.5], [-50.0, 8848.0]])
    path = _make_tile_dir(tmp_path, 0, 0, 0, elevation)
    result = _png2value(str(path), encoder="terrarium")
    np.testing.assert_allclose(result, elevation, atol=1.0)


def test_terrarium_nodata(tmp_path):
    """Packed value 0 (encoding -32768) is NoData."""
    elevation = np.array([[-32768.0, 0.0], [0.0, -32768.0]])
    path = _make_tile_dir(tmp_path, 0, 0, 0, elevation)
    result = _png2value(str(path), encoder="terrarium")
    assert np.isnan(result[0, 0])
    assert np.isnan(result[1, 1])
    assert not np.isnan(result[0, 1])


@pytest.mark.parametrize(
    ("encoder", "mode", "rgb_set", "expected"),
    [
        # uint8: R channel carries the value directly
        (
            "uint8",
            "RGB",
            [(0, 0, 42), (0, 1, 200), (1, 0, 255)],
            {(0, 0): 42, (0, 1): 200, (1, 0): -1},
        ),
        # uint16: R*256 + G
        (
            "uint16",
            "RGB",
            [((0, 0), (3, 232, 0)), ((0, 1), (1, 244, 0))],
            {(0, 0): 1000, (0, 1): 500},
        ),
        # uint24: R*65536 + G*256 + B; all-255 is nodata
        (
            "uint24",
            "RGB",
            [((0, 0), (1, 0, 0)), ((0, 1), (0, 1, 0)), ((1, 0), (255, 255, 255))],
            {(0, 0): 65536, (0, 1): 256, (1, 0): -1},
        ),
        # uint32: R*2^24 + G*2^16 + B*256 + A
        (
            "uint32",
            "RGBA",
            [((0, 0), (0, 0, 1, 0)), ((1, 1), (255, 255, 255, 255))],
            {(0, 0): 256, (1, 1): -1},
        ),
    ],
)
def test_uint_encoders(tmp_path, encoder, mode, rgb_set, expected):
    channels = 4 if mode == "RGBA" else 3
    rgb = np.zeros((2, 2, channels), dtype=np.uint8)
    for spec in rgb_set:
        # Allow either flat (row, col, value) for single-channel sets or
        # nested ((row, col), (R,G,B[,A])) for multi-channel sets.
        if isinstance(spec[0], tuple):
            (r, c), chans = spec
            rgb[r, c, : len(chans)] = chans
        else:
            r, c, v = spec
            rgb[r, c, 0] = v
    path = _make_tile_png(tmp_path, rgb, mode=mode)
    result = _png2value(path, encoder=encoder)
    for (r, c), v in expected.items():
        assert result[r, c] == v


@pytest.mark.parametrize("encoder", ["float8", "float16", "float24", "float32"])
def test_float_encoders_nodata_and_max(tmp_path, encoder):
    """Packed value 0 is NoData for every float encoder; non-zero decodes in [vmin, vmax]."""
    mode = "RGBA" if encoder == "float32" else "RGB"
    channels = 4 if mode == "RGBA" else 3
    rgb = np.zeros((2, 2, channels), dtype=np.uint8)
    # Put a non-zero byte somewhere so at least one pixel decodes.
    rgb[0, 1, 0] = 1
    path = _make_tile_png(tmp_path, rgb, mode=mode)
    result = _png2value(path, encoder=encoder, encoder_vmin=0.0, encoder_vmax=100.0)
    assert np.isnan(result[0, 0])
    assert 0.0 <= result[0, 1] <= 100.0


def test_unknown_encoder_raises(tmp_path):
    path = _make_tile_dir(tmp_path, 0, 0, 0, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="Unknown encoder"):
        _png2value(str(path), encoder="bogus")


# ---------------------------------------------------------------------------
# S3 download helpers (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_BOTO3, reason="Requires boto3")
class TestS3Download:
    def test_download_tile_success_and_failure(self, tmp_path):
        """_download_tile returns True on success and False on any exception."""
        mock_client = MagicMock()
        assert (
            _download_tile(
                mock_client, "bucket", "k/0/0/0.png", str(tmp_path / "0/0/0.png")
            )
            is True
        )

        mock_client.download_file.side_effect = Exception("network error")
        assert (
            _download_tile(
                mock_client, "bucket", "k/0/0/0.png", str(tmp_path / "fail.png")
            )
            is False
        )

    @patch("hydromt.data_catalog.drivers.raster.slippy_tile_driver.boto3")
    def test_download_missing_tiles_downloads_and_skips(self, mock_boto3, tmp_path):
        """Missing tiles are downloaded; tiles that already exist are skipped."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        # (1, 0, 0) is missing; (1, 1, 1) already exists locally.
        (tmp_path / "1" / "1").mkdir(parents=True)
        (tmp_path / "1" / "1" / "1.png").touch()

        result = _download_missing_tiles(
            str(tmp_path), "bucket", "tiles", "us-east-1", [(1, 0, 0), (1, 1, 1)]
        )
        assert result == 1
        mock_client.download_file.assert_called_once()


# ---------------------------------------------------------------------------
# Driver read() behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
def tile_dir(tmp_path):
    """Create a 2x2 tile grid at zoom 1 with 4px terrarium tiles."""
    for tx in range(2):
        for ty in range(2):
            _make_tile_dir(tmp_path, 1, tx, ty, np.full((4, 4), float(tx * 10 + ty)))
    return str(tmp_path)


@pytest.fixture
def mask_global():
    return _make_mask_gdf(-15000000, -15000000, 15000000, 15000000, crs=3857)


def test_read_happy_path(tile_dir, mask_global):
    """End-to-end read: Dataset shape, CRS, coords, and default variable."""
    driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4))
    ds = driver.read([tile_dir], mask=mask_global, zoom=1)
    assert isinstance(ds, xr.Dataset)
    assert "elevation" in ds
    assert ds["elevation"].shape == (8, 8)
    assert "x" in ds.coords
    assert "y" in ds.coords
    assert ds.rio.crs.to_epsg() == 3857


def test_read_custom_variable_name(tile_dir, mask_global):
    driver = SlippyTileDriver(
        options=SlippyTileOptions(tile_size=4, variable_name="bathymetry")
    )
    ds = driver.read([tile_dir], mask=mask_global, zoom=1)
    assert "bathymetry" in ds
    assert "elevation" not in ds


def test_read_mask_in_4326_is_reprojected(tile_dir):
    """A mask in EPSG:4326 is reprojected to 3857 before tile lookup."""
    mask_4326 = _make_mask_gdf(-170, -80, 170, 80, crs=4326)
    driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4))
    ds = driver.read([tile_dir], mask=mask_4326, zoom=1)
    assert isinstance(ds, xr.Dataset)


def test_read_requires_mask(tile_dir):
    driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4))
    with pytest.raises(ValueError, match="requires a spatial mask"):
        driver.read([tile_dir])


@pytest.mark.parametrize(
    "zoom",
    [1, 1.0, (50000, "metre"), (50000, "feet"), None],
    ids=["int", "float", "tuple_metre", "tuple_unknown_unit", "none_auto"],
)
def test_read_zoom_variants(tile_dir, mask_global, zoom):
    """Driver accepts zoom as int, float, (res, unit) tuple, or None (auto)."""
    driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4, max_zoom=1))
    ds = driver.read([tile_dir], mask=mask_global, zoom=zoom)
    assert isinstance(ds, xr.Dataset)


def test_read_no_tiles_behaviour(tmp_path, mask_global):
    """No tiles + RAISE → NoDataException; + IGNORE → None."""
    empty_dir = str(tmp_path / "empty")
    os.makedirs(os.path.join(empty_dir, "1", "0"), exist_ok=True)
    driver = SlippyTileDriver(options=SlippyTileOptions(tile_size=4, max_zoom=1))

    with pytest.raises(NoDataException, match="No tiles found"):
        driver.read([empty_dir], mask=mask_global, zoom=1)

    ds = driver.read(
        [empty_dir], mask=mask_global, zoom=1, handle_nodata=NoDataStrategy.IGNORE
    )
    assert ds is None


def test_write_raises():
    driver = SlippyTileDriver()
    with pytest.raises(NotImplementedError):
        driver.write("output", xr.Dataset())


def test_detect_max_zoom(tmp_path):
    """_detect_max_zoom finds numeric subdirs and ignores everything else."""
    for z in [0, 3, 5]:
        (tmp_path / str(z)).mkdir()
    (tmp_path / "metadata").mkdir()
    (tmp_path / "readme.txt").touch()
    assert SlippyTileDriver._detect_max_zoom(str(tmp_path)) == 5
    # Missing / empty dir → 0
    assert SlippyTileDriver._detect_max_zoom(str(tmp_path / "nope")) == 0


def test_options_defaults():
    """Pydantic defaults wire through — guards against future option renames."""
    opts = SlippyTileOptions()
    assert opts.encoder == "terrarium"
    assert opts.tile_size == 256
    assert opts.variable_name == "elevation"
    assert opts.max_zoom is None
    assert opts.s3_bucket is None


# ---------------------------------------------------------------------------
# S3 integration — real bucket, marked to skip without network / boto3
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gebco_2024_catalog_path(test_data_dir):
    path = test_data_dir / "gebco_2024" / "data_catalog.yml"
    assert path.is_file(), f"Expected catalog YAML at {path}"
    return path


@pytest.mark.integration
@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 required for S3 integration")
class TestSlippyTileDriverS3Integration:
    """End-to-end tests against the public deltares-ddb GEBCO 2024 tile set."""

    def test_downloads_north_sea(self, tmp_path, gebco_2024_catalog_path):
        """Download tiles for a North Sea bbox from the public S3 bucket.

        Catalog is copied to ``tmp_path`` so downloaded tiles land in the
        temp dir and the repository stays clean.
        """
        tmp_catalog = tmp_path / "data_catalog.yml"
        shutil.copy2(str(gebco_2024_catalog_path), str(tmp_catalog))
        cat = DataCatalog(data_libs=[str(tmp_catalog)])
        ds = cat.get_rasterdataset(
            "gebco_2024", bbox=(5.0, 52.0, 6.0, 56.0), zoom=(2000, "metre")
        )
        values = ds.values if isinstance(ds, xr.DataArray) else ds["elevation"].values
        finite = np.isfinite(values)
        assert finite.any()
        assert np.nanmin(values) < 0  # negative bathymetry expected
        assert list(tmp_catalog.parent.rglob("*.png"))  # tiles were downloaded

    def test_downloads_florida(self, tmp_path, gebco_2024_catalog_path):
        """Download tiles for a deep-Atlantic bbox from the public S3 bucket."""
        tmp_catalog = tmp_path / "data_catalog.yml"
        shutil.copy2(str(gebco_2024_catalog_path), str(tmp_catalog))
        cat = DataCatalog(data_libs=[str(tmp_catalog)])
        ds = cat.get_rasterdataset(
            "gebco_2024", bbox=(-80.0, 30.0, -78.0, 32.0), zoom=(2000, "metre")
        )
        values = ds.values if isinstance(ds, xr.DataArray) else ds["elevation"].values
        assert np.isfinite(values).any()
        assert np.nanmin(values) < -100  # deep Atlantic
