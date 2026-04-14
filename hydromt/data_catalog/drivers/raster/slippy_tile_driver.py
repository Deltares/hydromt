"""HydroMT driver for reading XYZ / slippy map tiles.

Reads terrain-encoded PNG tiles stored in the standard ``{zoom}/{x}/{y}.png``
directory structure and returns the data as an ``xr.Dataset`` in EPSG:3857.

Missing tiles are automatically downloaded from an S3 bucket when
``s3_bucket``, ``s3_key``, and ``s3_region`` are configured.

Includes tile-math and PNG decoding utilities adapted from
``cht_tiling.utils`` (Deltares).

Usage in a HydroMT data catalog YAML::

    my_tiles:
      data_type: RasterDataset
      driver:
        name: slippy_tile
        options:
          encoder: terrarium
          s3_bucket: deltares-ddb
          s3_key: data/bathymetry/my_dataset
          s3_region: eu-west-1
      uri: c:/data/tiles/my_dataset/
      metadata:
        crs: 3857

The driver requires a spatial *mask* (bounding box) to know which tiles to
read.  The zoom level can be passed explicitly or is derived from the mask
extent (targeting ~1024 pixels across the x-range).
"""

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Tuple

import numpy as np
import xarray as xr
from PIL import Image
from pydantic import Field

from hydromt.data_catalog.drivers.base_driver import DriverOptions
from hydromt.data_catalog.drivers.raster.raster_dataset_driver import (
    RasterDatasetDriver,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.typing import Geom, SourceMetadata, Variables, Zoom

logger = logging.getLogger(__name__)

# Optional S3 dependency
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config as BotoConfig

    _HAS_BOTO3 = True
except ImportError:
    # Bind the names to None so tests can ``mock.patch(..., 'boto3')`` and
    # downstream code can do a single ``if not _HAS_BOTO3`` guard.
    boto3 = None
    UNSIGNED = None
    BotoConfig = None
    _HAS_BOTO3 = False

# ---------------------------------------------------------------------------
# Tile utilities (adapted from cht_tiling.utils)
# ---------------------------------------------------------------------------

# Pixel size at zoom 0 in metres (at the equator)
_ZOOM0_PIXEL_SIZE = 156543.03

# Half the Web Mercator (EPSG:3857) extent in metres — i.e. the x/y coordinate
# at ±180° longitude / ±~85.0511° latitude.
_WEBMERCATOR_HALF_EXTENT = 20037508.34

# Powers of two used by the PNG elevation decoders below. Precomputed once
# so the inner loops don't recompute them per call / per tile.
_TWO_POW_8 = 256  # 2**8 — byte
_TWO_POW_15 = 32768  # 2**15 — terrarium zero offset
_TWO_POW_16 = 65536  # 2**16 — two-byte word
_TWO_POW_24 = 16777216  # 2**24 — three-byte word

# Max packed integer value at each bit-depth — reserved as NoData sentinel.
_MAX_UINT8 = 255  # 2**8 - 1
_MAX_UINT16 = 65535  # 2**16 - 1
_MAX_UINT24 = 16777215  # 2**24 - 1
_MAX_UINT32 = 4294967295  # 2**32 - 1

# Divisors for float encoders: both 0 and the max value are reserved, so the
# number of usable levels is 2**N - 2.
_FLOAT8_RANGE = 254  # 2**8 - 2
_FLOAT16_RANGE = 65534  # 2**16 - 2
_FLOAT24_RANGE = 16777214  # 2**24 - 2
_FLOAT32_RANGE = 4294967294  # 2**32 - 2

# Terrarium lower bound: the packed value 0 encodes the minimum (-32768); any
# decoded value below -(2**15 - 1) came from a NoData tile pixel.
_TERRARIUM_NODATA_THRESHOLD = -32767  # -(2**15 - 1)


def _get_zoom_level_for_resolution(dx: float) -> int:
    """Return the tile zoom level whose pixel size is just below *dx*.

    Parameters
    ----------
    dx : float
        Target pixel size in metres.

    Returns
    -------
    int
        Zoom level (0-23).
    """
    dxy = _ZOOM0_PIXEL_SIZE / 2 ** np.arange(24)
    izoom = np.nonzero(dxy < dx)[0]
    if len(izoom) == 0:
        return 23
    return int(izoom[0])


def _webmercator_to_latlon(easting: float, northing: float) -> Tuple[float, float]:
    """Convert Web Mercator (EPSG:3857) to latitude/longitude (degrees)."""
    lon = (easting / _WEBMERCATOR_HALF_EXTENT) * 180.0
    lat = (180.0 / math.pi) * (
        2.0 * math.atan(math.exp(northing / _WEBMERCATOR_HALF_EXTENT * math.pi))
        - (math.pi / 2.0)
    )
    return lat, lon


def _latlon_to_webmercator(lat: float, lon: float) -> Tuple[float, float]:
    """Convert latitude/longitude (degrees) to Web Mercator (EPSG:3857)."""
    x = lon * _WEBMERCATOR_HALF_EXTENT / 180.0
    y = (
        math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / math.pi
    ) * _WEBMERCATOR_HALF_EXTENT
    return x, y


def _latlon_to_tile_indices(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert latitude/longitude to tile column and row indices."""
    tile_x = int((lon + 180.0) / 360.0 * (2**zoom))
    tile_y = int(
        (
            1.0
            - (
                math.log(
                    math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))
                )
                / math.pi
            )
        )
        / 2.0
        * (2**zoom)
    )
    return tile_x, tile_y


def _xy2num(easting: float, northing: float, zoom: int) -> Tuple[int, int]:
    """Convert Web Mercator coordinates to tile indices.

    Parameters
    ----------
    easting, northing : float
        Coordinates in EPSG:3857 (metres).
    zoom : int
        Tile zoom level.

    Returns
    -------
    tuple[int, int]
        (tile_x, tile_y) indices.
    """
    lat, lon = _webmercator_to_latlon(easting, northing)
    return _latlon_to_tile_indices(lat, lon, zoom)


def _num2xy(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Convert tile indices to the upper-left Web Mercator coordinates.

    Parameters
    ----------
    xtile, ytile : int
        Tile column and row indices.
    zoom : int
        Tile zoom level.

    Returns
    -------
    tuple[float, float]
        (x, y) in EPSG:3857 metres.
    """
    n = 2**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return _latlon_to_webmercator(lat_deg, lon_deg)


def _png2elevation(
    png_file: str | Path,
    encoder: str = "terrarium",
    encoder_vmin: float = 0.0,
    encoder_vmax: float = 1.0,
) -> np.ndarray:
    """Decode a terrain-RGB PNG tile to an elevation array.

    Parameters
    ----------
    png_file : str or Path
        Path to the PNG tile.
    encoder : str, optional
        Encoding scheme.  One of ``'terrarium'``, ``'terrarium16'``,
        ``'uint8'``, ``'uint16'``, ``'uint24'``, ``'uint32'``,
        ``'float8'``, ``'float16'``, ``'float24'``, ``'float32'``.
        Default is ``'terrarium'``.
    encoder_vmin, encoder_vmax : float, optional
        Value range for float encoders.

    Returns
    -------
    np.ndarray
        2-D elevation array (npix x npix).  NoData pixels are ``np.nan``
        for float encoders and terrarium, or ``-1`` for uint encoders.
    """
    with Image.open(png_file) as img:
        if encoder == "terrarium":
            rgb = np.array(img.convert("RGB")).astype(float)
            elevation = (
                rgb[:, :, 0] * _TWO_POW_8 + rgb[:, :, 1] + rgb[:, :, 2] / _TWO_POW_8
            ) - _TWO_POW_15
            elevation[elevation < _TERRARIUM_NODATA_THRESHOLD] = np.nan
        elif encoder == "terrarium16":
            rgb = np.array(img.convert("RGB")).astype(float)
            elevation = (rgb[:, :, 0] * _TWO_POW_8 + rgb[:, :, 1]) - _TWO_POW_15
            elevation[elevation < _TERRARIUM_NODATA_THRESHOLD] = np.nan
        elif encoder == "uint8":
            rgb = np.array(img.convert("RGB")).astype(int)
            elevation = rgb[:, :, 0]
            elevation[elevation == _MAX_UINT8] = -1
        elif encoder == "uint16":
            rgb = np.array(img.convert("RGB")).astype(int)
            elevation = rgb[:, :, 0] * _TWO_POW_8 + rgb[:, :, 1]
            elevation[elevation == _MAX_UINT16] = -1
        elif encoder == "uint24":
            rgb = np.array(img.convert("RGB")).astype(int)
            elevation = (
                rgb[:, :, 0] * _TWO_POW_16 + rgb[:, :, 1] * _TWO_POW_8 + rgb[:, :, 2]
            )
            elevation[elevation == _MAX_UINT24] = -1
        elif encoder == "uint32":
            rgb = np.array(img.convert("RGBA")).astype(int)
            elevation = (
                rgb[:, :, 0] * _TWO_POW_24
                + rgb[:, :, 1] * _TWO_POW_16
                + rgb[:, :, 2] * _TWO_POW_8
                + rgb[:, :, 3]
            )
            elevation[elevation == _MAX_UINT32] = -1
        elif encoder == "float8":
            rgb = np.array(img.convert("RGB")).astype(float)
            idx = rgb[:, :, 0]
            elevation = (
                encoder_vmin + (encoder_vmax - encoder_vmin) * idx / _FLOAT8_RANGE
            )
            elevation[idx == 0] = np.nan
        elif encoder == "float16":
            rgb = np.array(img.convert("RGB")).astype(float)
            idx = rgb[:, :, 0] * _TWO_POW_8 + rgb[:, :, 1]
            elevation = (
                encoder_vmin + (encoder_vmax - encoder_vmin) * idx / _FLOAT16_RANGE
            )
            elevation[idx == 0] = np.nan
        elif encoder == "float24":
            rgb = np.array(img.convert("RGB")).astype(float)
            idx = rgb[:, :, 0] * _TWO_POW_16 + rgb[:, :, 1] * _TWO_POW_8 + rgb[:, :, 2]
            elevation = (
                encoder_vmin + (encoder_vmax - encoder_vmin) * idx / _FLOAT24_RANGE
            )
            elevation[idx == 0] = np.nan
        elif encoder == "float32":
            rgb = np.array(img.convert("RGBA")).astype(float)
            idx = (
                rgb[:, :, 0] * _TWO_POW_24
                + rgb[:, :, 1] * _TWO_POW_16
                + rgb[:, :, 2] * _TWO_POW_8
                + rgb[:, :, 3]
            )
            elevation = (
                encoder_vmin + (encoder_vmax - encoder_vmin) * idx / _FLOAT32_RANGE
            )
            elevation[idx == 0] = np.nan
        else:
            raise ValueError(f"Unknown encoder: {encoder!r}")

    return elevation


# ---------------------------------------------------------------------------
# S3 tile download
# ---------------------------------------------------------------------------


def _download_tile(s3_client: Any, bucket: str, key: str, filename: str) -> bool:
    """Download a single tile from S3.  Returns True on success."""
    try:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(Bucket=bucket, Key=key, Filename=filename)
        return True
    except Exception as e:
        logger.debug(f"Failed to download {key}: {e}")
        return False


def _download_missing_tiles(
    tile_root: str,
    s3_bucket: str,
    s3_key: str,
    s3_region: str,
    tile_indices: List[Tuple[int, int, int]],
) -> int:
    """Download missing tiles from S3 in parallel.

    Parameters
    ----------
    tile_root : str
        Local tile directory.
    s3_bucket : str
        S3 bucket name.
    s3_key : str
        S3 key prefix (e.g. ``'data/bathymetry/gebco_2024'``).
    s3_region : str
        AWS region.
    tile_indices : list of (zoom, x_tile, y_tile)
        Tiles to check and download if missing.

    Returns
    -------
    int
        Number of tiles downloaded.
    """
    if not _HAS_BOTO3:
        logger.warning("boto3 not installed — cannot download missing tiles from S3.")
        return 0

    # Collect missing tiles
    to_download: List[Tuple[str, str]] = []  # (s3_key, local_path)
    for izoom, itile, j in tile_indices:
        png_file = os.path.join(tile_root, str(izoom), str(itile), f"{j}.png")
        if not os.path.exists(png_file):
            key = f"{s3_key}/{izoom}/{itile}/{j}.png"
            to_download.append((key, png_file))

    if not to_download:
        return 0

    logger.info(
        f"Downloading {len(to_download)} missing tiles from s3://{s3_bucket}/{s3_key}/ ..."
    )
    s3_client = boto3.client(
        "s3", region_name=s3_region, config=BotoConfig(signature_version=UNSIGNED)
    )

    downloaded = 0
    with ThreadPoolExecutor() as pool:
        futures = [
            pool.submit(_download_tile, s3_client, s3_bucket, key, local)
            for key, local in to_download
        ]
        for future in futures:
            if future.result():
                downloaded += 1

    logger.info(f"Downloaded {downloaded}/{len(to_download)} tiles.")
    return downloaded


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class SlippyTileOptions(DriverOptions):
    """Options for the slippy tile driver.

    Parameters
    ----------
    encoder : str
        PNG encoding scheme (e.g. ``'terrarium'``, ``'float16'``).
    encoder_vmin, encoder_vmax : float
        Value range for float-type encoders.
    max_zoom : int or None
        Maximum available zoom level.  If *None*, the driver scans the
        tile directory to determine the highest available zoom.
    variable_name : str
        Name of the data variable in the returned Dataset.
    tile_size : int
        Pixel size of each tile (default 256).
    s3_bucket : str or None
        S3 bucket for downloading missing tiles.
    s3_key : str or None
        S3 key prefix (e.g. ``'data/bathymetry/gebco_2024'``).
    s3_region : str or None
        AWS region of the S3 bucket.
    """

    encoder: str = Field(default="terrarium", description="PNG encoding scheme")
    encoder_vmin: float = Field(default=0.0, description="Min value for float encoders")
    encoder_vmax: float = Field(default=1.0, description="Max value for float encoders")
    max_zoom: Optional[int] = Field(
        default=None, description="Max available zoom level"
    )
    variable_name: str = Field(default="elevation", description="Output variable name")
    tile_size: int = Field(default=256, description="Tile pixel size")
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket name")
    s3_key: Optional[str] = Field(default=None, description="S3 key prefix")
    s3_region: Optional[str] = Field(default=None, description="AWS region")


class SlippyTileDriver(RasterDatasetDriver):
    """Read elevation data from local XYZ / slippy map tile directories.

    Tiles must be stored as ``{uri}/{zoom}/{x}/{y}.png`` where *x* and *y*
    are standard Web Mercator tile indices.  The PNG files are decoded using
    the specified *encoder* (default: terrarium).

    When ``s3_bucket``, ``s3_key``, and ``s3_region`` are set, missing tiles
    are automatically downloaded from S3 (unsigned/public access) before
    reading.

    The returned ``xr.Dataset`` has coordinates in EPSG:3857.
    """

    name: ClassVar[str] = "slippy_tile"
    supports_writing: ClassVar[bool] = False
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = set()  # directory-based, no extension

    options: SlippyTileOptions = Field(default_factory=SlippyTileOptions)

    def read(
        self,
        uris: list[str],
        *,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        mask: Geom | None = None,
        variables: Variables | None = None,
        zoom: Zoom | None = None,
        chunks: dict[str, Any] | None = None,
        metadata: SourceMetadata | None = None,
    ) -> xr.Dataset:
        """Read slippy tiles within a bounding box and return as Dataset.

        Parameters
        ----------
        uris : list[str]
            Single-element list with the path to the tile root directory.
        mask : Geom, optional
            Geometry whose bounding box determines which tiles to read.
            **Required** — the driver cannot read without spatial bounds.
        zoom : Zoom, optional
            Zoom level as an ``int``, or ``(resolution, unit)`` tuple.
            If *None*, the zoom level is derived from the mask extent.
        handle_nodata : NoDataStrategy
            How to handle the case where no tiles are found.
        variables : Variables, optional
            Ignored (single-variable output).
        chunks : dict, optional
            Ignored (data is read eagerly).
        metadata : SourceMetadata, optional
            Source metadata (CRS, nodata, etc.).

        Returns
        -------
        xr.Dataset
            Dataset with one data variable (default ``'elevation'``) and
            ``x`` / ``y`` coordinates in EPSG:3857.
        """
        tile_root = uris[0]
        opts = self.options

        # --- resolve max zoom ------------------------------------------------
        max_zoom = opts.max_zoom
        if max_zoom is None:
            max_zoom = self._detect_max_zoom(tile_root)

        # --- resolve bounding box in EPSG:3857 --------------------------------
        if mask is None:
            raise ValueError("SlippyTileDriver requires a spatial mask (bounding box).")
        bbox_3857 = self._mask_to_bbox_3857(mask)
        xmin, ymin, xmax, ymax = bbox_3857

        # --- resolve zoom level -----------------------------------------------
        izoom = self._resolve_zoom(zoom, xmin, xmax, max_zoom, opts.tile_size)

        # --- compute tile index range -----------------------------------------
        ix0, iy0 = _xy2num(xmin, ymax, izoom)  # top-left
        ix1, iy1 = _xy2num(xmax, ymin, izoom)  # bottom-right
        ix0 = max(0, ix0)
        iy0 = max(0, iy0)
        iy1 = min(2**izoom - 1, iy1)
        ntiles = 2**izoom

        # --- download missing tiles from S3 if configured ---------------------
        if opts.s3_bucket and opts.s3_key and opts.s3_region:
            tile_list = [
                (izoom, i % ntiles, j)
                for i in range(ix0, ix1 + 1)
                for j in range(iy0, iy1 + 1)
            ]
            _download_missing_tiles(
                tile_root, opts.s3_bucket, opts.s3_key, opts.s3_region, tile_list
            )

        # --- read and assemble tiles ------------------------------------------
        npix = opts.tile_size
        nx = (ix1 - ix0 + 1) * npix
        ny = (iy1 - iy0 + 1) * npix
        z = np.full((ny, nx), np.nan, dtype=np.float64)
        tiles_read = 0

        for i in range(ix0, ix1 + 1):
            itile = i % ntiles  # wrap around dateline
            for j in range(iy0, iy1 + 1):
                png_file = os.path.join(tile_root, str(izoom), str(itile), f"{j}.png")
                if not os.path.exists(png_file):
                    continue
                tile_data = _png2elevation(
                    png_file,
                    encoder=opts.encoder,
                    encoder_vmin=opts.encoder_vmin,
                    encoder_vmax=opts.encoder_vmax,
                )
                ii0 = (i - ix0) * npix
                jj0 = (j - iy0) * npix
                z[jj0 : jj0 + npix, ii0 : ii0 + npix] = tile_data
                tiles_read += 1

        if tiles_read == 0:
            exec_nodata_strat(
                f"No tiles found in {tile_root} at zoom {izoom} for bbox {bbox_3857}",
                handle_nodata,
            )
            return None

        # --- compute coordinates ----------------------------------------------
        x0_m, y0_m = _num2xy(ix0, iy1 + 1, izoom)  # lower-left corner
        x1_m, y1_m = _num2xy(ix1 + 1, iy0, izoom)  # upper-right corner
        dx = (x1_m - x0_m) / nx
        dy = (y1_m - y0_m) / ny
        x = np.linspace(x0_m + 0.5 * dx, x1_m - 0.5 * dx, nx)
        y = np.linspace(y0_m + 0.5 * dy, y1_m - 0.5 * dy, ny)
        z = np.flipud(z)  # tiles are stored top-to-bottom

        # --- build xr.Dataset -------------------------------------------------
        var_name = opts.variable_name
        ds = xr.Dataset(
            {var_name: (["y", "x"], z.astype(np.float32))},
            coords={"x": x, "y": y},
        )
        ds.x.attrs["units"] = "m"
        ds.y.attrs["units"] = "m"
        ds.attrs["crs"] = 3857

        # set spatial dims for rioxarray compatibility
        ds = ds.rio.write_crs(3857)
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")

        logger.info(
            f"Read {tiles_read} tiles from {tile_root} at zoom {izoom} "
            f"({nx}x{ny} pixels)"
        )
        return ds

    def write(
        self,
        path: Path | str,
        data: xr.Dataset,
        *,
        write_kwargs: dict[str, Any] | None = None,
    ) -> Path:
        """Write is not supported for slippy tiles."""
        raise NotImplementedError(
            "SlippyTileDriver does not support writing. "
            "Use cht_tiling to create tile sets."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_max_zoom(tile_root: str) -> int:
        """Scan the tile directory for the highest available zoom level."""
        max_zoom = 0
        if not os.path.isdir(tile_root):
            return max_zoom
        for entry in os.listdir(tile_root):
            full = os.path.join(tile_root, entry)
            if os.path.isdir(full):
                try:
                    level = int(entry)
                    max_zoom = max(max_zoom, level)
                except ValueError:
                    continue
        return max_zoom

    @staticmethod
    def _mask_to_bbox_3857(mask: Geom) -> Tuple[float, float, float, float]:
        """Convert a mask geometry to a bounding box in EPSG:3857."""
        gdf = mask.to_crs(3857)
        return tuple(gdf.total_bounds)

    @staticmethod
    def _resolve_zoom(
        zoom: Zoom | None,
        xmin: float,
        xmax: float,
        max_zoom: int,
        tile_size: int,
    ) -> int:
        """Determine the tile zoom level from user input or bbox extent."""
        if zoom is None:
            pixel_size = (xmax - xmin) / 1024.0
            izoom = _get_zoom_level_for_resolution(pixel_size)
        elif isinstance(zoom, int):
            izoom = zoom
        elif isinstance(zoom, tuple):
            resolution, unit = zoom
            if unit not in ("m", "metre", "meter"):
                logger.warning(
                    f"Zoom unit '{unit}' not recognised, treating as metres."
                )
            izoom = _get_zoom_level_for_resolution(resolution)
        else:
            izoom = int(zoom)

        return min(max(izoom, 0), max_zoom)
