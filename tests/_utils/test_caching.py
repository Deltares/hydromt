import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import geopandas as gpd
import pytest
import rasterio as rio

from hydromt._compat import HAS_GDAL
from hydromt._utils.caching import (
    _cache_vrt_tiles,
    _copy_to_local,
    _overlaps,
)


def test__copy_to_local(tmp_path: Path, data_dir: Path):
    # Call the function
    _copy_to_local(
        Path(data_dir, "parameters_data.yml"),
        tmp_path,
    )

    # Assert the output, i.e. copying succeded
    assert Path(tmp_path, "parameters_data.yml").is_file()


def test__overlaps_true(vrt_dataset: ET.Element, vrt_affine: rio.Affine):
    # Get a source element
    source = vrt_dataset.find("VRTRasterBand/SimpleSource")

    # Call the function with a bbox overlapping with the tile
    out = _overlaps(source, affine=vrt_affine, bbox=[1, 0.5, 2.5, 2.5])

    # Tile bbox = [0, 0, 2, 3], so this should be true
    # Assert the output
    assert out


def test__overlaps_false(vrt_dataset: ET.Element, vrt_affine: rio.Affine):
    # Get a source element
    source = vrt_dataset.find("VRTRasterBand/SimpleSource")

    # Call the function with a bbox not overlapping with the tile (lies to the right)
    out = _overlaps(source, affine=vrt_affine, bbox=[2.5, 0.5, 4.5, 2.5])

    # Tile bbox = [0, 0, 2, 3], so this should be true
    # Assert the output
    assert not out


@pytest.mark.skipif(not HAS_GDAL, reason="GDAL not installed.")
def test__cache_vrt_files(
    vrt_path: Path,
    vrt_cache_dir: Path,
    geom_select1: gpd.GeoDataFrame,
):
    # Call the function
    out = _cache_vrt_tiles(
        vrt_uri=vrt_path.as_posix(),
        geom=geom_select1,
        cache_dir=vrt_cache_dir,
    )

    # Assert the output
    p = Path(vrt_cache_dir, "tmp.vrt")
    assert out == p
    assert out.is_file()
    # Assert its content
    with open(out, "r") as r:
        root = ET.fromstring(r.read())

    # Assert one element
    assert len(root.findall("VRTRasterBand/SimpleSource")) == 1


@pytest.mark.skipif(HAS_GDAL, reason="GDAL installed.")
def test__cache_vrt_files_import_error(
    vrt_path: Path,
):
    # With should error on the basis that gdal is not installed
    with pytest.raises(
        ImportError,
        match="Can't cache vrt's without GDAL installed.",
    ):
        # Call the function
        _ = _cache_vrt_tiles(
            vrt_uri=vrt_path.as_posix(),
        )


@pytest.mark.skipif(not HAS_GDAL, reason="GDAL not installed.")
def test__cache_vrt_files_append(
    vrt_path: Path,
    vrt_cache_dir: Path,
    geom_select1: gpd.GeoDataFrame,
):
    # Call the function
    out = _cache_vrt_tiles(
        vrt_uri=vrt_path.as_posix(),
        geom=geom_select1,
        cache_dir=vrt_cache_dir,
    )
    # Assert that the vrt now exists
    assert out.is_file()

    # Call the function again, but on a larger area that encompasses the first file
    out = _cache_vrt_tiles(
        vrt_uri=vrt_path.as_posix(),
        cache_dir=vrt_cache_dir,
    )
    # Assert the file is still there
    assert out.is_file()

    # Assert its content, with one extra added tile
    with open(out, "r") as r:
        root = ET.fromstring(r.read())

    # Assert one element
    assert len(root.findall("VRTRasterBand/SimpleSource")) == 2


@pytest.mark.skipif(not HAS_GDAL, reason="GDAL not installed.")
def test__cache_vrt_files_exists(
    vrt_path: Path,
    vrt_cache_dir: Path,
    geom_select1: gpd.GeoDataFrame,
):
    # Call the function
    out = _cache_vrt_tiles(
        vrt_uri=vrt_path.as_posix(),
        geom=geom_select1,
        cache_dir=vrt_cache_dir,
    )
    # Assert that the vrt now exists
    assert out.is_file()

    # Re-call with the same extent
    out2 = _cache_vrt_tiles(
        vrt_uri=vrt_path.as_posix(),
        geom=geom_select1,
        cache_dir=vrt_cache_dir,
    )
    # Assert that simply the file is returned as no new tiles were encountered
    assert out == out2


@pytest.mark.skipif(not HAS_GDAL, reason="GDAL not installed.")
def test__cache_vrt_files_not_found(
    vrt_path: Path,
    vrt_cache_dir: Path,
    geom_select2: gpd.GeoDataFrame,
):
    # Call the function with a geometry that lies above
    out = _cache_vrt_tiles(
        vrt_uri=vrt_path.as_posix(),
        geom=geom_select2,
        cache_dir=vrt_cache_dir,
    )

    # Assert the output
    assert out == vrt_path


@pytest.mark.skipif(not HAS_GDAL, reason="GDAL not installed.")
def test__cache_vrt_files_meta_errors(
    tmp_path: Path,
    vrt_dataset: ET.Element,
):
    # Without geotransform
    p = Path(tmp_path, "no_gtf.vrt")
    tmp_vrt = deepcopy(vrt_dataset)
    tmp_vrt.remove(tmp_vrt.find("GeoTransform"))
    ET.ElementTree(tmp_vrt).write(p)

    # Call the function, which should error
    with pytest.raises(ValueError, match="No GeoTransform found in:"):
        _ = _cache_vrt_tiles(p.as_posix())

    # Without srs
    p = Path(tmp_path, "no_srs.vrt")
    tmp_vrt = deepcopy(vrt_dataset)
    tmp_vrt.remove(tmp_vrt.find("SRS"))
    ET.ElementTree(tmp_vrt).write(p)

    # Call the function, which should error
    with pytest.raises(ValueError, match="No SRS info found at:"):
        _ = _cache_vrt_tiles(p.as_posix())


@pytest.mark.skipif(not HAS_GDAL, reason="GDAL not installed.")
def test__cache_vrt_files_source_errors(
    tmp_path: Path,
    vrt_dataset: ET.Element,
):
    # Without band
    p = Path(tmp_path, "no_band.vrt")
    tmp_vrt = deepcopy(vrt_dataset)
    tmp_vrt.remove(tmp_vrt.find("VRTRasterBand"))
    ET.ElementTree(tmp_vrt).write(p)

    # Call the function, which should error
    with pytest.raises(ValueError, match="Could not find VRTRasterBand in:"):
        _ = _cache_vrt_tiles(p.as_posix())

    # Without proper source data
    p = Path(tmp_path, "no_sources.vrt")
    tmp_vrt = deepcopy(vrt_dataset)
    band = tmp_vrt.find("VRTRasterBand")
    for item in list(band.findall("SimpleSource")):
        band.remove(item)
    ET.ElementTree(tmp_vrt).write(p)

    # Call the function, which should error
    with pytest.raises(ValueError, match="No Source information found at:"):
        _ = _cache_vrt_tiles(p.as_posix())

    # Without proper source data
    p = Path(tmp_path, "no_source_file.vrt")
    tmp_vrt = deepcopy(vrt_dataset)
    source = tmp_vrt.find("VRTRasterBand/SimpleSource")
    source.remove(source.find("SourceFilename"))
    ET.ElementTree(tmp_vrt).write(p)

    # Call the function, which should error
    with pytest.raises(ValueError, match="Could not find Source File in vrt:"):
        _ = _cache_vrt_tiles(p.as_posix())
