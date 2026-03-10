import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import shapely.geometry as sg


## For vrt caching
# Helper functions
def create_tile(
    path: Path,
    factor: int,
) -> Path:
    data = np.ones((3, 2), dtype=np.float32) * factor
    with rio.open(
        path,
        mode="w",
        driver="GTiff",
        height=3,
        width=2,
        count=1,
        dtype=np.float32,
        crs=rio.CRS.from_epsg(4326),
        transform=rio.Affine(1.0, 0.0, ((factor - 1) * 2.0), 0.0, -1.0, 3.0),
        # nodata=-9999.0,
    ) as r:
        r.write(data, 1)


def create_vrt_source(
    root: ET.Element,
    path: Path,
    idx: int,
) -> ET.Element:
    src = ET.SubElement(root, "SimpleSource")
    # Metadata
    src_fname = ET.SubElement(src, "SourceFilename", attrib={"relativeToVRT": "0"})
    src_fname.text = path.as_posix()
    src_band = ET.SubElement(src, "SourceBand")
    src_band.text = "1"
    # Positioning
    _ = ET.SubElement(
        src,
        "SrcRect",
        attrib={"xOff": "0", "yOff": "0", "xSize": "2", "ySize": "3"},
    )
    xoff = str(int(idx * 2))
    _ = ET.SubElement(
        src,
        "DstRect",
        attrib={"xOff": xoff, "yOff": "0", "xSize": "2", "ySize": "3"},
    )
    return src


# Actual fixtures
@pytest.fixture(scope="session")
def catalog_dummy() -> dict[str, Any]:
    catalog = {
        "raster1": {
            "crs": 4326,
            "data_type": "RasterDataset",
            "driver": {
                "name": "rasterio",
                "filesystem": "local",
            },
            "uri": "foo.tif",
        },
        "vector1": {
            "crs": 4326,
            "data_type": "GeoDataFrame",
            "driver": {
                "name": "pyogrio",
                "filesystem": "local",
            },
            "uri": "bar.fgb",
        },
        "raster2": {
            "crs": 3857,
            "data_type": "RasterDataset",
            "driver": {
                "name": "rasterio",
                "filesystem": "local",
            },
            "uri": "baz.tif",
        },
    }
    return catalog


@pytest.fixture(scope="session")
def geom_select1() -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        geometry=[sg.box(-0.5, 0.5, 1.5, 2.5)],
        crs=4326,
    )
    return gdf


@pytest.fixture(scope="session")
def geom_select2() -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        geometry=[sg.box(-0.5, 3.5, 1.5, 5.5)],
        crs=4326,
    )
    return gdf


@pytest.fixture
def source_tile1(tmp_path: Path) -> Path:
    p = Path(tmp_path, "tile1.tif")
    create_tile(path=p, factor=1)
    assert p.is_file()
    return p


@pytest.fixture
def source_tile2(tmp_path: Path) -> Path:
    p = Path(tmp_path, "tile2.tif")
    create_tile(path=p, factor=2)
    assert p.is_file()
    return p


@pytest.fixture(scope="session")
def vrt_affine() -> rio.Affine:
    aff = rio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0)
    return aff


@pytest.fixture
def vrt_dataset(source_tile1: Path, source_tile2: Path) -> ET.Element:
    # Set the root
    root = ET.Element(
        "VRTDataset",
        attrib={"rasterXSize": "4", "rasterYSize": "3"},
    )

    # Add dataset metadata
    srs = ET.SubElement(root, "SRS", attrib={"dataAxisToSRSAxisMapping": "2,1"})
    srs.text = rio.CRS.from_epsg(4326).to_wkt()
    gtf = ET.SubElement(root, "GeoTransform")
    gtf.text = "0.0, 1.0, 0.0, 3.0, 0.0, -1.0"

    # Define the band
    band = ET.SubElement(
        root, "VRTRasterBand", attrib={"dataType": "Float32", "band": "1"}
    )
    # Something to visualize
    color = ET.SubElement(band, "ColorInterp")
    color.text = "Gray"

    # Add tiles to the band
    for idx, path in enumerate([source_tile1, source_tile2]):
        _ = create_vrt_source(band, path, idx)

    # Return the root
    return root


@pytest.fixture
def vrt_path(tmp_path: Path, vrt_dataset: ET.Element) -> Path:
    p = Path(tmp_path, "tmp.vrt")
    # Create a tree
    tree = ET.ElementTree(vrt_dataset)
    # Format
    ET.indent(tree)
    # Write to file
    tree.write(p)
    assert p.is_file()
    return p


@pytest.fixture
def vrt_cache_dir(tmp_path: Path) -> Path:
    p = Path(tmp_path, "vrt_cache")
    p.mkdir(parents=True, exist_ok=True)
    assert p.is_dir()
    return p
