import os
from pathlib import Path

import geopandas as gpd
import pytest
from fsspec import AbstractFileSystem
from shapely.geometry import box

from hydromt.metadata_resolver.raster_tindex_resolver import RasterTindexResolver


@pytest.fixture()
def raster_tindex():
    raster_tindex_dict = {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "1",
                "type": "Feature",
                "properties": {
                    "location": "GRWL_mask_V01.01//NA18.tif",
                    "EPSG": "EPSG:32618",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-78.0081091502138, 4.000817142717678],
                            [-71.99333022015983, 4.000822441345486],
                            [-72.00061525372493, 0.0005420901698317547],
                            [-78.00082064189185, 0.0005420894530163727],
                            [-78.0081091502138, 4.000817142717678],
                        ]
                    ],
                },
            },
            {
                "id": "2",
                "type": "Feature",
                "properties": {
                    "location": "GRWL_mask_V01.01//NA19.tif",
                    "EPSG": "EPSG:32619",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-72.0081091502138, 4.000817142717678],
                            [-65.99333022015983, 4.000822441345486],
                            [-66.00061525372493, 0.0005420901698317547],
                            [-72.00082064189183, 0.0005420894530163727],
                            [-72.0081091502138, 4.000817142717678],
                        ]
                    ],
                },
            },
        ],
    }
    gdf = gpd.GeoDataFrame.from_features(raster_tindex_dict)
    gdf.set_crs(crs=4326, inplace=True)
    return gdf


def test_raster_tindex_resolver(raster_tindex, tmpdir):
    geom = raster_tindex.iloc[[0]]
    fp = os.path.join(tmpdir, "raster_tindex.gpkg")
    raster_tindex.to_file(fp)
    options = {"tileindex": "location"}
    resolver = RasterTindexResolver()
    paths = resolver.resolve(
        uri=fp, fs=AbstractFileSystem(), mask=geom, options=options
    )
    for file in raster_tindex["location"]:
        path = str(Path(os.path.join(tmpdir, file)))
        assert path in paths

    geom = gpd.GeoDataFrame(geometry=[box(-66, 1, 65, 3)], crs=4326)
    paths = resolver.resolve(
        uri=fp, fs=AbstractFileSystem(), mask=geom, options=options
    )
    assert len(paths) == 1
    path = str(Path(os.path.join(tmpdir, "GRWL_mask_V01.01\\NA19.tif")))
    assert path in paths

    options = {"tileindex": "file"}
    with pytest.raises(
        IOError,
        match='Tile index "file" column missing in tile index file.',
    ):
        resolver.resolve(uri=fp, fs=AbstractFileSystem(), mask=geom, options=options)

    geom = gpd.GeoDataFrame(geometry=[box(4, 52, 5, 53)], crs=4326)

    with pytest.raises(IOError, match="No intersecting tiles found."):
        resolver.resolve(uri=fp, fs=AbstractFileSystem(), mask=geom, options=options)
