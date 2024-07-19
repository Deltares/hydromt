from os.path import dirname, join
from pathlib import Path

import geopandas as gpd
import pytest
from fsspec import AbstractFileSystem
from shapely.geometry import box

from hydromt._typing import NoDataException, SourceMetadata
from hydromt.data_catalog.uri_resolvers.raster_tindex_resolver import (
    RasterTindexResolver,
)


class TestRasterTindexResolver:
    @pytest.fixture()
    def raster_tindex(self, tmpdir):
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
        fp = join(tmpdir, "raster_tindex.gpkg")
        gdf.to_file(fp)
        return fp

    def test_resolves_correctly(self, raster_tindex):
        geom = gpd.GeoDataFrame(geometry=[box(-78, 0.0005, -65, 4)], crs=4326)
        metadata = SourceMetadata()
        options = {"tileindex": "location"}
        resolver = RasterTindexResolver(filesystem=AbstractFileSystem())
        paths = resolver.resolve(
            uri=raster_tindex,
            metadata=metadata,
            mask=geom,
            options=options,
        )
        assert len(paths) == 2
        assert (
            str(Path(join(dirname(raster_tindex), "GRWL_mask_V01.01/NA19.tif")))
            in paths
        )
        assert (
            str(Path(join(dirname(raster_tindex), "GRWL_mask_V01.01/NA18.tif")))
            in paths
        )

        geom = gpd.GeoDataFrame(geometry=[box(-66, 1, 65, 3)], crs=4326)
        paths = resolver.resolve(
            uri=raster_tindex,
            metadata=metadata,
            mask=geom,
            options=options,
        )
        assert len(paths) == 1
        path = str(Path(join(dirname(raster_tindex), "GRWL_mask_V01.01/NA19.tif")))
        assert path in paths

    def test_raises_no_tileindex(self, raster_tindex):
        metadata = SourceMetadata()
        resolver = RasterTindexResolver(filesystem=AbstractFileSystem())
        geom = gpd.GeoDataFrame(geometry=[box(-78, 0.0005, -65, 4)], crs=4326)
        with pytest.raises(
            ValueError,
            match="RasterTindexResolver needs options specifying 'tileindex'",
        ):
            resolver.resolve(
                uri=raster_tindex,
                metadata=metadata,
                mask=geom,
                options={},
            )

    def test_raises_missing_tileindex(self, raster_tindex):
        resolver = RasterTindexResolver(filesystem=AbstractFileSystem())
        metadata = SourceMetadata()
        options = {"tileindex": "file"}
        geom = gpd.GeoDataFrame(geometry=[box(-78, 0.0005, -65, 4)], crs=4326)
        with pytest.raises(
            IOError,
            match='Tile index "file" column missing in tile index file.',
        ):
            resolver.resolve(
                uri=raster_tindex,
                metadata=metadata,
                mask=geom,
                options=options,
            )

    def test_raises_no_intersecting_files(self, raster_tindex):
        resolver = RasterTindexResolver(filesystem=AbstractFileSystem())
        metadata = SourceMetadata()
        options = {"tileindex": "file"}
        geom = gpd.GeoDataFrame(geometry=[box(4, 52, 5, 53)], crs=4326)
        with pytest.raises(NoDataException, match="found no intersecting tiles."):
            resolver.resolve(
                uri=raster_tindex,
                metadata=metadata,
                mask=geom,
                options=options,
            )
