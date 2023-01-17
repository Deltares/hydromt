from hydromt.geodataset import GeoDataset

import pytest
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import Polygon, MultiPolygon


@pytest.fixture
def dummy_shp():
    geom = [
        Polygon(((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))),
        MultiPolygon(
            [
                Polygon(((1, 0), (2, 0), (2, 1), (1, 1), (1, 0))),
                Polygon(((2, 0), (3, 0), (3, 1), (2, 1), (2, 0))),
            ]
        ),
    ]
    attrs = [
        {"Roman": "I"},
        {"Roman": "II"},
    ]
    gdf = GeoDataFrame(data=attrs, geometry=geom, crs=CRS.from_epsg(4326))
    return gdf


def test_vector(tmpdir, dummy_shp):
    path = str(tmpdir)

    # Create a geodataset and an ogr compliant version of it
    gd = GeoDataset.from_gdf(dummy_shp)
    oc = GeoDataset.ogr_compliant(gd)

    # Assert some ogr compliant stuff
    assert oc.ogr_layer_type == "MULTIPOLYGON"
    assert list(oc.dims)[0] == "record"
    assert len(oc.Roman) == 2

    # Write and load
    gd.geo.to_nc(f"{path}", fname="dummy_ogr")
    gd_nc = GeoDataset.from_nc(f"{path}\\dummy_ogr.nc")
