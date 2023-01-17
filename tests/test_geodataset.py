from hydromt.geodataset import GeoDataset

import pytest
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import Polygon

@pytest.fixture
def dummy_shp():
    geom = [
        Polygon(((0,0),(1,0),(1,1),(0,1),(0,0))),
        Polygon(((1,0),(2,0),(2,1),(1,1),(1,0))),
    ]
    attrs = [
        {"Roman": "I"},
        {"Roman": "II"},
    ]
    gdf = GeoDataFrame(data=attrs, geometry=geom, crs=CRS.from_epsg(4326))    
    return gdf

def test_vector(dummy_shp):
    gd = GeoDataset.from_gdf(dummy_shp)
    oc = GeoDataset.ogr_compliant(gd)
    pass
