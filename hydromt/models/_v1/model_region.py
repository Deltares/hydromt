"""The new model region class"""

import geopandas as gpd
from geopandas.geodataframe import CRS


class ModelRegion:
    def __init__(self, data: gpd.GeoDataFrame):
        self.set(data)

    def read(self):
        pass

    def write(self):
        pass

    def set(self, data: gpd.GeoDataFrame):
        self.data: gpd.GeoDataFrame = data
        self.crs: CRS = data.crs
