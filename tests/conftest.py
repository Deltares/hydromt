import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
import os
import glob
from os.path import join
from pathlib import Path
from rasterio.crs import CRS
from rasterio.transform import Affine, xy, rowcol
from shapely.geometry import box

from hydromt import rio, geo


@pytest.fixture
def rioda():
    return rio.full_from_transform(
        transform=[0.5, 0.0, 3.0, 0.0, -0.5, -9.0],
        shape=(4, 6),
        nodata=-1,
        name="test",
        crs=4326,
    )


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            "city": ["Buenos Aires", "Brasilia", "Santiago", "Bogota", "Caracas"],
            "country": ["Argentina", "Brazil", "Chile", "Colombia", "Venezuela"],
            "latitude": [-34.58, -15.78, -33.45, 4.60, 10.48],
            "longitude": [-58.66, -47.91, -70.66, -74.08, -66.86],
        }
    )
    return df


@pytest.fixture
def geodf(df):
    gdf = gpd.GeoDataFrame(
        data=df.drop(columns=["longitude", "latitude"]),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=4326,
    )
    return gdf


@pytest.fixture
def world():
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    return world


@pytest.fixture
def ts(geodf):
    dates = pd.date_range("01-01-2000", "12-31-2000", name="time")
    ts = pd.DataFrame(
        index=geodf.index,
        columns=dates,
        data=np.random.rand(geodf.index.size, dates.size),
    )
    return ts


@pytest.fixture
def geoda(geodf, ts):
    da = geo.GeoDataArray.from_gdf(geodf, ts, name="test", dims=("index", "time"))
    return da
