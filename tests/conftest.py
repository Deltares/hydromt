import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box

from hydromt import Model, GridModel, LumpedModel, NetworkModel
from hydromt.models.model_api import AuxmapsModel
from hydromt import raster, vector, gis_utils
import pyflwdir


@pytest.fixture
def rioda():
    return raster.full_from_transform(
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
def df_time():
    df_time = pd.DataFrame(
        {
            "precip": [0, 1, 2, 3, 4],
            "temp": [15, 16, 17, 18, 19],
            "pet": [1, 2, 3, 4, 5],
        },
        index=pd.date_range(start="2007-01-01", end="2007-01-05", freq="D"),
    )
    return df_time


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
    da = vector.GeoDataArray.from_gdf(geodf, ts, name="test", dims=("index", "time"))
    return da


@pytest.fixture
def demda():
    np.random.seed(11)
    da = xr.DataArray(
        data=np.random.rand(15, 10),
        dims=("y", "x"),
        coords={"y": -np.arange(0, 1500, 100), "x": np.arange(0, 1000, 100)},
        attrs=dict(_FillValue=-9999),
    )
    da.raster.set_crs(3785)
    return da


@pytest.fixture
def flwdir(demda):
    # NOTE: single basin!
    return pyflwdir.from_dem(
        demda.values,
        nodata=demda.raster.nodata,
        outlets="min",
        transform=demda.raster.transform,
        latlon=demda.raster.crs.is_geographic,
    )


@pytest.fixture
def flwda(flwdir):
    xcoords, ycoords = gis_utils.affine_to_coords(flwdir.transform, flwdir.shape)
    da = xr.DataArray(
        name="flwdir",
        data=flwdir.to_array("d8"),
        dims=("y", "x"),
        coords={"y": ycoords, "x": xcoords},
        attrs=dict(_FillValue=247),
    )
    da.raster.set_crs(3785)
    return da


@pytest.fixture
def hydds(flwda, flwdir):
    ds = flwda.copy().to_dataset()
    ds["uparea"] = xr.DataArray(
        data=flwdir.upstream_area("cell"),
        dims=flwda.raster.dims,
        attrs=dict(_FillValue=-9999),
    )
    ds.raster.set_crs(flwda.raster.crs)
    return ds


@pytest.fixture
def obsda():
    rng = np.random.default_rng(12345)
    da = xr.DataArray(
        data=rng.random(size=365) * 100,
        dims=("time"),
        coords={"time": pd.date_range(start="2020-01-01", periods=365, freq="1D")},
        attrs=dict(_FillValue=-9999),
    )
    da.raster.set_crs(4326)
    return da


@pytest.fixture
def demuda():
    import xugrid as xu

    uds = xu.data.adh_san_diego()
    uda = uds["elevation"]
    uda.ugrid.grid.set_crs(epsg=2230)
    return uda


@pytest.fixture
def model(demda, world, obsda):
    mod = Model()
    mod.setup_region({"geom": demda.raster.box})
    mod.setup_config(**{"header": {"setting": "value"}})
    mod.set_staticmaps(demda, "elevtn")  # will be deprecated
    mod.set_geoms(world, "world")
    mod.set_forcing(obsda, "waterlevel")
    mod.set_states(demda, "zsini")
    mod.set_results(obsda, "zs")
    return mod


@pytest.fixture
def auxmap_model(demda):
    mod = AuxmapsModel()
    mod.setup_region({"geom": demda.raster.box})
    mod.setup_config(**{"header": {"setting": "value"}})
    mod.set_auxmaps(demda, "elevtn")
    return mod


@pytest.fixture
def grid_model(demda, flwda):
    mod = GridModel()
    mod.setup_region({"geom": demda.raster.box})
    mod.setup_config(**{"header": {"setting": "value"}})
    mod.set_grid(demda, "elevtn")
    mod.set_grid(flwda, "flwdir")
    return mod


@pytest.fixture
def lumped_model(ts, geodf):
    mod = LumpedModel()
    # mod.setup_region({"bbox": geodf.total_bounds})
    mod.setup_config(**{"header": {"setting": "value"}})
    da = xr.DataArray(
        ts,
        dims=["index", "time"],
        coords={"index": ts.index, "time": ts.columns},
        name="zs",
    )
    da = da.assign_coords(geometry=(["index"], geodf["geometry"]))
    da.vector.set_crs(geodf.crs)
    mod.set_response_units(da)
    return mod


@pytest.fixture
def network_model():
    mod = NetworkModel()
    # TODO set data and attributes of mod
    return mod


@pytest.fixture
def mesh_model(demuda):
    from hydromt import MeshModel

    mod = MeshModel()
    # region = gpd.GeoDataFrame(
    #     geometry=[box(*demuda.ugrid.grid.bounds)], crs=demuda.ugrid.crs
    # )
    # mod.setup_region({"geom": region})
    mod.setup_config(**{"header": {"setting": "value"}})
    mod.set_mesh(demuda, "elevtn")
    return mod
