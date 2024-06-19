import logging
from os import sep
from os.path import abspath, dirname, join
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, Dict, Generator, Optional, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pyflwdir
import pytest
import xarray as xr
import xugrid as xu
import zarr
from dask import config as dask_config
from pytest_mock import MockerFixture
from shapely.geometry import box

from hydromt import (
    Model,
    raster,
    vector,
)
from hydromt.gis.raster_utils import affine_to_coords
from hydromt.plugins import Plugins

dask_config.set(scheduler="single-threaded")

from hydromt._typing import SourceMetadata
from hydromt.data_catalog import DataCatalog
from hydromt.data_catalog.adapters.geodataframe import GeoDataFrameAdapter
from hydromt.data_catalog.adapters.geodataset import GeoDatasetAdapter
from hydromt.data_catalog.drivers import GeoDataFrameDriver, RasterDatasetDriver
from hydromt.data_catalog.drivers.geodataset.geodataset_driver import GeoDatasetDriver
from hydromt.data_catalog.uri_resolvers import MetaDataResolver
from hydromt.model.components.config import ConfigComponent
from hydromt.model.components.geoms import GeomsComponent
from hydromt.model.components.spatial import SpatialModelComponent
from hydromt.model.components.spatialdatasets import SpatialDatasetsComponent
from hydromt.model.components.vector import VectorComponent
from hydromt.model.root import ModelRoot

dask_config.set(scheduler="single-threaded")

DATADIR = join(dirname(abspath(__file__)), "data")
DC_PARAM_PATH = join(DATADIR, "parameters_data.yml")


@pytest.fixture()
def PLUGINS() -> Plugins:
    # make sure to start each test with a clean state
    return Plugins()


@pytest.fixture(autouse=True)
def _local_catalog_eps(monkeypatch, PLUGINS):
    """Set entrypoints to local predefined catalogs."""
    cat_root = Path(__file__).parent.parent / "data" / "catalogs"
    for name, cls in PLUGINS.catalog_plugins.items():
        monkeypatch.setattr(
            f"hydromt.data_catalog.predefined_catalog.{cls.__name__}.base_url",
            str(cat_root / name),
        )


@pytest.fixture()
def example_zarr_file(tmp_dir: Path) -> Path:
    tmp_path: Path = tmp_dir / "0s.zarr"
    store = zarr.DirectoryStore(tmp_path)
    root: zarr.Group = zarr.group(store=store, overwrite=True)
    zarray_var: zarr.Array = root.zeros(
        "variable", shape=(10, 10), chunks=(5, 5), dtype="int8"
    )
    zarray_var[0, 0] = 42  # trigger write
    zarray_var.attrs.update(
        {
            "_ARRAY_DIMENSIONS": ["x", "y"],
            "coordinates": "xc yc",
            "long_name": "Test Array",
            "type_preferred": "int8",
        }
    )
    # create symmetrical coords
    xy = np.linspace(0, 9, 10, dtype=np.dtypes.Int8DType)
    xcoords, ycoords = np.meshgrid(xy, xy)

    zarray_x: zarr.Array = root.array("xc", xcoords, chunks=(5, 5), dtype="int8")
    zarray_x.attrs["_ARRAY_DIMENSIONS"] = ["x", "y"]
    zarray_y: zarr.Array = root.array("yc", ycoords, chunks=(5, 5), dtype="int8")
    zarray_y.attrs["_ARRAY_DIMENSIONS"] = ["x", "y"]
    zarr.consolidate_metadata(store)
    store.close()
    return tmp_path


@pytest.fixture()
def data_catalog(_local_catalog_eps) -> DataCatalog:
    """DataCatalog instance that points to local predefined catalogs."""
    return DataCatalog("artifact_data=v0.0.8")


@pytest.fixture(scope="session")
def latest_dd_version_uri():
    cat_root = Path(__file__).parent.parent / "data" / "catalogs" / "deltares_data"
    versions = [d.name for d in cat_root.iterdir() if d.is_dir()]
    latest_version = sorted(versions)[-1]
    return cat_root / latest_version / "data_catalog.yml"


@pytest.fixture(scope="class")
def tmp_dir() -> Generator[Path, None, None]:
    with TemporaryDirectory() as tempdirname:
        yield Path(tempdirname)


@pytest.fixture(scope="session")
def root() -> str:
    return abspath(sep)


@pytest.fixture()
def test_model(tmpdir) -> Model:
    return Model(root=tmpdir)


@pytest.fixture()
def rioda():
    return raster.full_from_transform(
        transform=[0.5, 0.0, 3.0, 0.0, -0.5, -9.0],
        shape=(4, 6),
        nodata=-1,
        name="test",
        crs=4326,
    )


@pytest.fixture()
def rioda_large():
    da = raster.full_from_transform(
        transform=[0.004166666666666666, 0.0, 0.0, 0.0, -0.004166666666666667, 0.0],
        shape=(1024, 1000),
        nodata=-9999,
        name="test",
        crs=4326,
    )
    return da


@pytest.fixture(scope="class")
def df():
    df = (
        pd.DataFrame(
            {
                "city": ["Buenos Aires", "Brasilia", "Santiago", "Bogota", "Caracas"],
                "country": ["Argentina", "Brazil", "Chile", "Colombia", "Venezuela"],
                "latitude": [-34.58, -15.78, -33.45, 4.60, 10.48],
                "longitude": [-58.66, -47.91, -70.66, -74.08, -66.86],
            }
        )
        .reset_index(drop=False)
        .rename({"index": "id"}, axis=1)
    )
    return df


@pytest.fixture()
def timeseries_df():
    # Create a date range from 2020-01-01 to 2020-12-31
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")

    # Generate random values for three columns
    np.random.seed(42)  # for reproducibility
    col1 = np.random.randint(0, 100, len(dates))  # integers from 0 to 99
    col2 = np.random.normal(
        50, 10, len(dates)
    )  # normal distribution with mean 50 and standard deviation 10
    col3 = np.random.choice(
        ["A", "B", "C"], len(dates)
    )  # categorical values from A, B, or C

    # Create a dataframe with the dates as index and the columns as values
    df = pd.DataFrame({"col1": col1, "col2": col2, "col3": col3}, index=dates)
    df.index.rename("time", inplace=True)
    return df


@pytest.fixture()
def timeseries_ds(timeseries_df):
    return timeseries_df[["col1", "col2"]].to_xarray()


@pytest.fixture()
def dfs_segmented_by_points(df):
    return {
        id: pd.DataFrame(
            {
                "time": pd.date_range("2023-08-22", periods=len(df), freq="1D"),
                "test1": np.arange(len(df)) * id,
                "test2": np.arange(len(df)) ** id,
            }
        ).set_index("time")
        for id in range(len(df))
    }


@pytest.fixture()
def dfs_segmented_by_vars(dfs_segmented_by_points):
    data_vars = [
        v
        for v in pd.concat(dfs_segmented_by_points.values()).columns
        if v not in ["id", "time"]
    ]

    tmp = dfs_segmented_by_points.copy()
    for i, df in tmp.items():
        df.insert(0, "id", i)
        df.reset_index(inplace=True)

    return {
        v: pd.concat(tmp.values()).pivot(index="time", columns="id", values=v)
        for v in data_vars
    }


@pytest.fixture()
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


@pytest.fixture(scope="class")
def geodf(df):
    gdf = gpd.GeoDataFrame(
        data=df.copy().drop(columns=["longitude", "latitude"]),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=4326,
    )
    return gdf


@pytest.fixture(scope="session")
def world() -> gpd.GeoDataFrame:
    return gpd.read_file(Path(DATADIR) / "world.gpkg")


@pytest.fixture()
def ts(geodf):
    dates = pd.date_range("01-01-2000", "12-31-2000", name="time")
    ts = pd.DataFrame(
        index=geodf.index.values,
        columns=dates,
        data=np.random.rand(geodf.index.size, dates.size),
    )
    return ts


@pytest.fixture()
def geoda(geodf, ts):
    da = vector.GeoDataArray.from_gdf(geodf, ts, name="test", dims=("index", "time"))
    da.vector.set_nodata(np.nan)
    return da


@pytest.fixture()
def geods(geoda):
    return geoda.to_dataset()


@pytest.fixture()
def demda():
    np.random.seed(11)
    da = xr.DataArray(
        data=np.random.rand(15, 10),
        dims=("y", "x"),
        coords={"y": -np.arange(0, 1500, 100), "x": np.arange(0, 1000, 100)},
        attrs=dict(_FillValue=-9999),
    )
    # NOTE epsg 3785 is deprecated https://epsg.io/3785
    da.raster.set_crs(3857)
    return da


@pytest.fixture()
def lulcda():
    """Imitate VITO land use land cover data."""
    np.random.seed(11)
    da = xr.DataArray(
        # vito lulc classes
        data=np.random.choice([20, 30, 40, 110], size=(15, 10)),
        dims=("y", "x"),
        coords={"y": -np.arange(0, 1500, 100), "x": np.arange(0, 1000, 100)},
        attrs=dict(_FillValue=255),
    )
    # NOTE epsg 3785 is deprecated https://epsg.io/3785
    da.raster.set_crs(3857)
    return da


@pytest.fixture()
def flwdir(demda):
    # NOTE: single basin!
    return pyflwdir.from_dem(
        demda.values,
        nodata=demda.raster.nodata,
        outlets="min",
        transform=demda.raster.transform,
        latlon=demda.raster.crs.is_geographic,
    )


@pytest.fixture()
def flwda(flwdir):
    da = xr.DataArray(
        name="flwdir",
        data=flwdir.to_array("d8"),
        dims=("y", "x"),
        coords=affine_to_coords(flwdir.transform, flwdir.shape),
        attrs=dict(_FillValue=247),
    )
    # NOTE epsg 3785 is deprecated https://epsg.io/3785
    da.raster.set_crs(3875)
    return da


@pytest.fixture()
def hydds(flwda, flwdir):
    ds = flwda.copy().to_dataset()
    ds["uparea"] = xr.DataArray(
        data=flwdir.upstream_area("cell"),
        dims=flwda.raster.dims,
        attrs=dict(_FillValue=-9999),
    )
    ds.raster.set_crs(flwda.raster.crs)
    return ds


@pytest.fixture()
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


@pytest.fixture()
def raster_ds():
    temp = 15 + 8 * np.random.randn(2, 2, 3)
    precip = 10 * np.random.rand(2, 2, 3)
    lon = [[-99.83, -99.32], [-99.79, -99.23]]
    lat = [[42.25, 42.21], [42.63, 42.59]]
    return xr.Dataset(
        {
            "temperature": (["x", "y", "time"], temp),
            "precipitation": (["x", "y", "time"], precip),
        },
        coords={
            "lon": (["x", "y"], lon),
            "lat": (["x", "y"], lat),
            "time": pd.date_range("2014-09-06", periods=3),
            "reference_time": pd.Timestamp("2014-09-05"),
        },
    )


@pytest.fixture()
def ts_extremes():
    rng = np.random.default_rng(12345)
    normal = pd.DataFrame(
        rng.random(size=(365 * 100, 2)) * 100,
        index=pd.date_range(start="2020-01-01", periods=365 * 100, freq="1D"),
    )
    ext = rng.gumbel(loc=100, scale=25, size=(200, 2))  # Create extremes
    for i in range(2):
        normal.loc[normal.nlargest(200, i).index, i] = ext[:, i].reshape(-1)
    da = xr.DataArray(
        data=normal.values,
        dims=("time", "stations"),
        coords={
            "time": pd.date_range(start="1950-01-01", periods=365 * 100, freq="D"),
            "stations": [1, 2],
        },
        attrs=dict(_FillValue=-9999),
    )
    da.raster.set_crs(4326)
    return da


@pytest.fixture()
def griduda():
    bbox = [12.09, 46.49, 12.10, 46.50]  # Piava river
    data_catalog = DataCatalog(data_libs=["artifact_data"])
    da = data_catalog.get_rasterdataset("merit_hydro", bbox=bbox, variables="elevtn")
    gdf_da = da.raster.vector_grid()
    gdf_da["value"] = da.values.flatten()
    gdf_da.index.name = "mesh2d_nFaces"
    uda = xu.UgridDataset.from_geodataframe(gdf_da)
    uda = uda["value"]
    uda = uda.rename("elevtn")
    uda.ugrid.grid.set_crs(epsg=gdf_da.crs.to_epsg())
    return uda


@pytest.fixture()
def bbox():
    bbox = [12.05, 45.30, 12.85, 45.65]
    return gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)


@pytest.fixture()
def grid_model(demda, world, obsda, tmpdir):
    mod = Model(
        root=str(tmpdir),
        data_libs=["artifact_data", DC_PARAM_PATH],
        components={"grid": {"type": "GridComponent"}},
        region_component="grid",
    )

    geoms_component = GeomsComponent(mod)
    geoms_component.set(world, "world")
    mod.add_component("geoms", geoms_component)

    config_component = ConfigComponent(mod)
    config_component.set("header.setting", "value")
    mod.add_component("config", config_component)

    forcing_component = SpatialDatasetsComponent(mod, region_component="geoms")
    forcing_component.set(obsda, "waterlevel")
    mod.add_component("forcing", forcing_component)

    maps_component = SpatialDatasetsComponent(mod, region_component="geoms")
    maps_component.set(demda, "elevtn")
    mod.add_component("maps", maps_component)

    states_component = SpatialDatasetsComponent(mod, region_component="geoms")
    states_component.set(demda, "zsini")
    mod.add_component("states", states_component)

    return mod


@pytest.fixture()
def mesh_model(tmpdir):
    mesh_model = Model(
        root=str(tmpdir),
        data_libs=["artifact_data", DC_PARAM_PATH],
        components={"mesh": {"type": "MeshComponent"}},
        region_component="mesh",
    )

    return mesh_model


def _create_vector_model(
    *,
    ts,
    geodf,
    mocker: MockerFixture,
) -> Model:
    region_component = mocker.Mock(spec_set=SpatialModelComponent)
    region_component.test_equal.return_value = (True, {})
    region_component.region = geodf
    components: Dict[str, Any] = {
        "area": region_component,
        "vector": {"type": VectorComponent.__name__, "region_component": "area"},
        "config": {"type": ConfigComponent.__name__},
    }

    mod = Model(components=components, region_component="area")
    cast(ConfigComponent, mod.config).set("header.setting", "value")

    # fill VectorComponent
    da = xr.DataArray(
        ts,
        dims=["index", "time"],
        coords={"index": ts.index, "time": ts.columns},
        name="zs",
    )
    da = da.assign_coords(geometry=(["index"], geodf["geometry"]))
    da.vector.set_crs(geodf.crs)
    cast(VectorComponent, mod.vector).set(da)
    return mod


@pytest.fixture()
def vector_model(ts, geodf, mocker: MockerFixture):
    return _create_vector_model(ts=ts, geodf=geodf, mocker=mocker)


@pytest.fixture()
def vector_model_no_defaults(ts, geodf, mocker: MockerFixture):
    return _create_vector_model(
        ts=ts,
        geodf=geodf,
        mocker=mocker,
    )


@pytest.fixture()
def mock_resolver() -> MetaDataResolver:
    class MockMetaDataResolver(MetaDataResolver):
        name = "mock_resolver"

        def resolve(self, uri, *args, **kwargs):
            return [uri]

    resolver = MockMetaDataResolver()
    return resolver


@pytest.fixture()
def mock_geodataframe_adapter():
    class MockGeoDataFrameAdapter(GeoDataFrameAdapter):
        def transform(
            self, gdf: gpd.GeoDataFrame, metadata: SourceMetadata, **kwargs
        ) -> Optional[gpd.GeoDataFrame]:
            return gdf

    return MockGeoDataFrameAdapter()


@pytest.fixture()
def mock_geo_ds_adapter():
    class MockGeoDatasetAdapter(GeoDatasetAdapter):
        def transform(self, ds, metadata: SourceMetadata, **kwargs):
            return ds

    return MockGeoDatasetAdapter()


@pytest.fixture()
def mock_geodf_driver(
    geodf: gpd.GeoDataFrame, mock_resolver: MetaDataResolver
) -> GeoDataFrameDriver:
    class MockGeoDataFrameDriver(GeoDataFrameDriver):
        name = "mock_geodf_driver"

        def read_data(self, *args, **kwargs) -> gpd.GeoDataFrame:
            return geodf

    return MockGeoDataFrameDriver(metadata_resolver=mock_resolver)


@pytest.fixture()
def mock_raster_ds_driver(
    raster_ds: xr.Dataset, mock_resolver: MetaDataResolver
) -> RasterDatasetDriver:
    class MockRasterDatasetDriver(RasterDatasetDriver):
        name = "mock_raster_ds_driver"
        supports_writing: ClassVar[bool] = True

        def read_data(self, *args, **kwargs) -> xr.Dataset:
            return raster_ds

    return MockRasterDatasetDriver(metadata_resolver=mock_resolver)


@pytest.fixture()
def mock_geo_ds_driver(
    geoda: xr.DataArray, mock_resolver: MetaDataResolver
) -> GeoDatasetDriver:
    class MockGeoDatasetDriver(GeoDatasetDriver):
        name = "mock_geo_ds_driver"
        supports_writing: ClassVar[bool] = True

        def read_data(self, *args, **kwargs) -> xr.Dataset:
            return geoda.to_dataset()

    return MockGeoDatasetDriver(metadata_resolver=mock_resolver)


@pytest.fixture()
def artifact_data():
    datacatalog = DataCatalog()
    datacatalog.from_predefined_catalogs("artifact_data")
    return datacatalog


@pytest.fixture()
def mock_model(tmpdir, mocker: MockerFixture):
    logger = logging.getLogger(__name__)
    logger.propagate = True
    model = mocker.create_autospec(Model)
    model.root = mocker.create_autospec(ModelRoot(tmpdir), instance=True)
    model.root.path.return_value = tmpdir
    model.data_catalog = mocker.create_autospec(DataCatalog)
    model.logger = logger
    return model


@pytest.fixture()
def basin_files():
    data_catalog = DataCatalog("artifact_data")
    ds = data_catalog.get_rasterdataset("merit_hydro_1k")
    gdf_bas_index = data_catalog.get_geodataframe("merit_hydro_index")
    bas_index = data_catalog.get_source("merit_hydro_index")

    return (data_catalog, ds, gdf_bas_index, bas_index)
