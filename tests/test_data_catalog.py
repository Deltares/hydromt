"""Tests for the hydromt.data_catalog submodule."""

import os
from os.path import abspath, dirname, join
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr

from hydromt.data_adapter import DataAdapter, RasterDatasetAdapter
from hydromt.data_catalog import (
    DataCatalog,
    _denormalise_data_dict,
    _parse_data_source_dict,
)

CATALOGDIR = join(dirname(abspath(__file__)), "..", "data", "catalogs")
DATADIR = join(dirname(abspath(__file__)), "data")


def test_parser():
    # valid abs root on windows and linux!
    root = "c:/root" if os.name == "nt" else "/c/root"
    # simple; abs path
    source = {
        "data_type": "RasterDataset",
        "path": f"{root}/to/data.tif",
    }
    adapter = _parse_data_source_dict("test", source, root=root)
    assert isinstance(adapter, RasterDatasetAdapter)
    assert adapter.path == abspath(source["path"])
    # test with Path object
    source.update(path=Path(source["path"]))
    adapter = _parse_data_source_dict("test", source, root=root)
    assert adapter.path == abspath(source["path"])
    # rel path
    source = {
        "data_type": "RasterDataset",
        "path": "path/to/data.tif",
        "kwargs": {"fn": "test"},
    }
    adapter = _parse_data_source_dict("test", source, root=root)
    assert adapter.path == abspath(join(root, source["path"]))
    # check if path in kwargs is also absolute
    assert adapter.driver_kwargs["fn"] == abspath(join(root, "test"))
    # alias
    dd = {
        "test": {
            "data_type": "RasterDataset",
            "path": "path/to/data.tif",
        },
        "test1": {"alias": "test"},
    }
    with pytest.deprecated_call():
        sources = _denormalise_data_dict(dd)
    assert len(sources) == 2
    for name, source in sources:
        adapter = _parse_data_source_dict(name, source, root=root, catalog_name="tmp")
        assert adapter.path == abspath(join(root, dd["test"]["path"]))
        assert adapter.catalog_name == "tmp"
    # placeholder
    dd = {
        "test_{p1}_{p2}": {
            "data_type": "RasterDataset",
            "path": "data_{p2}.tif",
            "placeholders": {"p1": ["a", "b"], "p2": ["1", "2", "3"]},
        },
    }
    sources = _denormalise_data_dict(dd)
    assert len(sources) == 6
    for name, source in sources:
        assert "placeholders" not in source
        adapter = _parse_data_source_dict(name, source, root=root)
        assert adapter.path == abspath(join(root, f"data_{name[-1]}.tif"))
    # variants
    dd = {
        "test": {
            "data_type": "RasterDataset",
            "variants": [
                {"path": "path/to/data1.tif", "version": "1"},
                {"path": "path/to/data2.tif", "provider": "local"},
            ],
        },
    }
    sources = _denormalise_data_dict(dd)
    assert len(sources) == 2
    for i, (name, source) in enumerate(sources):
        assert "variants" not in source
        adapter = _parse_data_source_dict(name, source, root=root, catalog_name="tmp")
        assert adapter.version == dd["test"]["variants"][i].get("version", None)
        assert adapter.provider == dd["test"]["variants"][i].get("provider", None)
        assert adapter.catalog_name == "tmp"

    # errors
    with pytest.raises(ValueError, match="Missing required path argument"):
        _parse_data_source_dict("test", {})
    with pytest.raises(ValueError, match="Data type error unknown"):
        _parse_data_source_dict("test", {"path": "", "data_type": "error"})
    with pytest.raises(
        ValueError, match="alias test not found in data_dict"
    ), pytest.deprecated_call():
        _denormalise_data_dict({"test1": {"alias": "test"}})


def test_data_catalog_io(tmpdir):
    data_catalog = DataCatalog()
    data_catalog.sources  # load artifact data as fallback
    # read / write
    fn_yml = join(tmpdir, "test.yml")
    data_catalog.to_yml(fn_yml)
    data_catalog1 = DataCatalog(data_libs=fn_yml)
    assert data_catalog.to_dict() == data_catalog1.to_dict()
    # test that no file is written for empty DataCatalog
    fn_yml = join(tmpdir, "test1.yml")
    DataCatalog(fallback_lib=None).to_yml(fn_yml)
    # test print
    print(data_catalog.get_source("merit_hydro"))


def test_versioned_catalogs(tmpdir):
    # make sure the catalogs individually still work
    legacy_yml_fn = join(DATADIR, "legacy_esa_worldcover.yml")
    legacy_data_catalog = DataCatalog(data_libs=[legacy_yml_fn])
    assert len(legacy_data_catalog) == 1
    source = legacy_data_catalog.get_source("esa_worldcover")
    assert Path(source.path).name == "esa-worldcover.vrt"
    assert source.version == "2020"
    # test round trip to and from dict
    legacy_data_catalog2 = DataCatalog().from_dict(legacy_data_catalog.to_dict())
    assert legacy_data_catalog2 == legacy_data_catalog
    # make sure we raise deprecation warning here
    with pytest.deprecated_call():
        _ = legacy_data_catalog["esa_worldcover"]

    # second catalog
    aws_yml_fn = join(DATADIR, "aws_esa_worldcover.yml")
    aws_data_catalog = DataCatalog(data_libs=[aws_yml_fn])
    assert len(aws_data_catalog) == 1
    # test get_source with all keyword combinations
    source = aws_data_catalog.get_source("esa_worldcover")
    assert source.path.endswith("ESA_WorldCover_10m_2020_v100_Map_AWS.vrt")
    assert source.version == "2021"
    source = aws_data_catalog.get_source("esa_worldcover", version=2021)
    assert source.path.endswith("ESA_WorldCover_10m_2020_v100_Map_AWS.vrt")
    assert source.version == "2021"
    source = aws_data_catalog.get_source("esa_worldcover", version=2021, provider="aws")
    assert source.path.endswith("ESA_WorldCover_10m_2020_v100_Map_AWS.vrt")
    # test round trip to and from dict
    aws_data_catalog2 = DataCatalog().from_dict(aws_data_catalog.to_dict())
    assert aws_data_catalog2 == aws_data_catalog

    # test errors
    with pytest.raises(KeyError):
        aws_data_catalog.get_source("esa_worldcover", version=2021, provider="asdfasdf")
    with pytest.raises(KeyError):
        aws_data_catalog.get_source(
            "esa_worldcover", version="asdfasdf", provider="aws"
        )
    with pytest.raises(KeyError):
        aws_data_catalog.get_source("asdfasdf", version=2021, provider="aws")

    # make sure we trigger user warning when overwriting versions
    with pytest.warns(UserWarning):
        aws_data_catalog.from_yml(aws_yml_fn)

    # make sure we can read merged catalogs
    merged_yml_fn = join(DATADIR, "merged_esa_worldcover.yml")
    merged_catalog = DataCatalog(data_libs=[merged_yml_fn])
    assert len(merged_catalog) == 3
    source_aws = merged_catalog.get_source("esa_worldcover")  # last variant is default
    assert source_aws.filesystem == "s3"
    assert merged_catalog.get_source("esa_worldcover", provider="aws") == source_aws
    source_loc = merged_catalog.get_source("esa_worldcover", provider="local")
    assert source_loc != source_aws
    assert source_loc.filesystem == "local"
    assert source_loc.version == "2021"  # get newest version
    # test get_source with version only
    assert merged_catalog.get_source("esa_worldcover", version=2021) == source_loc
    # test round trip to and from dict
    merged_catalog2 = DataCatalog().from_dict(merged_catalog.to_dict())
    assert merged_catalog2 == merged_catalog

    # Make sure we can query for the version we want
    aws_and_legacy_catalog = DataCatalog(data_libs=[legacy_yml_fn, aws_yml_fn])
    assert len(aws_and_legacy_catalog) == 2
    source_aws = aws_and_legacy_catalog.get_source("esa_worldcover")
    assert source_aws.filesystem == "s3"
    source_aws2 = aws_and_legacy_catalog.get_source("esa_worldcover", provider="aws")
    assert source_aws2 == source_aws
    source_loc = aws_and_legacy_catalog.get_source(
        "esa_worldcover", provider="legacy_esa_worldcover"  # provider is filename
    )
    assert Path(source_loc.path).name == "esa-worldcover.vrt"
    # test round trip to and from dict
    aws_and_legacy_catalog2 = DataCatalog().from_dict(aws_and_legacy_catalog.to_dict())
    assert aws_and_legacy_catalog2 == aws_and_legacy_catalog


def test_data_catalog(tmpdir):
    data_catalog = DataCatalog(data_libs=None)
    # initialized with empty dict
    assert len(data_catalog._sources) == 0
    # global data sources from artifacts are automatically added
    assert len(data_catalog.sources) > 0
    # test keys, getitem,
    keys = [key for key, _ in data_catalog.iter_sources()]
    source = data_catalog.get_source(keys[0])
    assert isinstance(source, DataAdapter)
    assert keys[0] in data_catalog.get_source_names()
    # add source from dict
    data_dict = {keys[0]: source.to_dict()}
    data_catalog.from_dict(data_dict)
    assert isinstance(data_catalog.__repr__(), str)
    assert isinstance(data_catalog._repr_html_(), str)
    assert isinstance(data_catalog.to_dataframe(), pd.DataFrame)
    with pytest.raises(ValueError, match="Value must be DataAdapter"):
        data_catalog.add_source("test", "string")
    # check that no sources are loaded if fallback_lib is None
    assert not DataCatalog(fallback_lib=None).sources
    # test artifact keys (NOTE: legacy code!)
    with pytest.deprecated_call():
        data_catalog = DataCatalog(deltares_data=False)
    assert len(data_catalog._sources) == 0
    with pytest.deprecated_call():
        data_catalog.from_artifacts("deltares_data")
    assert len(data_catalog._sources) > 0
    with pytest.raises(
        IOError, match="URL b'404: Not Found'"
    ), pytest.deprecated_call():
        data_catalog = DataCatalog(deltares_data="unknown_version")

    # test hydromt version in meta data
    fn_yml = join(tmpdir, "test.yml")
    data_catalog = DataCatalog()
    data_catalog.to_yml(fn_yml, meta={"hydromt_version": "0.7.0"})


def test_from_archive(tmpdir):
    data_catalog = DataCatalog()
    data_catalog._cache_dir = str(
        tmpdir.join(".hydromt_data")
    )  # change cache to tmpdir
    urlpath = data_catalog.predefined_catalogs["artifact_data"]["urlpath"]
    version_hash = list(
        data_catalog.predefined_catalogs["artifact_data"]["versions"].values()
    )[0]
    data_catalog.from_archive(urlpath.format(version=version_hash))
    assert len(data_catalog.iter_sources()) > 0
    source0 = data_catalog.get_source(
        next(iter([source_name for source_name, _ in data_catalog.iter_sources()]))
    )
    assert ".hydromt_data" in str(source0.path)
    # failed to download
    with pytest.raises(ConnectionError, match="Data download failed"):
        data_catalog.from_archive("https://asdf.com/asdf.zip")


def test_from_predefined_catalogs():
    data_catalog = DataCatalog()
    data_catalog.set_predefined_catalogs(
        join(CATALOGDIR, "..", "predefined_catalogs.yml")
    )
    for name in data_catalog.predefined_catalogs:
        data_catalog.from_predefined_catalogs(f"{name}=latest")
        assert len(data_catalog._sources) > 0
        data_catalog._sources = {}  # reset
    with pytest.raises(ValueError, match='Catalog with name "asdf" not found'):
        data_catalog.from_predefined_catalogs("asdf")


def test_export_global_datasets(tmpdir):
    DTYPES = {
        "RasterDatasetAdapter": (xr.DataArray, xr.Dataset),
        "GeoDatasetAdapter": (xr.DataArray, xr.Dataset),
        "GeoDataFrameAdapter": gpd.GeoDataFrame,
    }
    bbox = [12.0, 46.0, 13.0, 46.5]  # Piava river
    time_tuple = ("2010-02-10", "2010-02-15")
    data_catalog = DataCatalog("artifact_data")  # read artifacts
    source_names = [
        "era5[precip,temp]",
        "grwl_mask",
        "modis_lai",
        "osm_coastlines",
        "grdc",
        "corine",
        "gtsmv3_eu_era5",
    ]
    data_catalog.export_data(
        tmpdir,
        bbox=bbox,
        time_tuple=time_tuple,
        source_names=source_names,
        meta={"version": 1},
    )
    # test append and overwrite source
    data_catalog.export_data(
        tmpdir,
        bbox=bbox,
        source_names=["corine"],
        append=True,
        meta={"version": 2},
    )
    data_lib_fn = join(tmpdir, "data_catalog.yml")
    # check if meta is written
    with open(data_lib_fn, "r") as f:
        yml_list = f.readlines()
    assert yml_list[0].strip() == "meta:"
    assert yml_list[1].strip() == "version: 2"
    assert yml_list[2].strip().startswith("root:")
    # check if data is parsed correctly
    data_catalog1 = DataCatalog(data_lib_fn)
    for key, source in data_catalog1.iter_sources():
        source_type = type(source).__name__
        dtypes = DTYPES[source_type]
        obj = source.get_data()
        assert isinstance(obj, dtypes), key


def test_export_dataframe(tmpdir, df, df_time):
    # Write two csv files
    fn_df = str(tmpdir.join("test.csv"))
    df.to_csv(fn_df)
    fn_df_time = str(tmpdir.join("test_ts.csv"))
    df_time.to_csv(fn_df_time)

    # Test to_file method (needs reading)
    data_dict = {
        "test_df": {
            "path": fn_df,
            "driver": "csv",
            "data_type": "DataFrame",
            "kwargs": {
                "index_col": 0,
            },
        },
        "test_df_ts": {
            "path": fn_df_time,
            "driver": "csv",
            "data_type": "DataFrame",
            "kwargs": {
                "index_col": 0,
                "parse_dates": True,
            },
        },
    }

    data_catalog = DataCatalog()
    data_catalog.from_dict(data_dict)

    data_catalog.export_data(
        str(tmpdir),
        time_tuple=("2010-02-01", "2010-02-14"),
        bbox=[11.70, 45.35, 12.95, 46.70],
    )
    data_catalog1 = DataCatalog(str(tmpdir.join("data_catalog.yml")))
    assert len(data_catalog1.iter_sources()) == 1

    data_catalog.export_data(str(tmpdir))
    data_catalog1 = DataCatalog(str(tmpdir.join("data_catalog.yml")))
    assert len(data_catalog1.iter_sources()) == 2
    for key, source in data_catalog1.iter_sources():
        dtypes = pd.DataFrame
        obj = source.get_data()
        assert isinstance(obj, dtypes), key


def test_get_data(df, tmpdir):
    data_catalog = DataCatalog("artifact_data")  # read artifacts
    n = len(data_catalog)
    # raster dataset using three different ways
    name = "koppen_geiger"
    da = data_catalog.get_rasterdataset(data_catalog.get_source(name).path)
    assert len(data_catalog) == n + 1
    assert isinstance(da, xr.DataArray)
    da = data_catalog.get_rasterdataset(name, provider="artifact_data")
    assert isinstance(da, xr.DataArray)
    da = data_catalog.get_rasterdataset(da)
    assert isinstance(da, xr.DataArray)
    data = {"source": name, "provider": "artifact_data"}
    da = data_catalog.get_rasterdataset(data)
    with pytest.raises(ValueError, match='Unknown raster data type "list"'):
        data_catalog.get_rasterdataset([])
    with pytest.raises(FileNotFoundError):
        data_catalog.get_rasterdataset("test1.tif")
    with pytest.raises(ValueError, match="Unknown keys in requested data"):
        data_catalog.get_rasterdataset({"name": "test"})

    # vector dataset using three different ways
    name = "osm_coastlines"
    gdf = data_catalog.get_geodataframe(data_catalog.get_source(name).path)
    assert len(data_catalog) == n + 2
    assert isinstance(gdf, gpd.GeoDataFrame)
    gdf = data_catalog.get_geodataframe(name, provider="artifact_data")
    assert isinstance(gdf, gpd.GeoDataFrame)
    gdf = data_catalog.get_geodataframe(gdf)
    assert isinstance(gdf, gpd.GeoDataFrame)
    data = {"source": name, "provider": "artifact_data"}
    gdf = data_catalog.get_geodataframe(data)
    assert isinstance(gdf, gpd.GeoDataFrame)
    with pytest.raises(ValueError, match='Unknown vector data type "list"'):
        data_catalog.get_geodataframe([])
    with pytest.raises(FileNotFoundError):
        data_catalog.get_geodataframe("test1.gpkg")
    with pytest.raises(ValueError, match="Unknown keys in requested data"):
        data_catalog.get_geodataframe({"name": "test"})

    # geodataset using three different ways
    name = "gtsmv3_eu_era5"
    da = data_catalog.get_geodataset(data_catalog.get_source(name).path)
    assert len(data_catalog) == n + 3
    assert isinstance(da, xr.DataArray)
    da = data_catalog.get_geodataset(name, provider="artifact_data")
    assert isinstance(da, xr.DataArray)
    da = data_catalog.get_geodataset(da)
    assert isinstance(da, xr.DataArray)
    data = {"source": name, "provider": "artifact_data"}
    gdf = data_catalog.get_geodataset(data)
    assert isinstance(gdf, xr.DataArray)
    with pytest.raises(ValueError, match='Unknown geo data type "list"'):
        data_catalog.get_geodataset([])
    with pytest.raises(FileNotFoundError):
        data_catalog.get_geodataset("test1.nc")
    with pytest.raises(ValueError, match="Unknown keys in requested data"):
        data_catalog.get_geodataset({"name": "test"})

    # dataframe using single way
    name = "test.csv"
    fn = str(tmpdir.join(name))
    df.to_csv(fn)
    df = data_catalog.get_dataframe(fn, driver_kwargs=dict(index_col=0))
    assert len(data_catalog) == n + 4
    assert isinstance(df, pd.DataFrame)
    df = data_catalog.get_dataframe(name, provider="local")
    assert isinstance(df, pd.DataFrame)
    df = data_catalog.get_dataframe(df)
    assert isinstance(df, pd.DataFrame)
    data = {"source": name, "provider": "local"}
    gdf = data_catalog.get_dataframe(data)
    assert isinstance(gdf, pd.DataFrame)
    with pytest.raises(ValueError, match='Unknown tabular data type "list"'):
        data_catalog.get_dataframe([])
    with pytest.raises(FileNotFoundError):
        data_catalog.get_dataframe("test1.csv")
    with pytest.raises(ValueError, match="Unknown keys in requested data"):
        data_catalog.get_dataframe({"name": "test"})


def test_deprecation_warnings(artifact_data):
    with pytest.deprecated_call():
        # should be DataCatalog(data_libs=['artifact_data=v0.0.6'])
        DataCatalog(artifact_data="v0.0.6")
    with pytest.deprecated_call():
        cat = DataCatalog()
        # should be cat.from_predefined_catalogs('artifact_data', 'v0.0.6')
        cat.from_artifacts("artifact_data", version="v0.0.6")
    with pytest.deprecated_call():
        fn = artifact_data["chelsa"].path
        # should be driver_kwargs=dict(chunks={'x': 100, 'y': 100})
        artifact_data.get_rasterdataset(fn, chunks={"x": 100, "y": 100})
    with pytest.deprecated_call():
        fn = artifact_data["gadm_level1"].path
        # should be driver_kwargs=dict(assert_gtype='Polygon')
        artifact_data.get_geodataframe(fn, assert_gtype="MultiPolygon")
    with pytest.deprecated_call():
        fn = artifact_data["grdc"].path
        # should be driver_kwargs=dict(index_col=0)
        artifact_data.get_dataframe(fn, index_col=0)
    with pytest.deprecated_call():
        fn = artifact_data["gtsmv3_eu_era5"].path
        # should be driver_kwargs=dict(chunks={'time': 100})
        artifact_data.get_geodataset(fn, chunks={"time": 100})
