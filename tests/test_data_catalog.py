"""Tests for the hydromt.data_catalog submodule."""

import os
from os import mkdir
from os.path import abspath, dirname, join
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import requests
import xarray as xr
from yaml import dump

from hydromt._typing import NoDataStrategy
from hydromt.data_adapter import (
    DataAdapter,
    GeoDataFrameAdapter,
    GeoDatasetAdapter,
    RasterDatasetAdapter,
)
from hydromt.data_catalog import (
    DataCatalog,
    _denormalise_data_dict,
    _parse_data_source_dict,
    _yml_from_uri_or_path,
)
from hydromt.data_source import GeoDataFrameSource
from hydromt.gis.utils import to_geographic_bbox

CATALOGDIR = join(dirname(abspath(__file__)), "..", "data", "catalogs")
DATADIR = join(dirname(abspath(__file__)), "data")


def test_errors_on_no_root_found(tmpdir):
    d = {
        "meta": {
            "hydromt_version": ">=1.0a,<2",
            "roots": list(
                map(lambda p: join(tmpdir, p), ["a", "b", "c", "d", "4", "⽀"])
            ),
        },
    }
    with pytest.raises(ValueError, match="None of the specified roots were found"):
        _ = DataCatalog().from_dict(d)


def test_finds_later_roots(tmpdir):
    mkdir(join(tmpdir, "asasdfasdf"))
    d = {
        "meta": {
            "hydromt_version": ">=1.0a,<2",
            "roots": list(
                map(lambda p: join(tmpdir, p), ["a", "b", "c", "d", "4", "asasdfasdf"])
            ),
        },
    }
    cat = DataCatalog().from_dict(d)
    assert cat.root == Path(join(tmpdir, "asasdfasdf"))


def test_finds_roots_in_correct_order(tmpdir):
    paths = list(map(lambda p: join(tmpdir, p), ["a", "b", "c", "d", "4"]))
    for p in paths:
        mkdir(p)

    d = {
        "meta": {"hydromt_version": ">=1.0a,<2", "roots": paths},
    }
    cat = DataCatalog().from_dict(d)
    assert cat.root == Path(join(tmpdir, "a"))


def test_from_yml_no_root(tmpdir):
    d = {
        "meta": {"hydromt_version": ">=1.0a,<2"},
    }

    cat_file = join(tmpdir, "cat.yml")
    with open(cat_file, "w") as f:
        dump(d, f)

    cat = DataCatalog().from_yml(cat_file)
    assert cat.root == str(tmpdir)


def test_parser():
    # valid abs root on windows and linux!
    root = "c:/root" if os.name == "nt" else "/c/root"
    # simple; abs path
    source = {
        "driver": {"name": "pyogrio"},
        "data_type": "GeoDataFrame",
        "uri": f"{root}/to/data.gpkg",
    }
    datasource = _parse_data_source_dict("test", source, root=root)
    assert isinstance(datasource, GeoDataFrameSource)
    assert datasource.uri == abspath(source["uri"])
    # TODO: do we want to allow Path objects?
    # # test with Path object
    # source.update(uri=Path(source["uri"]))
    # datasource = _parse_data_source_dict("test", source, root=root)
    # assert datasource.uri == abspath(source["uri"])
    # rel path
    # source = {
    #     "data_adapter": {"name": "GeoDataFrame"},
    #     "driver": {"name": "pyogrio"},
    #     "data_type": "GeoDataFrame",
    #     "uri": "path/to/data.gpkg",
    #     "kwargs": {"fn": "test"},
    # }
    # datasource = _parse_data_source_dict("test", source, root=root)
    # assert datasource.uri == abspath(join(root, source["uri"]))
    # check if path in kwargs is also absolute
    # assert datasource.driver_kwargs["fn"] == abspath(join(root, "test"))
    # alias
    dd = {
        "test": {
            "driver": {"name": "pyogrio"},
            "data_type": "GeoDataFrame",
            "uri": "path/to/data.gpkg",
        },
        "test1": {"alias": "test"},
    }
    with pytest.deprecated_call():
        sources = _denormalise_data_dict(dd)
    assert len(sources) == 2
    for name, source in sources:
        datasource = _parse_data_source_dict(
            name,
            source,
            root=root,  # TODO: do we need catalog_name="tmp"
        )
        assert datasource.uri == abspath(join(root, dd["test"]["uri"]))
    # placeholder
    dd = {
        "test_{p1}_{p2}": {
            "driver": {"name": "pyogrio"},
            "data_type": "GeoDataFrame",
            "uri": "data_{p2}.gpkg",
            "placeholders": {"p1": ["a", "b"], "p2": ["1", "2", "3"]},
        },
    }
    sources = _denormalise_data_dict(dd)
    assert len(sources) == 6
    for name, source in sources:
        assert "placeholders" not in source
        datasource = _parse_data_source_dict(name, source, root=root)
        assert datasource.uri == abspath(join(root, f"data_{name[-1]}.gpkg"))
    # variants
    dd = {
        "test": {
            "driver": {"name": "pyogrio"},
            "data_type": "GeoDataFrame",
            "variants": [
                {"uri": "path/to/data1.gpkg", "version": "1"},
                {"uri": "path/to/data2.gpkg", "provider": "local"},
            ],
        },
    }
    sources = _denormalise_data_dict(dd)
    assert len(sources) == 2
    for i, (name, source) in enumerate(sources):
        assert "variants" not in source
        datasource = _parse_data_source_dict(
            name,
            source,
            root=root,  # TODO: do we need catalog_name="tmp"
        )
        assert datasource.version == dd["test"]["variants"][i].get("version", None)
        assert datasource.provider == dd["test"]["variants"][i].get("provider", None)
        # assert adapter.catalog_name == "tmp"

    # errors
    with pytest.raises(ValueError, match="DataSource needs 'data_type'"):
        _parse_data_source_dict("test", {})
    with pytest.raises(ValueError, match="Unknown 'data_type'"):
        _parse_data_source_dict("test", {"path": "", "data_type": "error"})
    with (
        pytest.raises(ValueError, match="alias test not found in data_dict"),
        pytest.deprecated_call(),
    ):
        _denormalise_data_dict({"test1": {"alias": "test"}})


@pytest.mark.skip("needs catalogs refactor")
def test_data_catalog_io(tmpdir, data_catalog):
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


@pytest.mark.skip("needs catalogs refactor")
def test_versioned_catalog_entries():
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
    source = aws_data_catalog.get_source("esa_worldcover", version="2021")
    assert source.path.endswith("ESA_WorldCover_10m_2020_v100_Map_AWS.vrt")
    assert source.version == "2021"
    source = aws_data_catalog.get_source(
        "esa_worldcover", version="2021", provider="aws"
    )
    assert source.path.endswith("ESA_WorldCover_10m_2020_v100_Map_AWS.vrt")
    # test round trip to and from dict
    aws_data_catalog2 = DataCatalog().from_dict(aws_data_catalog.to_dict())
    assert aws_data_catalog2 == aws_data_catalog

    # test errors
    with pytest.raises(KeyError):
        aws_data_catalog.get_source(
            "esa_worldcover", version="2021", provider="asdfasdf"
        )
    with pytest.raises(KeyError):
        aws_data_catalog.get_source(
            "esa_worldcover", version="asdfasdf", provider="aws"
        )
    with pytest.raises(KeyError):
        aws_data_catalog.get_source("asdfasdf", version="2021", provider="aws")

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
    assert merged_catalog.get_source("esa_worldcover", version="2021") == source_loc
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
        "esa_worldcover",
        provider="legacy_esa_worldcover",  # provider is filename
    )
    assert Path(source_loc.path).name == "esa-worldcover.vrt"
    # test round trip to and from dict
    aws_and_legacy_catalog2 = DataCatalog().from_dict(aws_and_legacy_catalog.to_dict())
    assert aws_and_legacy_catalog2 == aws_and_legacy_catalog


@pytest.mark.skip("needs catalogs refactor")
def test_versioned_catalogs(data_catalog):
    data_catalog._sources = {}  # reset
    data_catalog.from_predefined_catalogs("deltares_data")
    assert len(data_catalog.sources) > 0
    data_catalog._sources = {}  # reset
    data_catalog.from_predefined_catalogs("deltares_data", "v0.5.0")
    assert len(data_catalog.sources) > 0

    with pytest.raises(ValueError, match="Version v1993.7 not found "):
        _ = data_catalog.from_predefined_catalogs("deltares_data", "v1993.7")


@pytest.mark.skip("needs catalogs refactor")
def test_data_catalog(tmpdir, data_catalog):
    # initialized with empty dict
    data_catalog._sources = {}  # reset
    assert len(data_catalog._sources) == 0
    # global data sources from artifacts are automatically added
    assert len(data_catalog.sources) > 0
    # test keys, getitem,
    keys = [key for key, _ in data_catalog.iter_sources()]
    source = data_catalog.get_source(keys[0])
    assert data_catalog.contains_source(keys[0])
    assert data_catalog.contains_source(
        keys[0], version="asdfasdfasdf", permissive=True
    )
    assert not data_catalog.contains_source(
        keys[0], version="asdfasdf", permissive=False
    )
    assert isinstance(source, DataAdapter)
    assert keys[0] in data_catalog.get_source_names()
    # add source from dict
    data_dict = {keys[0]: source.to_dict()}
    data_catalog.from_dict(data_dict)
    assert isinstance(data_catalog.__repr__(), str)
    assert isinstance(data_catalog._repr_html_(), str)
    assert isinstance(data_catalog.to_dataframe(), pd.DataFrame)
    with pytest.raises(ValueError, match="Value must be DataAdapter"):
        data_catalog.add_source("test", "string")  # type: ignore
    # check that no sources are loaded if fallback_lib is None
    assert not DataCatalog(fallback_lib=None).sources
    # test artifact keys (NOTE: legacy code!)
    with pytest.deprecated_call():
        data_catalog = DataCatalog(deltares_data=False)
    assert len(data_catalog._sources) == 0
    data_catalog.from_predefined_catalogs("deltares_data")
    assert len(data_catalog._sources) > 0
    with (
        pytest.raises(ValueError, match="Version unknown_version not found"),
        pytest.deprecated_call(),
    ):
        data_catalog = DataCatalog(deltares_data="unknown_version")

    # test hydromt version in meta data
    fn_yml = join(tmpdir, "test.yml")
    data_catalog = DataCatalog()
    data_catalog.to_yml(fn_yml, meta={"hydromt_version": "0.7.0"})


@pytest.mark.skip("needs catalogs refactor")
def test_used_sources(tmpdir):
    merged_yml_fn = join(DATADIR, "merged_esa_worldcover.yml")
    data_catalog = DataCatalog(merged_yml_fn)
    source = data_catalog.get_source("esa_worldcover")
    source.mark_as_used()
    sources = data_catalog.list_sources(used_only=True)
    assert len(data_catalog) > 1
    assert len(sources) == 1
    assert sources[0][0] == "esa_worldcover"
    assert sources[0][1].provider == source.provider
    assert sources[0][1].version == source.version


@pytest.mark.skip("needs catalogs refactor")
def test_from_yml_with_archive(data_catalog):
    cache_dir = Path(data_catalog._cache_dir)
    data_catalog.from_predefined_catalogs("artifact_data=v0.0.8")
    sources = list(data_catalog.sources.keys())
    assert len(sources) > 0
    # as part of the getting the archive a a local
    # catalog file is written to the same folder
    # check if this file exists and we can read it
    yml_dst_fn = Path(cache_dir, "artifact_data", "v0.0.8", "data_catalog.yml")
    assert yml_dst_fn.exists()
    data_catalog1 = DataCatalog(yml_dst_fn)
    sources = list(data_catalog1.sources.keys())
    source = data_catalog1.get_source(sources[0])
    assert yml_dst_fn.parent == Path(source.path).parent.parent


@pytest.mark.skip("needs catalogs refactor")
def test_from_predefined_catalogs(data_catalog):
    assert len(data_catalog.predefined_catalogs) > 0
    for name in data_catalog.predefined_catalogs:
        data_catalog._sources = {}  # reset
        data_catalog.from_predefined_catalogs(f"{name}=latest")
        assert len(data_catalog._sources) > 0
    with pytest.raises(ValueError, match='Catalog with name "asdf" not found'):
        data_catalog.from_predefined_catalogs("asdf")


@pytest.mark.skip("needs catalogs refactor")
def test_export_global_datasets(tmpdir, data_catalog):
    DTYPES = {
        "RasterDatasetAdapter": (xr.DataArray, xr.Dataset),
        "GeoDatasetAdapter": (xr.DataArray, xr.Dataset),
        "GeoDataFrameAdapter": gpd.GeoDataFrame,
    }
    bbox = [12.0, 46.0, 13.0, 46.5]  # Piava river
    time_tuple = ("2010-02-10", "2010-02-15")
    data_catalog.from_predefined_catalogs("artifact_data")
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
        handle_nodata=NoDataStrategy.IGNORE,
    )
    # test append and overwrite source
    data_catalog.export_data(
        tmpdir,
        bbox=bbox,
        source_names=["corine"],
        append=True,
        meta={"version": 2},
        handle_nodata=NoDataStrategy.IGNORE,
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
    for key, source in data_catalog1.list_sources():
        source_type = type(source).__name__
        dtypes = DTYPES[source_type]
        obj = source.get_data()
        assert isinstance(obj, dtypes), key


@pytest.mark.skip(reason="needs implementation of all data types.")
def test_export_dataframe(tmpdir, df, df_time):
    # Write two csv files
    fn_df = str(tmpdir.join("test.csv"))
    fn_df_parquet = str(tmpdir.join("test.parquet"))
    df.to_csv(fn_df)
    df.to_parquet(fn_df_parquet)
    fn_df_time = str(tmpdir.join("test_ts.csv"))
    fn_df_time_parquet = str(tmpdir.join("test_ts.parquet"))
    df_time.to_csv(fn_df_time)
    df_time.to_parquet(fn_df_time_parquet)

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
        "test_df_parquet": {
            "path": fn_df_parquet,
            "driver": "parquet",
            "data_type": "DataFrame",
        },
        "test_df_ts_parquet": {
            "path": fn_df_time_parquet,
            "driver": "parquet",
            "data_type": "DataFrame",
        },
    }

    data_catalog = DataCatalog()
    data_catalog.from_dict(data_dict)

    data_catalog.export_data(
        str(tmpdir),
        time_tuple=("2010-02-01", "2010-02-14"),
        bbox=[11.70, 45.35, 12.95, 46.70],
        handle_nodata=NoDataStrategy.IGNORE,
    )
    data_catalog1 = DataCatalog(str(tmpdir.join("data_catalog.yml")))
    assert len(data_catalog1.list_sources()) == 2

    data_catalog.export_data(str(tmpdir))
    data_catalog1 = DataCatalog(str(tmpdir.join("data_catalog.yml")))
    assert len(data_catalog1.list_sources()) == 4
    for key, source in data_catalog1.list_sources():
        dtypes = pd.DataFrame
        obj = source.get_data()
        assert isinstance(obj, dtypes), key


@pytest.mark.skip("needs catalogs refactor")
def test_get_rasterdataset(data_catalog):
    n = len(data_catalog)
    # raster dataset using three different ways
    name = "koppen_geiger"
    da = data_catalog.get_rasterdataset(data_catalog.get_source(name).path)
    assert len(data_catalog) == n + 1
    assert isinstance(da, xr.DataArray)
    da = data_catalog.get_rasterdataset(name, provider="artifact_data")
    assert isinstance(da, xr.DataArray)
    bbox = [12.0, 46.0, 13.0, 46.5]
    da = data_catalog.get_rasterdataset(da, bbox=bbox)
    assert isinstance(da, xr.DataArray)
    assert np.allclose(da.raster.bounds, bbox)
    data = {"source": name, "provider": "artifact_data"}
    ds = data_catalog.get_rasterdataset(data, single_var_as_array=False)
    assert isinstance(ds, xr.Dataset)
    data = r"s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N29_00_E105_00_DEM/Copernicus_DSM_COG_10_N29_00_E105_00_DEM.tif"
    da = data_catalog.get_rasterdataset(data)
    assert isinstance(da, xr.DataArray)
    assert len(data_catalog) == n + 2
    with pytest.raises(ValueError, match='Unknown raster data type "list"'):
        data_catalog.get_rasterdataset([])
    with pytest.raises(FileNotFoundError):
        data_catalog.get_rasterdataset("test1.tif")
    with pytest.raises(ValueError, match="Unknown keys in requested data"):
        data_catalog.get_rasterdataset({"name": "test"})


@pytest.mark.skip("needs catalogs refactor")
def test_get_geodataframe(data_catalog):
    n = len(data_catalog)
    # vector dataset using three different ways
    name = "osm_coastlines"
    gdf = data_catalog.get_geodataframe(data_catalog.get_source(name).path)
    assert len(data_catalog) == n + 1
    assert isinstance(gdf, gpd.GeoDataFrame)
    gdf = data_catalog.get_geodataframe(name, provider="artifact_data")
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.index.size == 2
    gdf = data_catalog.get_geodataframe(gdf, geom=gdf.iloc[[0],], predicate="within")
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.index.size == 1
    data = {"source": name, "provider": "artifact_data"}
    gdf = data_catalog.get_geodataframe(data)
    assert isinstance(gdf, gpd.GeoDataFrame)
    with pytest.raises(ValueError, match='Unknown vector data type "list"'):
        data_catalog.get_geodataframe([])
    with pytest.raises(FileNotFoundError):
        data_catalog.get_geodataframe("test1.gpkg")
    with pytest.raises(ValueError, match="Unknown keys in requested data"):
        data_catalog.get_geodataframe({"name": "test"})


@pytest.mark.skip("needs catalogs refactor")
def test_get_geodataset(data_catalog):
    n = len(data_catalog)
    # geodataset using three different ways
    name = "gtsmv3_eu_era5"
    da = data_catalog.get_geodataset(data_catalog.get_source(name).path)
    assert len(data_catalog) == n + 1
    assert isinstance(da, xr.DataArray)
    da = data_catalog.get_geodataset(name, provider="artifact_data")
    assert da.vector.index.size == 19
    assert isinstance(da, xr.DataArray)
    bbox = [12.22412, 45.25635, 12.25342, 45.271]
    da = data_catalog.get_geodataset(
        da, bbox=bbox, time_tuple=("2010-02-01", "2010-02-05")
    )
    assert da.vector.index.size == 2
    assert da.time.size == 720
    assert isinstance(da, xr.DataArray)
    data = {"source": name, "provider": "artifact_data"}
    ds = data_catalog.get_geodataset(data, single_var_as_array=False)
    assert isinstance(ds, xr.Dataset)
    with pytest.raises(ValueError, match='Unknown geo data type "list"'):
        data_catalog.get_geodataset([])
    with pytest.raises(FileNotFoundError):
        data_catalog.get_geodataset("test1.nc")
    with pytest.raises(ValueError, match="Unknown keys in requested data"):
        data_catalog.get_geodataset({"name": "test"})


@pytest.mark.skip("needs catalogs refactor")
def test_get_dataset(timeseries_df, data_catalog):
    # get_dataset
    test_dataset = timeseries_df.to_xarray()
    subset_timeseries = timeseries_df.iloc[[0, len(timeseries_df) // 2]]
    time_tuple = (
        subset_timeseries.index[0].to_pydatetime(),
        subset_timeseries.index[1].to_pydatetime(),
    )
    ds = data_catalog.get_dataset(test_dataset, time_tuple=time_tuple)
    assert isinstance(ds, xr.Dataset)
    assert ds.time[-1].values == subset_timeseries.index[1].to_datetime64()

    ds = data_catalog.get_dataset(test_dataset, variables=["col1"])
    assert isinstance(ds, xr.DataArray)
    assert ds.name == "col1"


@pytest.mark.skip("needs catalogs refactor")
def test_get_dataframe(df, tmpdir, data_catalog):
    n = len(data_catalog)
    # dataframe using single way
    name = "test.csv"
    fn = str(tmpdir.join(name))
    df.to_csv(fn)
    df = data_catalog.get_dataframe(fn, driver_kwargs=dict(index_col=0))
    assert len(data_catalog) == n + 1
    assert isinstance(df, pd.DataFrame)
    df = data_catalog.get_dataframe(name, provider="user")
    assert isinstance(df, pd.DataFrame)
    df = data_catalog.get_dataframe(df, variables=["city"])
    assert isinstance(df, pd.DataFrame)
    assert df.columns == ["city"]
    data = {"source": name, "provider": "user"}
    gdf = data_catalog.get_dataframe(data)
    assert isinstance(gdf, pd.DataFrame)
    with pytest.raises(ValueError, match='Unknown tabular data type "list"'):
        data_catalog.get_dataframe([])
    with pytest.raises(FileNotFoundError):
        data_catalog.get_dataframe("test1.csv")
    with pytest.raises(ValueError, match="Unknown keys in requested data"):
        data_catalog.get_dataframe({"name": "test"})


@pytest.mark.skip("needs catalogs refactor")
def test_deprecation_warnings(data_catalog):
    with pytest.deprecated_call():
        # should be DataCatalog(data_libs=['artifact_data=v0.0.6'])
        DataCatalog(artifact_data="v0.0.8")
    with pytest.deprecated_call():
        fn = data_catalog["chelsa"].path
        # should be driver_kwargs=dict(chunks={'x': 100, 'y': 100})
        data_catalog.get_rasterdataset(fn, chunks={"x": 100, "y": 100})
    with pytest.deprecated_call():
        fn = data_catalog["gadm_level1"].path
        # should be driver_kwargs=dict(assert_gtype='Polygon')
        data_catalog.get_geodataframe(fn, assert_gtype="MultiPolygon")
    with pytest.deprecated_call():
        fn = data_catalog["grdc"].path
        # should be driver_kwargs=dict(index_col=0)
        data_catalog.get_dataframe(fn, index_col=0)
    with pytest.deprecated_call():
        fn = data_catalog["gtsmv3_eu_era5"].path
        # should be driver_kwargs=dict(chunks={'time': 100})
        data_catalog.get_geodataset(fn, chunks={"time": 100})


@pytest.mark.skip("needs catalogs refactor")
def test_detect_extent(data_catalog):
    # raster dataset
    name = "chirps_global"
    bbox = 11.60, 45.20, 13.00, 46.80
    expected_temporal_range = tuple(pd.to_datetime(["2010-02-02", "2010-02-15"]))
    ds = cast(RasterDatasetAdapter, data_catalog.get_source(name))
    detected_spatial_range = to_geographic_bbox(*ds.get_bbox(detect=True))
    detected_temporal_range = ds.get_time_range(detect=True)
    assert np.allclose(detected_spatial_range, bbox)
    assert detected_temporal_range == expected_temporal_range

    # geodataframe
    name = "gadm_level1"
    bbox = (6.63087893, 35.49291611, 18.52069473, 49.01704407)
    ds = cast(GeoDataFrameAdapter, data_catalog.get_source(name))

    detected_spatial_range = to_geographic_bbox(*ds.get_bbox(detect=True))
    assert np.all(np.equal(detected_spatial_range, bbox))

    # geodataset
    name = "gtsmv3_eu_era5"
    bbox = (12.22412, 45.22705, 12.99316, 45.62256)
    expected_temporal_range = (
        np.datetime64("2010-02-01"),
        np.datetime64("2010-02-14T23:50:00.000000000"),
    )
    ds = cast(GeoDatasetAdapter, data_catalog.get_source(name))
    detected_spatial_range = to_geographic_bbox(*ds.get_bbox(detect=True))
    detected_temporal_range = ds.get_time_range(detect=True)
    assert np.all(np.equal(detected_spatial_range, bbox))
    assert detected_temporal_range == expected_temporal_range


@pytest.mark.skip(reason="needs implementation of all data types.")
def test_to_stac(tmpdir, data_catalog):
    _ = data_catalog.get_rasterdataset("chirps_global")
    _ = data_catalog.get_geodataframe("gadm_level1")
    _ = data_catalog.get_geodataset("gtsmv3_eu_era5")

    sources = [
        "chirps_global",
        "gadm_level1",
        "gtsmv3_eu_era5",
    ]

    stac_catalog = data_catalog.to_stac_catalog(str(tmpdir), used_only=True)

    assert sorted(list(map(lambda x: x.id, stac_catalog.get_children()))) == sources
    # the two empty strings are for the root and self link which are destinct
    assert sorted(
        [
            Path(join(tmpdir, x.get_href())) if x != str(tmpdir) else tmpdir
            for x in stac_catalog.get_links()
        ]
    ) == sorted([Path(join(tmpdir, p, "catalog.json")) for p in ["", *sources, ""]])


@pytest.mark.skip(reason="Contains bug regarding switch to Pydantic.")
def test_from_stac():
    catalog_from_stac = DataCatalog().from_stac_catalog(
        "./tests/data/stac/catalog.json"
    )

    assert type(catalog_from_stac.get_source("chirps_global")) == RasterDatasetAdapter
    assert type(catalog_from_stac.get_source("gadm_level1")) == GeoDataFrameAdapter
    # assert type(catalog_from_stac.get_source("gtsmv3_eu_era5")) == GeoDatasetAdapter


def test_yml_from_uri_path():
    uri = "https://google.com/nothinghere"
    with pytest.raises(requests.HTTPError):
        _yml_from_uri_or_path(uri)
    uri = "https://raw.githubusercontent.com/Deltares/hydromt/main/.pre-commit-config.yaml"
    yml = _yml_from_uri_or_path(uri)
    assert isinstance(yml, dict)
    assert len(yml) > 0
