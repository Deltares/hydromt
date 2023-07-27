"""Tests for the hydromt.data_catalog submodule."""

import os
from os.path import abspath, dirname, join
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr

from hydromt.data_adapter import DataAdapter, RasterDatasetAdapter
from hydromt.data_catalog import DataCatalog, _denormalise_data_dict, _parse_data_dict

CATALOGDIR = join(dirname(abspath(__file__)), "..", "data", "catalogs")
DATADIR = join(dirname(abspath(__file__)), "data")


def test_parser():
    # valid abs root on windows and linux!
    root = "c:/root" if os.name == "nt" else "/c/root"
    # simple; abs path
    dd = {
        "test": {
            "data_type": "RasterDataset",
            "path": f"{root}/to/data.tif",
        }
    }
    dd_out = _parse_data_dict(dd, root=root)
    assert isinstance(dd_out["test"], RasterDatasetAdapter)
    assert dd_out["test"].path == abspath(dd["test"]["path"])
    # test with Path object
    dd["test"].update(path=Path(dd["test"]["path"]))
    dd_out = _parse_data_dict(dd, root=root)
    assert dd_out["test"].path == abspath(dd["test"]["path"])
    # rel path
    dd = {
        "test": {
            "data_type": "RasterDataset",
            "path": "path/to/data.tif",
            "kwargs": {"fn": "test"},
        },
        "root": root,
    }
    dd_out = _parse_data_dict(dd)
    assert dd_out["test"].path == abspath(join(root, dd["test"]["path"]))
    # check if path in kwargs is also absolute
    assert dd_out["test"].driver_kwargs["fn"] == abspath(join(root, "test"))
    # alias
    dd = {
        "test": {
            "data_type": "RasterDataset",
            "path": "path/to/data.tif",
        },
        "test1": {"alias": "test"},
    }
    with pytest.deprecated_call():
        dd = _denormalise_data_dict(dd, catalog_name="tmp")

    dd_out1 = _parse_data_dict(dd[0], root=root)
    dd_out2 = _parse_data_dict(dd[1], root=root)
    assert dd_out1["test"].path == dd_out2["test1"].path
    # placeholder
    dd = {
        "test_{p1}_{p2}": {
            "data_type": "RasterDataset",
            "path": "data_{p2}.tif",
            "placeholders": {"p1": ["a", "b"], "p2": ["1", "2", "3"]},
        },
    }
    dd_out = _parse_data_dict(dd, root=root)
    assert len(dd_out) == 6
    assert dd_out["test_a_1"].path == abspath(join(root, "data_1.tif"))
    assert "placeholders" not in dd_out["test_a_1"].to_dict()

    # errors
    with pytest.raises(ValueError, match="Missing required path argument"):
        _parse_data_dict({"test": {}})
    with pytest.raises(ValueError, match="Data type error unknown"):
        _parse_data_dict({"test": {"path": "", "data_type": "error"}})
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
    assert (
        Path(legacy_data_catalog.get_source("esa_worldcover").path).name
        == "esa-worldcover.vrt"
    )
    assert legacy_data_catalog.get_source("esa_worldcover").data_version == 2020

    # make sure we raise deprecation warning here
    with pytest.deprecated_call():
        _ = legacy_data_catalog["esa_worldcover"]

    aws_yml_fn = join(DATADIR, "aws_esa_worldcover.yml")
    aws_data_catalog = DataCatalog(data_libs=[aws_yml_fn])
    # test get_source with all keyword combinations
    assert (
        aws_data_catalog.get_source("esa_worldcover").path
        == "s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt"
    )
    assert aws_data_catalog.get_source("esa_worldcover").data_version == 2021
    assert (
        aws_data_catalog.get_source("esa_worldcover", data_version=2021).path
        == "s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt"
    )
    assert (
        aws_data_catalog.get_source("esa_worldcover", data_version=2021).data_version
        == 2021
    )
    assert (
        aws_data_catalog.get_source(
            "esa_worldcover", data_version=2021, provider="aws"
        ).path
        == "s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt"
    )

    with pytest.raises(KeyError):
        aws_data_catalog.get_source(
            "esa_worldcover", data_version=2021, provider="asdfasdf"
        )
    with pytest.raises(KeyError):
        aws_data_catalog.get_source(
            "esa_worldcover", data_version="asdfasdf", provider="aws"
        )
    with pytest.raises(KeyError):
        aws_data_catalog.get_source("asdfasdf", data_version=2021, provider="aws")

    # make sure we trigger user warning when overwriting versions
    with pytest.warns(UserWarning):
        aws_data_catalog.from_yml(aws_yml_fn)

    # make sure we can read merged catalogs
    merged_yml_fn = join(DATADIR, "merged_esa_worldcover.yml")
    read_merged_catalog = DataCatalog(data_libs=[merged_yml_fn])
    assert (
        read_merged_catalog.get_source("esa_worldcover").path
        == "s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt"
    )
    assert (
        read_merged_catalog.get_source("esa_worldcover", provider="aws").path
        == "s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt"
    )
    assert (
        Path(
            read_merged_catalog.get_source("esa_worldcover", provider="local").path
        ).name
        == "esa-worldcover.vrt"
    )

    # make sure dataframe doesn't merge different variants
    assert len(read_merged_catalog.to_dataframe()) == 2

    # Make sure we can queiry for the version we want
    aws_and_legacy_data_catalog = DataCatalog(data_libs=[legacy_yml_fn, aws_yml_fn])
    assert (
        aws_and_legacy_data_catalog.get_source("esa_worldcover").path
        == "s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt"
    )

    assert (
        aws_and_legacy_data_catalog.get_source("esa_worldcover", provider="aws").path
        == "s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt"
    )
    assert (
        Path(
            aws_and_legacy_data_catalog.get_source(
                "esa_worldcover", provider="legacy_esa_worldcover"
            ).path
        ).name
        == "esa-worldcover.vrt"
    )

    _ = aws_and_legacy_data_catalog.to_dict()


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


def test_get_data(df):
    data_catalog = DataCatalog("artifact_data")  # read artifacts

    # raster dataset using three different ways
    da = data_catalog.get_rasterdataset(data_catalog.get_source("koppen_geiger").path)
    assert isinstance(da, xr.DataArray)
    da = data_catalog.get_rasterdataset("koppen_geiger")
    assert isinstance(da, xr.DataArray)
    da = data_catalog.get_rasterdataset(da)
    assert isinstance(da, xr.DataArray)
    with pytest.raises(ValueError, match='Unknown raster data type "list"'):
        data_catalog.get_rasterdataset([])

    # vector dataset using three different ways
    gdf = data_catalog.get_geodataframe(data_catalog.get_source("osm_coastlines").path)
    assert isinstance(gdf, gpd.GeoDataFrame)
    gdf = data_catalog.get_geodataframe("osm_coastlines")
    assert isinstance(gdf, gpd.GeoDataFrame)
    gdf = data_catalog.get_geodataframe(gdf)
    assert isinstance(gdf, gpd.GeoDataFrame)
    with pytest.raises(ValueError, match='Unknown vector data type "list"'):
        data_catalog.get_geodataframe([])

    # geodataset using three different ways
    da = data_catalog.get_geodataset(data_catalog.get_source("gtsmv3_eu_era5").path)
    assert isinstance(da, xr.DataArray)
    da = data_catalog.get_geodataset("gtsmv3_eu_era5")
    assert isinstance(da, xr.DataArray)
    da = data_catalog.get_geodataset(da)
    assert isinstance(da, xr.DataArray)
    with pytest.raises(ValueError, match='Unknown geo data type "list"'):
        data_catalog.get_geodataset([])

    # dataframe using single way
    df = data_catalog.get_dataframe(df)
    assert isinstance(df, pd.DataFrame)
    with pytest.raises(ValueError, match='Unknown tabular data type "list"'):
        data_catalog.get_dataframe([])


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
