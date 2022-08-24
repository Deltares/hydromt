"""Tests for the hydromt.data_catalog submodule."""

import os
import pytest
from os.path import join, abspath
import pandas as pd
import geopandas as gpd
import xarray as xr
from hydromt.data_adapter import DataAdapter, RasterDatasetAdapter, DataFrameAdapter
from hydromt.data_catalog import (
    DataCatalog,
    _parse_data_dict,
)


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
    assert dd_out["test"].kwargs["fn"] == abspath(join(root, "test"))
    # alias
    dd = {
        "test": {
            "data_type": "RasterDataset",
            "path": "path/to/data.tif",
        },
        "test1": {"alias": "test"},
    }
    dd_out = _parse_data_dict(dd, root=root)
    assert dd_out["test"].path == dd_out["test1"].path
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
    # errors
    with pytest.raises(ValueError, match="Missing required path argument"):
        _parse_data_dict({"test": {}})
    with pytest.raises(ValueError, match="Data type error unknown"):
        _parse_data_dict({"test": {"path": "", "data_type": "error"}})
    with pytest.raises(ValueError, match="alias test not found in data_dict"):
        _parse_data_dict({"test1": {"alias": "test"}})


def test_data_catalog_io(tmpdir):
    data_catalog = DataCatalog()
    # read / write
    fn_yml = str(tmpdir.join("test.yml"))
    data_catalog.to_yml(fn_yml)
    data_catalog1 = DataCatalog(data_libs=fn_yml)
    assert data_catalog.to_dict() == data_catalog1.to_dict()
    # test print
    print(data_catalog["merit_hydro"])


def test_data_catalog(tmpdir):
    data_catalog = DataCatalog(data_libs=None)  # NOTE: legacy code!
    # initialized with empty dict
    assert len(data_catalog._sources) == 0
    # global data sources from artifacts are automatically added
    assert len(data_catalog) > 0
    # test keys, getitem,
    keys = data_catalog.keys
    source = data_catalog[keys[0]]
    assert isinstance(source, DataAdapter)
    assert keys[0] in data_catalog
    # add source from dict
    data_dict = {keys[0]: source.to_dict()}
    data_catalog.from_dict(data_dict)
    # printers
    assert isinstance(data_catalog.__repr__(), str)
    assert isinstance(data_catalog._repr_html_(), str)
    assert isinstance(data_catalog.to_dataframe(), pd.DataFrame)
    with pytest.raises(ValueError, match="Value must be DataAdapter"):
        data_catalog["test"] = "string"
    # test artifact keys (NOTE: legacy code!)
    data_catalog = DataCatalog(deltares_data=False)
    assert len(data_catalog._sources) == 0
    data_catalog.from_artifacts("deltares_data")
    assert len(data_catalog._sources) > 0
    with pytest.raises(IOError):
        data_catalog = DataCatalog(deltares_data="unknown_version")


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
    assert len(data_catalog._sources) > 0
    source0 = data_catalog._sources[[k for k in data_catalog.sources.keys()][0]]
    assert ".hydromt_data" in str(source0.path)
    # failed to download
    assert data_catalog.from_archive("https://asdf.com/asdf.zip") == 404


def test_from_predefined_catalogs():
    data_catalog = DataCatalog()
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
    bbox = [11.70, 45.35, 12.95, 46.70]  # Piava river
    time_tuple = ("2010-02-01", "2010-02-14")
    data_catalog = DataCatalog()  # read artifacts by default
    sns = [
        "era5",
        "grwl_mask",
        "modis_lai",
        "osm_coastlines",
        "grdc",
        "corine",
        "gtsmv3_eu_era5",
        "hydro_lakes",
        "eobs",
    ]
    data_catalog.export_data(
        str(tmpdir), bbox=bbox, time_tuple=time_tuple, source_names=sns
    )

    data_catalog1 = DataCatalog(str(tmpdir.join("data_catalog.yml")))
    for key, source in data_catalog1.sources.items():
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
    assert len(data_catalog1) == 1

    data_catalog.export_data(str(tmpdir))
    data_catalog1 = DataCatalog(str(tmpdir.join("data_catalog.yml")))
    assert len(data_catalog1) == 2
    for key, source in data_catalog1.sources.items():
        dtypes = pd.DataFrame
        obj = source.get_data()
        assert isinstance(obj, dtypes), key
