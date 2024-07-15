"""Tests for the hydromt.data_catalog submodule."""

import glob
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from os import mkdir
from os.path import abspath, basename, dirname, join
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from uuid import uuid4

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import requests
import xarray as xr
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem
from shapely import box
from yaml import dump

from hydromt._compat import HAS_GCSFS, HAS_OPENPYXL, HAS_S3FS
from hydromt._typing.error import ErrorHandleMethod, NoDataException, NoDataStrategy
from hydromt.config import Settings
from hydromt.data_catalog.adapters import (
    GeoDataFrameAdapter,
    GeoDatasetAdapter,
    RasterDatasetAdapter,
)
from hydromt.data_catalog.data_catalog import (
    DataCatalog,
    _denormalise_data_dict,
    _parse_data_source_dict,
    _yml_from_uri_or_path,
)
from hydromt.data_catalog.sources import (
    DataFrameSource,
    DataSource,
    GeoDataFrameSource,
    GeoDatasetSource,
    RasterDatasetSource,
)
from hydromt.gis.gis_utils import to_geographic_bbox
from hydromt.io.writers import write_xy

CATALOGDIR = join(dirname(abspath(__file__)), "..", "..", "data", "catalogs")
DATADIR = join(dirname(abspath(__file__)), "..", "data")


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
    assert datasource.full_uri == abspath(source["uri"])
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
    }
    sources = _denormalise_data_dict(dd)
    assert len(sources) == 1
    for name, source in sources:
        datasource = _parse_data_source_dict(
            name,
            source,
            root=root,  # TODO: do we need catalog_name="tmp"
        )
        assert datasource.full_uri == abspath(join(root, dd["test"]["uri"]))
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
        assert datasource.full_uri == abspath(join(root, f"data_{name[-1]}.gpkg"))
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


def test_data_catalog_io_round_trip(tmp_dir: Path, data_catalog: DataCatalog):
    # read / write
    uri_yml = str(tmp_dir / "test.yml")
    data_catalog.to_yml(uri_yml, root=str(data_catalog.root))
    data_catalog_read = DataCatalog(data_libs=uri_yml)
    assert data_catalog.to_dict() == data_catalog_read.to_dict()


def test_catalog_entry_no_variant(legacy_aws_worldcover):
    _, legacy_data_catalog = legacy_aws_worldcover
    # make sure the catalogs individually still work
    assert len(legacy_data_catalog) == 1
    source = legacy_data_catalog.get_source("esa_worldcover")
    assert Path(source.uri).name == "esa-worldcover.vrt"
    assert source.version == "2020"


def test_catalog_entry_no_variant_round_trip(
    legacy_aws_worldcover: Tuple[str, DataCatalog],
):
    _, legacy_data_catalog = legacy_aws_worldcover
    legacy_data_catalog2 = DataCatalog().from_dict(legacy_data_catalog.to_dict())
    assert legacy_data_catalog2 == legacy_data_catalog


def test_catalog_entry_single_variant(aws_worldcover):
    _, aws_data_catalog = aws_worldcover
    assert len(aws_data_catalog) == 1
    # test get_source with all keyword combinations
    source = aws_data_catalog.get_source("esa_worldcover")
    assert source.uri.endswith("ESA_WorldCover_10m_2020_v100_Map_AWS.vrt")
    assert source.version == "2021"
    source = aws_data_catalog.get_source("esa_worldcover", version="2021")
    assert source.uri.endswith("ESA_WorldCover_10m_2020_v100_Map_AWS.vrt")
    assert source.version == "2021"
    source = aws_data_catalog.get_source(
        "esa_worldcover", version="2021", provider="aws"
    )
    assert source.uri.endswith("ESA_WorldCover_10m_2020_v100_Map_AWS.vrt")


@pytest.fixture()
def aws_worldcover():
    aws_yml_path = join(DATADIR, "aws_esa_worldcover.yml")
    aws_data_catalog = DataCatalog(data_libs=[aws_yml_path])
    return (aws_yml_path, aws_data_catalog)


@pytest.fixture()
def merged_aws_worldcover():
    merged_yml_path = join(DATADIR, "merged_esa_worldcover.yml")
    merged_catalog = DataCatalog(data_libs=[merged_yml_path])
    return (merged_yml_path, merged_catalog)


@pytest.fixture()
def legacy_aws_worldcover():
    legacy_yml_path = join(DATADIR, "legacy_esa_worldcover.yml")
    legacy_data_catalog = DataCatalog(data_libs=[legacy_yml_path])
    return (legacy_yml_path, legacy_data_catalog)


def test_catalog_entry_single_variant_round_trip(
    aws_worldcover: Tuple[str, DataCatalog],
):
    _, aws_data_catalog = aws_worldcover
    aws_data_catalog_read = DataCatalog().from_dict(
        aws_data_catalog.to_dict(), root=aws_data_catalog.root
    )
    assert aws_data_catalog_read == aws_data_catalog


def test_catalog_entry_single_variant_unknown_provider(aws_worldcover):
    _, aws_data_catalog = aws_worldcover
    with pytest.raises(KeyError):
        aws_data_catalog.get_source(
            "esa_worldcover", version="2021", provider="asdfasdf"
        )


def test_catalog_entry_single_variant_unknown_version(aws_worldcover):
    _, aws_data_catalog = aws_worldcover
    with pytest.raises(KeyError):
        aws_data_catalog.get_source(
            "esa_worldcover", version="asdfasdf", provider="aws"
        )


def test_catalog_entry_single_variant_unknown_source(aws_worldcover):
    _, aws_data_catalog = aws_worldcover
    with pytest.raises(KeyError):
        aws_data_catalog.get_source("asdfasdf", version="2021", provider="aws")


def test_catalog_entry_warns_on_override_version(aws_worldcover):
    aws_yml_path, aws_data_catalog = aws_worldcover
    # make sure we trigger user warning when overwriting versions
    with pytest.warns(UserWarning):
        aws_data_catalog.from_yml(aws_yml_path)


def test_catalog_entry_merged_correct_version_provider(merged_aws_worldcover):
    _, merged_catalog = merged_aws_worldcover
    # make sure we can read merged catalogs
    assert len(merged_catalog) == 3
    source_aws = merged_catalog.get_source("esa_worldcover")  # last variant is default
    assert source_aws.driver.filesystem.protocol[0] == "s3"
    assert merged_catalog.get_source("esa_worldcover", provider="aws") == source_aws
    source_loc = merged_catalog.get_source("esa_worldcover", provider="local")
    assert source_loc != source_aws
    assert source_loc.driver.filesystem.protocol[0] == "file"
    assert source_loc.version == "2021"  # get newest version
    # test get_source with version only
    assert merged_catalog.get_source("esa_worldcover", version="2021") == source_loc
    # test round trip to and from dict


def test_catalog_entry_merged_round_trip(merged_aws_worldcover):
    _, merged_catalog = merged_aws_worldcover
    merged_dict = merged_catalog.to_dict()
    merged_catalog2 = DataCatalog().from_dict(merged_dict)

    merged_catalog2.root = merged_catalog.root
    assert merged_catalog2 == merged_catalog


def test_catalog_entry_merging(aws_worldcover, legacy_aws_worldcover):
    aws_yml_path, _ = aws_worldcover
    legacy_yml_path, _ = legacy_aws_worldcover
    # Make sure we can query for the version we want
    aws_and_legacy_catalog = DataCatalog(data_libs=[legacy_yml_path, aws_yml_path])
    assert len(aws_and_legacy_catalog) == 2
    source_aws = aws_and_legacy_catalog.get_source("esa_worldcover")
    assert source_aws.driver.filesystem.protocol[0] == "s3"
    source_aws2 = aws_and_legacy_catalog.get_source("esa_worldcover", provider="aws")
    assert source_aws2 == source_aws
    source_loc = aws_and_legacy_catalog.get_source(
        "esa_worldcover",
        provider="file",  # provider is filename
    )
    assert Path(source_loc.uri).name == "esa-worldcover.vrt"


def test_catalog_entry_merging_round_trip(aws_worldcover, legacy_aws_worldcover):
    aws_yml_path, _ = aws_worldcover
    legacy_yml_path, _ = legacy_aws_worldcover
    aws_and_legacy_catalog = DataCatalog(data_libs=[legacy_yml_path, aws_yml_path])
    # test round trip to and from dict
    d = aws_and_legacy_catalog.to_dict()

    aws_and_legacy_catalog2 = DataCatalog().from_dict(d)
    assert aws_and_legacy_catalog2 == aws_and_legacy_catalog


def test_versioned_catalogs_no_version(data_catalog):
    data_catalog._sources = {}  # reset
    data_catalog.from_predefined_catalogs("deltares_data")
    assert len(data_catalog.sources) > 0


def test_version_catalogs_errors_on_unknown_version(data_catalog):
    with pytest.raises(ValueError, match="Version v1993.7 not found "):
        _ = data_catalog.from_predefined_catalogs("deltares_data", "v1993.7")


def test_data_catalog_lazy_loading():
    data_catalog = DataCatalog()
    assert len(data_catalog._sources) == 0
    # global data sources from artifacts are automatically added
    assert len(data_catalog.sources) > 0


def test_data_catalog_contains_source_version_permissive(data_catalog):
    keys = data_catalog.get_source_names()
    assert data_catalog.contains_source(keys[0])
    assert data_catalog.contains_source(
        keys[0], version="asdfasdfasdf", permissive=True
    )
    assert not data_catalog.contains_source(
        keys[0], version="asdfasdf", permissive=False
    )


def test_data_catalog_repr(data_catalog):
    assert isinstance(data_catalog.__repr__(), str)
    assert isinstance(data_catalog._repr_html_(), str)
    assert isinstance(data_catalog.to_dataframe(), pd.DataFrame)
    with pytest.raises(ValueError, match="Value must be DataSource"):
        data_catalog.add_source("test", "string")  # type: ignore


def test_data_catalog_from_deltares_data():
    data_catalog = DataCatalog()
    assert len(data_catalog._sources) == 0
    data_catalog.from_predefined_catalogs("deltares_data")
    assert len(data_catalog._sources) > 0


def test_data_catalog_hydromt_version(tmpdir):
    yml_path = join(tmpdir, "test.yml")
    data_catalog = DataCatalog()
    data_catalog.to_yml(yml_path, meta={"hydromt_version": "0.7.0"})


def test_used_sources():
    merged_yml_path = join(DATADIR, "merged_esa_worldcover.yml")
    data_catalog = DataCatalog(merged_yml_path)
    source = data_catalog.get_source("esa_worldcover")
    source.mark_as_used()
    sources = data_catalog.list_sources(used_only=True)
    assert len(data_catalog) > 1
    assert len(sources) == 1
    assert sources[0][0] == "esa_worldcover"
    assert sources[0][1].provider == source.provider
    assert sources[0][1].version == source.version


def test_from_yml_with_archive(data_catalog: DataCatalog):
    data_catalog._sources = {}
    cache_dir = Path(data_catalog._cache_dir)
    data_catalog.from_predefined_catalogs("artifact_data=v1.0.0")
    sources = list(data_catalog.sources.keys())
    assert len(sources) > 0
    # as part of the getting the archive a a local
    # catalog file is written to the same folder
    # check if this file exists and we can read it
    yml_dst_path = Path(cache_dir, "artifact_data", "v1.0.0", "data_catalog.yml")
    assert yml_dst_path.exists()
    data_catalog1 = DataCatalog(yml_dst_path)
    sources = list(data_catalog1.sources.keys())
    source = data_catalog1.get_source(sources[0])
    assert yml_dst_path.parent == Path(source.full_uri).parent


def test_from_predefined_catalogs(data_catalog):
    assert len(data_catalog.predefined_catalogs) > 0
    for name in data_catalog.predefined_catalogs:
        data_catalog._sources = {}  # reset
        data_catalog.from_predefined_catalogs(f"{name}=latest")
        assert len(data_catalog._sources) > 0


def test_data_catalogs_raises_on_unknown_predefined_catalog(data_catalog):
    with pytest.raises(ValueError, match='Catalog with name "asdf" not found'):
        data_catalog.from_predefined_catalogs("asdf")


@pytest.fixture()
def export_test_slice_objects(tmpdir, data_catalog):
    data_catalog._sources = {}
    data_catalog.from_predefined_catalogs("artifact_data=v1.0.0")
    bbox = [12.0, 46.0, 13.0, 46.5]  # Piava river
    time_range = ("2010-02-10", "2010-02-15")
    data_lib_path = join(tmpdir, "data_catalog.yml")
    source_names = [
        "era5[precip,temp]",
        "grwl_mask",
        "modis_lai",
        "osm_coastlines",
        "grdc",
        "corine",
        "gtsmv3_eu_era5",
    ]

    return (data_catalog, bbox, time_range, source_names, data_lib_path)


@pytest.mark.skip("needs https://github.com/Deltares/hydromt/issues/886")
@pytest.mark.integration()
def test_export_global_datasets(tmpdir, export_test_slice_objects):
    (
        data_catalog,
        bbox,
        time_range,
        source_names,
        data_lib_path,
    ) = export_test_slice_objects
    data_catalog.export_data(
        tmpdir,
        bbox=bbox,
        time_range=time_range,
        source_names=source_names,
        meta={"version": 1},
        handle_nodata=NoDataStrategy.IGNORE,
    )
    with open(data_lib_path, "r") as f:
        yml_list = f.readlines()
    assert yml_list[0].strip() == "meta:"
    assert yml_list[1].strip() == "version: 1"
    assert yml_list[2].strip().startswith("root:")


@pytest.mark.skip("needs https://github.com/Deltares/hydromt/issues/886")
def test_export_global_datasets_overrwite(tmpdir, export_test_slice_objects):
    (
        data_catalog,
        bbox,
        time_range,
        source_names,
        data_lib_path,
    ) = export_test_slice_objects
    data_catalog.export_data(
        tmpdir,
        bbox=bbox,
        time_range=time_range,
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

    data_lib_path = join(tmpdir, "data_catalog.yml")
    # check if meta is written
    with open(data_lib_path, "r") as f:
        yml_list = f.readlines()
    assert yml_list[0].strip() == "meta:"
    assert yml_list[1].strip() == "version: 2"
    assert yml_list[2].strip().startswith("root:")


@pytest.mark.skip("needs https://github.com/Deltares/hydromt/issues/886")
@pytest.mark.integration()
def test_export_dataframe(tmpdir, df, df_time):
    # Write two csv files
    csv_path = str(tmpdir.join("test.csv"))
    parquet_path = str(tmpdir.join("test.parquet"))
    df.to_csv(csv_path)
    df.to_parquet(parquet_path)
    csv_ts_path = str(tmpdir.join("test_ts.csv"))
    parquet_ts_path = str(tmpdir.join("test_ts.parquet"))
    df_time.to_csv(csv_ts_path)
    df_time.to_parquet(parquet_ts_path)

    # Test to_file method (needs reading)
    data_dict = {
        "test_df": {
            "path": csv_path,
            "driver": "csv",
            "data_type": "DataFrame",
            "kwargs": {
                "index_col": 0,
            },
        },
        "test_df_ts": {
            "path": csv_ts_path,
            "driver": "csv",
            "data_type": "DataFrame",
            "kwargs": {
                "index_col": 0,
                "parse_dates": True,
            },
        },
        "test_df_parquet": {
            "path": parquet_path,
            "driver": "parquet",
            "data_type": "DataFrame",
        },
        "test_df_ts_parquet": {
            "path": parquet_ts_path,
            "driver": "parquet",
            "data_type": "DataFrame",
        },
    }

    data_catalog = DataCatalog()
    data_catalog.from_dict(data_dict)

    data_catalog.export_data(
        str(tmpdir),
        time_range=("2010-02-01", "2010-02-14"),
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


@pytest.mark.integration()
def test_http_data():
    dc = DataCatalog().from_dict(
        {
            "global_wind_atlas": {
                "data_type": "RasterDataset",
                "driver": {"name": "rasterio", "filesystem": "http"},
                "uri": "https://globalwindatlas.info/api/gis/global/wind-speed/10",
            }
        }
    )
    s: DataSource = dc.get_source("global_wind_atlas")
    # test inferred file system
    assert isinstance(s.driver.filesystem, fsspec.implementations.http.HTTPFileSystem)
    # test returns xarray DataArray
    da = s.read_data(bbox=[0, 0, 10, 10])
    assert isinstance(da, xr.DataArray)
    assert da.raster.shape == (4000, 4000)


class TestGetRasterDataset:
    @pytest.fixture()
    def era5_ds(self, data_catalog: DataCatalog) -> xr.Dataset:
        return data_catalog.get_rasterdataset("era5")

    @pytest.mark.integration()
    def test_zarr_and_netcdf_preprocessing_gives_same_results(
        self, era5_ds: xr.Dataset, tmp_path: Path
    ):
        path_zarr = tmp_path / "era5.zarr"
        era5_ds.to_zarr(path_zarr)
        data_dict = {
            "era5_zarr": {
                "data_type": "RasterDataset",
                "driver": {
                    "name": "raster_xarray",
                    "options": {
                        "preprocess": "round_latlon",
                    },
                },
                "metadata": {
                    "crs": 4326,
                },
                "uri": str(path_zarr),
            }
        }
        datacatalog = DataCatalog()
        datacatalog.from_dict(data_dict)
        era5_zarr = datacatalog.get_rasterdataset("era5_zarr")

        path_nc = tmp_path / "era5.nc"
        era5_ds.to_netcdf(path_nc)

        data_dict2 = {
            "era5_nc": {
                "data_type": "RasterDataset",
                "driver": {
                    "name": "raster_xarray",
                    "options": {
                        "preprocess": "round_latlon",
                    },
                },
                "metadata": {
                    "crs": 4326,
                },
                "uri": str(path_nc),
            }
        }
        datacatalog.from_dict(data_dict2)
        era5_nc = datacatalog.get_rasterdataset("era5_nc")
        assert era5_zarr.equals(era5_nc)
        dest: Path = tmp_path / "era5_copy.zarr"
        cast(RasterDatasetSource, datacatalog.get_source("era5_zarr")).to_file(dest)
        assert dest.exists()

    def test_rasterdataset_unit_attrs(self, data_catalog: DataCatalog):
        source = data_catalog.get_source("era5")
        attrs = {
            "temp": {"unit": "degrees C", "long_name": "temperature"},
            "temp_max": {"unit": "degrees C", "long_name": "maximum temperature"},
            "temp_min": {"unit": "degrees C", "long_name": "minimum temperature"},
        }
        source.metadata.attrs.update(**attrs)
        data_catalog.add_source("era5_new", source)
        raster = data_catalog.get_rasterdataset("era5_new")
        assert raster["temp"].attrs["unit"] == attrs["temp"]["unit"]
        assert raster["temp_max"].attrs["long_name"] == attrs["temp_max"]["long_name"]

    @pytest.mark.integration()
    def test_to_stac(self, data_catalog: DataCatalog):
        # raster dataset
        name = "chirps_global"
        source = cast(RasterDatasetSource, data_catalog.get_source(name))
        bbox, _ = source.get_bbox()
        start_dt, end_dt = source.get_time_range(detect=True)
        start_dt = pd.to_datetime(start_dt)
        end_dt = pd.to_datetime(end_dt)
        raster_stac_catalog = StacCatalog(id=name, description=name)
        raster_stac_item = StacItem(
            name,
            geometry=None,
            bbox=list(bbox),
            properties=source.metadata.model_dump(exclude_none=True),
            datetime=None,
            start_datetime=start_dt,
            end_datetime=end_dt,
        )
        raster_stac_asset = StacAsset(str(source.uri))
        raster_base_name = basename(source.uri)
        raster_stac_item.add_asset(raster_base_name, raster_stac_asset)

        raster_stac_catalog.add_item(raster_stac_item)

        outcome = cast(
            StacCatalog, source.to_stac_catalog(on_error=ErrorHandleMethod.RAISE)
        )

        assert raster_stac_catalog.to_dict() == outcome.to_dict()  # type: ignore
        source.metadata.crs = (
            -3.14
        )  # manually create an invalid adapter by deleting the crs
        assert source.to_stac_catalog(on_error=ErrorHandleMethod.SKIP) is None

    @pytest.fixture()
    def zoom_dict(self, tmp_dir: Path, zoom_level_tif: str) -> Dict[str, Any]:
        return {
            "test_zoom": {
                "data_type": "RasterDataset",
                "driver": {"name": "rasterio"},
                "uri": f"{str(tmp_dir)}/test_zl{{overview_level:d}}.tif",  # test with str format for zoom level
                "metadata": {
                    "crs": 4326,
                    "zls_dict": {0: 0.1, 1: 0.3},
                },
            }
        }

    @pytest.fixture()
    def zoom_level_tif(self, rioda_large: xr.DataArray, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test_zl1.tif")
        # write tif with zoom level 1 in name
        # NOTE zl 0 not written to check correct functioning
        rioda_large.raster.to_raster(uri)  # , overviews=[0, 1])
        return uri

    @pytest.mark.integration()
    @pytest.mark.usefixtures("zoom_level_tif")
    def test_zoom_levels_normal_tif(
        self, data_catalog: DataCatalog, zoom_dict: Dict[str, Any]
    ):
        # test zoom levels in name
        name: str = next(iter(zoom_dict.keys()))
        data_catalog.from_dict(zoom_dict)
        da1 = data_catalog.get_rasterdataset(name, zoom=(0.3, "degree"))
        assert isinstance(da1, xr.DataArray)

    @pytest.fixture()
    def zoom_level_cog(self, tmp_dir: Path, rioda_large: xr.DataArray) -> str:
        # write COG
        cog_uri = str(tmp_dir / "test_cog.tif")
        rioda_large.raster.to_raster(cog_uri, driver="COG", overviews="auto")
        return cog_uri

    def test_zoom_levels_cog(
        self, zoom_level_cog: str, rioda_large: xr.DataArray, data_catalog: DataCatalog
    ):
        # test COG zoom levels
        # return native resolution
        res = np.asarray(rioda_large.raster.res)
        da = data_catalog.get_rasterdataset(zoom_level_cog, zoom=0)
        assert np.allclose(da.raster.res, res)

    def test_zoom_levels_cog_zoom_resolution(
        self, zoom_level_cog: str, data_catalog: DataCatalog, rioda_large: xr.DataArray
    ):
        # reurn zoom level 1
        res = np.asarray(rioda_large.raster.res)
        da1 = data_catalog.get_rasterdataset(
            zoom_level_cog, zoom=(res[0] * 2, "degree")
        )
        assert np.allclose(da1.raster.res, res * 2)

    @pytest.fixture()
    def tif_no_overviews(self, tmp_dir: Path, rioda_large: xr.DataArray) -> str:
        uri = str(tmp_dir / "test_tif_no_overviews.tif")
        rioda_large.raster.to_raster(uri, driver="GTiff")
        return uri

    @pytest.mark.integration()
    def test_zoom_levels_no_overviews(
        self,
        tif_no_overviews: str,
        rioda_large: xr.DataArray,
        data_catalog: DataCatalog,
    ):
        # test if file hase no overviews
        da = data_catalog.get_rasterdataset(tif_no_overviews, zoom=(0.01, "degree"))
        xr.testing.assert_allclose(da, rioda_large)

    @pytest.mark.integration()
    def test_zoom_levels_with_variable(self, data_catalog: DataCatalog):
        # test if file has {variable} in path
        da = data_catalog.get_rasterdataset("merit_hydro", zoom=(0.01, "degree"))
        assert isinstance(da, xr.Dataset)

    @pytest.mark.integration()
    def test_get_koppen_geiger(self, data_catalog: DataCatalog):
        name = "koppen_geiger"
        source = data_catalog.get_source(name)
        da = data_catalog.get_rasterdataset(source)
        assert isinstance(da, xr.DataArray)

    @pytest.mark.integration()
    def test_bbox(self, data_catalog: DataCatalog):
        name = "koppen_geiger"
        da = data_catalog.get_rasterdataset(name)
        bbox = [12.0, 46.0, 13.0, 46.5]
        da = data_catalog.get_rasterdataset(da, bbox=bbox)
        assert isinstance(da, xr.DataArray)
        assert np.allclose(da.raster.bounds, bbox)

    @pytest.mark.integration()
    @pytest.mark.skipif(not HAS_S3FS, reason="S3FS not installed.")
    def test_s3(self, data_catalog: DataCatalog):
        data = r"s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N29_00_E105_00_DEM/Copernicus_DSM_COG_10_N29_00_E105_00_DEM.tif"
        # TODO: use filesystem in driver
        da = data_catalog.get_rasterdataset(
            data,
            driver={
                "name": "rasterio",
                "filesystem": {"protocol": "s3", "anon": "true"},
            },
        )

        assert isinstance(da, xr.DataArray)

    @pytest.mark.skipif(not HAS_S3FS, reason="S3FS not installed.")
    def test_aws_worldcover(self, test_settings: Settings):
        catalog_fn = join(CATALOGDIR, "aws_data", "v1.0.0", "data_catalog.yml")
        data_catalog = DataCatalog(data_libs=[catalog_fn])
        da = data_catalog.get_rasterdataset(
            "esa_worldcover_2020_v100",
            bbox=[12.0, 46.0, 12.5, 46.50],
        )
        assert da.name == "landuse"

        # check that the relevant file was cached
        cached_files: List[Path] = list(
            (
                test_settings.cache_root
                / "ESA_WorldCover_10m_2020_v100_Map_AWS"
                / "map"
            ).iterdir()
        )
        assert len(cached_files) == 1

        # check that the vrt was cached and the relevant file updated
        with (
            test_settings.cache_root
            / "ESA_WorldCover_10m_2020_v100_Map_AWS"
            / "ESA_WorldCover_10m_2020_v100_Map_AWS.vrt"
        ).open() as f:
            tree: ET.ElementTree = ET.parse(f)
        root: ET.Element = tree.getroot()
        assert (
            len(
                list(
                    filter(
                        lambda el: el.text.startswith("map"),
                        root.findall("VRTRasterBand/ComplexSource/SourceFilename"),
                    )
                )
            )
            == 1
        )

    @pytest.mark.skip(
        reason="Waiting for: https://github.com/Deltares/hydromt/issues/492"
    )
    @pytest.mark.skipif(not HAS_GCSFS, reason="GCSFS not installed.")
    def test_gcs_cmip6(self):
        # TODO switch to pre-defined catalogs when pushed to main
        catalog_fn = join(CATALOGDIR, "gcs_cmip6_data", "v1.0.0", "data_catalog.yml")
        data_catalog = DataCatalog(data_libs=[catalog_fn])
        ds = data_catalog.get_rasterdataset(
            "cmip6_NOAA-GFDL/GFDL-ESM4_historical_r1i1p1f1_Amon",
            variables=["precip", "temp"],
            time_range=(("1990-01-01", "1990-03-01")),
        )
        # Check reading and some preprocess
        assert "precip" in ds
        assert not np.any(ds[ds.raster.x_dim] > 180)

    @pytest.mark.integration()
    def test_reads_slippy_map_output(self, tmp_dir: Path, rioda_large: xr.DataArray):
        # write vrt data
        name = "tiled"
        root = tmp_dir / name
        rioda_large.raster.to_xyz_tiles(
            root=root,
            tile_size=256,
            zoom_levels=[0],
        )
        cat = DataCatalog(str(root / f"{name}.yml"), cache=True)
        cat.get_rasterdataset(name)
        assert len(glob.glob(join(root, "*", "*", "*.tif"))) == 16

    @pytest.mark.integration()
    def test_get_rasterdataset_unknown_datatype(self, data_catalog: DataCatalog):
        with pytest.raises(ValueError, match='Unknown raster data type "list"'):
            data_catalog.get_rasterdataset([])

    @pytest.mark.integration()
    def test_unknown_file(self, data_catalog: DataCatalog):
        with pytest.raises(NoDataException):
            data_catalog.get_rasterdataset("test1.tif")


def test_get_rasterdataset_unknown_key(data_catalog):
    with pytest.raises(ValueError, match="Unknown keys in requested data"):
        data_catalog.get_rasterdataset({"name": "test"})


class TestGetGeoDataFrame:
    @pytest.fixture()
    def uri_geojson(self, tmp_dir: Path, geodf: gpd.GeoDataFrame) -> str:
        uri_gdf = tmp_dir / "test.geojson"
        geodf.to_file(uri_gdf, driver="GeoJSON")
        return uri_gdf

    @pytest.fixture()
    def uri_shp(self, tmp_dir: Path, geodf: gpd.GeoDataFrame) -> str:
        uri_shapefile = tmp_dir / "test.shp"  # shapefile what a horror
        geodf.to_file(uri_shapefile)
        return uri_shapefile

    @pytest.mark.integration()
    def test_read_geojson_bbox(
        self, uri_geojson: str, geodf: gpd.GeoDataFrame, data_catalog: DataCatalog
    ):
        gdf = data_catalog.get_geodataframe(uri_geojson, bbox=geodf.total_bounds)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert np.all(gdf == geodf)

    @pytest.mark.integration()
    def test_read_shapefile_bbox(
        self, uri_shp: str, geodf: gpd.GeoDataFrame, data_catalog: DataCatalog
    ):
        gdf = data_catalog.get_geodataframe(uri_shp, bbox=geodf.total_bounds)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert np.all(gdf == geodf)

    @pytest.mark.integration()
    def test_read_shapefile_mask(
        self, uri_shp: str, geodf: gpd.GeoDataFrame, data_catalog: DataCatalog
    ):
        mask = gpd.GeoDataFrame({"geometry": [box(*geodf.total_bounds)]}, crs=geodf.crs)
        gdf = data_catalog.get_geodataframe(uri_shp, geom=mask)
        assert np.all(gdf == geodf)

    @pytest.mark.integration()
    def test_read_geojson_buffer_rename(
        self, uri_geojson: str, geodf: gpd.GeoDataFrame, data_catalog: DataCatalog
    ):
        gdf = data_catalog.get_geodataframe(
            uri_geojson,
            bbox=geodf.total_bounds,
            buffer=1000,
            data_adapter={"rename": {"test": "test1"}},
        )
        assert np.all(gdf == geodf)

    @pytest.mark.integration()
    def test_read_shp_buffer_rename(
        self, uri_shp: str, geodf: gpd.GeoDataFrame, data_catalog: DataCatalog
    ):
        gdf = data_catalog.get_geodataframe(
            uri_shp,
            bbox=geodf.total_bounds,
            buffer=1000,
            data_adapter={"rename": {"test": "test1"}},
        )
        assert np.all(gdf == geodf)

    @pytest.mark.integration()
    def test_read_unit_attrs(self, data_catalog: DataCatalog):
        gadm_level1: GeoDataFrameSource = data_catalog.get_source("gadm_level1")
        attrs = {"NAME_0": {"long_name": "Country names"}}
        gadm_level1.metadata.attrs.update(**attrs)
        gadm_level1_gdf = data_catalog.get_geodataframe("gadm_level1")
        assert gadm_level1_gdf["NAME_0"].attrs["long_name"] == "Country names"

    @pytest.mark.integration()
    def test_read_geojson_nodata_ignore(
        self, uri_geojson: str, data_catalog: DataCatalog
    ):
        gdf1 = data_catalog.get_geodataframe(
            uri_geojson,
            # only really care that the bbox doesn't intersect with anythign
            bbox=[12.5, 12.6, 12.7, 12.8],
            predicate="within",
            handle_nodata=NoDataStrategy.IGNORE,
        )

        assert gdf1 is None

    @pytest.mark.integration()
    def test_read_geojson_nodata_raise(
        self, uri_geojson: str, data_catalog: DataCatalog
    ):
        with pytest.raises(NoDataException):
            data_catalog.get_geodataframe(
                uri_geojson,
                # only really care that the bbox doesn't intersect with anythign
                bbox=[12.5, 12.6, 12.7, 12.8],
                predicate="within",
                handle_nodata=NoDataStrategy.RAISE,
            )

    @pytest.mark.integration()
    def test_raises_filenotfound(self, data_catalog: DataCatalog):
        with pytest.raises(NoDataException):
            data_catalog.get_geodataframe("no_file.geojson")

    @pytest.mark.integration()
    def test_to_stac_geodataframe(self, data_catalog: DataCatalog):
        # geodataframe
        name = "gadm_level1"
        source = cast(GeoDataFrameSource, data_catalog.get_source(name))
        bbox, _ = source.get_bbox()
        gdf_stac_catalog = StacCatalog(id=name, description=name)
        gds_stac_item = StacItem(
            name,
            geometry=None,
            bbox=list(bbox),
            properties=source.metadata,
            datetime=datetime(1, 1, 1),
        )
        gds_stac_asset = StacAsset(str(source.metadata.url))
        gds_base_name = basename(source.uri)
        gds_stac_item.add_asset(gds_base_name, gds_stac_asset)

        gdf_stac_catalog.add_item(gds_stac_item)
        outcome = cast(
            StacCatalog, source.to_stac_catalog(on_error=ErrorHandleMethod.RAISE)
        )
        assert gdf_stac_catalog.to_dict() == outcome.to_dict()  # type: ignore
        source.metadata.crs = (
            -3.14
        )  # manually create an invalid adapter by deleting the crs
        assert source.to_stac_catalog(on_error=ErrorHandleMethod.SKIP) is None


def test_get_geodataframe_path(data_catalog):
    n = len(data_catalog)

    name = "osm_coastlines"
    uri = data_catalog.get_source(name).uri
    p = Path(data_catalog.root) / uri

    # vector dataset using three different ways
    gdf = data_catalog.get_geodataframe(p)
    assert len(data_catalog) == n + 1
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_get_geodataframe_artifact_data(data_catalog):
    name = "osm_coastlines"
    gdf = data_catalog.get_geodataframe(name)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.index.size == 2


def test_get_geodataframe_artifact_data_geom(data_catalog):
    name = "osm_coastlines"
    gdf = data_catalog.get_geodataframe(name)
    gdf = data_catalog.get_geodataframe(gdf, geom=gdf.iloc[[0],], predicate="within")
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.index.size == 1


def test_get_geodataframe_unknown_data_type(data_catalog):
    with pytest.raises(ValueError, match='Unknown vector data type "list"'):
        data_catalog.get_geodataframe([])


@pytest.mark.integration()
def test_get_geodataframe_unknown_file(data_catalog):
    with pytest.raises(NoDataException):
        data_catalog.get_geodataframe("test1.gpkg")


def test_get_geodataframe_unknown_key(data_catalog):
    with pytest.raises(ValueError, match="Unknown keys in requested data"):
        data_catalog.get_geodataframe({"name": "test"})


class TestGetGeoDataset:
    @pytest.fixture()
    def geojson_dataset(self, geodf: gpd.GeoDataFrame, tmp_dir: Path) -> str:
        uri_gdf = str(tmp_dir / "test.geojson")
        geodf.to_file(uri_gdf, driver="GeoJSON")
        return uri_gdf

    @pytest.fixture()
    def csv_dataset(self, ts: pd.DataFrame, tmp_dir: Path) -> str:
        uri_csv = str(tmp_dir / "test.csv")
        ts.to_csv(uri_csv)
        return uri_csv

    @pytest.fixture()
    def xy_dataset(self, geodf: gpd.GeoDataFrame, tmp_dir: Path) -> str:
        uri_csv_locs = str(tmp_dir / "test_locs.xy")
        write_xy(uri_csv_locs, geodf)
        return uri_csv_locs

    @pytest.fixture()
    def nc_dataset(self, geoda: xr.Dataset, tmp_path: Path) -> str:
        backslash: str = "\\"
        uri_nc: str = str(
            tmp_path / f"{uuid4().hex.replace(backslash, '')}.nc"
        )  # generate random name for netcdf blocking
        geoda.vector.to_netcdf(uri_nc)
        return uri_nc

    @pytest.mark.integration()
    def test_geojson_vector_with_csv_data(
        self,
        geojson_dataset: str,
        data_catalog: DataCatalog,
        csv_dataset: str,
        geoda: xr.DataArray,
    ):
        da: Union[xr.DataArray, xr.Dataset, None] = data_catalog.get_geodataset(
            geojson_dataset,
            driver={"name": "geodataset_vector", "options": {"data_path": csv_dataset}},
        )
        assert isinstance(da, xr.DataArray), type(da)
        da = da.sortby("index")
        assert np.allclose(da, geoda)

    @pytest.mark.integration()
    def test_netcdf_with_variable_name(
        self, nc_dataset: str, data_catalog: DataCatalog, geoda: xr.DataArray
    ):
        da: Union[xr.DataArray, xr.Dataset, None] = data_catalog.get_geodataset(
            nc_dataset,
            variables=["test1"],
            bbox=geoda.vector.bounds,
            driver="geodataset_xarray",
        )
        assert isinstance(da, xr.DataArray)
        da = da.sortby("index")
        assert np.allclose(da, geoda)
        assert da.name == "test1"

    @pytest.mark.integration()
    def test_netcdf_single_var_as_array_false(
        self, nc_dataset: str, data_catalog: DataCatalog
    ):
        ds: Union[xr.DataArray, xr.Dataset, None] = data_catalog.get_geodataset(
            nc_dataset, single_var_as_array=False, driver="geodataset_xarray"
        )
        assert isinstance(ds, xr.Dataset)
        assert "test" in ds

    @pytest.mark.integration()
    def test_xy_locs_with_csv_data(
        self,
        xy_dataset: str,
        csv_dataset: str,
        data_catalog: DataCatalog,
        geoda: xr.DataArray,
        geodf: gpd.GeoDataFrame,
    ):
        da: Union[xr.DataArray, xr.Dataset, None] = data_catalog.get_geodataset(
            xy_dataset,
            driver={
                "name": "geodataset_vector",
                "options": {"data_path": csv_dataset},
            },
            metadata={"crs": geodf.crs},
        )
        assert isinstance(da, xr.DataArray)
        da = da.sortby("index")
        assert np.allclose(da, geoda)
        assert da.vector.crs.to_epsg() == 4326

    @pytest.mark.integration()
    def test_nodata_filenotfound(self, data_catalog: DataCatalog):
        with pytest.raises(NoDataException, match="no files"):
            data_catalog.get_geodataset("no_file.geojson")

    @pytest.mark.integration()
    def test_nodata_ignore(self, nc_dataset: str, data_catalog: DataCatalog):
        da: Optional[xr.DataArray] = data_catalog.get_geodataset(
            nc_dataset,
            # only really care that the bbox doesn't intersect with anythign
            driver="geodataset_xarray",
            bbox=[12.5, 12.6, 12.7, 12.8],
            handle_nodata=NoDataStrategy.IGNORE,
        )
        assert da is None

    @pytest.mark.integration()
    def test_nodata_raises_nodata(
        geodf: gpd.GeoDataFrame, nc_dataset: str, data_catalog: DataCatalog
    ):
        with pytest.raises(NoDataException):
            data_catalog.get_geodataset(
                nc_dataset,
                driver="geodataset_xarray",
                # only really care that the bbox doesn't intersect with anythign
                bbox=[12.5, 12.6, 12.7, 12.8],
                handle_nodata=NoDataStrategy.RAISE,
            )

    @pytest.mark.integration()
    def test_geodataset_unit_attrs(self, data_catalog: DataCatalog):
        source: DataSource = data_catalog.get_source("gtsmv3_eu_era5")
        attrs = {
            "waterlevel": {
                "long_name": "sea surface height above mean sea level",
                "unit": "meters",
            }
        }
        source.metadata.attrs = attrs
        gtsm_geodataarray = data_catalog.get_geodataset(source)
        assert gtsm_geodataarray.attrs["long_name"] == attrs["waterlevel"]["long_name"]
        assert gtsm_geodataarray.attrs["unit"] == attrs["waterlevel"]["unit"]

    @pytest.mark.integration()
    def test_geodataset_unit_conversion(self, data_catalog: DataCatalog):
        gtsm_geodataarray = data_catalog.get_geodataset("gtsmv3_eu_era5")
        source = data_catalog.get_source("gtsmv3_eu_era5")
        source.data_adapter.unit_mult = {"waterlevel": 1000}
        datacatalog = DataCatalog()
        gtsm_geodataarray1000 = datacatalog.get_geodataset(source)
        assert gtsm_geodataarray1000.equals(gtsm_geodataarray * 1000)

    @pytest.mark.integration()
    def test_geodataset_set_nodata(self, data_catalog: DataCatalog):
        source = data_catalog.get_source("gtsmv3_eu_era5")
        source.metadata.nodata = -99
        datacatalog = DataCatalog()
        ds = datacatalog.get_geodataset(source)
        assert ds.vector.nodata == -99

    def test_to_stac_geodataset(self, data_catalog: DataCatalog):
        # geodataset
        name = "gtsmv3_eu_era5"
        source = cast(GeoDatasetSource, data_catalog.get_source(name))
        bbox, _ = source.get_bbox()
        start_dt, end_dt = source.get_time_range(detect=True)
        start_dt = pd.to_datetime(start_dt)
        end_dt = pd.to_datetime(end_dt)
        gds_stac_catalog = StacCatalog(id=name, description=name)
        gds_stac_item = StacItem(
            name,
            geometry=None,
            bbox=list(bbox),
            properties=source.metadata.model_dump(exclude_none=True),
            datetime=None,
            start_datetime=start_dt,
            end_datetime=end_dt,
        )
        gds_stac_asset = StacAsset(str(source.uri))
        gds_base_name = basename(source.uri)
        gds_stac_item.add_asset(gds_base_name, gds_stac_asset)

        gds_stac_catalog.add_item(gds_stac_item)

        outcome = cast(
            StacCatalog, source.to_stac_catalog(on_error=ErrorHandleMethod.RAISE)
        )
        assert gds_stac_catalog.to_dict() == outcome.to_dict()  # type: ignore
        source.metadata.crs = (
            -3.14
        )  # manually create an invalid adapter by deleting the crs
        assert source.to_stac_catalog(ErrorHandleMethod.SKIP) is None


def test_get_geodataset_artifact_data(data_catalog):
    name = "gtsmv3_eu_era5"
    da = data_catalog.get_geodataset(name)
    assert da.vector.index.size == 19
    assert isinstance(da, xr.DataArray)


def test_get_geodataset_bbox_time_range(data_catalog: DataCatalog):
    name = "gtsmv3_eu_era5"
    uri = data_catalog.get_source(name).uri
    p = Path(data_catalog.root) / uri

    # vector dataset using three different ways
    da = data_catalog.get_geodataset(p, driver="geodataset_xarray")
    bbox = [12.22412, 45.25635, 12.25342, 45.271]
    da = data_catalog.get_geodataset(
        da,
        bbox=bbox,
        time_range=("2010-02-01", "2010-02-05"),
        driver="geodataset_xarray",
    )
    assert da.vector.index.size == 2
    assert da.time.size == 720
    assert isinstance(da, xr.DataArray)


def test_get_geodataset_unknown_data_type(data_catalog):
    with pytest.raises(ValueError, match='Unknown geo data type "list"'):
        data_catalog.get_geodataset([])


@pytest.mark.integration()
def test_get_geodataset_unknown_file(data_catalog):
    with pytest.raises(NoDataException, match="no files"):
        data_catalog.get_geodataset("test1.nc")


def test_get_geodataset_unknown_keys(data_catalog):
    with pytest.raises(ValueError, match="Unknown keys in requested data"):
        data_catalog.get_geodataset({"name": "test"})


def test_get_dataset(timeseries_df: pd.DataFrame, data_catalog: DataCatalog):
    test_dataset: xr.Dataset = timeseries_df.to_xarray()
    subset_timeseries = timeseries_df.iloc[[0, len(timeseries_df) // 2]]
    time_range = (
        subset_timeseries.index[0].to_pydatetime(),
        subset_timeseries.index[1].to_pydatetime(),
    )
    ds = data_catalog.get_dataset(test_dataset, time_range=time_range)
    assert isinstance(ds, xr.Dataset)
    assert ds.time[-1].values == subset_timeseries.index[1].to_datetime64()


def test_get_dataset_variables(timeseries_df: pd.DataFrame, data_catalog: DataCatalog):
    test_dataset: xr.Dataset = timeseries_df.to_xarray()
    ds = data_catalog.get_dataset(test_dataset, variables=["col1"])
    assert isinstance(ds, xr.DataArray)
    assert ds.name == "col1"


class TestGetDataFrame:
    @pytest.fixture()
    def uri_csv(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri: str = str(tmp_dir / "test.csv")
        df.to_csv(uri)
        return uri

    @pytest.fixture()
    def uri_parquet(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri: str = str(tmp_dir / "test.parquet")
        df.to_parquet(uri)
        return uri

    @pytest.fixture()
    def uri_fwf(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.txt")
        df.to_string(uri, index=False)
        return uri

    @pytest.fixture()
    def uri_xlsx(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.xlsx")
        df.to_excel(uri, index=False)
        return uri

    def test_reads_csv(self, df: pd.DataFrame, uri_csv: str, data_catalog: DataCatalog):
        df1 = data_catalog.get_dataframe(
            uri_csv, driver={"name": "pandas", "options": {"index_col": 0}}
        )
        assert isinstance(df1, pd.DataFrame)
        pd.testing.assert_frame_equal(df, df1)

    def test_reads_parquet(
        self, df: pd.DataFrame, uri_parquet: str, data_catalog: DataCatalog
    ):
        df1 = data_catalog.get_dataframe(uri_parquet)
        assert isinstance(df1, pd.DataFrame)
        pd.testing.assert_frame_equal(df, df1)

    def test_reads_fwf(self, df: pd.DataFrame, uri_fwf: str, data_catalog: DataCatalog):
        df1 = data_catalog.get_dataframe(
            uri_fwf, driver={"name": "pandas", "options": {"colspecs": "infer"}}
        )
        assert isinstance(df1, pd.DataFrame)
        pd.testing.assert_frame_equal(df1, df)

    @pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl is not installed.")
    def test_reads_excel(
        self, df: pd.DataFrame, uri_xlsx: str, data_catalog: DataCatalog
    ):
        df1 = data_catalog.get_dataframe(
            uri_xlsx, driver={"name": "pandas", "options": {"index_col": 0}}
        )
        assert isinstance(df1, pd.DataFrame)
        pd.testing.assert_frame_equal(df1, df.set_index("id"))

    def test_dataframe_unit_attrs(
        self, df: pd.DataFrame, tmp_dir: Path, data_catalog: DataCatalog
    ):
        df_path = tmp_dir / "cities.csv"
        df["test_na"] = -9999
        df.to_csv(df_path)
        cities = {
            "cities": {
                "uri": str(df_path),
                "data_type": "DataFrame",
                "driver": "pandas",
                "metadata": {
                    "nodata": -9999,
                    "attrs": {
                        "city": {"long_name": "names of cities"},
                        "country": {"long_name": "names of countries"},
                    },
                },
            }
        }
        data_catalog.from_dict(cities)
        cities_df = data_catalog.get_dataframe("cities")
        assert cities_df["city"].attrs["long_name"] == "names of cities"
        assert cities_df["country"].attrs["long_name"] == "names of countries"
        assert np.all(cities_df["test_na"].isna())

    @pytest.fixture()
    def csv_uri_time(self, tmp_dir: Path, df_time: pd.DataFrame) -> str:
        uri = str(tmp_dir / "test_ts.csv")
        df_time.to_csv(uri)
        return uri

    def test_time(
        self, df_time: pd.DataFrame, csv_uri_time: str, data_catalog: DataCatalog
    ):
        dfts = data_catalog.get_dataframe(
            csv_uri_time,
            driver={"name": "pandas", "options": {"index_col": 0, "parse_dates": True}},
        )
        assert isinstance(dfts, pd.DataFrame)
        assert np.all(
            dfts == df_time
        )  # indexes have different freq when parse_dates is used.

    def test_time_rename(self, csv_uri_time: str, data_catalog: DataCatalog):
        # Test renaming
        rename = {
            "precip": "P",
            "temp": "T",
            "pet": "ET",
        }
        dfts = data_catalog.get_dataframe(
            csv_uri_time,
            driver={"name": "pandas", "options": {"index_col": 0, "parse_dates": True}},
            data_adapter={"rename": rename},
        )
        assert np.all(list(dfts.columns) == list(rename.values()))

    def test_time_unit_mult_add(
        self, csv_uri_time: str, data_catalog: DataCatalog, df_time: pd.DataFrame
    ):
        unit_mult = {
            "precip": 0.75,
            "temp": 2,
            "pet": 1,
        }
        unit_add = {
            "precip": 0,
            "temp": -1,
            "pet": 2,
        }
        dfts = data_catalog.get_dataframe(
            csv_uri_time,
            driver={"name": "pandas", "options": {"index_col": 0, "parse_dates": True}},
            data_adapter={"unit_mult": unit_mult, "unit_add": unit_add},
        )
        # Do checks
        for var in df_time.columns:
            assert np.all(df_time[var] * unit_mult[var] + unit_add[var] == dfts[var])

    def test_time_slice(self, csv_uri_time: str, data_catalog: DataCatalog):
        dfts = data_catalog.get_dataframe(
            csv_uri_time,
            time_range=("2007-01-02", "2007-01-04"),
            driver={"name": "pandas", "options": {"index_col": 0, "parse_dates": True}},
        )
        assert len(dfts) == 3

    def test_time_variable_slice(self, csv_uri_time: str, data_catalog: DataCatalog):
        # Test variable slice
        vars_slice = ["precip", "temp"]
        dfts = data_catalog.get_dataframe(
            csv_uri_time,
            variables=vars_slice,
            driver={
                "name": "pandas",
                "options": {"parse_dates": True, "index_col": 0},
            },
        )
        assert np.all(dfts.columns == vars_slice)

    def test_to_stac(self, df: pd.DataFrame, tmp_dir: Path):
        uri_df = str(tmp_dir / "test.csv")
        name = "test_dataframe"
        df.to_csv(uri_df)
        dc = DataCatalog().from_dict(
            {name: {"data_type": "DataFrame", "uri": uri_df, "driver": "pandas"}}
        )

        source = cast(DataFrameSource, dc.get_source(name))

        with pytest.raises(
            NotImplementedError,
            match="DataFrameSource does not support full stac conversion ",
        ):
            source.to_stac_catalog(on_error=ErrorHandleMethod.RAISE)

        assert source.to_stac_catalog(on_error=ErrorHandleMethod.SKIP) is None

        stac_catalog = StacCatalog(
            name,
            description=name,
        )
        stac_item = StacItem(
            name,
            geometry=None,
            bbox=[0, 0, 0, 0],
            properties=source.metadata.model_dump(exclude_none=True),
            datetime=datetime(1, 1, 1),
        )
        stac_asset = StacAsset(str(uri_df))
        stac_item.add_asset("hydromt_path", stac_asset)

        stac_catalog.add_item(stac_item)
        outcome = cast(
            StacCatalog, source.to_stac_catalog(on_error=ErrorHandleMethod.COERCE)
        )
        assert stac_catalog.to_dict() == outcome.to_dict()  # type: ignore


def test_get_dataframe(df, tmpdir, data_catalog):
    n = len(data_catalog)
    name = "test.csv"
    csv_path = str(tmpdir.join(name))
    df.to_csv(csv_path)
    df = data_catalog.get_dataframe(csv_path)
    assert len(data_catalog) == n + 1
    assert isinstance(df, pd.DataFrame)


def test_get_dataframe_variables(df, data_catalog):
    df = data_catalog.get_dataframe(df, variables=["city"])
    assert isinstance(df, pd.DataFrame)
    assert df.columns == ["city"]


def test_get_dataframe_custom_data(tmp_dir, df, data_catalog):
    name = "test.csv"
    path = Path(tmp_dir, name)
    df.to_csv(path)

    gdf = data_catalog.get_dataframe(df)
    assert isinstance(gdf, pd.DataFrame)


def test_get_dataframe_unknown_data_type(data_catalog):
    with pytest.raises(ValueError, match='Unknown tabular data type "list"'):
        data_catalog.get_dataframe([])


@pytest.mark.integration()
def test_get_dataframe_unknown_file(data_catalog):
    with pytest.raises(NoDataException, match="no files"):
        data_catalog.get_dataframe("test1.csv")


def test_get_dataframe_unknown_keys(data_catalog):
    with pytest.raises(ValueError, match="Unknown keys in requested data"):
        data_catalog.get_dataframe({"name": "test"})


def test_detect_extent_rasterdataset(data_catalog):
    # raster dataset
    name = "chirps_global"
    bbox = 11.60, 45.20, 13.00, 46.80
    expected_temporal_range = tuple(pd.to_datetime(["2010-02-02", "2010-02-15"]))
    ds = cast(RasterDatasetAdapter, data_catalog.get_source(name))
    detected_spatial_range = to_geographic_bbox(*ds.get_bbox(detect=True))
    detected_temporal_range = ds.get_time_range(detect=True)
    assert np.allclose(detected_spatial_range, bbox)
    assert detected_temporal_range == expected_temporal_range


def test_detect_extent_geodataframe(data_catalog):
    # geodataframe
    name = "gadm_level1"
    bbox = (6.63087893, 35.49291611, 18.52069473, 49.01704407)
    ds = cast(GeoDataFrameAdapter, data_catalog.get_source(name))

    detected_spatial_range = to_geographic_bbox(*ds.get_bbox(detect=True))
    assert np.all(np.equal(detected_spatial_range, bbox))


def test_detect_extent_geodataset(data_catalog):
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


def test_to_stac_raster_dataset(tmpdir, data_catalog):
    data_catalog._sources = {}
    _ = data_catalog.get_rasterdataset("chirps_global")

    sources = [
        "chirps_global",
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


def test_from_stac():
    catalog_from_stac = DataCatalog().from_stac_catalog(
        "./tests/data/stac/catalog.json"
    )

    assert type(catalog_from_stac.get_source("chirps_global")) == RasterDatasetSource
    assert type(catalog_from_stac.get_source("gadm_level1")) == GeoDataFrameSource
    assert type(catalog_from_stac.get_source("gtsmv3_eu_era5")) == RasterDatasetSource


def test_yml_from_uri_path():
    uri = "https://google.com/nothinghere"
    with pytest.raises(requests.HTTPError):
        _yml_from_uri_or_path(uri)
    uri = "https://raw.githubusercontent.com/Deltares/hydromt/main/.pre-commit-config.yaml"
    yml = _yml_from_uri_or_path(uri)
    assert isinstance(yml, dict)
    assert len(yml) > 0
