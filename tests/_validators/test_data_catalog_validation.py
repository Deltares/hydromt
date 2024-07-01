"""Testing of the Pydantic models for validation of Data catalogs."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_core import Url

from hydromt._validators.data_catalog import (
    DataCatalogItem,
    DataCatalogMetaData,
    DataCatalogValidator,
)
from hydromt.io.readers import _yml_from_uri_or_path


@pytest.mark.skip("validators need  to be updated to newest format")
def test_deltares_data_catalog(latest_dd_version_uri):
    yml_dict = _yml_from_uri_or_path(latest_dd_version_uri)
    # whould raise error if something goes wrong
    _ = DataCatalogValidator.from_dict(yml_dict)


def test_geodataframe_entry_validation():
    d = {
        "crs": 4326,
        "data_type": "GeoDataFrame",
        "driver": "vector",
        "kwargs": {"layer": "BasinATLAS_v10_lev12"},
        "meta": {
            "category": "hydrography",
            "notes": "renaming and units might require some revision",
            "paper_doi": "10.1038/s41597-019-0300-6",
            "paper_ref": "Linke et al. (2019)",
            "source_license": "CC BY 4.0",
            "source_url": "https://www.hydrosheds.org/hydroatlas",
            "source_version": "10",
        },
        "path": "hydrography/hydro_atlas/basin_atlas_v10.gpkg",
    }
    entry = DataCatalogItem.from_dict(d, name="basin_atlas_level12_v10")

    assert entry.crs == 4326
    assert entry.data_type == "GeoDataFrame"
    assert entry.driver == "vector"
    assert entry.kwargs == {"layer": "BasinATLAS_v10_lev12"}
    assert entry.meta is not None
    assert entry.meta.category == "hydrography"
    assert entry.meta.notes == "renaming and units might require some revision"
    assert entry.meta.paper_doi == "10.1038/s41597-019-0300-6"
    assert entry.meta.paper_ref == "Linke et al. (2019)"
    assert entry.meta.source_license == "CC BY 4.0"
    assert entry.meta.source_url == Url("https://www.hydrosheds.org/hydroatlas")
    assert entry.meta.source_version == "10"
    assert entry.path == Path("hydrography/hydro_atlas/basin_atlas_v10.gpkg")


def test_valid_catalog_variants():
    d = {
        "meta": {"hydromt_version": ">=1.0a,<2", "roots": [""]},
        "esa_worldcover": {
            "crs": 4326,
            "data_type": "RasterDataset",
            "driver": "raster",
            "filesystem": "local",
            "kwargs": {"chunks": {"x": 36000, "y": 36000}},
            "meta": {
                "category": "landuse",
                "source_license": "CC BY 4.0",
                "source_url": "https://doi.org/10.5281/zenodo.5571936",
            },
            "variants": [
                {
                    "provider": "local",
                    "version": 2021,
                    "path": "landuse/esa_worldcover_2021/esa-worldcover.vrt",
                },
                {
                    "provider": "local",
                    "version": 2020,
                    "path": "landuse/esa_worldcover/esa-worldcover.vrt",
                },
                {
                    "provider": "aws",
                    "version": 2020,
                    "path": "s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt",
                    "rename": {"ESA_WorldCover_10m_2020_v100_Map_AWS": "landuse"},
                    "filesystem": "s3",
                    "storage_options": {"anon": True},
                },
            ],
        },
    }
    _ = DataCatalogValidator.from_dict(d)


def test_no_hydrmt_version_loggs_warning(caplog: pytest.LogCaptureFixture):
    d = {
        "meta": {"roots": [""]},
    }

    _ = DataCatalogValidator.from_dict(d)
    assert "No hydromt version" in caplog.text


def test_catalog_metadata_validation():
    d = {
        "hydromt_version": ">=1.0a,<2",
        "roots": ["p:/wflow_global/hydromt", "/mnt/p/wflow_global/hydromt"],
        "version": "2023.3",
    }
    catalog_metadata = DataCatalogMetaData.from_dict(d)
    assert catalog_metadata.roots == [
        Path("p:/wflow_global/hydromt"),
        Path("/mnt/p/wflow_global/hydromt"),
    ]
    assert catalog_metadata.version == "2023.3"


def test_raster_dataset_entry_validation():
    d = {
        "crs": 4326,
        "data_type": "RasterDataset",
        "driver": "raster",
        "kwargs": {
            "chunks": {
                "x": 3600,
                "y": 3600,
            }
        },
        "meta": {
            "category": "meteo",
            "paper_doi": "10.1038/sdata.2017.122",
            "paper_ref": "Karger et al. (2017)",
            "source_license": "CC BY 4.0",
            "source_url": "https://chelsa-climate.org/downloads/",
            "source_version": "1.2",
        },
        "path": "meteo/chelsa_clim_v1.2/CHELSA_bio10_12.tif",
    }

    entry = DataCatalogItem.from_dict(d, name="chelsa_v1.2")
    assert entry.name == "chelsa_v1.2"
    assert entry.crs == 4326
    assert entry.data_type == "RasterDataset"
    assert entry.driver == "raster"
    assert entry.path == Path("meteo/chelsa_clim_v1.2/CHELSA_bio10_12.tif")
    assert entry.kwargs == {"chunks": {"x": 3600, "y": 3600}}
    assert entry.meta is not None
    assert entry.meta.category == "meteo"
    assert entry.meta.paper_doi == "10.1038/sdata.2017.122"
    assert entry.meta.paper_ref == "Karger et al. (2017)"
    assert entry.meta.source_license == "CC BY 4.0"
    assert entry.meta.source_url == Url("https://chelsa-climate.org/downloads/")
    assert entry.meta.source_version == "1.2"


def test_dataset_entry_with_typo_validation():
    d = {
        "crs_num": 4326,
        "datatype": "RasterDataset",
        "diver": "raster",
        "kw_args": {
            "chunks": {
                "x": 3600,
                "y": 3600,
            }
        },
        "filepath": "meteo/chelsa_clim_v1.2/CHELSA_bio10_12.tif",
    }

    # 8 errors are:
    #  - missing crs, data_type and driver (3)
    #  - extra crs_num, datatype, diver, and filepath (5)
    with pytest.raises(ValidationError, match="7 validation errors"):
        _ = DataCatalogItem.from_dict(d, name="chelsa_v1.2")


def test_data_type_typo():
    d = {
        "crs": 4326,
        "data_type": "RaserDataset",
        "driver": "raster",
        "path": ".",
    }
    with pytest.raises(ValidationError, match="1 validation error"):
        _ = DataCatalogItem.from_dict(d, name="chelsa_v1.2")


def test_data_invalid_crs():
    d = {
        "crs": 123456789,
        "data_type": "RasterDataset",
        "driver": "raster",
        "path": ".",
    }
    with pytest.raises(ValidationError, match="1 validation error"):
        _ = DataCatalogItem.from_dict(d, name="chelsa_v1.2")
