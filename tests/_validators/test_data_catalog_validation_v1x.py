"""Testing of the Pydantic models for validation of Data catalogs."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from hydromt._validators.data_catalog_v0x import DataCatalogV0Validator
from hydromt._validators.data_catalog_v1x import (
    DataCatalogV1Item,
    DataCatalogV1MetaData,
    DataCatalogV1Validator,
)
from hydromt.io import _yml_from_uri_or_path
from tests.conftest import TEST_DATA_DIR


def test_deltares_data_catalog_v1(latest_dd_version_uri):
    yml_dict = _yml_from_uri_or_path(latest_dd_version_uri)
    # would raise error if something goes wrong
    _ = DataCatalogV1Validator.from_dict(yml_dict)


def test_geodataframe_v1_entry_validation():
    d = {
        "data_type": "GeoDataFrame",
        "version": 10,
        "uri": "hydrography/hydro_atlas/basin_atlas_v10.gpkg",
        "driver": {"name": "pyogrio", "options": {"layer": "BasinATLAS_v10_lev12"}},
        "metadata": {
            "category": "hydrography",
            "crs": 4326,
            "notes": "renaming and units might require some revision",
            "paper_doi": "10.1038/s41597-019-0300-6",
            "paper_ref": "Linke et al. (2019)",
            "source_url": "https://www.hydrosheds.org/hydroatlas",
            "source_license": "CC BY 4.0",
            "extent": {
                "bbox": {
                    "West": -180.0,
                    "South": -55.988,
                    "East": 180.001,
                    "North": 83.626,
                }
            },
        },
    }
    entry = DataCatalogV1Item.from_dict(d, name="basin_atlas_level12_v10")

    assert entry.metadata is not None
    assert entry.driver is not None
    assert entry.metadata.crs == 4326
    assert entry.data_type == "GeoDataFrame"
    assert entry.driver.name == "pyogrio"
    assert entry.driver.options.model_dump() == {"layer": "BasinATLAS_v10_lev12"}

    assert entry.metadata.category == "hydrography"
    assert entry.metadata.notes == "renaming and units might require some revision"
    assert entry.metadata.paper_doi == "10.1038/s41597-019-0300-6"
    assert entry.metadata.paper_ref == "Linke et al. (2019)"
    assert entry.metadata.source_license == "CC BY 4.0"
    assert entry.metadata.source_url == "https://www.hydrosheds.org/hydroatlas"
    assert entry.uri == Path("hydrography/hydro_atlas/basin_atlas_v10.gpkg")


def test_valid_v1_catalog_variants():
    d = {
        "meta": {"hydromt_version": ">=1.0a,<2", "roots": [""]},
        "esa_worldcover": {
            "data_type": "RasterDataset",
            "driver": {
                "name": "raster",
                "options": {"chunks": {"x": 36000, "y": 36000}},
            },
            "metadata": {
                "crs": 4326,
                "category": "landuse",
                "source_license": "CC BY 4.0",
                "source_url": "https://doi.org/10.5281/zenodo.5571936",
            },
            "variants": [
                {
                    "provider": "local",
                    "version": 2021,
                    "uri": "landuse/esa_worldcover_2021/esa-worldcover.vrt",
                },
                {
                    "provider": "local",
                    "version": 2020,
                    "uri": "landuse/esa_worldcover/esa-worldcover.vrt",
                },
                {
                    "provider": "aws",
                    "version": 2020,
                    "uri": "s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt",
                    "rename": {"ESA_WorldCover_10m_2020_v100_Map_AWS": "landuse"},
                    "driver": {
                        "name": "raster",
                        "filesystem": "s3",
                        "options": {"anon": "true"},
                    },
                },
            ],
        },
    }
    _ = DataCatalogV1Validator.from_dict(d)


def test_no_hydromt_version_in_v1_catalog_logs_warning(
    caplog: pytest.LogCaptureFixture,
):
    d = {
        "meta": {"roots": [""]},
    }

    _ = DataCatalogV1Validator.from_dict(d)
    assert "No hydromt version" in caplog.text


def test_catalog_v1_metadata_validation():
    d = {
        "hydromt_version": ">=1.0a,<2",
        "roots": ["p:/wflow_global/hydromt", "/mnt/p/wflow_global/hydromt"],
        "version": "2023.3",
    }
    catalog_metadata = DataCatalogV1MetaData.from_dict(d)
    assert catalog_metadata.roots == [
        Path("p:/wflow_global/hydromt"),
        Path("/mnt/p/wflow_global/hydromt"),
    ]
    assert catalog_metadata.version == "2023.3"


def test_raster_dataset_v1_entry_validation():
    d = {
        "data_type": "RasterDataset",
        "metadata": {
            "category": "meteo",
            "paper_doi": "10.1038/sdata.2017.122",
            "paper_ref": "Karger et al. (2017)",
            "source_license": "CC BY 4.0",
            "source_url": "https://chelsa-climate.org/downloads/",
            "source_version": "1.2",
            "crs": 4326,
        },
        "driver": {
            "name": "raster",
            "options": {
                "chunks": {
                    "x": 3600,
                    "y": 3600,
                }
            },
        },
        "uri": Path("meteo/chelsa_clim_v1.2/CHELSA_bio10_12.tif"),
    }

    entry = DataCatalogV1Item.from_dict(d, name="chelsa_v1.2")
    assert entry.metadata is not None
    assert entry.driver is not None

    assert entry.name == "chelsa_v1.2"
    assert entry.metadata.crs == 4326
    assert entry.data_type == "RasterDataset"
    assert entry.driver.name == "raster"
    assert entry.uri == Path("meteo/chelsa_clim_v1.2/CHELSA_bio10_12.tif")
    assert entry.driver.options.model_dump() == {"chunks": {"x": 3600, "y": 3600}}
    assert entry.metadata is not None
    assert entry.metadata.category == "meteo"
    assert entry.metadata.paper_doi == "10.1038/sdata.2017.122"
    assert entry.metadata.paper_ref == "Karger et al. (2017)"
    assert entry.metadata.source_license == "CC BY 4.0"
    assert entry.metadata.source_url == "https://chelsa-climate.org/downloads/"
    assert entry.metadata.source_version == "1.2"


def test_dataset_v1_entry_with_typo_validation():
    d = {
        "meta_data": {
            "crs_num": 4326,
        },
        "datatype": "RasterDataset",
        "driver": {
            "name": "raster",
            "kw_options": {
                "chunks": {
                    "x": 3600,
                    "y": 3600,
                }
            },
        },
        "path": "meteo/chelsa_clim_v1.2/CHELSA_bio10_12.tif",
    }

    # 4 errors are:
    #  - missing data_type field
    #  - unknown extra field meta_data (should be metadata)
    #  - extra field datatype (should be data_type)
    #  - extra field in crs_num (should be crs)
    with pytest.raises(ValidationError, match="4 validation errors"):
        _ = DataCatalogV1Item.from_dict(d, name="chelsa_v1.2")


def test_data_type_v1_typo_data_type():
    d = {
        "metadata": {
            "crs": 4326,
        },
        "data_type": "RaserDataset",
        "driver": {"name": "raster"},
        "uri": Path("."),
    }
    with pytest.raises(ValidationError, match="1 validation error"):
        _ = DataCatalogV1Item.from_dict(d, name="chelsa_v1.2")


def test_data_invalid_crs_v1():
    d = {
        "metadata": {
            "crs": 123456789,
        },
        "data_type": "RasterDataset",
        "uri": Path("."),
    }
    with pytest.raises(ValidationError, match="validation error for chelsa_v1.2"):
        _ = DataCatalogV1Item.from_dict(d, name="chelsa_v1.2")


def test_upgrade_v0_data_catalog():
    expected_upgraded_data_catalog = DataCatalogV1Validator.from_yml(
        Path(TEST_DATA_DIR) / "test_v0_data_catalog_upgraded.yml"
    )
    v0_catalog = DataCatalogV0Validator.from_yml(
        Path(TEST_DATA_DIR) / "test_v0_data_catalog.yml"
    )

    upgraded_catalog = DataCatalogV1Validator.from_v0(v0_catalog)

    assert upgraded_catalog == expected_upgraded_data_catalog
