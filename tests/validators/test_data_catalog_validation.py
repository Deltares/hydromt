"""Testing of the Pydantic models for validation of Data catalogs."""


from pathlib import Path

import pytest
from pydantic_core import Url

from hydromt.validators.data_catalog import (
    DataCatalogItem,
    DataCatalogMetaData,
    DataCatalogValidator,
)


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


def test_valid_catalog_with_alias():
    d = {
        "chelsa": {"alias": "chelsa_v1.2"},
        "chelsa_v1.2": {
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
                "source_url": "http://chelsa-climate.org/downloads/",
                "source_version": "1.2",
            },
            "path": "meteo/chelsa_clim_v1.2/CHELSA_bio10_12.tif",
        },
    }
    _ = DataCatalogValidator.from_dict(d)


def test_dangling_alias_catalog_entry():
    d = {
        "chelsa": {"alias": "chelsa_v1.2"},
    }

    with pytest.raises(AssertionError):
        _ = DataCatalogValidator.from_dict(d)


def test_valid_alias_catalog_entry():
    d = {
        "chelsa": {"alias": "chelsa_v1.2"},
        "chelsa_v1.2": {
            "crs": 4326,
            "data_type": "RasterDataset",
            "driver": "raster",
            "path": ".",
        },
    }
    entry = DataCatalogValidator.from_dict(d)

    assert entry.aliases == {"chelsa": "chelsa_v1.2"}


def test_catalog_metadata_validation():
    d = {
        "root": "p:/wflow_global/hydromt",
        "version": "2023.3",
    }
    catalog_metadata = DataCatalogMetaData.from_dict(d)
    assert catalog_metadata.root == Path("p:/wflow_global/hydromt")
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
            "source_url": "http://chelsa-climate.org/downloads/",
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
    assert entry.meta.source_url == Url("http://chelsa-climate.org/downloads/")
    assert entry.meta.source_version == "1.2"
