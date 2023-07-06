import json

import numpy as np
import pytest

import hydromt._compat as compat
from hydromt.cli.api import (
    get_datasets,
    get_model_components,
    get_predifined_catalogs,
    get_region,
)


def test_get_region():
    region = {"basin": [12.111195, 46.088866]}  # Piave basin
    region_geom = get_region(region=region)
    assert isinstance(region_geom, str)
    try:
        region_json = json.loads(region_geom)
    except ValueError:
        raise ValueError("Returned region is not in valid json")

    assert region_json["type"] == "FeatureCollection"
    assert len(region_json["features"]) == 1
    assert region_json["features"][0]["properties"]["area"] != 0

    with pytest.raises(
        ValueError,
        match="Only basin, subbasin, and interbasin are accepted region definitions",
    ):
        get_region(region={"bbox": [4.416504, 51.597548, 6.256714, 52.281602]})


def test_api_datasets():
    # datasets
    assert "artifact_data" in get_predifined_catalogs()
    datasets = get_datasets("artifact_data")
    assert isinstance(datasets, dict)
    assert isinstance(datasets["RasterDatasetSource"], list)


def test_api_model_components():
    # models
    components = get_model_components("lumped_model", component_types=["write"])
    name = "write_response_units"
    assert name in components
    assert np.all([k.startswith("write") for k in components])
    keys = ["doc", "required", "optional", "kwargs"]
    assert np.all([k in components[name] for k in keys])
    if compat.HAS_XUGRID:
        components = get_model_components("mesh_model")
