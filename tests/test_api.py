import pytest
import json
import numpy as np

from hydromt.cli.api import *


def test_get_region():
    data_libs = "deltares_data"
    region = {"basin": [6.338331593204473, 52.521107365953334]}  # vecht basin
    region_geom = get_region(region=region, data_libs=data_libs)
    assert isinstance(region_geom, str)
    try:
        region_json = json.loads(region_geom)
    except:
        raise ValueError("Returned region is not in valid json")
    assert region_json["type"] == "FeatureCollection"
    assert len(region_json["features"]) == 1
    assert region_json["features"][0]["properties"]["area"] != 0

    with pytest.raises(
        ValueError,
        match="Only basin, subbasin, and interbasin are accepted region definitions",
    ):
        get_region(
            region={"bbox": [4.416504, 51.597548, 6.256714, 52.281602]},
            data_libs=data_libs,
        )


def test_api_datasets():
    # datasets
    assert "artifact_data" in get_predifined_catalogs()
    datasets = get_datasets("artifact_data")
    assert isinstance(datasets, dict)
    assert isinstance(datasets["RasterDatasetSource"], list)


@pytest.mark.skipif("lumped_model" not in ENTRYPOINTS, reason="HydroMT not installed.")
def test_api_model_components():
    # models
    components = get_model_components("lumped_model", component_types=["write"])
    name = "write_response_units"
    assert name in components
    assert np.all([k.startswith("write") for k in components])
    keys = ["doc", "required", "optional", "kwargs"]
    assert np.all([k in components[name] for k in keys])
    components = get_model_components("mesh_model")
