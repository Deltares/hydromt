import numpy as np
import pytest

import hydromt._compat as compat
from hydromt.cli.api import (
    get_datasets,
    get_model_components,
    get_predifined_catalogs,
)


@pytest.mark.skip(reason="Needs implementation of RasterDataSet.")
def test_api_datasets():
    # datasets
    assert "artifact_data" in get_predifined_catalogs()
    datasets = get_datasets("artifact_data")
    assert isinstance(datasets, dict)
    assert isinstance(datasets["RasterDatasetSource"], list)


def test_api_model_components():
    # models
    components = get_model_components("vector_model", component_types=["write"])
    name = "write_vector"
    assert name in components
    assert np.all([k.startswith("write") for k in components])
    keys = ["doc", "required", "optional", "kwargs"]
    assert np.all([k in components[name] for k in keys])
    if compat.HAS_XUGRID:
        components = get_model_components("mesh_model")
