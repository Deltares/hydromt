"""Tests for the cli submodule."""

import pytest
from click.testing import CliRunner
import numpy as np
import os
from hydromt import __version__
from hydromt.cli.main import main as hydromt_cli
from hydromt.cli import api as hydromt_api


def test_cli(tmpdir):
    r = CliRunner().invoke(hydromt_cli, "--version")
    assert r.exit_code == 0
    assert r.output.split()[-1] == __version__

    r = CliRunner().invoke(hydromt_cli, "--models")
    assert r.exit_code == 0
    assert r.output.startswith("model plugins")

    r = CliRunner().invoke(hydromt_cli, "--help")
    assert r.exit_code == 0
    # NOTE: when called from CliRunner we get "Usage: main" instead of "Usage: hydromt"
    assert r.output.startswith("Usage: main [OPTIONS] COMMAND [ARGS]...")

    r = CliRunner().invoke(hydromt_cli, ["build", "--help"])
    assert r.exit_code == 0
    assert r.output.startswith("Usage: main build [OPTIONS] MODEL MODEL_ROOT")

    r = CliRunner().invoke(hydromt_cli, ["update", "--help"])
    assert r.exit_code == 0
    assert r.output.startswith("Usage: main update [OPTIONS] MODEL MODEL_ROOT")

    r = CliRunner().invoke(hydromt_cli, ["clip", "--help"])
    assert r.exit_code == 0
    assert r.output.startswith(
        "Usage: main clip [OPTIONS] MODEL MODEL_ROOT MODEL_DESTINATION REGION"
    )

    r = CliRunner().invoke(
        hydromt_cli,
        ["build", "model", str(tmpdir), "{'subbasin': [-7.24, 62.09], 'strord': 4}"],
    )
    with pytest.raises(ValueError, match="Unknown model"):
        raise r.exception

    r = CliRunner().invoke(
        hydromt_cli,
        ["update", "test_model", str(tmpdir), "-c", "component", "--opt", "key=value"],
    )
    with pytest.raises(ValueError, match="Unknown model"):
        raise r.exception

    r = CliRunner().invoke(
        hydromt_cli,
        ["clip", "test_model", str(tmpdir), str(tmpdir), "{'bbox': [1,2,3,4]}"],
    )
    with pytest.raises(NotImplementedError):
        raise r.exception


def test_api_datasets():
    # datasets
    assert "artifact_data" in hydromt_api.get_predifined_catalogs()
    datasets = hydromt_api.get_datasets("artifact_data")
    assert isinstance(datasets, dict)
    assert isinstance(datasets["RasterDatasetSource"], list)


def test_api_model_components():
    # models
    components = hydromt_api.get_model_components(
        "lumped_model", component_types=["write"]
    )
    name = "write_response_units"
    assert name in components
    assert np.all([k.startswith("write") for k in components])
    keys = ["doc", "required", "optional", "kwargs"]
    assert np.all([k in components[name] for k in keys])
