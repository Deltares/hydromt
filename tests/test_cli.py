"""Tests for the cli submodule."""

import os

import numpy as np
import pytest
from click.testing import CliRunner

from hydromt import __version__
from hydromt.cli import api as hydromt_api
from hydromt.cli.main import main as hydromt_cli


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

    root = str(tmpdir.join("grid_model_region"))
    cmd = [
        "build",
        "grid_model",
        root,
        "-r",
        "{'bbox': [12.05,45.30,12.85,45.65]}",
        "--opt",
        "setup_grid.res=0.05",
        "-vv",
    ]
    r = CliRunner().invoke(hydromt_cli, cmd)
    assert os.path.isfile(os.path.join(root, "geoms", "region.geojson"))

    # test force overwrite
    with pytest.raises(IOError, match="Model dir already exists"):
        r = CliRunner().invoke(hydromt_cli, cmd)
        raise r.exception
    r = CliRunner().invoke(hydromt_cli, cmd + ["--fo"])
    assert r.exit_code == 0

    root = str(tmpdir.join("empty_region"))
    r = CliRunner().invoke(
        hydromt_cli,
        ["build", "grid_model", root, "-vv"],
    )
    assert r.exit_code == 0

    r = CliRunner().invoke(
        hydromt_cli,
        [
            "build",
            "test_model",
            str(tmpdir),
            "-r",
            "{'subbasin': [-7.24, 62.09], 'strord': 4}",
        ],
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
