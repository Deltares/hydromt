"""Tests for the cli submodule."""


import numpy as np
import pytest
from click.testing import CliRunner

from hydromt import __version__
from hydromt.cli import api as hydromt_api
from hydromt.cli.main import main as hydromt_cli


def test_cli_verison(tmpdir):
    r = CliRunner().invoke(hydromt_cli, "--version")
    assert r.exit_code == 0
    assert r.output.split()[-1] == __version__


def test_cli_models(tmpdir):
    r = CliRunner().invoke(hydromt_cli, "--models")
    assert r.exit_code == 0
    assert r.output.startswith("model plugins")


def test_cli_help(tmpdir):
    r = CliRunner().invoke(hydromt_cli, "--help")
    assert r.exit_code == 0
    # NOTE: when called from CliRunner we get "Usage: main" instead of "Usage: hydromt"
    assert r.output.startswith("Usage: main [OPTIONS] COMMAND [ARGS]...")


def test_cli_build_help(tmpdir):
    r = CliRunner().invoke(hydromt_cli, ["build", "--help"])
    assert r.exit_code == 0
    assert r.output.startswith("Usage: main build [OPTIONS] MODEL MODEL_ROOT")


def test_cli_update_help(tmpdir):
    r = CliRunner().invoke(hydromt_cli, ["update", "--help"])
    assert r.exit_code == 0
    assert r.output.startswith("Usage: main update [OPTIONS] MODEL MODEL_ROOT")


def test_cli_clip_help(tmpdir):
    r = CliRunner().invoke(hydromt_cli, ["clip", "--help"])
    assert r.exit_code == 0
    assert r.output.startswith(
        "Usage: main clip [OPTIONS] MODEL MODEL_ROOT MODEL_DESTINATION REGION"
    )


def test_cli_build_grid_model(tmpdir):
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
    _ = CliRunner().invoke(hydromt_cli, cmd)

    # test force overwrite
    with pytest.raises(IOError, match="Model dir already exists"):
        _ = CliRunner().invoke(hydromt_cli, cmd, catch_exceptions=False)

    r = CliRunner().invoke(hydromt_cli, cmd + ["--fo"])
    assert r.exit_code == 0


def test_cli_build_unknown_model(tmpdir):
    with pytest.raises(ValueError, match="Unknown model"):
        _ = CliRunner().invoke(
            hydromt_cli,
            [
                "build",
                "test_model",
                str(tmpdir),
                "-r",
                "{'subbasin': [-7.24, 62.09], 'strord': 4}",
            ],
            catch_exceptions=False,
        )


def test_cli_update_unknown_model(tmpdir):
    with pytest.raises(ValueError, match="Unknown model"):
        _ = CliRunner().invoke(
            hydromt_cli,
            [
                "update",
                "test_model",
                str(tmpdir),
                "-c",
                "component",
                "--opt",
                "key=value",
            ],
            catch_exceptions=False,
        )


def test_cli_clip_unknown_model(tmpdir):
    with pytest.raises(NotImplementedError):
        _ = CliRunner().invoke(
            hydromt_cli,
            ["clip", "test_model", str(tmpdir), str(tmpdir), "{'bbox': [1,2,3,4]}"],
            catch_exceptions=False,
        )


def test_export_cli_deltared_data(tmpdir):
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmpdir),
            "-t",
            "river_atlas_v10",
            "-r",
            "{'subbasin': [-7.24, 62.09], 'uparea': 50}",
            "--dd",
        ],
    )

    assert r.exit_code == 0, r.output


def test_export_cli_catalog(tmpdir):
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmpdir),
            "-t",
            "river_atlas_v10",
            "-d",
            "tests/data/test_sources.yml",
        ],
    )
    assert r.exit_code == 0, r.output


def test_export_cli_config_file(tmpdir):
    r = CliRunner().invoke(
        hydromt_cli,
        ["export", ".", "-f", "tests/data/export_config.yml"],
    )
    assert r.exit_code == 0, r.output


def test_api_datasets():
    # datasets
    assert "artifact_data" in hydromt_api.get_predifined_catalogs()
    datasets = hydromt_api.get_datasets("artifact_data")
    assert isinstance(datasets, dict)
    assert isinstance(datasets["RasterDatasetSource"], list)


def test_api_model_components():
    # models
    components = hydromt_api.get_model_components(
        "vector_model", component_types=["write"]
    )
    name = "write_vector"
    assert name in components
    assert np.all([k.startswith("write") for k in components])
    keys = ["doc", "required", "optional", "kwargs"]
    assert np.all([k in components[name] for k in keys])
