"""Tests for the cli submodule."""

import logging
from os.path import join
from pathlib import Path
from typing import Generator

import pytest
from click.testing import CliRunner

from hydromt import __version__
from hydromt._typing import NoDataException
from hydromt.cli.main import main as hydromt_cli
from hydromt.model.components.grid import GridComponent
from tests.conftest import TEST_DATA_DIR

BUILD_CONFIG_PATH = join(TEST_DATA_DIR, "build_config.yml")
UPDATE_CONFIG_PATH = join(TEST_DATA_DIR, "update_config.yml")


def test_cli_version():
    r = CliRunner().invoke(hydromt_cli, "--version")
    assert r.exit_code == 0
    assert r.output.split()[-1] == __version__


def test_cli_models():
    r = CliRunner().invoke(hydromt_cli, "--models")
    assert r.exit_code == 0
    assert "Model plugins" in r.output
    assert "Model" in r.output


def test_cli_components():
    r = CliRunner().invoke(hydromt_cli, "--components")
    assert r.exit_code == 0
    assert "Component plugins" in r.output
    assert GridComponent.__name__ in r.output


def test_cli_help():
    r = CliRunner().invoke(hydromt_cli, "--help")
    assert r.exit_code == 0
    # NOTE: when called from CliRunner we get "Usage: main" instead of "Usage: hydromt"
    assert r.output.startswith("Usage: main [OPTIONS] COMMAND [ARGS]...")


def test_cli_build_help():
    r = CliRunner().invoke(hydromt_cli, ["build", "--help"])
    assert r.exit_code == 0
    assert r.output.startswith("Usage: main build [OPTIONS] MODEL MODEL_ROOT")


def test_cli_update_help():
    r = CliRunner().invoke(hydromt_cli, ["update", "--help"])
    assert r.exit_code == 0
    assert r.output.startswith("Usage: main update [OPTIONS] MODEL MODEL_ROOT")


@pytest.fixture
def _reset_log_level() -> Generator[None, None, None]:
    yield
    main_logger = logging.getLogger("hydromt")
    main_logger.setLevel(logging.NOTSET)  # Most verbose so all messages get passed


def test_cli_build_missing_arg_workflow(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    cmd = [
        "build",
        "model",
        str(tmp_path),
        "-i",
        str(join(TEST_DATA_DIR, "missing_data_workflow.yml")),
        "-vv",
    ]
    with caplog.at_level(logging.NOTSET):
        r = CliRunner().invoke(hydromt_cli, cmd)

    assert r.exit_code == 1
    assert (
        "Validation of step 1 (config.update) failed because of the following error:"
        in caplog.text
    )


def test_cli_build_v0x_workflow(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    cmd = [
        "build",
        "model",
        str(tmp_path),
        "-i",
        str(join(TEST_DATA_DIR, "v0x_workflow.yml")),
        "-vv",
    ]
    with caplog.at_level(logging.NOTSET):
        r = CliRunner().invoke(hydromt_cli, cmd)

    assert r.exit_code == 1
    assert (
        "does not contain a `steps` section. Perhaps you're using a v0.x format?"
        in caplog.text
    )


def test_cli_update_missing_arg(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    cmd = [
        "update",
        "model",
        str(tmp_path),
        "-i",
        str(join(TEST_DATA_DIR, "missing_data_workflow.yml")),
        "-vv",
    ]
    with caplog.at_level(logging.NOTSET):
        r = CliRunner().invoke(hydromt_cli, cmd)

    assert r.exit_code == 1
    assert (
        "Validation of step 1 (config.update) failed because of the following error:"
        in caplog.text
    )


def test_cli_update_v0x_workflow(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    cmd = [
        "update",
        "model",
        str(tmp_path),
        "-i",
        str(join(TEST_DATA_DIR, "v0x_workflow.yml")),
        "-vv",
    ]
    with caplog.at_level(logging.NOTSET):
        r = CliRunner().invoke(hydromt_cli, cmd)

    assert r.exit_code == 1
    assert (
        "does not contain a `steps` section. Perhaps you're using a v0.x format?"
        in caplog.text
    )


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_build_update_model(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    root = tmp_path / "model_region"
    cmd = [
        "build",
        "model",
        str(root),
        "-i",
        BUILD_CONFIG_PATH,
        "-d",
        "artifact_data",
        "-vv",
    ]
    r = CliRunner().invoke(hydromt_cli, cmd)

    assert r.exit_code == 0
    assert Path(root, "run_config.toml").exists()
    # Open and check content
    with open(root / "run_config.toml") as f:
        content = f.read()
    assert "starttime = 2010-01-01" in content
    assert '[model]\ntype = "model"' in content
    assert "endtime " not in content

    # We need to build before we can update
    root_out = tmp_path / "model_region_update"
    cmd = [
        "update",
        "model",
        str(root),
        "-o",
        str(root_out),
        "-i",
        UPDATE_CONFIG_PATH,
        "-vv",
    ]
    r = CliRunner().invoke(hydromt_cli, cmd)
    assert r.exit_code == 0
    assert Path(root_out, "run_config.toml").exists()
    # Open and check content
    with open(root_out / "run_config.toml") as f:
        content = f.read()
    assert "starttime = 2020-01-01" in content
    assert '[model]\ntype = "model"' in content
    assert "endtime " in content


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_build_no_config(tmp_path: Path):
    result = CliRunner().invoke(
        hydromt_cli,
        [
            "build",
            "model",
            str(tmp_path / "model_region"),
            "-d",
            "artifact_data",
            "-vv",
        ],
    )
    assert result.exit_code == 2
    assert "Error: Missing option '-i' / '--config'." in result.output


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_build_unknown_option(tmp_path: Path):
    cmd = [
        "build",
        "model",
        str(tmp_path / "model_region"),
        "--opt",
        "setup_grid.res=0.05",
        "-vv",
    ]
    r = CliRunner().invoke(hydromt_cli, cmd)

    # Check that "Error: No such option: --opt" is in the output
    assert r.exit_code == 2
    assert "Error: No such option: --opt" in r.output


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_build_unknown_model(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown model"):
        _ = CliRunner().invoke(
            hydromt_cli,
            [
                "build",
                "test_model",
                str(tmp_path),
                "-i",
                BUILD_CONFIG_PATH,
            ],
            catch_exceptions=False,
        )


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_update_unknown_model(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown model"):
        _ = CliRunner().invoke(
            hydromt_cli,
            [
                "update",
                "test_model",
                str(tmp_path),
                "-i",
                UPDATE_CONFIG_PATH,
            ],
            catch_exceptions=False,
        )


@pytest.mark.usefixtures("_reset_log_level")
def test_export_cli_deltares_data(tmp_path: Path):
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmp_path),
            "-s",
            "hydro_lakes",
            "--bbox",
            "[12.05,45.30,12.85,45.65]",
            "--dd",
        ],
        catch_exceptions=False,
    )

    assert r.exit_code == 0, r.output


def test_export_cli_no_data_ignore(tmp_path: Path):
    with pytest.raises(NoDataException):
        _ = CliRunner().invoke(
            hydromt_cli,
            [
                "export",
                str(tmp_path),
                "-s",
                "hydro_lakes",
                "--bbox",
                "[1,2,3,4]",
                "--error-on-empty",
            ],
            catch_exceptions=False,
        )


def test_export_skips_overwrite(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        # export twice
        for _i in range(2):
            _ = CliRunner().invoke(
                hydromt_cli,
                [
                    "export",
                    str(tmp_path),
                    "-s",
                    "hydro_lakes",
                ],
                catch_exceptions=False,
            )
    assert "already exists and not in forced overwrite mode" in caplog.text


def test_export_does_not_warn_on_forced_overwrite(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    _ = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmp_path),
            "-s",
            "hydro_lakes",
        ],
        catch_exceptions=False,
    )

    _ = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmp_path),
            "-s",
            "hydro_lakes",
            "--fo",
        ],
        catch_exceptions=False,
    )
    assert "already exists and not in forced overwrite mode" not in caplog.text


@pytest.mark.usefixtures("_reset_log_level")
def test_export_cli_catalog(tmp_path: Path):
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmp_path),
            "-s",
            "hydro_lakes",
            "-d",
            str(join(TEST_DATA_DIR, "test_sources1.yml")),
        ],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output


@pytest.mark.usefixtures("_reset_log_level")
def test_export_multiple_sources(tmp_path: Path):
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmp_path),
            "-s",
            "hydro_lakes",
            "-s",
            "gtsmv3_eu_era5",
            "-d",
            "tests/data/test_sources1.yml",
            "-d",
            "tests/data/test_sources2.yml",
        ],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output


@pytest.mark.usefixtures("_reset_log_level")
def test_export_cli_config_file(tmp_path: Path):
    r = CliRunner().invoke(
        hydromt_cli,
        ["export", str(tmp_path), "-i", "tests/data/export_config.yml"],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output


@pytest.mark.usefixtures("_reset_log_level")
def test_v0_catalog_is_not_valid_v1_catalog(caplog):
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "check",
            "-d",
            "data/catalogs/artifact_data/v0.0.9/data_catalog.yml",
            "--format",
            "v1",
        ],
        catch_exceptions=False,
    )

    assert r.exit_code == 1
    assert "No hydromt version was specified for the data catalog" in caplog.text
    assert "has the following error(s): " in caplog.text


@pytest.mark.usefixtures("_reset_log_level")
def test_validate_v0_catalog():
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "check",
            "-d",
            "data/catalogs/artifact_data/v0.0.9/data_catalog.yml",
            "--format",
            "v0",
        ],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_check_v0x_workflow_format_v0(caplog):
    cmd = [
        "check",
        "-i",
        Path(TEST_DATA_DIR) / "v0x_workflow.yml",
        "-vvv",
        "--format",
        "v0",
    ]

    r = CliRunner().invoke(hydromt_cli, cmd, catch_exceptions=False)

    assert r.exit_code == 1
    assert "v0.x workflow files cannot be validated by hydromt v1." in caplog.text


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_check_v0x_workflow(caplog):
    cmd = [
        "check",
        "-i",
        Path(TEST_DATA_DIR) / "v0x_workflow.yml",
        "-vv",
    ]

    r = CliRunner().invoke(hydromt_cli, cmd, catch_exceptions=False)

    assert r.exit_code == 1
    error_msg = f"It seems your workflow file at {Path(TEST_DATA_DIR) / 'v0x_workflow.yml'} does not contain a `steps` section. Perhaps you're using a v0.x format?"
    assert error_msg.lower() in caplog.text.lower()
