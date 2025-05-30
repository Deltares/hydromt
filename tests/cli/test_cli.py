"""Tests for the cli submodule."""

from logging import NOTSET, WARNING, Logger, getLogger
from os.path import join
from pathlib import Path
from typing import Generator

import pytest
from click.testing import CliRunner, Result

from hydromt import __version__
from hydromt._typing import NoDataException
from hydromt.cli.main import main as hydromt_cli
from hydromt.model.components.grid import GridComponent
from tests.conftest import TEST_DATA_DIR


def test_cli_verison():
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
    main_logger: Logger = getLogger("hydromt")
    main_logger.setLevel(NOTSET)  # Most verbose so all messages get passed


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_build_grid_model(tmpdir):
    root = str(tmpdir.join("grid_model_region"))
    cmd = [
        "build",
        "grid_model",
        root,
        "--opt",
        "setup_grid.res=0.05",
        "-vv",
    ]
    _ = CliRunner().invoke(hydromt_cli, cmd)


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_build_unknown_model(tmpdir):
    with pytest.raises(ValueError, match="Unknown model"):
        _ = CliRunner().invoke(
            hydromt_cli,
            [
                "build",
                "test_model",
                str(tmpdir),
            ],
            catch_exceptions=False,
        )


@pytest.mark.usefixtures("_reset_log_level")
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


@pytest.mark.usefixtures("_reset_log_level")
def test_export_cli_deltares_data(tmpdir):
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmpdir),
            "-s",
            "hydro_lakes",
            "--bbox",
            "[12.05,45.30,12.85,45.65]",
            "--dd",
        ],
        catch_exceptions=False,
    )

    assert r.exit_code == 0, r.output


def test_export_cli_no_data_ignore(tmpdir):
    with pytest.raises(NoDataException):
        _ = CliRunner().invoke(
            hydromt_cli,
            [
                "export",
                str(tmpdir),
                "-s",
                "hydro_lakes",
                "--bbox",
                "[1,2,3,4]",
                "--error-on-empty",
            ],
            catch_exceptions=False,
        )


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_build_override(tmpdir):
    root = str(tmpdir.join("grid_model_region"))
    cmd = [
        "build",
        "test_model",
        root,
        "-i",
        "tests/data/test_model_config.yml",
        "-d",
        "artifact_data",
        "-d",
        Path(__name__).absolute().parents[0]
        / "examples"
        / "data"
        / "vito_reclass.yml",  # for reclass data
    ]
    res: Result = CliRunner().invoke(hydromt_cli, cmd)
    assert not res.exception

    # test force overwrite
    with pytest.raises(IOError, match="File.*already exists"):
        CliRunner().invoke(hydromt_cli, cmd, catch_exceptions=False)

    r = CliRunner().invoke(hydromt_cli, cmd + ["--fo"])
    assert r.exit_code == 0


def test_export_skips_overwrite(tmpdir, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(WARNING):
        # export twice
        for _i in range(2):
            _ = CliRunner().invoke(
                hydromt_cli,
                [
                    "export",
                    str(tmpdir),
                    "-s",
                    "hydro_lakes",
                ],
                catch_exceptions=False,
            )
    assert "already exists and not in forced overwrite mode" in caplog.text


def test_export_does_not_warn_on_fo(tmpdir, caplog):
    _ = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmpdir),
            "-s",
            "hydro_lakes",
        ],
        catch_exceptions=False,
    )

    _ = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmpdir),
            "-s",
            "hydro_lakes",
            "--fo",
        ],
        catch_exceptions=False,
    )
    assert "already exists and not in forced overwrite mode" not in caplog.text


@pytest.mark.usefixtures("_reset_log_level")
def test_export_cli_catalog(tmpdir):
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmpdir),
            "-s",
            "hydro_lakes",
            "-d",
            join(TEST_DATA_DIR, "test_sources1.yml"),
        ],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output


@pytest.mark.usefixtures("_reset_log_level")
def test_export_multiple_sources(tmpdir):
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmpdir),
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
def test_export_cli_config_file(tmpdir):
    r = CliRunner().invoke(
        hydromt_cli,
        ["export", str(tmpdir), "-i", "tests/data/export_config.yml"],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output


@pytest.mark.skip("Needs validator overhaul")
def test_check_cli():
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "check",
            "-d",
            "tests/data/test_sources1.yml",
            "-i",
            "tests/data/test_model_config.yml",
        ],
    )
    assert r.exit_code == 0, r.output
