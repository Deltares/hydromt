"""Tests for the cli submodule."""

import logging
import shutil
from datetime import date
from pathlib import Path
from textwrap import dedent
from typing import Generator

import pytest
import yaml
from click.testing import CliRunner

from hydromt import __version__
from hydromt.cli.main import DataCatalogV1Validator
from hydromt.cli.main import main as hydromt_cli
from hydromt.error import NoDataException
from hydromt.model.components.grid import GridComponent
from hydromt.readers import read_toml

LEGACY_V1_CATALOG = dedent("""\
    cmip6_historical:
      data_type: RasterDataset
      uri: gs://cmip6/CMIP6/CMIP/MPI-ESM1-2-HR/historical/r1i1p1f1/day/tas/*/*
      driver:
        name: raster_xarray
        filesystem:
          protocol: gcs
    """)

CURRENT_V1_CATALOG = dedent("""\
    cmip6_historical:
      data_type: RasterDataset
      uri: gs://cmip6/CMIP6/CMIP/MPI-ESM1-2-HR/historical/r1i1p1f1/day/tas/*/*
      driver:
        name: raster_xarray
      filesystem:
        protocol: gcs
    """)

CONFLICTING_V1_CATALOG = dedent("""\
    cmip6_historical:
      data_type: RasterDataset
      uri: gs://cmip6/data/*
      driver:
        name: raster_xarray
        filesystem: {protocol: gcs}
      uri_resolver:
        name: convention
        filesystem: {protocol: s3}
    """)


def _check(*args: str):
    return CliRunner().invoke(hydromt_cli, ["check", *args], catch_exceptions=False)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "catalog.yml"
    path.write_text(text)
    return path


def _upgraded(path: Path) -> Path:
    return path.with_stem(f"{path.stem}_upgraded")


@pytest.fixture(scope="session")
def build_config_path(test_data_dir: Path) -> Path:
    path = test_data_dir / "build_config.yml"
    if not path.exists():
        raise FileNotFoundError(f"Test build config file not found at {path}")
    return path


@pytest.fixture(scope="session")
def update_config_path(test_data_dir: Path) -> Path:
    path = test_data_dir / "update_config.yml"
    if not path.exists():
        raise FileNotFoundError(f"Test update config file not found at {path}")
    return path


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
    tmp_path: Path, test_data_dir: Path, caplog: pytest.LogCaptureFixture
):
    cmd = [
        "build",
        "model",
        str(tmp_path),
        "-i",
        str(test_data_dir / "missing_data_workflow.yml"),
        "-vv",
    ]
    with caplog.at_level(logging.NOTSET):
        r = CliRunner().invoke(hydromt_cli, cmd)

    assert r.exit_code == 1
    assert (
        "Validation of step 1 (config.update) failed because of the following error:"
        in caplog.text
    )


def test_cli_build_v0x_workflow(
    tmp_path: Path, test_data_dir: Path, caplog: pytest.LogCaptureFixture
):
    cmd = [
        "build",
        "model",
        str(tmp_path),
        "-i",
        str(test_data_dir / "v0x_workflow.yml"),
        "-vv",
    ]
    with caplog.at_level(logging.NOTSET):
        r = CliRunner().invoke(hydromt_cli, cmd)

    assert r.exit_code == 1
    assert (
        "does not contain a `steps` section. Perhaps you're using a v0.x format?"
        in caplog.text
    )


def test_cli_update_missing_arg(
    tmp_path: Path, test_data_dir: Path, caplog: pytest.LogCaptureFixture
):
    cmd = [
        "update",
        "model",
        str(tmp_path),
        "-i",
        str(test_data_dir / "missing_data_workflow.yml"),
        "-vv",
    ]
    with caplog.at_level(logging.NOTSET):
        r = CliRunner().invoke(hydromt_cli, cmd)

    assert r.exit_code == 1
    assert (
        "Validation of step 1 (config.update) failed because of the following error:"
        in caplog.text
    )


def test_cli_update_v0x_workflow(
    tmp_path: Path, test_data_dir: Path, caplog: pytest.LogCaptureFixture
):
    cmd = [
        "update",
        "model",
        str(tmp_path),
        "-i",
        str(test_data_dir / "v0x_workflow.yml"),
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
def test_cli_build_update_model(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    build_config_path: Path,
    update_config_path: Path,
):
    root = tmp_path / "model_region"
    cmd = [
        "build",
        "model",
        str(root),
        "-i",
        build_config_path.as_posix(),
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
        update_config_path.as_posix(),
        "-vv",
    ]
    r = CliRunner().invoke(hydromt_cli, cmd)
    assert r.exit_code == 0
    assert Path(root_out, "run_config.toml").exists()

    # Open and check content
    content = read_toml(root_out / "run_config.toml")
    assert content["starttime"] == date(year=2020, month=1, day=1)
    assert content["model"]["type"] == "model"
    assert "endtime" in content


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

    # Check that an error about '--opt' is raised.
    # (error string changes between click versions, so we check for parts)
    assert r.exit_code == 2
    assert "Error: No such option" in r.output
    assert "--opt" in r.output


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_build_unknown_model(tmp_path: Path, build_config_path: Path):
    with pytest.raises(ValueError, match="Unknown model"):
        _ = CliRunner().invoke(
            hydromt_cli,
            [
                "build",
                "test_model",
                str(tmp_path),
                "-i",
                build_config_path.as_posix(),
            ],
            catch_exceptions=False,
        )


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_update_unknown_model(tmp_path: Path, update_config_path: Path):
    with pytest.raises(ValueError, match="Unknown model"):
        _ = CliRunner().invoke(
            hydromt_cli,
            [
                "update",
                "test_model",
                str(tmp_path),
                "-i",
                update_config_path.as_posix(),
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
                "-d",
                "artifact_data",
            ],
            catch_exceptions=False,
        )


def test_export_skips_overwrite(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        # export twice
        for _ in range(2):
            _ = CliRunner().invoke(
                hydromt_cli,
                [
                    "export",
                    str(tmp_path),
                    "-s",
                    "hydro_lakes",
                    "-d",
                    "artifact_data",
                ],
                catch_exceptions=False,
            )
    assert "skipping export" in caplog.text


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
            "-d",
            "artifact_data",
        ],
        catch_exceptions=False,
    )

    _ = CliRunner().invoke(
        hydromt_cli,
        ["export", str(tmp_path), "-s", "hydro_lakes", "--fo", "-d", "artifact_data"],
        catch_exceptions=False,
    )
    assert "already exists and not in forced overwrite mode" not in caplog.text


@pytest.mark.usefixtures("_reset_log_level")
def test_export_cli_catalog(tmp_path: Path, test_data_dir: Path):
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmp_path),
            "-s",
            "hydro_lakes",
            "-d",
            str(test_data_dir / "test_sources1.yml"),
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
    r = _check(
        "-d", "data/catalogs/artifact_data/v0.0.9/data_catalog.yml", "--format", "v1"
    )

    assert r.exit_code == 1
    assert "outdated layout" in caplog.text
    assert "--upgrade" in caplog.text


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
def test_cli_check_v0x_workflow_format_v0(caplog, test_data_dir: Path):
    cmd = [
        "check",
        "-i",
        str(test_data_dir / "v0x_workflow.yml"),
        "-vvv",
        "--format",
        "v0",
    ]

    r = CliRunner().invoke(hydromt_cli, cmd, catch_exceptions=False)

    assert r.exit_code == 1
    assert "v0.x workflow files cannot be validated by hydromt v1." in caplog.text


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_check_v0x_workflow(caplog, test_data_dir: Path):
    cmd = [
        "check",
        "-i",
        str(test_data_dir / "v0x_workflow.yml"),
        "-vv",
    ]

    r = CliRunner().invoke(hydromt_cli, cmd, catch_exceptions=False)

    assert r.exit_code == 1
    error_msg = f"It seems your workflow file at {test_data_dir / 'v0x_workflow.yml'} does not contain a `steps` section. Perhaps you're using a v0.x format?"
    assert error_msg.lower() in caplog.text.lower()


@pytest.mark.usefixtures("_reset_log_level")
@pytest.mark.parametrize(
    "extra", [["--format", "v0"], []], ids=["format-v0", "default-format"]
)
def test_cli_upgrade_datacatalog(test_data_dir: Path, tmp_path: Path, extra):
    target = tmp_path / "test_v0_data_catalog.yml"
    shutil.copy(test_data_dir / "test_v0_data_catalog.yml", target)

    r = _check("-d", str(target), *extra, "--upgrade")

    assert r.exit_code == 0, r.output
    out_path = _upgraded(target)  # was f"{stem}_v1" before
    assert out_path.exists()
    expected = DataCatalogV1Validator.from_yml(
        test_data_dir / "test_v0_data_catalog_upgraded.yml"
    )
    assert DataCatalogV1Validator.from_yml(out_path) == expected


@pytest.mark.usefixtures("_reset_log_level")
def test_invalid_v1_catalog_still_reports_validation_errors(tmp_path, caplog):
    path = _write(tmp_path, "src:\n  data_type: RaserDataset\n  uri: x.tif\n")

    r = _check("-d", str(path))

    assert r.exit_code == 1
    assert "has the following error(s)" in caplog.text


@pytest.mark.usefixtures("_reset_log_level")
def test_legacy_v1_layout_passes_with_warning_and_writes_nothing(tmp_path, caplog):
    path = _write(tmp_path, LEGACY_V1_CATALOG)

    r = _check("-d", str(path))

    assert r.exit_code == 0, r.output
    assert "moved to the source root" in caplog.text
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_upgrade_v1_datacatalog(tmp_path):
    path = _write(tmp_path, LEGACY_V1_CATALOG)

    r = _check("-d", str(path), "--upgrade")

    assert r.exit_code == 0, r.output
    assert path.read_text() == LEGACY_V1_CATALOG  # original untouched
    out = _upgraded(path)
    upgraded = yaml.safe_load(out.read_text())
    source = upgraded["cmip6_historical"]
    assert source["filesystem"] == {"protocol": "gcs"}
    assert source["driver"] == {"name": "raster_xarray"}
    assert upgraded["meta"]["hydromt_version"].startswith(">=1.5")
    DataCatalogV1Validator.from_yml(out)  # and it is valid


@pytest.mark.usefixtures("_reset_log_level")
def test_upgraded_catalog_has_no_pending_migrations(tmp_path, caplog):
    path = _write(tmp_path, LEGACY_V1_CATALOG)
    _check("-d", str(path), "--upgrade")
    caplog.clear()

    r = _check("-d", str(_upgraded(path)))

    assert r.exit_code == 0, r.output
    assert "moved to the source root" not in caplog.text


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_upgrade_is_noop_for_current_catalog(tmp_path):
    path = _write(tmp_path, CURRENT_V1_CATALOG)

    r = _check("-d", str(path), "--upgrade")

    assert r.exit_code == 0, r.output
    assert list(tmp_path.iterdir()) == [path]  # nothing written


@pytest.mark.usefixtures("_reset_log_level")
def test_cli_fails_on_unmigratable_catalog(tmp_path, caplog):
    path = _write(tmp_path, CONFLICTING_V1_CATALOG)

    r = _check("-d", str(path), "--upgrade")

    assert r.exit_code == 1
    assert "can't be upgraded automatically" in caplog.text
    assert "cmip6_historical" in caplog.text  # names the offending source
    assert not _upgraded(path).exists()
