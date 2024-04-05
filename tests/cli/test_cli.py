"""Tests for the cli submodule."""

from os.path import abspath, dirname, join

import pytest
from click.testing import CliRunner

from hydromt import __version__
from hydromt._typing import NoDataException
from hydromt.cli.main import main as hydromt_cli
from hydromt.components.region import ModelRegionComponent

DATADIR = join(dirname(abspath(__file__)), "..", "data")


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
    assert ModelRegionComponent.__name__ in r.output


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


def test_cli_clip_help():
    r = CliRunner().invoke(hydromt_cli, ["clip", "--help"])
    assert r.exit_code == 0
    assert r.output.startswith(
        "Usage: main clip [OPTIONS] MODEL MODEL_ROOT MODEL_DESTINATION REGION"
    )


@pytest.mark.skip(reason="GridModel has been removed")
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


@pytest.mark.skip(
    "needs implementaion of components to tell what is allowed to be overwritten"
)
def test_cli_build_override(tmpdir):
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


@pytest.mark.skip(reason="needs translation to new entrypoint structure")
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


@pytest.mark.skip(reason="needs translation to new entrypoint structure")
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


@pytest.mark.skip(reason="Needs refactoring from path to uri.")
def test_export_cli_deltares_data(tmpdir):
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmpdir),
            "-s",
            "hydro_lakes",
            "-r",
            "{'bbox': [12.05,45.30,12.85,45.65]}",
            "--dd",
        ],
        catch_exceptions=False,
    )

    assert r.exit_code == 0, r.output


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_export_cli_no_data_ignore(tmpdir):
    with pytest.raises(NoDataException):
        _ = CliRunner().invoke(
            hydromt_cli,
            [
                "export",
                str(tmpdir),
                "-s",
                "hydro_lakes",
                "-r",
                "{'bbox': [12.05,12.06,12.07,12.08]}",
                "--error-on-empty",
            ],
            catch_exceptions=False,
        )


@pytest.mark.skip(reason="Needs refactoring from path to uri.")
def test_export_cli_unsupported_region(tmpdir):
    with pytest.raises(NotImplementedError):
        _ = CliRunner().invoke(
            hydromt_cli,
            [
                "export",
                str(tmpdir),
                "-s",
                "hydro_lakes",
                "-r",
                "{'subbasin': [-7.24, 62.09], 'uparea': 50}",
                "--dd",
            ],
            catch_exceptions=False,
        )


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_export_cli_catalog(tmpdir):
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmpdir),
            "-s",
            "hydro_lakes",
            "-d",
            join(DATADIR, "test_sources.yml"),
        ],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_export_time_tuple(tmpdir):
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "export",
            str(tmpdir),
            "-s",
            "hydro_lakes",
            "-t",
            "['2010-01-01','2020-12-31']",
            "-d",
            "tests/data/test_sources.yml",
        ],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
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
            "-t",
            "['2010-01-01','2014-12-31']",
            "-d",
            "tests/data/test_sources.yml",
            "-d",
            "tests/data/test_sources.yml",
        ],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_export_cli_config_file(tmpdir):
    r = CliRunner().invoke(
        hydromt_cli,
        ["export", str(tmpdir), "-i", "tests/data/export_config.yml"],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output


@pytest.mark.skip(reason="GridComponent should remove region argument in create().")
def test_check_cli():
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "check",
            "-d",
            "tests/data/test_sources.yml",
            "-i",
            "tests/data/test_model_config.yml",
        ],
    )
    assert r.exit_code == 0, r.output


@pytest.mark.skip(reason="GridComponent should remove region argument in create().")
def test_check_cli_unsupported_region():
    with pytest.raises(Exception, match="is not supported in region validation yet"):
        _ = CliRunner().invoke(
            hydromt_cli,
            [
                "check",
                "-r",
                "{'subbasin': [-7.24, 62.09], 'uparea': 50}",
                "-i",
                "tests/data/test_model_config.yml",
            ],
            catch_exceptions=False,
        )


@pytest.mark.skip(reason="GridComponent should remove region argument in create().")
def test_check_cli_known_region():
    with pytest.raises(Exception, match="Unknown region kind"):
        _ = CliRunner().invoke(
            hydromt_cli,
            [
                "check",
                "-r",
                "{'asdfasdfasdf': [-7.24, 62.09], 'uparea': 50}",
                "-i",
                "tests/data/test_model_config.yml",
            ],
            catch_exceptions=False,
        )


@pytest.mark.skip(reason="GridComponent should remove region argument in create().")
def test_check_cli_bbox_valid():
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "check",
            "-r",
            "{'bbox': [12.05,45.30,12.85,45.65]}",
            "-i",
            "tests/data/test_model_config.yml",
        ],
    )
    assert r.exit_code == 0, r.output


@pytest.mark.skip(reason="GridComponent should remove region argument in create().")
def test_check_cli_geom_valid():
    r = CliRunner().invoke(
        hydromt_cli,
        [
            "check",
            "-r",
            "{'geom': 'tests/data/naturalearth_lowres.geojson'}",
            "-i",
            "tests/data/test_model_config.yml",
        ],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output


@pytest.mark.skip(reason="GridComponent should remove region argument in create().")
def test_check_cli_geom_missing_file():
    with pytest.raises(Exception, match="Path not found at asdf"):
        _ = CliRunner().invoke(
            hydromt_cli,
            [
                "check",
                "-r",
                "{'geom': 'asdfasdf'}",
                "-i",
                "tests/data/test_model_config.yml",
            ],
            catch_exceptions=False,
        )
