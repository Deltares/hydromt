"""Tests for the cli submodule."""

import pytest
from click.testing import CliRunner
import numpy as np
from hydromt import __version__
from hydromt.cli.main import main as hydromt_cli


def test_cli(tmpdir):

    r = CliRunner().invoke(hydromt_cli, "--version")
    assert r.exit_code == 0
    assert r.output.split()[-1] == __version__

    r = CliRunner().invoke(hydromt_cli, "--models")
    assert r.exit_code == 0
    assert r.output.startswith("hydroMT model plugins:")

    r = CliRunner().invoke(hydromt_cli, "--help")
    assert r.exit_code == 0
    # NOTE: when called from CliRunner we get "Usage: main" instead of "Usage: hydromt"
    assert r.output.startswith("Usage: main [OPTIONS] COMMAND [ARGS]...")

    r = CliRunner().invoke(hydromt_cli, ["build", "--help"])
    assert r.exit_code == 0
    assert r.output.startswith("Usage: main build [OPTIONS] MODEL MODEL_ROOT REGION")

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
    with pytest.raises(ValueError, match="Model unknown"):
        raise r.exception

    r = CliRunner().invoke(
        hydromt_cli,
        ["update", "model", str(tmpdir), "-c", "component", "--opt", "key=value"],
    )
    with pytest.raises(ValueError, match="Model unknown"):
        raise r.exception

    r = CliRunner().invoke(
        hydromt_cli, ["clip", "model", str(tmpdir), str(tmpdir), "{'bbox': [1,2,3,4]}"]
    )
    with pytest.raises(NotImplementedError):
        raise r.exception
