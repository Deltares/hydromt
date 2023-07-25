"""HydroMT: Automated and reproducible model building and analysis."""


# Set environment variables (this will be temporary)
# to use shapely 2.0 in favor of pygeos (if installed)
import os

os.environ["USE_PYGEOS"] = "0"

# pkg_resource deprication warnings originate from dependencies
# so silence them for now
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# required for accessor style documentation
from subprocess import run

from xarray import DataArray, Dataset

# submodules
from . import cli, flw, raster, stats, vector, workflows
from .data_catalog import *
from .io import *

# high-level methods
from .models import *


def _run_clean(s: str) -> str:
    return (
        run(s.split(" "), capture_output=True)
        .stdout.decode("utf-8")
        .strip()
        .split("\n")
    )


last_release = _run_clean("git tag --list")[-1]

last_release_sha = _run_clean(f"git rev-list -n 1 {last_release}")

num_commits_since_release = (
    len(_run_clean(f"git rev-list {last_release_sha}..main")) + 1
)

# version number without 'v' at start
__version__ = f"0.8.1.dev{num_commits_since_release}"
