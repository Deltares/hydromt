"""Implementation of the predefined data catalogs entry points."""

import logging
import shutil
import sys
from pathlib import Path
from typing import Callable, ClassVar, Optional

import packaging.version
import pooch

from hydromt.data_adapter.caching import HYDROMT_DATADIR, _copyfile, _uri_validator

logger = logging.getLogger(__name__)

# this is the default location of the predefined catalogs
# in the test environment this is set to local data/catalogs directory using a global fixture
GIT_ROOT = r"https://github.com/Deltares/hydromt/blob/main/data/catalogs"

__all__ = [
    "PredefinedCatalog",
    "DeltaresDataCatalog",
    "ArtifactDataCatalog",
    "AWSDataCatalog",
    "GCSCMIP6DataCatalog",
    "create_registry_file",
]


def create_registry_file(root: Path, registry_path: Optional[Path] = None) -> None:
    """Create a registry file for all catalog files in the root directory.

    The root directory should contain a <version>/data_catalog.yml file per version.
    By default the root directory is the cache directory of the catalog instance.

    Parameters
    ----------
    root: Path
        root directory to search for data_catalog.yml files
    """
    # we don't use pooch.create_registry here as we want to only include vaild data_catalog.yml files
    registry = {}
    for path in root.glob("**/data_catalog.yml"):
        key = path.relative_to(root).as_posix()
        if not _valid_key(key):
            raise ValueError(f"No valid version found in {key}")
        if sys.platform == "win32":
            # The line endings need to be replaced when operating from windows in order to maintain equality of hashes
            _replace_line_endings(path)
        file_hash = pooch.file_hash(path)
        registry[key] = file_hash

    if not registry:
        raise FileNotFoundError(f"No data_catalog.yml files found in {root}")

    if registry_path is None:
        registry_path = Path(root / "registry.txt")
    with open(registry_path, "w") as f:
        for fname, hash in registry.items():
            f.write(f"{fname} {hash}\n")


class PredefinedCatalog(object):
    """Predefined data catalog.

    A predefined data catalog is a collection of data_catalog.yml files that are stored in a
    specific directory structure. The catalog is defined by a base_url and a name. The predefined
    catalog can be used to retrieve data_catalog.yml files for specific versions.

    Directory structure:
    - <base_url>/registry.txt
    - <base_url>/<version>/data_catalog.yml

    Cached directory structure:
    - <cache_dir>/<name>/registry.txt
    - <cache_dir>/<name>/<version>/data_catalog.yml
    """

    # required class variables to be defined in subclasses
    base_url: ClassVar[str] = GIT_ROOT
    name: ClassVar[str] = "predefined_catalog"

    def __init__(self, format_version: str = "v0", cache_dir=HYDROMT_DATADIR) -> None:
        # init arguments passed by DataCatalog
        self._format_version = format_version
        self._cache_dir: Path = Path(cache_dir)
        # placeholders set by the class
        self._pooch: Optional[pooch.Pooch] = None
        self._versions: Optional[list[str]] = None

    @property
    def registry(self) -> dict:
        """Return the registry."""
        return self.pooch.registry

    @property
    def pooch(self) -> pooch.Pooch:
        """Return a pooch instance with all data catalog files in registry."""
        if self._pooch is None:
            self._create_pooch()
            self._load_registry_file()
        return self._pooch

    @property
    def versions(self) -> list[str]:
        """Return the versions of the catalog."""
        if not self._versions:
            self._versions = self._set_versions()
        return self._versions

    def _create_pooch(self) -> None:
        self._pooch = pooch.create(
            path=self._cache_dir / self.name,
            base_url=self.base_url,
            retry_if_failed=3,
        )

    def _set_versions(self) -> list[str]:
        """Set valid catalog versions."""
        # parse versions from registry, assume registry key is <version>/data_catalog.yml
        # keep only versions that match the format_version
        keys = self.registry.keys()
        _versions = [
            v.split("/")[0] for v in keys if _valid_key(v, self._format_version)
        ]
        if len(_versions) == 0:
            raise RuntimeError(
                f"No compatible catalog version could be found for {self.name}."
            )
        self._versions = sorted(_versions, key=packaging.version.parse)
        return self._versions

    def _load_registry_file(self, overwrite: bool = False) -> None:
        """Create a catalog from a yaml file."""
        if self._pooch is None:
            self._create_pooch()
        if self.registry and not overwrite:
            return
        registry_path = Path(self._cache_dir / self.name / "registry.txt")
        if registry_path.exists():
            registry_path.unlink()
        try:  # try to retrieve and cache the registry file
            _copyfile(f"{self.base_url}/registry.txt", registry_path)
        except (ConnectionError, FileNotFoundError):
            logger.warning(
                f"Failed to retrieve {self.name} versions file from {self.base_url}."
                " Creating registry file from cached catalog files."
            )
            create_registry_file(registry_path.parent)
        if not registry_path.exists():
            raise FileNotFoundError(
                f"No cached file found. Failed to retrieve {self.name} versions file"
            )
        self.pooch.load_registry(registry_path)

    def get_catalog_file(self, version: Optional[str] = None) -> Optional[Path]:
        """Get the cached catalog file path for a specific version.

        Parameters
        ----------
        version: str, optional
            The version of the catalog to retrieve. If None, the latest version is retrieved.

        Returns
        -------
        Path
            The path to the cachd catalog file.
        """
        if version is None or version == "latest":  # get latest version
            version = self.versions[-1]
        if version not in self.versions:
            raise ValueError(f"Version {version} not found in {self.name} catalog")
        # get the catalog file
        key = f"{version}/data_catalog.yml"
        # fetch the file (download if not cached)
        path = self.pooch.fetch(key, downloader=self._downloader)
        return Path(path) if path else None

    @property
    def _downloader(self) -> Optional[Callable]:
        if not _uri_validator(self.base_url):
            return _copy_file
        return None


def _valid_key(v: str, format_version: Optional[str] = None) -> bool:
    """Check if the key is a valid version."""
    try:
        packaging.version.parse(v.split("/")[0])
        return v.startswith(format_version) if format_version else True
    except (packaging.version.InvalidVersion, AttributeError):
        return False


def _copy_file(
    url: str,
    output_file: str,
    pooch: Optional[pooch.Pooch] = None,
    check_only: bool = False,
):
    """Copy a local file to the cache directory for testing purposes.

    for more info, see: https://www.fatiando.org/pooch/latest/downloaders.html
    """
    url = Path(url)
    output_file = Path(output_file)
    file_exists = url.is_file()
    if check_only:
        return file_exists
    if not file_exists:
        raise FileNotFoundError(f"Local file {url} does not exist.")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(url, output_file)
    return output_file


class DeltaresDataCatalog(PredefinedCatalog):
    """Deltares data catalog."""

    base_url = f"{GIT_ROOT}/deltares_data"
    name = "deltares_data"


class ArtifactDataCatalog(PredefinedCatalog):
    """Artifact data catalog."""

    base_url = f"{GIT_ROOT}/artifact_data"
    name = "artifact_data"


class AWSDataCatalog(PredefinedCatalog):
    """AWS data catalog."""

    base_url = f"{GIT_ROOT}/aws_data"
    name = "aws_data"


class GCSCMIP6DataCatalog(PredefinedCatalog):
    """GCS CMIP6 data catalog."""

    base_url = f"{GIT_ROOT}/gcs_cmip6_data"
    name = "gcs_cmip6_data"


# TODO: replace with a entrypoint plugin structure in v1
PREDEFINED_CATALOGS = {
    "artifact_data": ArtifactDataCatalog,
    "deltares_data": DeltaresDataCatalog,
    "aws_data": AWSDataCatalog,
    "gcs_cmip6_data": GCSCMIP6DataCatalog,
}


def _replace_line_endings(file_path: Path):
    WINDOWS_LINE_ENDING = b"\r\n"
    UNIX_LINE_ENDING = b"\n"
    with open(file_path, "rb") as open_file:
        content = open_file.read()
    content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)
    with open(file_path, "wb") as open_file:
        open_file.write(content)
