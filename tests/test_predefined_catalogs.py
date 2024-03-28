from pathlib import Path

import pytest
import yaml

from hydromt.predefined_catalogs import _get_catalog_eps, _get_catalog_versions


@pytest.fixture()
def cat_root():
    return Path(__file__).parent.parent / "data/catalogs"


def test_eps():
    # TODO mock actual entrypoints
    eps = _get_catalog_eps()
    assert "artifact_data" in eps


def test_get_versions(tmpdir):
    # create a dummy catalog file
    root = tmpdir.mkdir("dummy_data")
    cat_path = root.join("data_catalog.yml")
    cat_path.write("meta:\n  version: v0.1.0")
    versions = _get_catalog_versions(root)
    assert len(versions) == 1
    assert versions[0].get("version") == "v0.1.0"
    assert versions[0].get("hash") == "ded043c2ad73df6ea118340cc55583c7"


def test_get_versions_artifacts(cat_root):
    versions = _get_catalog_versions(cat_root / "artifact_data")
    assert len(versions) > 0
    assert any(v["version"] == "v0.0.8" for v in versions)
    version = [v for v in versions if v["version"] == "v0.0.8"][0]
    assert all([key in version for key in ["version", "hash", "path"]])


def test_catalog_versions(cat_root):
    # assert all subdirs are catalogs and have a versions.yml file
    catalogs = [d for d in cat_root.iterdir() if d.is_dir()]
    for cat in catalogs:
        version_yml = cat / "versions.yml"
        assert version_yml.exists()
        with open(version_yml, "r") as f:
            versions_file = yaml.safe_load(f)["versions"]
        versions = _get_catalog_versions(cat_root / cat)
        # compare list of dicts
        assert versions == versions_file
