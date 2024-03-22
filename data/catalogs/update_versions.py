"""Script to update the versions.yml file for each predefined catalog."""
from pathlib import Path

import yaml

from hydromt.predefined_catalogs import _get_catalog_versions

BASE_URL = r"https://github.com/Deltares/hydromt/blob/main/data/catalogs"


def write_versions_file(root):
    """Write the versions file for a predefined catalog.

    We assume that each catalog has a data_catalog.yml file in each subdirectory.
    """
    root = Path(root)
    versions = _get_catalog_versions(root)
    for v in versions:
        assert (
            v["version"] == v["path"].split("/")[0]
        ), f"Catalog {root.name} version {v['version']} does not match path {v['path']}"
    yml_dict = {
        "name": root.name,
        "base_url": f"{BASE_URL}/{root.name}",
        "versions": versions,
    }
    with open(root / "versions.yml", "w") as f:
        yaml.dump(yml_dict, f, sort_keys=False)


if __name__ == "__main__":
    root = Path(__file__).parent
    cat_roots = [d for d in root.iterdir() if d.is_dir()]
    for cat_root in cat_roots:
        write_versions_file(cat_root)
