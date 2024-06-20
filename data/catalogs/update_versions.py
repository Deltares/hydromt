"""Script to update the registry.txt file for each predefined catalog."""

from pathlib import Path

from hydromt.data_catalog.predefined_catalog import create_registry_file

if __name__ == "__main__":
    root = Path(__file__).parent
    cat_roots = [d for d in root.iterdir() if d.is_dir()]
    for cat_root in cat_roots:
        create_registry_file(cat_root)
