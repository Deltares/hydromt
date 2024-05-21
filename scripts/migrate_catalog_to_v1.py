"""Script to migrate your catalog to the v1 version."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Set

from yaml import Loader, load

from hydromt.data_catalog import DataCatalog

DRIVER_RENAME_MAPPING: Dict[str, Dict[str, str]] = {
    "RasterDataset": {
        "raster": "rasterio",
        "zarr": "raster_xarray",
        "netcdf": "raster_xarray",
        "raster_tindex": "raster_tindex",  # TODO: https://github.com/Deltares/hydromt/issues/856
    },
    "GeoDataset": {
        "vector": "geodataset_vector",
        "zarr": "geodataset_xarray",
        "netcdf": "geodataset_xarray",
    },
    "GeoDataFrame": {
        "csv": "geodataframe_table",
        "parquet": "geodataframe_table",
        "xls": "geodataframe_table",
        "xlsx": "geodataframe_table",
        "xy": "geodataframe_table",
        "vector": "pyogrio",
    },
    "DataFrame": {
        "csv": "pandas",
        "parquet": "pandas",
        "xls": "pandas",
        "xlsx": "pandas",
        "fwf": "pandas",
    },
    "DataSet": {
        "zarr": "dataset_xarray",  # TODO: https://github.com/Deltares/hydromt/issues/878
        "netcdf": "dataset_xarray",  # TODO: https://github.com/Deltares/hydromt/issues/878
    },
}


def prepare_path_out(path_out: Path, overwrite: bool):
    """Make sure path_out parameter is correct."""
    if path_out.exists():
        if not overwrite:
            raise RuntimeError(f"path out at {path_out} already exists.")
        if path_out.is_dir():
            raise RuntimeError(f"path out at {path_out} is directory.")
    else:
        path_out.parent.mkdir(parents=True, exist_ok=True)


def load_catalog(catalog_path: Path):
    """Load the catalog into dict."""
    with open(catalog_path) as f:
        return load(f, Loader)


def migrate_meta(catalog_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate metadata of catalog."""
    if catalog_meta := catalog_dict.pop("meta", None):
        if version := catalog_meta.get("hydromt_version"):
            catalog_meta["hydromt_version"] = version

        return catalog_meta


def migrate_entries(catalog_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate all entries of the data catalog."""
    new_catalog_meta: Dict[str, Any] = migrate_meta(catalog_dict)
    new_catalog_dict = {
        item[0]: migrate_entry(item[1]) for item in catalog_dict.items()
    }
    new_catalog_dict["meta"] = new_catalog_meta
    return new_catalog_dict


def migrate_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate the v0.x entry to v1."""
    # Check if alias
    if entry.get("alias"):
        raise RuntimeError(
            "Warning: aliases have been deprecated. Please migrate alias"
            f"{entry} to variants or remove completely."
        )
    # migrate path to uri
    if path := entry.pop("path", None):
        entry["uri"] = path

    # migrate driver str to dict
    data_type: Optional[str] = entry.get("data_type")
    if data_type and entry.get("driver"):
        driver_name: str = DRIVER_RENAME_MAPPING[data_type][entry.pop("driver")]
        entry["driver"] = {"name": driver_name}

    # move kwargs and driver_kwargs to driver options
    old_kwarg_names: Set[str] = {"kwargs", "driver_kwargs"}
    if any(old_kwarg_names.intersection(entry.keys())):
        entry["driver"]["options"] = {}
        for field in old_kwarg_names:
            if value := entry.pop(field, None):
                entry["driver"]["options"][field] = value

    # move fsspec filesystem to driver
    if filesystem := entry.pop("filesystem", None):
        entry["driver"]["filesystem"] = filesystem

    # migrate meta to metadata
    if metadata := entry.pop("meta"):
        for before, after in {
            "source_url": "url",
            "source_author": "author",
            "source_version": "version",
            "source_license": "license",
            "source_info": "info",
        }.items():
            if value := metadata.pop(before, None):
                metadata[after] = value
            entry["metadata"] = metadata

    # move crs and nodata to metadata
    new_meta: Set[str] = {"crs", "nodata"}
    for field in new_meta:
        if value := entry.pop(field, None):
            entry["metadata"][field] = value

    # Migrate postprocessing parameters to data_adapter
    postprocessing_params: Set[str] = {"unit_add", "unit_mult", "rename"}
    if any(postprocessing_params.intersection(entry.keys())):
        entry["data_adapter"] = {}

        for param in postprocessing_params:
            if val := entry.pop(param, None):
                entry["data_adapter"][param] = val

    # Migrate variants
    if variants := entry.get("variants"):
        entry["variants"] = [migrate_entry(variant) for variant in variants]

    return entry


def write_out(new_catalog_dict: Dict[str, Any], path_out: Path):
    """Write the catalog out to the new structure."""
    if meta := new_catalog_dict.pop("meta", None):
        root = meta.pop("root", None)
    else:
        root = None
    catalog = DataCatalog().from_dict(new_catalog_dict, root=root)
    catalog.to_yml(path_out)


def main(path_in: Path, path_out: Path, overwrite: bool):
    """Read in data catalog and write out."""
    prepare_path_out(path_out, overwrite)
    old_catalog_dict: Dict[str, Any] = load_catalog(path_in)
    new_catalog_dict: Dict[str, Any] = migrate_entries(old_catalog_dict)
    write_out(new_catalog_dict, path_out)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="migrate_catalogs", description="migrate data catalog to v1."
    )
    parser.add_argument("filein")
    parser.add_argument("fileout")
    parser.add_argument("-o", "--overwrite", action="store_true", default=True)
    args: Namespace = parser.parse_args()
    file_in: Path = Path(args.filein)
    file_out: Path = Path(args.fileout)
    overwrite: bool = args.overwrite
    main(file_in, file_out, overwrite)
