"""Script to migrate your catalog to the v1 version."""

from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Set

from yaml import Loader, dump, load

from hydromt.data_catalog import DataCatalog

DRIVER_RENAME_MAPPING: Dict[str, Dict[str, str]] = {
    "RasterDataset": {
        "raster": "rasterio",
        "zarr": "raster_xarray",
        "netcdf": "raster_xarray",
        "raster_tindex": "rasterio",
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


def load_catalog(catalog_path: Path) -> Dict:
    """Load the catalog into dict."""
    with open(catalog_path) as f:
        return load(f, Loader)


def migrate_meta(catalog_dict: Dict[str, Any], version: str) -> Dict[str, Any]:
    """Migrate metadata of catalog."""
    if catalog_meta := catalog_dict.pop("meta", None):
        if hydromt_version := catalog_meta.get("hydromt_version"):
            catalog_meta["hydromt_version"] = hydromt_version
        else:
            catalog_meta["hydromt_version"] = ">1.0a,<2"

        catalog_meta["version"] = version

        return catalog_meta


def migrate_entries(catalog_dict: Dict[str, Any], version: str) -> Dict[str, Any]:
    """Migrate all entries of the data catalog."""
    new_catalog_meta: Dict[str, Any] = migrate_meta(catalog_dict, version)
    new_catalog_dict = {
        "meta": new_catalog_meta,
        **{item[0]: migrate_entry(*item) for item in catalog_dict.items()},
    }
    return new_catalog_dict


def migrate_entry(
    entry_name: str, entry: Dict[str, Any], variant_ref: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Migrate the v0.x entry to v1."""
    if variant_ref is None:
        variant_ref = {}
    # Check if alias
    if entry.get("alias"):
        raise RuntimeError(
            "Warning: aliases have been deprecated. Please migrate alias"
            f"{entry} to variants or remove completely."
        )
    print(f"migrating entry: {entry_name}")
    # migrate path to uri
    if path := entry.pop("path", None):
        entry["uri"] = path

    # migrate driver str to dict
    data_type: Optional[str] = entry.get("data_type", variant_ref.get("data_type"))
    if data_type and entry.get("driver"):
        old_driver: str = entry.pop("driver")
        driver_name: str = DRIVER_RENAME_MAPPING[data_type][old_driver]
        entry["driver"] = {"name": driver_name}
        if old_driver == "raster_tindex":
            entry["uri_resolver"] = {"name": "raster_tindex"}

    # move kwargs and driver_kwargs to driver options
    old_kwarg_names: Set[str] = {"kwargs", "driver_kwargs"}
    if any(old_kwarg_names.intersection(entry.keys())):
        entry["driver"]["options"] = {}
        for field in old_kwarg_names:
            if value := entry.pop(field, None):
                if tindex := value.pop("tileindex", None):
                    entry["uri_resolver"]["options"] = {"tileindex": tindex}
                entry["driver"]["options"] = value

    # move fsspec filesystem to driver
    if filesystem := entry.pop("filesystem", None):
        if isinstance(filesystem, str):
            entry["driver"]["filesystem"] = {"protocol": filesystem}
        else:
            entry["driver"]["filesystem"] = filesystem
        if storage_options := entry.pop("storage_options", None):
            entry["driver"]["filesystem"] = {
                **entry["driver"]["filesystem"],
                **storage_options,
            }

    # migrate meta to metadata
    if metadata := entry.pop("meta", None):
        for before, after in {
            "source_url": "url",
            "source_author": "author",
            "source_license": "license",
            "source_info": "info",
        }.items():
            if value := metadata.pop(before, None):
                metadata[after] = value

        first_entry: bool = True
        for before, after in {
            "source_spatial_extent": "bbox",
            "source_temporal_extent": "time_range",
        }.items():
            if value := metadata.pop(before, None):
                if first_entry:
                    metadata["extent"] = {}
                    first_entry = False
                metadata["extent"][after] = value

        entry["metadata"] = metadata

    # move crs and nodata to metadata
    new_meta: Set[str] = {"crs", "nodata"}
    if "metadata" not in entry:
        entry["metadata"] = {}
    for field in new_meta:
        value: Any = entry.pop(field, None)
        if value is not None:
            entry["metadata"][field] = value
    if not entry["metadata"]:
        entry.pop("metadata")

    # Migrate postprocessing parameters to data_adapter
    postprocessing_params: Set[str] = {"unit_add", "unit_mult", "rename"}
    if any(postprocessing_params.intersection(entry.keys())):
        entry["data_adapter"] = {}

        for param in postprocessing_params:
            val: Any = entry.pop(param, None)
            if val is not None:
                entry["data_adapter"][param] = val

    # Migrate variants
    if variants := entry.get("variants", None):
        entry["variants"] = [
            migrate_entry(
                entry_name=f"{entry_name} variant",
                entry=variant,
                variant_ref={"data_type": entry.get("data_type")},
            )
            for variant in variants
        ]

    return entry


def write_out(new_catalog_dict: Dict[str, Any], path_out: Path):
    """Write the catalog out to the new structure."""
    # validate catalog, need a copy because catalog_dict is changed by DataCatalog.from_dict
    copy_catalog = deepcopy(new_catalog_dict)
    meta = copy_catalog.pop("meta", None)
    if meta:
        root = meta.pop("root", None)
    else:
        root = None
    DataCatalog().from_dict(copy_catalog, root=root)

    # dump dict
    with open(path_out, mode="w") as f:
        dump(new_catalog_dict, f, default_flow_style=False, sort_keys=False)


def main(path_in: Path, path_out: Path, overwrite: bool, version: str):
    """Read in data catalog and write out."""
    prepare_path_out(path_out, overwrite)
    old_catalog_dict: Dict[str, Any] = load_catalog(path_in)
    new_catalog_dict: Dict[str, Any] = migrate_entries(old_catalog_dict, version)
    write_out(new_catalog_dict, path_out)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="migrate_catalogs", description="migrate data catalog to v1."
    )
    parser.add_argument("filein")
    parser.add_argument("fileout")
    parser.add_argument("-o", "--overwrite", action="store_true", default=True)
    parser.add_argument("-v", "--version", default="v1.0.0")
    args: Namespace = parser.parse_args()
    file_in: Path = Path(args.filein)
    file_out: Path = Path(args.fileout)
    overwrite: bool = args.overwrite
    version: str = args.version
    main(file_in, file_out, overwrite, version)
