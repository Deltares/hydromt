import copy
from textwrap import dedent

import pytest
import yaml
from packaging.version import Version

from hydromt import __version__ as HYDROMT_VERSION
from hydromt._validators.data_catalog_v1x import DataCatalogV1Validator
from hydromt._validators.upgrade_data_catalog import (
    _MIGRATIONS,
    MigrationError,
    migrate_catalog,
    migrate_source,
)
from hydromt.readers import _yml_from_uri_or_path

# ---------------------------------------------------------------- fixtures

LEGACY_DRIVER_FS = {
    "data_type": "RasterDataset",
    "uri": "gs://cmip6/data/*",
    "driver": {"name": "raster_xarray", "filesystem": {"protocol": "gcs"}},
}

CURRENT_SOURCE = {
    "data_type": "RasterDataset",
    "uri": "gs://cmip6/data/*",
    "driver": {"name": "raster_xarray"},
    "filesystem": {"protocol": "gcs"},
}

V0_GEODATAFRAME = {
    "crs": 4326,
    "data_type": "GeoDataFrame",
    "driver": "vector",
    "kwargs": {"layer": "BasinATLAS_v10_lev12"},
    "meta": {"category": "hydrography", "source_license": "CC BY 4.0"},
    "path": "hydrography/hydro_atlas/basin_atlas_v10.gpkg",
}

V0_ESA_WORLDCOVER = {
    "crs": 4326,
    "data_type": "RasterDataset",
    "driver": "raster",
    "filesystem": "local",
    "kwargs": {"chunks": {"x": 36000, "y": 36000}},
    "meta": {"category": "landuse", "source_license": "CC BY 4.0"},
    "variants": [
        {"provider": "local", "version": 2021, "path": "landuse/esa_2021/esa.vrt"},
        {
            "provider": "aws",
            "version": 2020,
            "path": "s3://esa-worldcover/v100/2020/map.vrt",
            "rename": {"map": "landuse"},
            "filesystem": "s3",
            "storage_options": {"anon": "true"},
        },
    ],
}


def _v0(*, drop: tuple[str, ...] = (), **overrides) -> dict:
    d = {**copy.deepcopy(V0_GEODATAFRAME), **overrides}
    for key in drop:
        d.pop(key)
    return d


def _migrate(source: dict, name: str = "src") -> dict:
    migrated, _ = migrate_catalog({name: source})
    return migrated[name]


# ------------------------------------------------- filesystem-to-root (1.5)


def test_filesystem_moves_from_driver_to_root():
    out = _migrate(LEGACY_DRIVER_FS)
    assert out["filesystem"] == {"protocol": "gcs"}
    assert out["driver"] == {"name": "raster_xarray"}


def test_filesystem_moves_from_uri_resolver_to_root():
    src = {
        "data_type": "RasterDataset",
        "uri": "abfs://container/data.tif",
        "driver": {"name": "rasterio"},
        "uri_resolver": {
            "name": "convention",
            "filesystem": {"protocol": "abfs", "account_name": "acc"},
        },
    }
    out = _migrate(src)
    assert out["filesystem"] == {"protocol": "abfs", "account_name": "acc"}
    assert "filesystem" not in out["uri_resolver"]


def test_identical_filesystems_on_driver_and_resolver_collapse_to_one():
    fs = {"protocol": "s3", "anon": True}
    src = {
        "data_type": "RasterDataset",
        "uri": "s3://b/x.tif",
        "driver": {"name": "rasterio", "filesystem": copy.deepcopy(fs)},
        "uri_resolver": {"name": "convention", "filesystem": copy.deepcopy(fs)},
    }
    out = _migrate(src)
    assert out["filesystem"] == fs
    assert "filesystem" not in out["driver"]
    assert "filesystem" not in out["uri_resolver"]


def test_string_filesystem_is_normalised_to_mapping():
    # same shape as the variant in test_valid_v1_catalog_variants
    src = {
        "data_type": "RasterDataset",
        "uri": "s3://b/x.vrt",
        "driver": {"name": "raster", "filesystem": "s3", "options": {"anon": "true"}},
    }
    out = _migrate(src)
    assert out["filesystem"] == {"protocol": "s3"}
    assert out["driver"] == {"name": "raster", "options": {"anon": "true"}}


def test_filesystem_is_migrated_inside_variants():
    src = {
        "data_type": "RasterDataset",
        "driver": {"name": "rasterio"},
        "variants": [
            {
                "provider": "aws",
                "uri": "s3://b/x.tif",
                "driver": {"name": "rasterio", "filesystem": {"protocol": "s3"}},
            },
        ],
    }
    variant = _migrate(src)["variants"][0]
    assert variant["filesystem"] == {"protocol": "s3"}
    assert variant["driver"] == {"name": "rasterio"}


@pytest.mark.parametrize(
    "src",
    [
        {  # driver and uri_resolver disagree
            "data_type": "RasterDataset",
            "uri": "x",
            "driver": {"name": "rasterio", "filesystem": {"protocol": "gcs"}},
            "uri_resolver": {"name": "convention", "filesystem": {"protocol": "s3"}},
        },
        {  # root and nested
            "data_type": "RasterDataset",
            "uri": "x",
            "filesystem": {"protocol": "s3"},
            "driver": {"name": "rasterio", "filesystem": {"protocol": "gcs"}},
        },
    ],
    ids=["driver-vs-resolver", "root-vs-nested"],
)
def test_conflicting_filesystems_raise_with_source_name(src):
    with pytest.raises(MigrationError, match="src"):
        migrate_catalog({"src": src})


SHARED_DRIVER = dedent("""\
    a: {data_type: RasterDataset, uri: x.tif, driver: &drv {name: rasterio, filesystem: {protocol: gcs}}}
    b: {data_type: RasterDataset, uri: y.tif, driver: *drv}
    """)
SHARED_FILESYSTEM = dedent("""\
    a: {data_type: RasterDataset, uri: x.tif, driver: {name: rasterio, filesystem: &fs {protocol: gcs}}}
    b: {data_type: RasterDataset, uri: y.tif, driver: {name: rasterio, filesystem: *fs}}
    """)


@pytest.mark.parametrize(
    "text",
    [SHARED_DRIVER, SHARED_FILESYSTEM],
    ids=["shared-driver", "shared-filesystem"],
)
def test_yaml_anchors_do_not_leak_between_sources(text):
    migrated, _ = migrate_catalog(yaml.safe_load(text))
    for name in ("a", "b"):
        assert migrated[name]["filesystem"] == {"protocol": "gcs"}
        assert migrated[name]["driver"] == {"name": "rasterio"}
    assert migrated["a"]["filesystem"] is not migrated["b"]["filesystem"]
    assert "&id" not in yaml.safe_dump(migrated)


def test_v0_geodataframe_is_converted():
    assert _migrate(copy.deepcopy(V0_GEODATAFRAME)) == {
        "data_type": "GeoDataFrame",
        "uri": "hydrography/hydro_atlas/basin_atlas_v10.gpkg",
        "driver": {"name": "pyogrio", "options": {"layer": "BasinATLAS_v10_lev12"}},
        "metadata": {
            "crs": 4326,
            "category": "hydrography",
            "source_license": "CC BY 4.0",
        },
    }


def test_v0_variants_are_converted():
    out = _migrate(copy.deepcopy(V0_ESA_WORLDCOVER), "esa_worldcover")
    assert out["driver"] == {
        "name": "rasterio",
        "options": {"chunks": {"x": 36000, "y": 36000}},
    }
    assert "filesystem" not in out  # v0 "local" is the default
    assert out["variants"][0] == {
        "provider": "local",
        "version": 2021,
        "uri": "landuse/esa_2021/esa.vrt",
    }
    assert out["variants"][1] == {
        "provider": "aws",
        "version": 2020,
        "uri": "s3://esa-worldcover/v100/2020/map.vrt",
        "rename": {"map": "landuse"},
        "filesystem": {"protocol": "s3", "anon": "true"},
    }


def test_v0_filesystem_and_storage_options_become_one_mapping():
    out = _migrate(_v0(filesystem="s3", storage_options={"anon": True}))
    assert out["filesystem"] == {"protocol": "s3", "anon": True}
    assert "storage_options" not in out


def test_v0_protocol_is_inferred_from_path():
    out = _migrate(_v0(path="s3://bucket/x.gpkg", storage_options={"anon": True}))
    assert out["filesystem"] == {"protocol": "s3", "anon": True}


def test_v0_windows_drive_letter_is_not_a_protocol():
    out = _migrate(_v0(path=r"C:\data\x.gpkg"))
    assert "filesystem" not in out
    assert out["uri"] == r"C:\data\x.gpkg"


def test_v0_kwargs_override_driver_kwargs():
    out = _migrate(_v0(driver_kwargs={"layer": "a", "x": 1}, kwargs={"layer": "b"}))
    assert out["driver"]["options"] == {"layer": "b", "x": 1}


def test_v0_nodata_zero_is_kept():
    assert _migrate(_v0(nodata=0))["metadata"]["nodata"] == 0


def test_v0_adapter_fields_and_placeholders():
    out = _migrate(
        _v0(
            rename={"a": "b"},
            unit_add={"b": 1},
            unit_mult={"b": 2},
            placeholders={"year": ["2020", "2021"]},
        )
    )
    assert out["data_adapter"] == {
        "rename": {"a": "b"},
        "unit_add": {"b": 1},
        "unit_mult": {"b": 2},
    }
    assert out["placeholders"] == {"year": ["2020", "2021"]}


def test_v0_provider_and_version_are_kept():
    out = _migrate(_v0(provider="local", version=2))
    assert (out["provider"], out["version"]) == ("local", 2)


@pytest.mark.parametrize(
    ("src", "match"),
    [
        (_v0(driver="vector_table"), "no v1 driver"),
        (_v0(storage_options={"anon": True}), "storage_options"),
        (
            _v0(path=r"C:\data\x.gpkg", storage_options={"anon": True}),
            "storage_options",
        ),
        (_v0(drop=("driver",)), "without a v0"),  # kwargs but no driver
    ],
    ids=[
        "unmapped-driver",
        "options-without-protocol",
        "windows-path-options",
        "kwargs-without-driver",
    ],
)
def test_v0_unmappable_constructs_raise(src, match):
    with pytest.raises(MigrationError, match=match):
        migrate_catalog({"src": src})


@pytest.mark.parametrize(
    ("meta", "expected", "rewritten"),
    [
        ({"hydromt_version": ">=0.9,<1"}, ">=1.0", True),  # contradicts
        ({"roots": [""]}, ">=1.0", True),  # absent
        (None, ">=1.0", True),  # no meta at all
        ({"hydromt_version": ">=1.0a,<2"}, ">=1.0a,<2", False),  # already admits
    ],
    ids=["contradicting", "absent", "no-meta", "admitting"],
)
def test_v0_migration_fixes_meta_hydromt_version(meta, expected, rewritten):
    raw = {"src": copy.deepcopy(V0_GEODATAFRAME)}
    if meta is not None:
        raw = {"meta": meta, **raw}
    migrated, report = migrate_catalog(raw)
    assert migrated["meta"]["hydromt_version"] == expected
    assert bool(report.notes) is rewritten


@pytest.mark.parametrize(
    "source",
    [LEGACY_DRIVER_FS, V0_GEODATAFRAME, V0_ESA_WORLDCOVER, CURRENT_SOURCE],
    ids=["legacy-v1", "v0", "v0-variants", "current"],
)
def test_migrations_are_idempotent(source):
    once, _ = migrate_catalog({"src": copy.deepcopy(source)})
    twice, report = migrate_catalog(once)
    assert twice == once
    assert report.applied == []


def test_migrate_catalog_does_not_mutate_its_input():
    raw = {"src": copy.deepcopy(LEGACY_DRIVER_FS)}
    snapshot = copy.deepcopy(raw)
    migrate_catalog(raw)
    assert raw == snapshot


def test_only_outdated_sources_are_reported():
    _, report = migrate_catalog(
        {"old": copy.deepcopy(V0_GEODATAFRAME), "new": copy.deepcopy(CURRENT_SOURCE)}
    )
    assert {name for name, _ in report.applied} == {"old"}


def test_runtime_mode_skips_v0_but_applies_current_migrations():
    v0 = copy.deepcopy(V0_GEODATAFRAME)
    assert migrate_source(v0, runtime_only=True) == []
    assert v0 == V0_GEODATAFRAME  # untouched

    applied = migrate_source(copy.deepcopy(LEGACY_DRIVER_FS), runtime_only=True)
    assert [m.id for m in applied] == ["filesystem-to-root"]


def test_migration_ids_are_unique():
    ids = [m.id for m in _MIGRATIONS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    "m", [m for m in _MIGRATIONS if m.removed_in], ids=lambda m: m.id
)
def test_runtime_shim_is_dropped_on_time(m):
    # Migrations are never deleted (`check --upgrade` must keep working for old
    # catalogs); after `removed_in` they only stop applying at runtime.
    if Version(HYDROMT_VERSION) >= Version(m.removed_in):
        assert not m.runtime, f"Set runtime=False on '{m.id}' (due in {m.removed_in})"


def test_deltares_v0_catalog_migrates_to_valid_v1(dd_v0_catalog):
    migrated, report = migrate_catalog(_yml_from_uri_or_path(dd_v0_catalog))
    assert report.applied
    DataCatalogV1Validator.from_dict(migrated)  # raises on failure


def test_latest_deltares_v1_catalog_has_no_pending_migrations(latest_dd_version_uri):
    _, report = migrate_catalog(_yml_from_uri_or_path(latest_dd_version_uri))
    assert not report.applied, [(n, m.id) for n, m in report.applied]


def test_filesystem_migration_replaces_min_version():
    raw = {
        "meta": {"hydromt_version": ">1.0a,<2"},
        "src": {
            "uri": "x",
            "driver": {"name": "rasterio", "filesystem": {"protocol": "gcs"}},
        },
    }
    migrated, _ = migrate_catalog(raw)
    assert migrated["meta"]["hydromt_version"] == ">=1.5.0,<2"


def test_no_filesystem_keeps_version():
    raw = {"meta": {"hydromt_version": ">1.0a,<2"}, "src": {"uri": "x"}}
    migrated, _ = migrate_catalog(raw)
    assert migrated["meta"]["hydromt_version"] == ">1.0a,<2"


def test_higher_existing_min_version_untouched():
    raw = {
        "meta": {"hydromt_version": ">=1.6,<2"},
        "src": {
            "uri": "x",
            "driver": {"name": "rasterio", "filesystem": {"protocol": "gcs"}},
        },
    }
    migrated, _ = migrate_catalog(raw)
    assert migrated["meta"]["hydromt_version"] == ">=1.6,<2"
