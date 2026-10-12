from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit

from hydromt._utils.catalog_format import LATEST_CATALOG_FORMAT, catalog_format
from hydromt._validators.data_catalog_v1x import DRIVER_RENAME_MAPPING

logger = logging.getLogger(__name__)

__all__ = [
    "MigrationError",
    "Migration",
    "MigrationReport",
    "migrate_source",
    "migrate_catalog",
]

SourceDict = dict[str, Any]


class MigrationError(Exception):
    """Custom exception for migration errors."""


@dataclass(frozen=True)
class Migration:
    id: str
    """Unique identifier for the migration"""
    introduced_in: str
    """First release that writes the new form"""
    description: str
    """Description of the migration"""
    apply: Callable[[SourceDict], bool]
    """Mutates SourceDict in place, return True if it changed something"""
    removed_in: str | None = None
    """First release that removes support for the previous form."""
    runtime: bool = True
    """If True, the migration will always be applied. If False, only hydromt check --upgrade will apply this migration."""


def migration(
    id: str,
    introduced_in: str,
    description: str,
    removed_in: str | None = None,
    runtime: bool = True,
):
    def register(fn):
        _MIGRATIONS.append(
            Migration(
                id=id,
                introduced_in=introduced_in,
                description=description,
                apply=fn,
                removed_in=removed_in,
                runtime=runtime,
            )
        )
        return fn

    return register


@dataclass
class MigrationReport:
    applied: list[tuple[str, Migration]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def migrate_source(
    source: SourceDict, *, runtime_only: bool = False
) -> list[Migration]:
    return [
        m for m in _MIGRATIONS if (m.runtime or not runtime_only) and m.apply(source)
    ]


def migrate_catalog(raw: dict | None) -> tuple[dict, MigrationReport]:
    catalog = _clone(raw or {})
    report = MigrationReport()
    for name, source in catalog.items():
        if name == "meta" or not isinstance(source, dict):
            continue
        try:
            report.applied += [(name, m) for m in migrate_source(source)]
        except MigrationError as e:
            raise MigrationError(f"{name}: {e}") from e

    if report.applied:
        meta = dict(catalog.get("meta") or {})
        old = meta.get("format_version")
        if old is None:
            old = "v1"
            logger.warning("No format_version found in catalog meta, assuming 'v1'.")
        try:
            current = catalog_format(old)
        except ValueError as e:
            raise MigrationError(str(e)) from e
        if current is None or current < LATEST_CATALOG_FORMAT:
            meta["format_version"] = LATEST_CATALOG_FORMAT
            report.notes.append(
                f"meta.format_version set to {LATEST_CATALOG_FORMAT} (was {old!r})."
            )
            catalog = {
                "meta": meta,
                **{k: v for k, v in catalog.items() if k != "meta"},
            }
    return catalog, report


def _clone(obj: Any) -> Any:
    """Copy that does NOT preserve shared references (YAML anchors/aliases)."""
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone(v) for v in obj]
    return obj


def _variants(source: SourceDict) -> list[SourceDict]:
    return [v for v in source.get("variants") or [] if isinstance(v, dict)]


### Migrations ###
# Order matters: migrations will be applied in the order they are listed in this file.

_MIGRATIONS: list[Migration] = []  # registry


@migration(
    id="v0-to-v1",
    introduced_in="1.0",
    description="Source uses the v0 data catalog layout",
    runtime=False,
)
def _v0_to_v1(source: SourceDict) -> bool:
    changed = False
    if _is_v0(source, _V0_SOURCE_KEYS):
        _convert_v0_source(source)
        changed = True
    for variant in _variants(source):
        if _is_v0(variant, _V0_VARIANT_KEYS):
            _convert_v0_variant(variant)
            changed = True
    return changed


_V0_SOURCE_KEYS = {
    "path",
    "meta",
    "crs",
    "nodata",
    "attrs",
    "unit_add",
    "unit_mult",
    "kwargs",
    "driver_kwargs",
}
_V0_VARIANT_KEYS = {"path"}


def _is_v0(d: SourceDict, keys: set[str]) -> bool:
    # v0 spells driver/filesystem as strings; v1 uses mappings
    return (
        isinstance(d.get("driver"), str)
        or isinstance(d.get("filesystem"), str)
        or not keys.isdisjoint(d)
    )


def _scheme(path: Any) -> str | None:
    scheme = urlsplit(str(path or "")).scheme
    return (
        scheme if len(scheme) > 1 else None
    )  # "C:\data" is a drive letter, not a protocol


def _v0_filesystem(d: SourceDict) -> dict | None:
    """v0 `filesystem: s3` + `storage_options` -> `{protocol: s3, **storage_options}`."""
    explicit = d.pop("filesystem", None)
    options = d.pop("storage_options", None) or {}
    if isinstance(explicit, dict):  # already the new form
        return {**explicit, **options}
    if explicit is None and not options:
        return None
    protocol = explicit or _scheme(d.get("path"))
    if protocol is None:
        raise MigrationError(
            "`storage_options` given, but no `filesystem` and no protocol in `path`."
        )
    if protocol in {"local", "file"} and not options:
        return None  # the default
    return {"protocol": protocol, **options}


def _convert_v0_source(source: SourceDict) -> None:
    filesystem = _v0_filesystem(source)  # first: may need `path` to infer the protocol

    if "path" in source:
        source["uri"] = source.pop("path")
    if filesystem:
        source["filesystem"] = filesystem

    # driver: name mapping + kwargs -> options (kwargs wins, like from_v0)
    options = {
        **(source.pop("driver_kwargs", None) or {}),
        **(source.pop("kwargs", None) or {}),
    }
    driver = source.get("driver")
    if isinstance(driver, str):
        data_type = source.get("data_type", "undefined")
        new_name = DRIVER_RENAME_MAPPING.get(data_type, {}).get(driver)
        if new_name is None:
            raise MigrationError(
                f"no v1 driver for v0 driver '{driver}' on a {data_type}."
            )
        source["driver"] = {
            "name": new_name,
            **({"options": options} if options else {}),
        }
    elif options:
        raise MigrationError(
            "`kwargs`/`driver_kwargs` given without a v0 `driver` name."
        )

    # metadata: `meta` plus the root-level crs/nodata/attrs
    metadata = dict(source.pop("meta", None) or {})
    for key in ("crs", "nodata", "attrs"):
        if (value := source.pop(key, None)) is not None:  # keeps nodata: 0
            metadata[key] = value
    if metadata:
        source["metadata"] = metadata

    # data adapter
    adapter = {}
    for key in ("rename", "unit_add", "unit_mult"):
        if value := source.pop(key, None):
            adapter[key] = value
    if adapter:
        source["data_adapter"] = adapter


def _convert_v0_variant(variant: SourceDict) -> None:
    filesystem = _v0_filesystem(variant)
    if "path" in variant:
        variant["uri"] = variant.pop("path")
    if filesystem:
        variant["filesystem"] = filesystem


@migration(
    "filesystem-to-root",
    introduced_in="1.5.0",
    description="`filesystem` under `driver`/`uri_resolver` moved to the source root",
    removed_in="1.6.0",
)
def _filesystem_to_root(source: SourceDict) -> bool:
    changed = _upgrade_filesystem_inplace(source)
    for variant in _variants(source):
        changed |= _upgrade_filesystem_inplace(variant)
    return changed


def _as_filesystem(value: Any) -> dict:
    if isinstance(value, str):
        return {"protocol": value}
    if isinstance(value, dict):
        return value
    raise MigrationError(f"unsupported `filesystem` value: {value!r}")


def _upgrade_filesystem_inplace(source: SourceDict) -> bool:
    legacy = {}
    for key in ("driver", "uri_resolver"):
        section = source.get(key)
        if isinstance(section, dict) and "filesystem" in section:
            legacy[key] = _as_filesystem(section.pop("filesystem"))
    if not legacy:
        return False
    if "filesystem" in source:
        raise MigrationError(
            "`filesystem` is set at the source root and under driver/uri_resolver."
        )
    if len({json.dumps(v, sort_keys=True, default=str) for v in legacy.values()}) > 1:
        raise MigrationError("driver and uri_resolver define different filesystems.")
    source["filesystem"] = copy.deepcopy(next(iter(legacy.values())))
    return True
