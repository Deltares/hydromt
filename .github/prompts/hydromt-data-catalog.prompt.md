---
mode: agent
description: >
  Expert guidance for working on the HydroMT DataCatalog subsystem:
  adding or modifying DataSource types, Drivers, URIResolvers, and
  DataAdapters. Auto-invoked when editing files under
  hydromt/data_catalog/.
applyTo: "hydromt/data_catalog/**"
---

# HydroMT DataCatalog skill

You are working on the **HydroMT DataCatalog** subsystem
(`hydromt/data_catalog/`). Follow the rules below precisely.

---

## Pipeline architecture (must preserve)

Data access is a strict 4-layer pipeline. Never collapse layers:

```
DataSource  →  URIResolver  →  Driver  →  DataAdapter
(metadata)     (URI lookup)   (raw I/O)  (normalise/slice)
```

| Layer | Base class | File | Responsibility |
|---|---|---|---|
| DataSource | `DataSource` (Pydantic) | `sources/data_source.py` | Metadata, config, `read_data()`, `to_file()`, `to_stac_catalog()` |
| URIResolver | `URIResolver` (ABC) | `uri_resolvers/uri_resolver.py` | `resolve(source_config) -> list[URI]` |
| Driver | `BaseDriver` (Pydantic) | `drivers/base_driver.py` | `read(uris, **kwargs)`, `write(path, data)` — raw I/O only |
| DataAdapter | `DataAdapterBase` (Pydantic) | `adapters/data_adapter_base.py` | `rename`, `unit_add`, `unit_mult`, bbox/time/variable slicing |

---

## Creating a new DataSource type

1. Subclass `DataSource` in `hydromt/data_catalog/sources/`.
2. Add a Pydantic field `data_type: Literal["YourTypeName"]`.
3. Implement `read_data(bbox, time_range, variables, …) -> XArray/GeoDataFrame`.
4. Implement `to_file(data_root, …) -> DataSource`.
5. Register in `sources/__init__.py` and `sources/factory.py`.
6. Add entry-point `"YourTypeName" = "hydromt.data_catalog.sources.your_module:YourSource"` under `[project.entry-points."hydromt.catalogs"]` in `pyproject.toml` if externally discoverable.

**`DataType` literal consistency**: the `data_type` string in your source must
exactly match the key in `factory.py`. Note existing case-mismatch bug:
`type_def.py` uses `"DataSet"`/`"GeoDataSet"` but factory uses
`"Dataset"`/`"GeoDataset"` — be consistent with factory, not type_def.

---

## Creating a new Driver

1. Subclass `BaseDriver` in `hydromt/data_catalog/drivers/`.
2. Create a companion `DriverOptions(pydantic.BaseModel)` for driver-specific config.
3. Implement `read(uris: list[str], **kwargs) -> DataType`.
4. Implement `write(path: str | Path, data: DataType, …) -> None` if writable.
5. Register in `drivers/__init__.py`.
6. Add entry-point under `[project.entry-points."hydromt.drivers"]`.

Drivers do **raw I/O only** — no renaming, unit conversion, or spatial slicing.
All post-load normalisation belongs in the adapter layer.

---

## Creating a new URIResolver

1. Subclass `URIResolver` in `hydromt/data_catalog/uri_resolvers/`.
2. Implement `resolve(source_config) -> list[str]`.
3. Register in `uri_resolvers/__init__.py`.
4. Add entry-point under `[project.entry-points."hydromt.uri_resolvers"]`.

---

## NoDataStrategy — required everywhere

All `read_data` implementations must accept a `handle_nodata: NoDataStrategy`
parameter and call `exec_nodata_strat` from `hydromt/error.py` when data is
absent:

```python
from hydromt.error import NoDataStrategy, exec_nodata_strat

def read_data(self, …, handle_nodata: NoDataStrategy = NoDataStrategy.RAISE):
    if data is None:
        exec_nodata_strat(
            f"No data found for {self.name}",
            strategy=handle_nodata,
        )
        return None
    return data
```

Never silently return `None` or raise without going through this helper.

---

## DataCatalog public API

Core methods on `DataCatalog` (`hydromt/data_catalog/data_catalog.py`):

| Method | Returns |
|---|---|
| `get_rasterdataset(source, …)` | `xr.Dataset` |
| `get_geodataframe(source, …)` | `gpd.GeoDataFrame` |
| `get_geodataset(source, …)` | `xr.Dataset` (vector cube) |
| `get_dataset(source, …)` | `xr.Dataset` |
| `get_dataframe(source, …)` | `pd.DataFrame` |
| `export_data(data_root, …)` | writes files + catalog YAML |
| `from_stac_catalog(stac_url)` | populates catalog from STAC |
| `to_stac_catalog(path)` | writes STAC catalog |

Use `add_source(name, source)` to register at runtime.

### `contains_source` gotcha
The `contains_source` method (~line 509) has a suspected boolean-inversion bug
in provider/version logic. Verify its behaviour with a test before relying on it.

---

## Testing DataCatalog changes

- Test files live in `tests/data_catalog/`.
- Shared fixtures (sample rasters, GeoDataFrames, catalog YAML paths) in
  `tests/conftest.py` — reuse them.
- Mark tests needing external data: `@pytest.mark.integration`.
- Guard optional-dep paths: `@pytest.mark.skipif(not HAS_S3FS, reason="s3fs not installed")`.
- Run: `pixi run test tests/data_catalog/`

---

## Checklist before opening a PR

- [ ] New source/driver/resolver registered in `__init__.py` and `pyproject.toml`.
- [ ] `DataType` literal consistent with `factory.py` casing.
- [ ] `NoDataStrategy` accepted and passed to `exec_nodata_strat`.
- [ ] NumPy docstrings on all public methods.
- [ ] Tests added under `tests/data_catalog/`.
- [ ] `docs/changelog.rst` updated.
