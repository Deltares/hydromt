# HydroMT — Agent Guide

HydroMT is a Python framework for building and analysing hydrology/hydraulics
models. The core library (this repo) provides the data catalogue, model
abstraction, GIS utilities, and a plugin system that downstream packages
(`hydromt_sfincs`, `hydromt_wflow`, …) extend.

---

## Essential commands

```bash
# Install / sync environment (pixi manages all deps + lockfile)
pixi install

# Run tests (default env)
pixi run test

# Run tests with coverage
pixi run test-cov

# Lint (runs pre-commit on all files)
pixi run lint

# Type-check
pixi run mypy

# Build docs locally
pixi run docs
```

> **Never use `pip install` directly.** All dependency management goes through
> `pixi`. The lockfile `pixi.lock` is committed; update it with
> `pixi update` when adding or bumping deps in `pyproject.toml`.

---

## Repo layout

```
hydromt/
  data_catalog/       # DataCatalog + 4-layer pipeline (source/resolver/driver/adapter)
    sources/          # DataSource subclasses (RasterDataset, GeoDataFrame, …)
    drivers/          # I/O drivers (rasterio, geodataframe, zarr, …)
    uri_resolvers/    # URI resolution (ConventionResolver, …)
    adapters/         # Post-load normalisation (unit convert, rename, slice)
  model/
    model.py          # Model base class
    components/       # ModelComponent ABC + built-in implementations
    processes/        # Domain helpers (basin_mask, meteo, …)
  gis/                # Raster/vector GIS utilities
  stats/              # Statistical helpers
  _validators/        # Pydantic validators for workflow YAML
  _utils/             # Internal helpers (not public API)
  typing/             # Typed aliases and type definitions
  cli/main.py         # CLI entry points (build, update, check, export)
  plugins.py          # Plugin discovery via entry points
  error.py            # NoDataStrategy enum + exec_nodata_strat
  log.py              # Logging helpers

tests/                # Mirrors hydromt/ layout; shared fixtures in conftest.py
docs/                 # Sphinx docs; dev guides in docs/dev/
data/catalogs/        # Predefined data catalogue YAML files
.github/workflows/    # CI pipelines (tests, lint, docs, release, sonar)
pyproject.toml        # All project + tooling + pixi config
```

---

## Architecture invariants — do not violate

### 1. Plugin contracts
Every pluggable class (`Model`, `ModelComponent`, `BaseDriver`, `URIResolver`,
`DataSource`) must:
- Be discoverable via its entry-point group (`hydromt.models`,
  `hydromt.components`, `hydromt.drivers`, `hydromt.catalogs`,
  `hydromt.uri_resolvers`).
- Pass the subclass check performed by `hydromt/plugins.py`.
- Declare `__hydromt_eps__: ClassVar[list[str]]` when registering multiple names.

### 2. DataCatalog pipeline split
Keep the four layers separate. Do not collapse Driver + Adapter logic or merge
Resolver concerns into a Source:
- **DataSource** — owns metadata, Pydantic config, `read_data()`/`to_file()`.
- **URIResolver** — translates source config into concrete URIs.
- **Driver** — raw I/O only (`read(uris)` / `write(path, data)`).
- **DataAdapter** — post-load normalisation only (rename, unit add/mult, bbox/time/var slice).

### 3. `@hydromt_step` decorator
Every `Model` method intended to be callable from a workflow YAML **must** be
decorated with `@hydromt_step`. All positional arguments must be JSON-safe
primitives (`str`, `int`, `float`, `bool`, `None`, `list`, `dict`). Methods
decorated this way are validated at workflow-load time.

### 4. `NoDataStrategy`
All data-access paths must accept and honour the `NoDataStrategy` enum
(`RAISE`, `WARN`, `IGNORE`). Never silently swallow missing-data errors
without passing through `exec_nodata_strat`.

### 5. `ModelComponent.write()` signature
`write()` on any `ModelComponent` subclass must have **no required positional
arguments** — it is called as `component.write()` by the model's batch writer.

### 6. Pydantic v2 throughout
All config/validation models use **Pydantic v2**. Do not introduce dataclasses
or `TypedDict` for validated config; use `pydantic.BaseModel` or
`AbstractBaseModel` (for polymorphic deserialization by `"name"` field).

---

## Conventions

### Naming
| Thing | Convention | Example |
|---|---|---|
| Model subclass | `<Name>Model` | `SfincsModel` |
| Driver subclass | `<Format>Driver` | `RasterioDriver` |
| Component subclass | `<Name>Component` | `GridComponent` |
| Arg meaning "catalog key or file path" | `_fn` suffix | `dem_fn` |
| Internal/private module | `_` prefix | `_utils/`, `_validators/` |
| Data type literal strings | PascalCase | `"RasterDataset"`, `"GeoDataFrame"` |

### Typing
- Add type hints to all public functions/methods.
- Use typed aliases from `hydromt/typing/type_def.py` for catalog-specific types.
- Optional deps are guarded via `hydromt/_compat.py` flags (`HAS_S3FS`, etc.).

### Docstrings
NumPy docstring format is required for all public API. Example:

```python
def my_func(x: int, y: str) -> bool:
    """Short one-line summary.

    Longer description if needed.

    Parameters
    ----------
    x : int
        Description of x.
    y : str
        Description of y.

    Returns
    -------
    bool
        Description of return value.
    """
```

### Tests
- Mirror the source layout: `tests/data_catalog/`, `tests/model/`, etc.
- Fixtures go in `tests/conftest.py` (shared) or local `conftest.py`.
- Use `pytest.mark.integration` for tests requiring external resources.
- Use `pytest.mark.skipif(not HAS_X, reason="...")` for optional-dep tests.
- `test_equal` helpers return `tuple[bool, dict[str, str]]` accumulating all
  errors rather than raising on the first failure.

---

## PR checklist (enforced by PR template)

Every PR must:
1. Update `docs/changelog.rst` with a note under the correct version header.
2. Include or update tests for changed behaviour.
3. Pass `pixi run lint` (ruff + ruff-format) and the test suite.
4. Not modify `data/catalogs/predefined_catalogs.yml` unless that is the
   explicit goal of the PR.

---

## Gotchas

| # | File | Issue |
|---|---|---|
| 1 | `hydromt/data_catalog/adapters/rasterdataset.py` | Walrus-precedence bug: `if nodata := metadata.nodata is not None:` assigns a `bool`, not the nodata value. Check before touching this method. |
| 2 | `hydromt/typing/type_def.py` vs `sources/factory.py` | `DataType` literal uses `"DataSet"`/`"GeoDataSet"` but factory/source files use `"Dataset"`/`"GeoDataset"` — case mismatch. |
| 3 | `hydromt/data_catalog/data_catalog.py` ~line 509 | `contains_source` provider/version lookup logic appears to invert a boolean — verify carefully before refactoring. |
| 4 | `hydromt/_compat.py` | Python <3.10 entry-point fallback is dead code; project requires >=3.11. Do not add more compat branches. |
| 5 | `hydromt/__init__.py` | Contains intentional import-time side effects: patches a Rasterio `sys.excepthook` bug and forces `netCDF4` import ordering. Do not reorder or remove these. |
| 6 | `docs/dev/core_dev/code_conventions.rst` | Docs still reference "black" and pip-first examples; actual tooling is `ruff-format` + pixi. Ignore the docs, follow the tooling. |
| 7 | `pyproject.toml` pixi tasks | `pypi-git-resore` (missing `t`) — typo in task name; do not rely on that task until fixed. |

---

## Domain vocabulary

| Term | Meaning |
|---|---|
| `flwdir` | Flow direction grid (D8, LDD, or nextxy encoding) |
| `basid` | Basin/sub-basin identifier integer |
| PET | Potential evapotranspiration |
| LDD | Local drain direction (PCRaster encoding) |
| `_fn` arg suffix | "catalog key or file path" — resolves via `DataCatalog.get_*` |
| `NoDataStrategy` | RAISE / WARN / IGNORE policy for missing data |
| `hydromt_step` | Decorator that gates a method for YAML workflow invocation |
| STAC | SpatioTemporal Asset Catalog — supported export/import format for DataCatalog |

---

## Release process (brief)

1. Releases live on `release/vX.Y` branches.
2. Use `.github/workflows/create-release-branch.yml` to cut a release branch.
3. Use `.github/workflows/create-release.yml` to tag and publish.
4. PyPI publish is gated by `.github/workflows/publish-pypi.yml`.
5. A `record-release` PR automatically merges the release back to `main`.

Do not manually edit `hydromt/__init__.py` version; the release scripts manage it.
