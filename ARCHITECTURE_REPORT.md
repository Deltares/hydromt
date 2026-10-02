# HydroMT Repository Architecture & Conventions Report (AI-agent oriented)

> Scope: direct inspection of source, tests, docs, and CI config in this repo (not README-only).
> Confidence markers: **High** (direct code/config), **Medium** (inference from behavior/docs), **Low** (explicitly uncertain).

---

## 1) Structure & entry points

## High-level module/package map

- `hydromt/` is the core library.
  - Core packages by size: `data_catalog` (largest), `model`, `gis`, `stats`, plus internal helper layers `_utils`, `_validators`, `typing`.
    - Evidence: `hydromt/` tree and per-folder file counts from direct listing (`hydromt/data_catalog`, `hydromt/model`, etc.; `pyproject.toml`).
- `tests/` mirrors package domains (`data_catalog`, `model`, `gis`, `components`, etc.), with heavy shared fixture setup in `tests/conftest.py`.
  - Evidence: `tests/` tree, `tests/conftest.py`.
- `docs/` includes end-user + dev docs, architecture docs, release process docs, and Sphinx build config.
  - Evidence: `docs/dev/**`, `docs/conf.py`.
- `data/catalogs/` stores predefined catalog YAMLs used by `DataCatalog`.
  - Evidence: `data/`, `hydromt/data_catalog/predefined_catalog.py`.
- `.github/workflows/` has CI/testing, docs, release, PyPI publish, downstream compatibility, and maintenance automations.
  - Evidence: `.github/workflows/tests.yml`, `docs.yml`, `create-release.yml`, `publish-pypi.yml`, etc.

## Main entry points (CLI/public API)

- CLI entrypoint:
  - Script: `hydromt` -> `hydromt.cli.main:main`.
  - Commands: `build`, `update`, `check`, `export`.
  - Evidence: `pyproject.toml` `[project.scripts]`; `hydromt/cli/main.py`.
- Public Python API exported from package root:
  - `DataCatalog`, `Model`, `hydromt_step`, `PLUGINS`, subpackages.
  - Evidence: `hydromt/__init__.py`.
- Plugin/public extension entrypoint groups:
  - `hydromt.models`, `hydromt.components`, `hydromt.drivers`, `hydromt.catalogs`, `hydromt.uri_resolvers`.
  - Evidence: `pyproject.toml` `[project.entry-points."..."]`, `hydromt/plugins.py`.

## Conceptual dependency/import graph

Core flow is layered:

1. **CLI** (`hydromt/cli/main.py`) -> loads model plugin + workflow YAML -> calls `Model.build/update` or `DataCatalog.export_data`.
2. **Model** (`hydromt/model/model.py`) owns:
   - `ModelRoot` for FS mode/state,
   - `DataCatalog` for data access,
   - `ModelComponent`s for model artifacts.
3. **DataCatalog** (`hydromt/data_catalog/data_catalog.py`) resolves source definitions -> delegates to:
   - `DataSource` (`sources/data_source.py`)
   - `URIResolver` (`uri_resolvers/uri_resolver.py`)
   - `Driver` (`drivers/base_driver.py`)
   - `DataAdapter` (`adapters/data_adapter_base.py`)
4. **Plugin registry** (`hydromt/plugins.py`) discovers extensible classes via Python entry points + `__hydromt_eps__`.

This is the central architecture to preserve when changing behavior.

---

## 2) Architecture & design decisions

## Core abstractions/patterns

- **Plugin architecture via entry points** with runtime validation against expected base classes and duplicate-name detection.
  - Evidence: `hydromt/plugins.py`.
- **Pydantic-based typed config models** in extension layers:
  - `AbstractBaseModel` supports polymorphic deserialization by `"name"` field.
  - `BaseDriver` and related options use typed Pydantic models.
  - Workflow validator (`HydromtModelSetup`) binds functions/signatures.
  - Evidence: `hydromt/_abstract_base.py`, `hydromt/data_catalog/drivers/base_driver.py`, `hydromt/_validators/model_config.py`.
- **Workflow-step gating**:
  - Only methods decorated with `@hydromt_step` are callable from workflow YAML.
  - Enforced at runtime (`_validate_steps`) and validator level.
  - Evidence: `hydromt/model/steps.py`, `hydromt/_utils/steps_validator.py`, `hydromt/_validators/model_config.py`.
- **Data access pipeline split** (resolver/driver/adapter):
  - Resolver finds URIs, driver loads raw data, adapter normalizes/slices/units.
  - Evidence: `hydromt/data_catalog/sources/rasterdataset.py`, `.../uri_resolvers/convention_resolver.py`, `.../drivers/raster/rasterio_driver.py`, `.../adapters/rasterdataset.py`.

## Notable design decisions + tradeoffs

- **Strong extensibility over simplicity**:
  - Many base types + plugin discovery means flexibility but more indirection.
  - Evidence: `hydromt/plugins.py`, `docs/dev/architecture/architecture.rst`.
- **Runtime strategy-based no-data handling** (`RAISE/WARN/IGNORE`) instead of a single exception policy.
  - Good for batch export/use-cases; increases branching complexity.
  - Evidence: `hydromt/error.py`, `data_catalog` methods using `handle_nodata`.
- **Cross-platform and cloud-first IO** through `fsspec` + protocol-aware providers.
  - Evidence: drivers/sources; `hydromt/data_catalog/drivers/base_driver.py`.
- **Release process intentionally branch-family based** (`release/vX.Y`) with automated record-back PR flow.
  - Evidence: `.github/workflows/create-release*.yml`, `docs/dev/core_dev/release.rst`.

## Deviations from “obvious” approach

- `hydromt/__init__.py` patches a Rasterio `sys.excepthook` bug and forces `netCDF4` import ordering to avoid xarray issue.
  - Non-obvious import-time side effects, intentionally defensive.
  - Evidence: `hydromt/__init__.py`.
- `exec_nodata_strat` introspects caller module/logger dynamically.
  - Non-obvious logging behavior based on call stack.
  - Evidence: `hydromt/error.py`, tests `tests/_typing/test_error_*.py`.

---

## 3) Conventions

## Naming conventions

- Stated conventions: PEP8-ish, CamelCase classes, snake_case methods, NumPy docstrings.
  - Evidence: `docs/dev/core_dev/code_conventions.rst`, `docs/dev/architecture/conventions.rst`.
- In practice:
  - Internal modules often prefixed with `_` for helper/private semantics (`_utils`, `_validators`, `_compat`).
  - Data catalog type names use PascalCase strings (`RasterDataset`, `GeoDataFrame`, etc.).
  - Evidence: package structure + class names in `data_catalog/sources/*.py`.

## Typing conventions

- Strong type hints throughout core interfaces (`Model`, `DataCatalog`, drivers/adapters/sources).
- Pydantic v2 is heavily used for config/runtime model validation.
- Typed aliases and typed models in `hydromt/typing`.
  - Evidence: `pyproject.toml` (`pydantic>=2.11`), `hydromt/typing/type_def.py`, `hydromt/_validators/model_config.py`, `drivers/base_driver.py`.

## Testing conventions

- Test structure mostly mirrors source package layout.
- Pytest fixtures are extensive and centralized in `tests/conftest.py`.
- Mostly function-style tests; selective class-based grouping (e.g., driver option tests).
- Markers used: `integration`, `manual`, plus many `skipif` for optional deps.
  - Evidence: `pyproject.toml` `[tool.pytest.ini_options]`, `tests/conftest.py`, `tests/data_catalog/test_data_catalog.py`, `tests/data_catalog/drivers/test_base_driver.py`.

## Error handling / logging patterns

- Explicit `NoDataStrategy` enum used per call path.
- Logging is centralized under `hydromt` root logger with helper context manager `log.to_file(...)`.
- Build/update workflows write `hydromt.log`.
  - Evidence: `hydromt/error.py`, `hydromt/log.py`, `hydromt/model/model.py`, `hydromt/cli/main.py`.

## Docstring/comment style

- Declared style: NumPy docstring format.
- Mostly followed; but some files/functions have sparse or inconsistent docs (“Args:” placeholders, typos).
  - Evidence: `docs/dev/core_dev/code_conventions.rst`; compare with `hydromt/data_catalog/sources/rasterdataset.py` (`Args:` placeholders), misc typos in comments/docs.

---

## 4) Tooling & workflows

## Build/dependency management

- Build backend: **flit** (`flit_core`).
- Dependency definition: `pyproject.toml`.
- Locking/runtime envs/tasks: **pixi** (`pixi.lock` committed).
- Pre-commit hooks: ruff + ruff-format + hygiene hooks + nbstripout.
  - Evidence: `pyproject.toml`, `pixi.lock`, `.pre-commit-config.yaml`.

## Local run workflows (from config)

- CLI: `hydromt ...`
- Tests: `pixi run test`, `pixi run test-cov`
- Lint: `pixi run lint` (pre-commit all files)
- Type-check: `pixi run mypy`
- Docs: `pixi run docs` / `pixi run doc`
  - Evidence: `pyproject.toml` `[tool.pixi.feature.*.tasks]`.

## CI/CD pipeline structure

- PR/push pipelines:
  - Linting: `.github/workflows/linting.yml`
  - Test matrix (OS x Python x dependency mode): `.github/workflows/tests.yml`
  - Sonar + coverage: `.github/workflows/sonar.yml`
  - Docs build/publish: `.github/workflows/docs.yml`
  - Data catalog validation: `.github/workflows/check-data-catalogs.yml`
  - Binder/docker smoke: `.github/workflows/test-docker.yml`
- Maintenance automation:
  - monthly pixi lock + SBOM update PR,
  - monthly pre-commit autoupdate PR.
  - Evidence: `pixi_auto_update.yml`, `pre-commit_auto_update.yml`.
- Release/deploy:
  - release branch creation, tag/release creation, docs publish, PyPI publish.
  - Evidence: `create-release-branch.yml`, `create-release.yml`, `publish-pypi.yml`.

**Merge gating note (uncertain):** workflows indicate expected quality gates, but actual required-status enforcement is repository settings (not visible here).

## Versioning/release process

- SemVer + changelog-based release notes.
- Version source of truth in `hydromt/__init__.py` (`__version__`).
- Release families on `release/vX.Y` branches; `record-release` PR merges back to `main`.
  - Evidence: `hydromt/__init__.py`, `docs/changelog.rst`, `docs/dev/core_dev/release.rst`, release workflows.

---

## 5) Domain-specific context

Key hydrology/geospatial terms used in code:

- **Basin / subbasin / interbasin**, **basid**: watershed delineation semantics.
  - Evidence: `hydromt/model/processes/basin_mask.py`.
- **Flow direction** (`flwdir`, D8/LDD/nextxy), stream maps, outlets.
  - Evidence: `hydromt/model/processes/basin_mask.py`, `hydromt/gis/flw.py`.
- **PET** (potential evapotranspiration) methods:
  - `debruin`, `makkink`, `penman-monteith_*`.
  - Evidence: `hydromt/model/processes/meteo.py`.
- **Raster/vector/geodataset** distinction as first-class data types in catalog APIs.
  - Evidence: `hydromt/data_catalog/data_catalog.py`, `sources/*.py`.
- **STAC** conversion support for catalog/sources.
  - Evidence: `data_catalog.py` (`to_stac_catalog/from_stac_catalog`), source `to_stac_catalog`.

Non-obvious external coupling:

- Heavy dependence on geospatial stack (`rasterio`, `geopandas`, `xarray`, `pyproj`, `pyflwdir`, `fsspec`).
- Predefined catalog fallback to GitHub raw URLs when local catalogs absent.
  - Evidence: `hydromt/data_catalog/predefined_catalog.py`.
- Optional capability flags (`HAS_S3FS`, `HAS_ADLFS`, `HAS_PYET`, etc.) alter behavior/tests.
  - Evidence: `hydromt/_compat.py`, many `pytest.mark.skipif`.

---

## 6) Gotchas & inconsistencies (important for AI agents)

1. **Potential typo/bug in pixi task wiring**
   - `pypi` task depends on `pypi-git-restore`, but task appears defined as `pypi-git-resore` (missing `t`).
   - Evidence: `pyproject.toml` around `[tool.pixi.feature.dev.tasks]`.
   - Confidence: **High** (direct string mismatch).

2. **Type alias inconsistency**
   - `DataType` literal in `hydromt/typing/type_def.py` uses `"DataSet"` / `"GeoDataSet"`, while factory and sources use `"Dataset"` / `"GeoDataset"`.
   - Evidence: `hydromt/typing/type_def.py`, `hydromt/data_catalog/sources/factory.py`.
   - Confidence: **High** (direct mismatch).

3. **Possible logic bug in `contains_source`**
   - Provider/version lookup logic appears inconsistent with internal `_sources` structure and may invert a boolean.
   - Evidence: `hydromt/data_catalog/data_catalog.py` (`contains_source` block around lines ~509–521) versus `add_source` shape.
   - Confidence: **Medium** (strongly suspicious; not fully validated by execution).

4. **Possible walrus-precedence bug in nodata handling**
   - In `RasterDatasetAdapter._set_nodata`, expression `if nodata := metadata.nodata is not None:` assigns boolean, not nodata object.
   - Evidence: `hydromt/data_catalog/adapters/rasterdataset.py`.
   - Confidence: **High** (syntax semantics), runtime impact depends on metadata usage.

5. **Docs/conventions drift vs current tooling**
   - Docs still describe “black style” and pip-first examples, while repo runs `ruff-format` and pixi-centric workflows.
   - Evidence: `docs/dev/core_dev/code_conventions.rst`, `.pre-commit-config.yaml`, `pyproject.toml`.
   - Confidence: **High**.

6. **Dead compatibility branches likely obsolete**
   - `_compat.py` has logic for Python <3.10 entry-point fallback, but project requires >=3.11.
   - Evidence: `_compat.py`, `pyproject.toml` (`requires-python`).
   - Confidence: **High**.

7. **Known flaky/external tests are explicitly excluded**
   - `manual` and skipped flaky HTTP tests exist; default pytest excludes manual.
   - Evidence: `pyproject.toml` markers/addopts, `tests/data_catalog/test_data_catalog.py`.
   - Confidence: **High**.

---

## Practical “agent mental model” before editing

- Treat **plugin contracts** (`name`, entry points, `__hydromt_eps__`, subclass checks) as critical compatibility surface.
- For workflow features, ensure methods are `@hydromt_step` and signature-bindable from YAML-safe args.
- Preserve the **DataCatalog pipeline split** (resolver/driver/adapter); don’t collapse responsibilities.
- Respect `NoDataStrategy` behavior in new data paths.
- Expect optional-dependency guarded behavior in both code and tests.
- Update `docs/changelog.rst` and tests when changing behavior (repo conventions and PR template explicitly ask this).
