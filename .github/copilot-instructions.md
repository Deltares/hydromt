# HydroMT — Copilot instructions

HydroMT is a Python (>=3.11) framework for building and analysing
hydrology/hydraulics models. Core subsystems: `data_catalog` (DataSource /
URIResolver / Driver / DataAdapter pipeline), `model` (Model base +
ModelComponents), `gis` utilities, and a plugin system via Python entry points.

---

## Commands

```bash
pixi run test        # run tests
pixi run test-cov    # tests + coverage
pixi run lint        # ruff + ruff-format (pre-commit)
pixi run mypy        # type-check
pixi run docs        # build docs
```

Use **pixi** for all dependency management. Never use `pip install` directly.

---

## Architecture rules

- **DataCatalog pipeline has four layers** — keep them separate:
  `DataSource` (metadata + config) → `URIResolver` (URI lookup) → `Driver`
  (raw I/O) → `DataAdapter` (normalise/slice). Do not merge responsibilities.
- **`@hydromt_step`** — any `Model` method callable from workflow YAML must
  carry this decorator. All its positional args must be JSON-safe primitives.
- **`NoDataStrategy`** (`RAISE`/`WARN`/`IGNORE`) must be accepted and honoured
  in every data-access path via `exec_nodata_strat` in `hydromt/error.py`.
- **`ModelComponent.write()`** must have no required positional arguments.
- **Pydantic v2** for all config/validation. Use `AbstractBaseModel` (from
  `hydromt/_abstract_base.py`) when you need polymorphic deserialization.
- **Plugin contracts** — pluggable classes must be registered via entry-point
  groups in `pyproject.toml` and pass subclass checks in `hydromt/plugins.py`.

---

## Conventions

- **Docstrings**: NumPy format on all public API.
- **Naming**:
  - Model/Component/Driver subclasses: `<Name>Model`, `<Name>Component`, `<Name>Driver`.
  - Arg meaning "catalog key or file path": `_fn` suffix (e.g. `dem_fn`).
  - Internal modules: `_` prefix (`_utils/`, `_validators/`).
  - Data type literal strings: PascalCase (`"RasterDataset"`, `"GeoDataFrame"`).
- **Tests**: mirror source layout in `tests/`; fixtures in `conftest.py`;
  `test_equal` helpers return `(bool, dict[str, str])` accumulating all errors.
- **Optional deps**: guard with flags from `hydromt/_compat.py`; use
  `pytest.mark.skipif(not HAS_X, ...)` in tests.

---

## Every PR must

1. Update `docs/changelog.rst`.
2. Include or update tests for changed behaviour.
3. Pass `pixi run lint` and the test suite.

---

## Known gotchas

- `hydromt/__init__.py` has intentional import-time side effects (Rasterio
  excepthook patch, netCDF4 import ordering). Do not reorder or remove them.
- Walrus-precedence bug in `data_catalog/adapters/rasterdataset.py`
  (`_set_nodata`): `if nodata := metadata.nodata is not None:` assigns a bool.
- `DataType` literal has a case mismatch: `type_def.py` uses `"DataSet"` /
  `"GeoDataSet"` but factory/sources use `"Dataset"` / `"GeoDataset"`.
- `contains_source` in `data_catalog.py` (~line 509) has suspicious boolean
  inversion — verify carefully before refactoring.
- Docs reference "black" and pip examples; actual tooling is ruff-format + pixi.
- `_compat.py` Python <3.10 fallback is dead code; do not add more.
