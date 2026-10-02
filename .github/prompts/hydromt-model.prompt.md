---
mode: agent
description: >
  Expert guidance for working on HydroMT's Model abstraction and
  ModelComponent system: adding model steps, implementing new components,
  or changing Model base-class behaviour. Auto-invoked when editing files
  under hydromt/model/.
applyTo: "hydromt/model/**"
---

# HydroMT Model & Component skill

You are working on the **HydroMT Model** subsystem (`hydromt/model/`). Follow
the rules below precisely.

---

## Key classes and files

| Class | File | Purpose |
|---|---|---|
| `Model` | `model/model.py` | Base class for all HydroMT models |
| `ModelRoot` | `model/root.py` | Filesystem mode/path management |
| `ModelComponent` | `model/components/base.py` | ABC for all model artefact components |
| `HydromtModelSetup` | `_validators/model_config.py` | Pydantic validator for workflow YAML |
| `@hydromt_step` | `model/steps.py` | Decorator gating YAML-callable methods |

---

## `Model` base class — core methods

```python
model.build(region, res, write=True)   # orchestrate build workflow
model.update(…)                         # update existing model
model.read()                            # read all components from disk
model.write()                           # write all components to disk
model.test_equal(other) -> (bool, dict) # compare two model instances
```

`model.components["name"]` gives access to registered `ModelComponent`
instances. Components are registered by passing them to the `Model` constructor
or via plugin entry points.

---

## Adding a new `@hydromt_step` method

A step is a public `Model` method callable from workflow YAML:

```python
from hydromt.model.steps import hydromt_step

class MyModel(Model):

    @hydromt_step
    def setup_dem(
        self,
        dem_fn: str,          # catalog key or file path
        res: float = 100.0,
        crs: int | None = None,
    ) -> None:
        """Set up DEM grid.

        Parameters
        ----------
        dem_fn : str
            Catalog key or path to DEM source.
        res : float, optional
            Target resolution in metres.
        crs : int, optional
            EPSG code. Defaults to model CRS.
        """
        ...
```

### Rules for `@hydromt_step` methods
1. **All positional arguments must be JSON-safe**: `str`, `int`, `float`,
   `bool`, `None`, `list`, `dict`. No xarray objects, GeoDataFrames, etc.
2. Argument names ending in `_fn` are resolved as catalog keys or file paths
   via the model's `DataCatalog`.
3. The method is validated against the YAML step schema at workflow-load time
   (`_validators/model_config.py`). Signature mismatches fail at load, not at
   runtime.
4. Do not decorate private helpers or utility methods with `@hydromt_step`.

---

## Implementing a new `ModelComponent`

```python
from hydromt.model.components.base import ModelComponent

class MyComponent(ModelComponent):
    # Required — declare component data type
    _data: MyDataType | None = None

    def read(self, fn: str = "mydata.nc", **kwargs) -> None:
        """Read component from disk."""
        ...

    def write(self, fn: str = "mydata.nc", **kwargs) -> None:
        """Write component to disk.

        IMPORTANT: write() must have NO required positional arguments.
        It is called as component.write() by Model.write().
        """
        ...

    def test_equal(self, other: "MyComponent") -> tuple[bool, dict[str, str]]:
        """Return (all_equal, {field: error_message}) accumulating all diffs."""
        errors: dict[str, str] = {}
        # ... compare fields, add to errors dict
        return len(errors) == 0, errors

    def close(self) -> None:
        """Release any open file handles."""
        ...
```

### `ModelComponent` contract checklist
- [ ] `write()` has **no required positional args**.
- [ ] `test_equal()` returns `tuple[bool, dict[str, str]]` and accumulates
  all errors rather than raising on the first difference.
- [ ] `read()` and `write()` use `self._root` (a `ModelRoot`) for FS paths.
- [ ] Register via entry-point `[project.entry-points."hydromt.components"]`
  in `pyproject.toml` if intended to be discoverable by downstream plugins.

---

## Workflow YAML structure (for reference)

```yaml
modeltype: MyModel
global:
  data_libs: [artifact_data]
  region: {geom: path/to/region.geojson}
build:
  - setup_dem:
      dem_fn: merit_hydro
      res: 100.0
  - setup_rivers:
      river_fn: rivers_glob_v1
```

Each step name must match a `@hydromt_step`-decorated method on the model.
Step arguments are validated by `HydromtModelSetup` against the method
signature.

---

## `ModelRoot` — filesystem state

`model._root` is a `ModelRoot` instance managing:
- `model._root.path` — absolute root directory.
- `model._root.mode` — `"r"` (read-only), `"w"` (new/overwrite), `"w+"` (append).

Access paths for components via:
```python
component_path = self._root.path / "subdir" / filename
```

Do not hardcode absolute paths; always go through `_root`.

---

## Testing model changes

- Test files: `tests/model/`, `tests/components/`.
- Fixtures: `tests/conftest.py` (shared model instances, temp dirs).
- `test_equal` pattern — write tests that call `model.test_equal(other)` and
  assert `equal is True` with `errors == {}`.
- Mark tests requiring disk I/O with `tmp_path` fixture (pytest built-in).
- Run: `pixi run test tests/model/ tests/components/`

---

## Domain vocabulary

| Term | Meaning in model context |
|---|---|
| `_fn` arg suffix | Catalog key or file path resolved via `DataCatalog` |
| `region` | Spatial bounding geometry/mask defining the model domain |
| `res` | Spatial resolution (usually in metres or degrees) |
| `flwdir` | Flow direction grid (D8/LDD encoding) |
| `basid` | Sub-basin identifier integer |
| PET | Potential evapotranspiration (see `model/processes/meteo.py`) |

---

## Checklist before opening a PR

- [ ] New `@hydromt_step` methods have only JSON-safe arg types.
- [ ] `ModelComponent.write()` has no required positional args.
- [ ] `test_equal()` accumulates errors into a dict rather than raising.
- [ ] NumPy docstrings on all public methods.
- [ ] Tests added under `tests/model/` or `tests/components/`.
- [ ] `docs/changelog.rst` updated.
