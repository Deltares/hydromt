---
mode: agent
description: >
  Expert guidance for working on the HydroMT plugin system: registering new
  plugins (models, components, drivers, catalogs, URI resolvers), writing
  downstream plugin packages, and understanding plugin discovery and
  validation. Auto-invoked when editing hydromt/plugins.py or
  pyproject.toml entry-point sections.
applyTo: "hydromt/plugins.py"
---

# HydroMT Plugin system skill

You are working on the **HydroMT plugin system** (`hydromt/plugins.py`). This
system enables downstream packages (`hydromt_sfincs`, `hydromt_wflow`, …) to
extend HydroMT by registering their own models, components, drivers, and more.

---

## Five plugin entry-point groups

| Group | Base class | Registered by |
|---|---|---|
| `hydromt.models` | `hydromt.Model` | Downstream model packages |
| `hydromt.components` | `hydromt.ModelComponent` | Core + downstream |
| `hydromt.drivers` | `hydromt.data_catalog.drivers.BaseDriver` | Core + downstream |
| `hydromt.catalogs` | `hydromt.data_catalog.sources.DataSource` | Core + downstream |
| `hydromt.uri_resolvers` | `hydromt.data_catalog.uri_resolvers.URIResolver` | Core + downstream |

---

## Registering a plugin in `pyproject.toml`

```toml
[project.entry-points."hydromt.models"]
"MyModel" = "my_package.models.my_model:MyModel"

[project.entry-points."hydromt.components"]
"MyComponent" = "my_package.components.my_component:MyComponent"

[project.entry-points."hydromt.drivers"]
"MyDriver" = "my_package.drivers.my_driver:MyDriver"

[project.entry-points."hydromt.catalogs"]
"MyDataSource" = "my_package.sources.my_source:MyDataSource"

[project.entry-points."hydromt.uri_resolvers"]
"MyResolver" = "my_package.resolvers.my_resolver:MyResolver"
```

The key string (e.g. `"MyModel"`) is the **plugin name** used at runtime. It
must be unique across all installed packages for a given group.

---

## Multiple names for one class (`__hydromt_eps__`)

If a class should be discoverable under more than one name (e.g. for backwards
compatibility), declare:

```python
class MyDriver(BaseDriver):
    __hydromt_eps__: ClassVar[list[str]] = ["MyDriver", "my_driver_legacy"]
```

And register both names in `pyproject.toml`. The plugin registry will validate
that both names point to the same class.

---

## Plugin discovery at runtime

`hydromt/plugins.py` builds a global `PLUGINS` registry object. It:

1. Calls `importlib.metadata.entry_points(group="hydromt.*")` for each group.
2. Loads each entry point and checks `issubclass(cls, expected_base)`.
3. Raises `PluginError` on duplicate names or subclass failures.
4. Exposes `PLUGINS.models`, `PLUGINS.components`, `PLUGINS.drivers`, etc.

`PLUGINS` is importable from `hydromt` directly:
```python
from hydromt import PLUGINS
my_model_cls = PLUGINS.models["MyModel"]
```

### Plugin validation rules (enforced at import time)
- The registered class **must** be a subclass of the expected base.
- Plugin names **must** be unique within their group.
- If `__hydromt_eps__` is declared, all listed names must be registered and
  resolve to the same class.

---

## Writing a downstream plugin package (checklist)

When creating a new downstream HydroMT plugin package:

```
my_package/
  __init__.py
  models/
    my_model.py          # subclasses hydromt.Model
  components/            # optional, if adding new component types
  drivers/               # optional, if adding custom I/O drivers
  sources/               # optional, if adding new DataSource types
pyproject.toml           # registers entry points
```

**`pyproject.toml` minimum for a model plugin:**

```toml
[project]
name = "hydromt-mypackage"
dependencies = ["hydromt>=X.Y"]

[project.entry-points."hydromt.models"]
"MyModel" = "my_package.models.my_model:MyModel"

[build-system]
requires = ["flit_core>=3.2"]
build-backend = "flit_core.buildapi"
```

---

## `AbstractBaseModel` — polymorphic Pydantic deserialization

For Pydantic models that need to deserialize from a `"name"` discriminator field
(e.g. driver options, resolver configs), subclass `AbstractBaseModel`:

```python
from hydromt._abstract_base import AbstractBaseModel

class MyDriverOptions(AbstractBaseModel):
    name: Literal["MyDriver"] = "MyDriver"
    compression: str = "lzw"
```

`AbstractBaseModel` enables `Union[DriverOptionsA, DriverOptionsB]` to
deserialize correctly by inspecting the `"name"` field.

---

## Testing plugin registration

Tests for plugin discovery live in `tests/` (look for `test_plugins.py` or
`test_*plugin*`). When adding a new plugin:

1. Verify it is discoverable: `PLUGINS.models["MyModel"]` should not raise.
2. Verify subclass check: `issubclass(PLUGINS.models["MyModel"], Model)`.
3. Test duplicate-name rejection if applicable.

Run: `pixi run test tests/` (or target the specific test file).

---

## Optional dependency guards

If a plugin requires an optional dependency, guard the import:

```python
# hydromt/_compat.py pattern
try:
    import s3fs
    HAS_S3FS = True
except ImportError:
    HAS_S3FS = False
```

Then in the plugin:
```python
from hydromt._compat import HAS_S3FS

class S3Driver(BaseDriver):
    def read(self, uris, **kwargs):
        if not HAS_S3FS:
            raise ImportError("s3fs is required for S3Driver. Install with: pip install s3fs")
        ...
```

And in tests:
```python
import pytest
from hydromt._compat import HAS_S3FS

@pytest.mark.skipif(not HAS_S3FS, reason="s3fs not installed")
def test_s3_driver(): ...
```

---

## Checklist before opening a PR

- [ ] Entry points registered in `pyproject.toml` under correct group.
- [ ] Class is a subclass of the expected base (verified by `issubclass`).
- [ ] `__hydromt_eps__` declared if multiple names are needed.
- [ ] Optional deps guarded via `_compat.py` + `pytest.mark.skipif`.
- [ ] Plugin discovery tested (assert `PLUGINS.group["Name"]` works).
- [ ] `docs/changelog.rst` updated.
