---
name: code-review
description: Perform a code review of the current session's changes against HydroMT conventions. Use when the user requests a code review, clicks the Run Code Review button, or asks to review changes.
---

# Code Review (HydroMT)

You are a coding agent acting as a code reviewer for the **HydroMT** repository.
Review changed files and surface concrete, actionable issues as inline comments.

---

## Workflow

1. Identify changed files: `git diff --name-only HEAD` and `git status --short`.
2. For each changed file read the relevant ranges and review them against the codebase.
3. For every issue found, use the `addComment` tool to attach a comment to the exact file URI and line range.
4. Do **not** modify files, commit, or push. Output is review comments only.
5. When all changed files are reviewed, stop.

---

## HydroMT-specific review checklist

Work through these checks for **every changed file** before moving on.

### Architecture invariants

- [ ] **DataCatalog pipeline layers are not collapsed.** `DataSource`, `URIResolver`, `Driver`, and `DataAdapter` must stay separate — flag any method that does raw I/O *and* normalisation, or resolves URIs *and* reads.
- [ ] **`@hydromt_step` usage.** Any new `Model` method intended to be called from workflow YAML must carry `@hydromt_step`. All its positional args must be JSON-safe primitives (`str`, `int`, `float`, `bool`, `None`, `list`, `dict`) — flag objects, xarray types, GeoDataFrames, etc.
- [ ] **`NoDataStrategy` honoured.** New data-access paths must accept `handle_nodata: NoDataStrategy` and call `exec_nodata_strat` from `hydromt/error.py`. Flag silent returns of `None` or bare raises.
- [ ] **`ModelComponent.write()` signature.** `write()` must have no required positional arguments. Flag any required positional params added to `write`.
- [ ] **Plugin contracts intact.** New pluggable classes must declare the correct entry-point group and pass subclass checks. Flag missing `__hydromt_eps__` when multiple names are needed.
- [ ] **Pydantic v2 only.** No new plain dataclasses or `TypedDict` for validated config — use `pydantic.BaseModel` or `AbstractBaseModel`.

### Correctness & bugs

- [ ] **Walrus-operator precedence.** Flag any `if x := expr is not None:` — the `is not None` binds tighter than `:=`, assigning a bool. Use `if (x := expr) is not None:` instead.
- [ ] **`DataType` literal casing.** Any new `data_type` literal must match `factory.py` casing (`"Dataset"`, `"GeoDataset"`), **not** `type_def.py` (`"DataSet"`, `"GeoDataSet"`).
- [ ] **Import-time side effects in `__init__.py`.** Do not reorder or remove the Rasterio `sys.excepthook` patch or the `netCDF4` import workaround.
- [ ] **Optional-dep guards.** Code that imports optional packages (`s3fs`, `adlfs`, `pyet`, …) must check the `HAS_*` flag from `hydromt/_compat.py` and raise a helpful `ImportError` if absent.

### Conventions

- [ ] **NumPy docstrings** on all public functions/methods/classes. Flag missing or `Args:`-style docstrings on public API.
- [ ] **`_fn` suffix** on arguments that accept a catalog key or file path.
- [ ] **Naming conventions**: model subclasses `<Name>Model`, driver subclasses `<Name>Driver`, component subclasses `<Name>Component`.
- [ ] **Type hints** on all public function signatures.

### Tests & docs

- [ ] **Tests cover the change.** If behaviour changed, flag missing or insufficient test coverage.
- [ ] **Test structure mirrors source** (`tests/data_catalog/` for catalog changes, etc.). Flag misplaced tests.
- [ ] **`test_equal` pattern**: helpers that compare model state must return `tuple[bool, dict[str, str]]` accumulating all diffs — not raise on first failure.
- [ ] **`docs/changelog.rst` updated.** Flag if the PR modifies user-facing behaviour but `changelog.rst` is not touched.
- [ ] **No modifications to `data/catalogs/predefined_catalogs.yml`** unless that is the explicit goal.

### General

- [ ] Correctness and edge cases
- [ ] Security and data-handling issues
- [ ] Code clarity and consistency with surrounding code

---

## Comment quality

- Prefer **fewer, higher-signal** comments over stylistic nits.
- Each comment must explain *what* is wrong and *why* it matters.
- Be specific to the exact line range — do not leave per-file summary comments.
- Do not comment on things that are already correct.
