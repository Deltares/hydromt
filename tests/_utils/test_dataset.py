import numpy as np
import xarray as xr

from hydromt._utils.dataset import _test_equal_grid_data


def make_grid(
    data: dict[str, np.ndarray] | None = None,
    crs: int = 4326,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> xr.Dataset:
    """Build a minimal raster-aware xr.Dataset."""
    if x is None:
        x = np.linspace(0, 1, 5)
    if y is None:
        y = np.linspace(0, 1, 4)
    if data is None:
        data = {"var1": np.ones((len(y), len(x)))}

    data_vars = {name: xr.DataArray(arr, dims=["y", "x"]) for name, arr in data.items()}
    ds = xr.Dataset(data_vars, coords={"x": x, "y": y})
    ds = ds.rio.write_crs(crs)
    return ds


def test_both_empty():
    """Two empty datasets are considered equal."""
    eq, errors = _test_equal_grid_data(xr.Dataset(), xr.Dataset())
    assert eq
    assert errors == {}


def test_first_empty_second_not():
    """An empty first grid compared to a non-empty second grid returns an error."""
    other = make_grid()
    eq, errors = _test_equal_grid_data(xr.Dataset(), other)
    assert not eq
    assert "grid" in errors


def test_second_empty_first_not():
    """A non-empty first grid compared to an empty second grid flags all maps as missing."""
    grid = make_grid()
    eq, errors = _test_equal_grid_data(grid, xr.Dataset())
    assert not eq
    assert any("missing" in k for k in errors)


def test_different_crs():
    """Grids with different CRS are flagged with a crs error."""
    grid = make_grid(crs=4326)
    other = make_grid(crs=32631)
    eq, errors = _test_equal_grid_data(grid, other)
    assert not eq
    assert "crs" in errors


def test_identical_crs():
    """Grids with the same CRS and data produce no errors."""
    grid = make_grid(crs=4326)
    other = make_grid(crs=4326)
    eq, errors = _test_equal_grid_data(grid, other)
    assert eq
    assert errors == {}


def test_different_dim_values():
    """Grids with the same shape but different coordinate values are flagged."""
    grid = make_grid(x=np.linspace(0, 1, 5))
    other = make_grid(x=np.linspace(0, 2, 5))
    eq, errors = _test_equal_grid_data(grid, other)
    assert not eq
    assert "dims" in errors


def test_missing_dim_in_grid():
    """A dimension present in other but absent from grid is flagged."""
    grid = make_grid()
    other = make_grid()
    other = other.expand_dims("time")
    eq, errors = _test_equal_grid_data(grid, other)
    assert not eq
    assert "dims" in errors


def test_other_has_additional_map():
    """A map present in other but not in grid is reported as additional."""
    grid = make_grid(data={"var1": np.ones((4, 5))})
    other = make_grid(data={"var1": np.ones((4, 5)), "var2": np.zeros((4, 5))})
    eq, errors = _test_equal_grid_data(grid, other)
    assert not eq
    assert any("additional" in k for k in errors)


def test_other_missing_map():
    """A map present in grid but absent from other is reported as missing."""
    grid = make_grid(data={"var1": np.ones((4, 5)), "var2": np.zeros((4, 5))})
    other = make_grid(data={"var1": np.ones((4, 5))})
    eq, errors = _test_equal_grid_data(grid, other)
    assert not eq
    assert any("missing" in k for k in errors)


def test_identical_grids():
    """Two grids with identical data, coords, and CRS are considered equal."""
    data = {"var1": np.ones((4, 5)), "var2": np.arange(20).reshape(4, 5).astype(float)}
    grid = make_grid(data=data)
    other = make_grid(data=data)
    eq, errors = _test_equal_grid_data(grid, other)
    assert eq
    assert errors == {}


def test_map_value_mismatch():
    """Maps with different values are reported as invalid."""
    grid = make_grid(data={"var1": np.ones((4, 5))})
    other = make_grid(data={"var1": np.zeros((4, 5))})
    eq, errors = _test_equal_grid_data(grid, other)
    assert not eq
    assert any("invalid" in k for k in errors)


def test_map_value_within_tolerance():
    """Map values differing by less than atol=1e-3 are considered equal."""
    grid = make_grid(data={"var1": np.ones((4, 5))})
    other = make_grid(data={"var1": np.ones((4, 5)) + 1e-4})
    eq, errors = _test_equal_grid_data(grid, other)
    assert eq
    assert errors == {}


def test_map_value_outside_tolerance():
    """Map values differing by more than atol=1e-3 are reported as invalid."""
    grid = make_grid(data={"var1": np.ones((4, 5))})
    other = make_grid(data={"var1": np.ones((4, 5)) + 0.1})
    eq, errors = _test_equal_grid_data(grid, other)
    assert not eq
    assert any("invalid" in k for k in errors)


def test_map_dtype_mismatch():
    """Maps with the same values but different dtypes are reported as invalid."""
    grid = make_grid(data={"var1": np.ones((4, 5), dtype=np.float32)})
    other = make_grid(data={"var1": np.ones((4, 5), dtype=np.float64)})
    eq, errors = _test_equal_grid_data(grid, other)
    assert not eq
    invalid_map_errors = next(v for k, v in errors.items() if "invalid" in k)
    assert "var1" in invalid_map_errors
    assert (
        "float64" in invalid_map_errors["var1"]
        or "float32" in invalid_map_errors["var1"]
    )


def test_map_nodata_mismatch():
    """Maps with different nodata values are reported as invalid."""
    arr = np.ones((4, 5))
    grid = make_grid(data={"var1": arr})
    other = make_grid(data={"var1": arr})
    grid["var1"].rio.write_nodata(-9999, inplace=True)
    other["var1"].rio.write_nodata(-1, inplace=True)
    eq, errors = _test_equal_grid_data(grid, other)
    assert not eq
    invalid_map_errors = next(v for k, v in errors.items() if "invalid" in k)
    assert "nodata" in invalid_map_errors["var1"]


def test_map_nodata_both_nan():
    """Maps where both nodata values are NaN are considered equal despite NaN != NaN."""
    arr = np.ones((4, 5))
    grid = make_grid(data={"var1": arr})
    other = make_grid(data={"var1": arr})
    grid["var1"].rio.write_nodata(np.nan, inplace=True)
    other["var1"].rio.write_nodata(np.nan, inplace=True)
    eq, errors = _test_equal_grid_data(grid, other)
    assert eq
    assert errors == {}


def test_coord_mismatch():
    """Coordinates present in both grids but with different values are flagged."""
    grid = make_grid()
    other = make_grid()
    grid = grid.assign_coords(mask=xr.DataArray(np.ones((4, 5)), dims=["y", "x"]))
    other = other.assign_coords(mask=xr.DataArray(np.zeros((4, 5)), dims=["y", "x"]))
    eq, errors = _test_equal_grid_data(grid, other)
    assert not eq
    assert any("invalid coords" in k for k in errors)


def test_coord_only_in_grid_not_other():
    """Coordinates present only in grid and not in other are silently skipped."""
    grid = make_grid()
    other = make_grid()
    grid = grid.assign_coords(extra=xr.DataArray(np.ones((4, 5)), dims=["y", "x"]))
    eq, errors = _test_equal_grid_data(grid, other)
    assert "invalid coords" not in str(errors)


def test_3d_map_value_mismatch():
    """3D maps with different values are reported as invalid using only the first slice."""
    arr_a = np.ones((3, 4, 5))
    arr_b = np.zeros((3, 4, 5))
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 4)
    t = np.arange(3)
    grid = xr.Dataset(
        {"var1": xr.DataArray(arr_a, dims=["time", "y", "x"])},
        coords={"x": x, "y": y, "time": t},
    ).rio.write_crs(4326)
    other = xr.Dataset(
        {"var1": xr.DataArray(arr_b, dims=["time", "y", "x"])},
        coords={"x": x, "y": y, "time": t},
    ).rio.write_crs(4326)
    eq, errors = _test_equal_grid_data(grid, other)
    assert not eq
    invalid_map_errors = next(v for k, v in errors.items() if "invalid" in k)
    assert "var1" in invalid_map_errors


def test_multiple_errors_collected():
    """All error categories — crs, missing, additional, invalid — are collected in one pass."""
    grid = make_grid(data={"var1": np.ones((4, 5)), "var2": np.ones((4, 5))}, crs=4326)
    other = make_grid(
        data={"var1": np.zeros((4, 5)), "var3": np.ones((4, 5))}, crs=32631
    )
    eq, errors = _test_equal_grid_data(grid, other)
    assert not eq
    assert "crs" in errors
    assert any("missing" in k for k in errors)
    assert any("additional" in k for k in errors)
    assert any("invalid" in k for k in errors)
