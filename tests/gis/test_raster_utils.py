import numpy as np
import pytest
import xarray as xr
from dask.array import Array

from hydromt.gis.raster_utils import full


def test_full():
    # Call the function
    da = full(
        coords={"x": np.array([1, 2, 3]), "y": np.array([6, 7, 8])},
        nodata=-1,
        shape=(3, 3),
        name="test",
        attrs={"foo": "bar"},
        crs=4326,
    )

    # Assert the output
    # Typing
    assert isinstance(da, xr.DataArray)
    assert isinstance(da.data, np.ndarray)
    # Content
    assert np.all(da.values == -1)
    assert da.shape == (3, 3)
    assert da.raster.nodata == -1
    assert da.raster.dims == ("y", "x")
    assert da.name == "test"
    assert da.raster.crs.to_epsg() == 4326


def test_full_lazy():
    # Call the function
    da = full(
        coords={"x": np.array([1, 2, 3]), "y": np.array([6, 7, 8])},
        nodata=-1,
        lazy=True,
    )

    # Assert the output, it's lazy now
    assert isinstance(da.data, Array)
    assert da.shape == (3, 3)
    assert np.all(da.values == -1)  # Upon request


def test_full_fill_value():
    # Call the function
    da = full(
        coords={"x": np.array([1, 2, 3]), "y": np.array([6, 7, 8])},
        nodata=-1,
        fill_value=1,
    )

    # Assert the output
    assert da.raster.nodata == -1
    assert np.all(da.values == 1)


def test_full_rotate():
    # Call the function
    da = full(
        coords={"x": np.array([1, 2, 3]), "y": np.array([6, 7, 8])},
    )

    # Assert the output
    assert isinstance(da, xr.DataArray)


def test_full_errors():
    # Call the function with the wrong shape
    with pytest.raises(
        ValueError,
        match="conflicting sizes for dimension 'x'",
    ):
        _ = full(
            coords={"x": np.array([1, 2, 3]), "y": np.array([6, 7, 8])},
            shape=[4, 5],  # Obviously wrong
        )
