import numpy as np
import pytest
import xarray as xr
from affine import Affine
from dask.array import Array

from hydromt.gis.raster_utils import full, full_from_transform, full_like


def test_full():
    # Call the function
    da = full(
        coords={"y": np.array([8, 7, 6]), "x": np.array([1, 2, 3])},
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
        coords={"y": np.array([8, 7, 6]), "x": np.array([1, 2, 3])},
        nodata=-1,
        lazy=True,
    )

    # Assert the output, it's lazy now
    assert isinstance(da.data, Array)
    assert da.shape == (3, 3)
    assert np.all(da.values == -1)  # Upon request


def test_full_nan():
    # Call the function
    da = full(
        coords={"y": np.array([8, 7, 6]), "x": np.array([1, 2, 3])},
    )

    # Assert the output, nodata should be nan and the data should be nan
    assert np.isnan(da.raster.nodata)
    assert np.all(np.isnan(da.values))


def test_full_fill_value():
    # Call the function
    da = full(
        coords={"y": np.array([8, 7, 6]), "x": np.array([1, 2, 3])},
        nodata=-1,
        fill_value=1,
    )

    # Assert the output
    assert da.raster.nodata == -1
    assert np.all(da.values == 1)


def test_full_rotated():
    # Create coordinates
    yc = xr.DataArray(data=np.array([[2.0, 1.5], [1.0, 0.5]]), dims=["y", "x"])
    xc = xr.DataArray(data=np.array([[0.5, 1.5], [0.0, 1.0]]), dims=["y", "x"])
    # Call the function
    da = full(coords={"yc": yc, "xc": xc}, nodata=-1)

    # Assert the output
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("y", "x")
    assert tuple(da.coords) == ("yc", "xc", "spatial_ref")
    # Assert the rotation
    np.testing.assert_almost_equal(da.raster.transform[1], -0.5, decimal=4)


def test_full_not_rotated():
    # Call the function
    da = full(
        coords={"y": np.array([8, 7, 6]), "x": np.array([1, 2, 3])},
        nodata=-1,
    )

    # Assert no rotation, i.e. equal to zero
    np.testing.assert_almost_equal(da.raster.transform[1], -0.0)


def test_full_errors():
    # Call the function with the wrong shape
    with pytest.raises(
        ValueError,
        match="conflicting sizes for dimension 'y'",
    ):
        _ = full(
            coords={"y": np.array([8, 7, 6]), "x": np.array([1, 2, 3])},
            shape=(4, 5),  # Obviously wrong
        )


def test_full_from_transform():
    # Call the function
    da = full_from_transform(
        transform=(1.0, 0.0, 0.5, 0.0, -1.0, 8.5),
        shape=(3, 3),
        nodata=-1,
    )

    # Assert the output
    # which should be equal to our `test_full` in terms of coordinates
    assert da.shape == (3, 3)  # Yeah..
    np.testing.assert_array_almost_equal(da.y.values, [8, 7, 6])
    np.testing.assert_array_almost_equal(da.x.values, [1, 2, 3])


def test_full_from_transform_affine():
    # Call the function
    da = full_from_transform(
        transform=Affine(1.0, 0.0, 0.5, 0.0, -1.0, 8.5),
        shape=(5, 5),  # Lets do something else
    )

    # Assert the output
    # Coordinates should have a few extra
    assert da.shape == (5, 5)  # Yeah..
    np.testing.assert_array_almost_equal(da.y.values, [8, 7, 6, 5, 4])
    np.testing.assert_array_almost_equal(da.x.values, [1, 2, 3, 4, 5])


def test_full_from_transform_errors():
    # Call the function with to many dimensions
    with pytest.raises(
        ValueError,
        match="Only 2D and 3D data arrays supported.",
    ):
        _ = full_from_transform(
            transform=Affine(1.0, 0.0, 0.5, 0.0, -1.0, 8.5),
            shape=(1, 2, 3, 4),  # Cant do 4D data
        )

    # Call the function with to few dimensions
    with pytest.raises(
        ValueError,
        match="Only 2D and 3D data arrays supported.",
    ):
        _ = full_from_transform(
            transform=Affine(1.0, 0.0, 0.5, 0.0, -1.0, 8.5),
            shape=(1,),  # Cant do 1D data
        )


def test_full_like():
    # Create dummy DataArray
    dd = xr.DataArray(
        data=np.ones((3, 3)),
        coords={"y": np.array([8, 7, 6]), "x": np.array([1, 2, 3])},
        dims=("y", "x"),
        name="test",
    )

    # Call the function
    da = full_like(dd, nodata=-1)

    # Assert the output, coordinates should be the same
    assert np.all(da.y.values == dd.y.values)
    assert np.all(da.x.values == dd.x.values)
    assert da.raster.transform == dd.raster.transform


def test_full_like_errors():
    # Supply a nonsense object
    with pytest.raises(
        ValueError,
        match="other should be xarray.DataArray.",
    ):
        _ = full_like(2)
