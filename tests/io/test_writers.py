import io
import logging
import sys
from pathlib import Path

import pytest
import xarray as xr

from hydromt.io.writers import write_nc


def test_write_nc(tmp_path: Path, demda: xr.DataArray):
    # Call the function in its default state
    write_nc(ds=demda, file_path=Path(tmp_path, "foo.nc"))

    # Assert the file is there
    assert Path(tmp_path, "foo.nc").is_file()

    # Assert some simple things regarding the content
    ds = xr.open_dataset(Path(tmp_path, "foo.nc"))
    assert "foo" in ds.data_vars
    assert "x" in ds.dims
    assert ds.attrs == {}
    assert ds.foo.attrs == {}
    assert "foo" not in ds.encoding


def test_write_nc_compress(tmp_path: Path, raster_ds: xr.Dataset):
    # Call the function and write normally
    write_nc(ds=raster_ds, file_path=Path(tmp_path, "foo.nc"))

    # Call the function but compress
    write_nc(ds=raster_ds, file_path=Path(tmp_path, "foo-c.nc"), compress=True)

    # Assert that the output should be larger due to extra information for compression
    # This is funny as compression only makes a file smaller when it had a
    # considerable size to begin with
    assert (
        Path(tmp_path, "foo-c.nc").stat().st_size
        > Path(tmp_path, "foo.nc").stat().st_size
    )


def test_write_nc_gdal_compliant(tmp_path: Path, rioda: xr.DataArray):
    # Call the function with gdal compliant set to True
    write_nc(ds=rioda, file_path=Path(tmp_path, "foo.nc"), gdal_compliant=True)
    write_nc(
        ds=rioda,
        file_path=Path(tmp_path, "foo_r.nc"),
        gdal_compliant=True,
        rename_dims=True,
    )

    # Assert it's content, also compared to the default
    ds = xr.open_dataset(Path(tmp_path, "foo.nc"))
    assert "grid_mapping" in ds.test.attrs
    assert "x" in ds.dims
    ds = xr.open_dataset(Path(tmp_path, "foo_r.nc"))
    assert "x" not in ds.dims
    assert "longitude" in ds.dims


def test_write_nc_no_progress(
    tmp_path: Path,
    rioda: xr.DataArray,
):
    # Redirect the stdout as caplog only catches logging from the logging module
    s = io.StringIO()
    sys.stdout = s
    # Call the function with the progressbar disabled
    write_nc(ds=rioda, file_path=Path(tmp_path, "foo.nc"), progressbar=False)

    # Assert the logging output
    s.seek(0)
    assert "#####" not in s.read()


def test_write_nc_with_progress(
    tmp_path: Path,
    rioda: xr.DataArray,
):
    # Redirect the stdout as caplog only catches logging from the logging module
    s = io.StringIO()
    sys.stdout = s
    # Call the function with the progressbar enabled
    write_nc(ds=rioda, file_path=Path(tmp_path, "foo.nc"), progressbar=True)

    # Assert the logging output
    s.seek(0)
    assert "#####" in s.read()


def test_write_nc_wrong_type(caplog: pytest.LogCaptureFixture, tmp_path: Path):
    caplog.set_level(logging.WARNING)

    # Call the function with dumb input like an integer for a dataset
    write_nc(ds=2, file_path=Path(tmp_path, "foo.nc"))

    # Assert the logging message
    assert "Dataset object of type int not recognized" in caplog.text


def test_write_nc_errors(tmp_path: Path, raster_ds: xr.Dataset):
    # Create the file path and touch the file
    p = Path(tmp_path, "foo.nc")
    p.touch()

    # Now it exists, try to write without force_overwrite
    with pytest.raises(
        IOError,
        match=f"File {p.as_posix()} already exists",
    ):
        write_nc(ds=raster_ds, file_path=p)


def test_write_nc_compute_warns(tmp_path: Path, rioda: xr.DataArray):
    with pytest.raises(
        ValueError, match="'compute' argument is ignored in ds.to_netcdf function."
    ):
        write_nc(
            ds=rioda,
            file_path=Path(tmp_path, "foo.nc"),
            to_netcdf_kwargs={"compute": False},
        )
