from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from hydromt.drivers.zarr_driver import ZarrDriver
from hydromt.metadata_resolver.convention_resolver import ConventionResolver


class TestZarrDriver:
    @pytest.fixture()
    def example_zarr_file(self, tmp_dir: Path) -> Path:
        tmp_path: Path = tmp_dir / "0s.zarr"
        store = zarr.DirectoryStore(tmp_path)
        root: zarr.Group = zarr.group(store=store, overwrite=True)
        zarray_var: zarr.Array = root.zeros(
            "variable", shape=(10, 10), chunks=(5, 5), dtype="int8"
        )
        zarray_var[0, 0] = 42  # trigger write
        zarray_var.attrs.update(
            {
                "_ARRAY_DIMENSIONS": ["x", "y"],
                "coordinates": "xc yc",
                "long_name": "Test Array",
                "type_preferred": "int8",
            }
        )
        # create symmetrical coords
        xy = np.linspace(0, 9, 10, dtype=np.dtypes.Int8DType)
        xcoords, ycoords = np.meshgrid(xy, xy)

        zarray_x: zarr.Array = root.array("xc", xcoords, chunks=(5, 5), dtype="int8")
        zarray_x.attrs["_ARRAY_DIMENSIONS"] = ["x", "y"]
        zarray_y: zarr.Array = root.array("yc", ycoords, chunks=(5, 5), dtype="int8")
        zarray_y.attrs["_ARRAY_DIMENSIONS"] = ["x", "y"]
        zarr.consolidate_metadata(store)
        store.close()
        return (root, tmp_path)

    def test_zarr_read(self, example_zarr_file: Path):
        res: xr.Dataset = ZarrDriver(metadata_resolver=ConventionResolver()).read(
            str(example_zarr_file[1])
        )
        assert list(res.data_vars.keys()) == ["variable"]
        assert res["variable"].shape == (10, 10)
        assert list(res.coords.keys()) == ["xc", "yc"]
        assert res["variable"].values[0, 0] == 42

    def test_zarr_write(self, raster_ds: xr.Dataset, tmp_dir: Path):
        zarr_path: Path = tmp_dir / "raster.zarr"
        driver = ZarrDriver()
        driver.write(zarr_path, raster_ds)
        assert np.all(driver.read(str(zarr_path)) == raster_ds)
