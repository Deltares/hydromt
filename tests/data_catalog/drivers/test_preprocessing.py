import numpy as np
import pandas as pd
import xarray as xr

from hydromt.data_catalog.drivers.preprocessing import (
    harmonise_dims,
    round_latlon,
)
from hydromt.gis.raster_utils import full_from_transform


def test_round_latlon():
    da: xr.DataArray = full_from_transform(
        transform=[1.0, 0.0, 0.0005, 0.0, -1.0, 0.0],
        shape=(4, 6),
        nodata=-1,
        name="test",
        crs=4326,
    )

    ds: xr.Dataset = da.to_dataset(name="test")
    res: xr.Dataset = round_latlon(ds, decimals=3)
    assert np.equal(np.round(res.x.data[0], 3), 0.5)


def _make_ds(x, y, crs=None):
    t = pd.date_range("2020-01-01", periods=2, freq="D")
    data = np.zeros((len(t), len(y), len(x)))
    ds = xr.Dataset(
        {"precip": (["time", "y", "x"], data)},
        coords={"x": x, "y": y, "time": t},
    )
    if crs is not None:
        ds.raster.set_crs(crs)
    return ds


def test_harmonise_dims_geographic_normalises_longitude():
    # A geographic dataset in the 0-360 convention should be wrapped to
    # -180-180 and sorted west-to-east.
    x = np.array([0.0, 90.0, 270.0, 350.0])
    y = np.array([10.0, 5.0, 0.0])
    res = harmonise_dims(_make_ds(x, y, crs=4326))
    np.testing.assert_array_equal(res.x.values, [-90.0, -10.0, 0.0, 90.0])


def test_harmonise_dims_no_crs_does_not_shift_projected(caplog):
    # Projected coordinates (metres, all > 180) without an embedded CRS must
    # NOT be normalised by -360, and a warning should be emitted (#1476).
    x = np.arange(10490.0, 10490.0 + 30 * 30, 30.0)
    y = np.arange(39010.0, 39010.0 + 30 * 30, 30.0)[::-1]
    res = harmonise_dims(_make_ds(x, y))
    assert res.x.values[0] == 10490.0
    np.testing.assert_array_equal(res.x.values, x)
    assert any("no CRS" in rec.message for rec in caplog.records)


def test_harmonise_dims_projected_crs_does_not_shift():
    # A dataset with a projected CRS (metres) must be left untouched, silently.
    x = np.arange(10490.0, 10490.0 + 30 * 30, 30.0)
    y = np.arange(39010.0, 39010.0 + 30 * 30, 30.0)[::-1]
    res = harmonise_dims(_make_ds(x, y, crs=3414))
    np.testing.assert_array_equal(res.x.values, x)
