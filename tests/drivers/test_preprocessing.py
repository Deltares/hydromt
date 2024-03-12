import xarray as xr

from hydromt.driver.preprocessing import (
    round_latlon,
)
from hydromt.gis.raster import full_from_transform


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
    assert round(res.x.data[0], 3) == 0.5
