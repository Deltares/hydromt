# -*- coding: utf-8 -*-
"""Tests for the hydromt.data_catalog.adapters submodule."""

import glob
from os.path import abspath, dirname, join
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr

from hydromt import _compat as compat
from hydromt.data_catalog import DataCatalog
from hydromt.data_catalog.sources import RasterDatasetSource

TESTDATADIR = join(dirname(abspath(__file__)), "..", "data")
CATALOGDIR = join(dirname(abspath(__file__)), "..", "..", "data", "catalogs")


@pytest.mark.skip(reason="Needs refactor from path to uri.")
@pytest.mark.skipif(not compat.HAS_GCSFS, reason="GCSFS not installed.")
def test_gcs_cmip6():
    # TODO switch to pre-defined catalogs when pushed to main
    catalog_fn = join(CATALOGDIR, "gcs_cmip6_data", "v0.1.0", "data_catalog.yml")
    data_catalog = DataCatalog(data_libs=[catalog_fn])
    ds = data_catalog.get_rasterdataset(
        "cmip6_NOAA-GFDL/GFDL-ESM4_historical_r1i1p1f1_Amon",
        variables=["precip", "temp"],
        time_tuple=(("1990-01-01", "1990-03-01")),
    )
    # Check reading and some preprocess
    assert "precip" in ds
    assert not np.any(ds[ds.raster.x_dim] > 180)
    # Skip as I don't think this adds value to testing a gcs cloud archive
    # Write and compare
    # fn_nc = str(tmpdir.join("test.nc"))
    # ds.to_netcdf(fn_nc)
    # ds1 = data_catalog.get_rasterdataset(fn_nc)
    # assert np.allclose(ds["precip"][0, :, :], ds1["precip"][0, :, :])


@pytest.mark.skip(reason="Needs implementation of all Drivers.")
@pytest.mark.skipif(not compat.HAS_S3FS, reason="S3FS not installed.")
def test_aws_worldcover():
    catalog_fn = join(CATALOGDIR, "aws_data", "v0.1.0", "data_catalog.yml")
    data_catalog = DataCatalog(data_libs=[catalog_fn])
    da = data_catalog.get_rasterdataset(
        "esa_worldcover_2020_v100",
        bbox=[12.0, 46.0, 12.5, 46.50],
    )
    assert da.name == "landuse"


@pytest.mark.skip(
    "Needs implementation of https://github.com/Deltares/hydromt/issues/875"
)
@pytest.mark.integration()
def test_rasterdataset_zoomlevels(
    rioda_large: xr.DataArray, tmp_dir: Path, data_catalog: DataCatalog
):
    # write tif with zoom level 1 in name
    # NOTE zl 0 not written to check correct functioning
    name = "test_zoom"
    rioda_large.raster.to_raster(str(tmp_dir / "test_zl1.tif"))
    yml_dict = {
        name: {
            "data_type": "RasterDataset",
            "driver": {"name": "rasterio"},
            "uri": f"{str(tmp_dir)}/test_zl{{zoom_level:d}}.tif",  # test with str format for zoom level
            "metadata": {
                "crs": 4326,
            },
            "zoom_levels": {0: 0.1, 1: 0.3},
        }
    }
    # test zoom levels in name
    data_catalog.from_dict(yml_dict)
    rds: RasterDatasetSource = cast(RasterDatasetSource, data_catalog.get_source(name))
    assert rds._parse_zoom_level(None) is None
    assert rds._parse_zoom_level(zoom_level=1) == 1
    assert rds._parse_zoom_level(zoom_level=(0.3, "degree")) == 1
    assert rds._parse_zoom_level(zoom_level=(0.29, "degree")) == 0
    assert rds._parse_zoom_level(zoom_level=(0.1, "degree")) == 0
    assert rds._parse_zoom_level(zoom_level=(1, "meter")) == 0
    with pytest.raises(TypeError, match="zoom_level unit"):
        rds._parse_zoom_level(zoom_level=(1, "asfd"))
    with pytest.raises(TypeError, match="zoom_level not understood"):
        rds._parse_zoom_level(zoom_level=(1, "asfd", "asdf"))
    da1 = data_catalog.get_rasterdataset(name, zoom_level=(0.3, "degree"))
    assert isinstance(da1, xr.DataArray)
    # write COG
    cog_fn = str(tmp_dir / "test_cog.tif")
    rioda_large.raster.to_raster(cog_fn, driver="COG", overviews="auto")
    # test COG zoom levels
    # return native resolution
    res = np.asarray(rioda_large.raster.res)
    da1 = data_catalog.get_rasterdataset(cog_fn, zoom_level=0)
    assert np.allclose(da1.raster.res, res)
    # reurn zoom level 1
    da1 = data_catalog.get_rasterdataset(cog_fn, zoom_level=(res[0] * 2, "degree"))
    assert np.allclose(da1.raster.res, res * 2)
    # test if file hase no overviews
    tif_fn = str(tmp_dir / "test_tif_no_overviews.tif")
    rioda_large.raster.to_raster(tif_fn, driver="GTiff")
    da1 = data_catalog.get_rasterdataset(tif_fn, zoom_level=(0.01, "degree"))
    xr.testing.assert_allclose(da1, rioda_large)
    # test if file has {variable} in path
    da1 = data_catalog.get_rasterdataset("merit_hydro", zoom_level=(0.01, "degree"))
    assert isinstance(da1, xr.Dataset)


@pytest.mark.skip(
    reason="Needs implementation https://github.com/Deltares/hydromt/issues/875."
)
def test_reads_slippy_map_output(tmp_dir: Path, rioda_large: xr.DataArray):
    # write vrt data
    name = "tiled"
    root = tmp_dir / name
    rioda_large.raster.to_xyz_tiles(
        root=root,
        tile_size=256,
        zoom_levels=[0],
    )
    cat = DataCatalog(str(root / f"{name}.yml"), cache=True)
    cat.get_rasterdataset(name)
    assert len(glob.glob(join(cat._cache_dir, name, name, "*", "*", "*.tif"))) == 16
