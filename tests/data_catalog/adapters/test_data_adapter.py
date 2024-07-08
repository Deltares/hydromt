# -*- coding: utf-8 -*-
"""Tests for the hydromt.data_catalog.adapters submodule."""

from os.path import abspath, dirname, join

import numpy as np
import pytest

from hydromt import _compat as compat
from hydromt.data_catalog import DataCatalog

TESTDATADIR = join(dirname(abspath(__file__)), "..", "data")
CATALOGDIR = join(dirname(abspath(__file__)), "..", "..", "data", "catalogs")


@pytest.mark.skip(reason="https://github.com/Deltares/hydromt/issues/1025")
@pytest.mark.skipif(not compat.HAS_GCSFS, reason="GCSFS not installed.")
def test_gcs_cmip6():
    # TODO switch to pre-defined catalogs when pushed to main
    catalog_path = join(CATALOGDIR, "gcs_cmip6_data", "v0.1.0", "data_catalog.yml")
    data_catalog = DataCatalog(data_libs=[catalog_path])
    ds = data_catalog.get_rasterdataset(
        "cmip6_NOAA-GFDL/GFDL-ESM4_historical_r1i1p1f1_Amon",
        variables=["precip", "temp"],
        time_range=(("1990-01-01", "1990-03-01")),
    )
    # Check reading and some preprocess
    assert "precip" in ds
    assert not np.any(ds[ds.raster.x_dim] > 180)


@pytest.mark.skip(reason="https://github.com/Deltares/hydromt/issues/1025")
@pytest.mark.skipif(not compat.HAS_S3FS, reason="S3FS not installed.")
def test_aws_worldcover():
    catalog_path = join(CATALOGDIR, "aws_data", "v0.1.0", "data_catalog.yml")
    data_catalog = DataCatalog(data_libs=[catalog_path])
    da = data_catalog.get_rasterdataset(
        "esa_worldcover_2020_v100",
        bbox=[12.0, 46.0, 12.5, 46.50],
    )
    assert da.name == "landuse"
