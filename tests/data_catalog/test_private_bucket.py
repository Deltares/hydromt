from __future__ import annotations

import os
from pathlib import Path

import pytest
import xarray as xr
from botocore.exceptions import ClientError, ProfileNotFound

from hydromt._compat import HAS_BOTO3
from hydromt.data_catalog import DataCatalog
from hydromt.readers import open_mfdataset, open_zarrs

CATALOG_YAML = """
meta:
  version: v1
  hydromt_version: ">1.0a,<2"

merit_hydro:
  data_type: RasterDataset
  uri: s3://hydromt-data/merit_hydro/merit_hydro.zarr
  driver:
    name: raster_xarray
    filesystem:
      protocol: s3
      anon: false
      profile: hydromt-data
    options:
      zarr_format: 3
  metadata:
    crs: EPSG:4326
"""


def _require_or_skip_aws_profile(profile_name: str) -> None:
    """In CI, missing profile is a hard failure. Locally, skip the test."""
    if not HAS_BOTO3:
        pytest.skip("boto3 is required for this test. Install the `io` extra.")
    import boto3

    try:
        boto3.Session(profile_name=profile_name)
    except ProfileNotFound:
        if os.environ.get("CI"):
            # This env var is always True in gh actions
            # https://docs.github.com/en/actions/reference/workflows-and-actions/variables
            pytest.fail(
                f"AWS profile '{profile_name}' not configured in CI -- "
                "credentials setup step likely broken or missing."
            )
        pytest.skip(
            f"Needs the '{profile_name}' AWS profile configured locally "
            "and network access to the private bucket. "
            "See https://github.com/Deltares-research/hydromt_data_pipelines/blob/main/README.md "
            "for a guide on setting up the AWS profile."
        )


@pytest.fixture
def catalog_yaml_path(tmp_path: Path) -> Path:
    path = tmp_path / "private_bucket_catalog.yml"
    path.write_text(CATALOG_YAML)
    return path


def test_yaml_parses_expected_filesystem_config(catalog_yaml_path: Path) -> None:
    """Pure parsing check -- no network/credentials, runs in normal CI."""
    dc = DataCatalog(data_libs=[str(catalog_yaml_path)])

    assert "merit_hydro" in dc.sources
    source = dc.get_source("merit_hydro")

    fs_config = source.driver.filesystem.serialize(include_credentials=True)
    assert fs_config["protocol"] == "s3"
    assert fs_config["profile"] == "hydromt-data"
    assert fs_config["anon"] is False


def _make_botocore_403_error():
    return ClientError(
        error_response={
            "Error": {"Code": "AccessDenied", "Message": "Access Denied"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        operation_name="GetObject",
    )


def test_open_zarrs_wraps_s3_403_as_permission_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_open_zarr(uri, **kwargs):
        raise _make_botocore_403_error()

    monkeypatch.setattr(xr, "open_zarr", fake_open_zarr)

    with pytest.raises(PermissionError, match="Unauthorized access"):
        open_zarrs(["s3://hydromt-data/merit_hydro/merit_hydro.zarr"], {})


def test_open_mfdataset_wraps_s3_403_as_permission_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_open_mfdataset(uris, **kwargs):
        raise _make_botocore_403_error()

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    with pytest.raises(PermissionError, match="Unauthorized access"):
        open_mfdataset(
            ["s3://hydromt-data/merit_hydro/merit_hydro.nc"], lambda ds: ds, {}
        )


def test_get_rasterdataset_reads_private_bucket(catalog_yaml_path: Path) -> None:
    """Real end-to-end read against the private bucket."""
    _require_or_skip_aws_profile("hydromt-data")

    dc = DataCatalog(data_libs=[str(catalog_yaml_path)])

    ds = dc.get_rasterdataset("merit_hydro")

    assert isinstance(ds, xr.Dataset)
    assert "elv" in ds.data_vars
