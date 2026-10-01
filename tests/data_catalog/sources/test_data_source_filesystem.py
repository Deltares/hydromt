import pytest
from pydantic import ValidationError

from hydromt._utils.warnings import HydroMTDeprecationWarning
from hydromt.data_catalog.drivers import RasterioDriver
from hydromt.data_catalog.sources import RasterDatasetSource
from hydromt.data_catalog.uri_resolvers import ConventionResolver
from hydromt.typing import FSSpecFileSystem


def _source(**kw):
    return RasterDatasetSource.model_validate(
        {"name": "cmip6", "uri": "gs://cmip6/x/*", **kw}
    )


def test_legacy_driver_filesystem_loads_with_warning():
    with pytest.warns(
        HydroMTDeprecationWarning,
        match="`filesystem` under `driver`/`uri_resolver` is deprecated",
    ):
        source = _source(
            driver={"name": "raster_xarray", "filesystem": {"protocol": "gcs"}}
        )
    assert source.filesystem.protocol == "gcs"
    assert source.driver.filesystem is source.filesystem
    assert source.uri_resolver.filesystem is source.filesystem


def test_root_filesystem_is_shared_by_reference_without_warning(recwarn):
    source = _source(driver={"name": "raster_xarray"}, filesystem={"protocol": "gcs"})
    assert source.driver.filesystem is source.filesystem
    assert source.uri_resolver.filesystem is source.filesystem
    assert not [w for w in recwarn if issubclass(w.category, HydroMTDeprecationWarning)]


def test_conflicting_component_filesystems_raise():
    with pytest.raises(ValidationError, match="different filesystems"):
        RasterDatasetSource(
            name="x",
            uri="gs://b/x",
            driver=RasterioDriver(filesystem=FSSpecFileSystem(protocol="s3")),
            uri_resolver=ConventionResolver(
                filesystem=FSSpecFileSystem(protocol="gcs")
            ),
        )


def test_component_dump_has_no_filesystem():
    source = _source(driver={"name": "raster_xarray"}, filesystem={"protocol": "gcs"})
    assert "filesystem" not in source.driver.model_dump()
    assert "filesystem" not in source.uri_resolver.model_dump()


def test_v0_layout_is_still_rejected_at_runtime():
    with pytest.raises(ValidationError):
        RasterDatasetSource.model_validate(
            {"name": "x", "path": "x.tif", "driver": "raster"}
        )
