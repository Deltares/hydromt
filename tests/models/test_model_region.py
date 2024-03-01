from os.path import exists, join

from geopandas.testing import assert_geodataframe_equal

from hydromt._typing.model_mode import ModelMode


def test_model_reads_region_jit(test_model, geodf):
    test_dir = test_model.root.path
    test_model.mode = ModelMode.READ
    region_file_path = join(test_dir, "region.geojson")
    geodf.to_file(region_file_path)
    assert test_model.region._data is None
    _ = test_model.region.data
    assert_geodataframe_equal(test_model.region._data, geodf)


def test_model_writes_region(test_model, geodf):
    test_dir = test_model.root.path
    test_model.mode = ModelMode.WRITE
    region_file_path = join(test_dir, "region.geojson")
    assert not exists(region_file_path)
    test_model.region.create({"bbox": geodf.total_bounds})
    test_model.region.write()
    assert exists(region_file_path)
