from os.path import exists, join

from geopandas.testing import assert_geodataframe_equal

from hydromt._typing.model_mode import ModelMode


def test_model_reads_region(test_model, geodf):
    test_dir = test_model.root.path
    test_model.root.mode = ModelMode.READ
    region_file_path = join(test_dir, "region.geojson")
    geodf.to_file(region_file_path)
    assert_geodataframe_equal(test_model.region.region_data, geodf)


def test_model_writes_region(test_model, geodf):
    test_dir = test_model.root.path
    test_model.root.mode = ModelMode.WRITE
    region_file_path = join(test_dir, "region.geojson")
    assert not exists(region_file_path)
    test_model.region.create(region={"bbox": geodf.total_bounds})
    test_model.region.write()
    assert exists(region_file_path)
