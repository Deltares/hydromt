from os import makedirs
from os.path import join
from typing import cast

import geopandas as gpd
from shapely.geometry import box

from hydromt.components.geoms import GeomsComponent
from hydromt.models import Model


def test_model_set_geoms(tmpdir):
    bbox = box(*[4.221067, 51.949474, 4.471006, 52.073727], ccw=True)
    geom = gpd.GeoDataFrame(geometry=[bbox], crs=4326)

    model = Model(root=str(tmpdir), mode="w")
    model.add_component("geom", GeomsComponent(model))

    geom_component = model.get_component("geom", GeomsComponent)

    geom_component.set(geom, "geom_wgs84")

    assert list(geom_component.data.keys()) == ["geom_wgs84"]
    assert list(geom_component.data.values())[0].equals(geom)


def test_model_read_geoms(tmpdir):
    bbox = box(*[4.221067, 51.949474, 4.471006, 52.073727], ccw=True)
    geom = gpd.GeoDataFrame(geometry=[bbox], crs=4326)
    write_folder = join(tmpdir, "geoms")
    makedirs(write_folder, exist_ok=True)
    write_path = join(write_folder, "test_geom.geojson")
    geom.to_file(write_path)

    model = Model(root=tmpdir, mode="r")
    model.add_component("geom", GeomsComponent(model))

    geom_component = model.get_component("geom", GeomsComponent)

    component_data = geom_component.data["test_geom"]
    assert geom.equals(component_data)


def test_model_write_geoms_wgs84_with_model_crs(tmpdir):
    bbox = box(*[4.221067, 51.949474, 4.471006, 52.073727], ccw=True)
    geom_4326 = gpd.GeoDataFrame(geometry=[bbox], crs=4326)
    geom_3857 = cast(gpd.GeoDataFrame, geom_4326.copy().to_crs(3857))

    model = Model(root=str(tmpdir), target_model_crs=3857, mode="w")
    model.add_component("test_geom", GeomsComponent(model))

    geom_component = model.get_component("test_geom", GeomsComponent)
    geom_component.set(geom_4326, "test_geom")

    assert geom_component.data["test_geom"].equals(geom_3857)
    write_folder = join(tmpdir, "geoms")
    write_path = join(write_folder, "test_geom.geojson")
    geom_component.write(to_wgs84=True)

    gdf = gpd.read_file(write_path)
    assert gdf is not None
    assert gdf.equals(geom_4326)


def test_model_write_geoms(tmpdir):
    model = Model(root=str(tmpdir), mode="w", target_model_crs=3857)
    model.add_component("geom", GeomsComponent(model))
    geom_component = model.get_component("geom", GeomsComponent)

    bbox = box(*[4.221067, 51.949474, 4.471006, 52.073727], ccw=True)
    geom = gpd.GeoDataFrame(geometry=[bbox], crs=4326)
    geom.to_crs(3857, inplace=True)
    assert geom.crs.to_epsg() == 3857

    write_folder = join(tmpdir, "geoms")
    write_path = join(write_folder, "test_geom.geojson")
    geom_component.set(geom, "test_geom")
    geom_component.write()
    region_geom = gpd.read_file(write_path)

    assert region_geom.crs.to_epsg() == 3857
