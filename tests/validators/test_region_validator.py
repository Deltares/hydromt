"""Testing for the validation of region specifications."""


from pathlib import Path

import pytest
from pydantic_core import ValidationError

from hydromt.validators.region import (
    BoundingBoxBasinRegion,
    BoundingBoxInterBasinRegion,
    BoundingBoxRegion,
    BoundingBoxSubBasinRegion,
    GeometryBasinRegion,
    GeometryInterBasinRegion,
    GeometryRegion,
    GeometrySubBasinRegion,
    GridRegion,
    MeshRegion,
    MultiPointBasinRegion,
    MultiPointSubBasinRegion,
    PointBasinRegion,
    PointSubBasinRegion,
    WGS84Point,
    validate_region,
)


def test_bbox_point_validator():
    b = {"bbox": [-1.0, -1.0, 1.0, 1.0]}

    region = validate_region(b)
    assert region == BoundingBoxRegion(xmin=-1.0, ymin=-1.0, xmax=1.0, ymax=1.0)


def test_invalid_bbox_point_validator():
    b = {"bbox": [1.0, 1.0, -1.0, -1.0]}

    with pytest.raises(ValidationError):
        _ = validate_region(b)


def test_unknown_region_type_validator():
    b = {"asdfasdf": [1.0, 1.0, -1.0, -1.0]}

    with pytest.raises(ValueError, match="Unknown region kind"):
        _ = validate_region(b)


def test_geom_validator():
    b = {"geom": "tests/data/naturalearth_lowres.geojson"}

    region = validate_region(b)
    assert region == GeometryRegion(path=Path("tests/data/naturalearth_lowres.geojson"))


def test_geom_non_existant_path_validator():
    b = {"geom": "tests/data/masdfasdfasdf.geojson"}

    with pytest.raises(ValueError, match="1 validation error"):
        _ = validate_region(b)


def test_grid_validator():
    b = {"grid": "tests/data/naturalearth_lowres.geojson"}

    region = validate_region(b)
    assert region == GridRegion(path=Path("tests/data/naturalearth_lowres.geojson"))


def test_mesh_validator():
    b = {"mesh": "tests/data/naturalearth_lowres.geojson"}

    region = validate_region(b)
    assert region == MeshRegion(path=Path("tests/data/naturalearth_lowres.geojson"))


def test_point_sub_basin_validator():
    b = {"subbasin": [0, 0]}

    region = validate_region(b)
    assert region == PointSubBasinRegion(points=[WGS84Point(x=0, y=0)])


def test_multipoint_sub_basin_validator():
    b = {"subbasin": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]}

    region = validate_region(b)
    assert region == MultiPointSubBasinRegion(
        points=[
            WGS84Point(x=1, y=1),
            WGS84Point(x=2, y=2),
            WGS84Point(x=3, y=3),
            WGS84Point(x=4, y=4),
            WGS84Point(x=5, y=5),
        ]
    )


def test_bounding_box_sub_basin_validator():
    b = {"subbasin": [-1.0, -1.0, 1.0, 1.0]}

    region = validate_region(b)
    assert region == BoundingBoxSubBasinRegion(xmin=-1.0, ymin=-1.0, xmax=1.0, ymax=1.0)


def test_geometry_sub_basin_validator():
    b = {"subbasin": "tests/data/naturalearth_lowres.geojson"}

    region = validate_region(b)
    assert region == GeometrySubBasinRegion(
        path=Path("tests/data/naturalearth_lowres.geojson")
    )


def test_bounding_box_inter_basin_validator():
    b = {"interbasin": [-1.0, -1.0, 1.0, 1.0]}

    region = validate_region(b)
    assert region == BoundingBoxInterBasinRegion(
        xmin=-1.0, ymin=-1.0, xmax=1.0, ymax=1.0
    )


def test_geometry_inter_basin_validator():
    b = {"interbasin": "tests/data/naturalearth_lowres.geojson"}

    region = validate_region(b)
    assert region == GeometryInterBasinRegion(
        path=Path("tests/data/naturalearth_lowres.geojson")
    )


def test_point_basin_validator():
    b = {"basin": [0, 0]}

    region = validate_region(b)
    assert region == PointBasinRegion(points=[WGS84Point(x=0, y=0)])


def test_multipoint_basin_validator():
    b = {"basin": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]}

    region = validate_region(b)
    assert region == MultiPointBasinRegion(
        points=[
            WGS84Point(x=1, y=1),
            WGS84Point(x=2, y=2),
            WGS84Point(x=3, y=3),
            WGS84Point(x=4, y=4),
            WGS84Point(x=5, y=5),
        ]
    )


def test_bounding_box_basin_validator():
    b = {"basin": [-1.0, -1.0, 1.0, 1.0]}

    region = validate_region(b)
    assert region == BoundingBoxBasinRegion(xmin=-1.0, ymin=-1.0, xmax=1.0, ymax=1.0)


def test_geometry_basin_validator():
    b = {"basin": "tests/data/naturalearth_lowres.geojson"}

    region = validate_region(b)
    assert region == GeometryBasinRegion(
        path=Path("tests/data/naturalearth_lowres.geojson")
    )
