"""Testing for the validation of region specifications."""

from pathlib import Path

import pytest
from pydantic_core import ValidationError

from hydromt._validators.region import (
    BoundingBoxRegion,
    PathRegion,
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

    with pytest.raises(NotImplementedError, match="Unknown region kind"):
        _ = validate_region(b)


def test_geom_validator():
    b = {"geom": "tests/data/naturalearth_lowres.geojson"}

    region = validate_region(b)
    assert region == PathRegion(path=Path("tests/data/naturalearth_lowres.geojson"))


def test_geom_non_existant_path_validator():
    b = {"geom": "tests/data/masdfasdfasdf.geojson"}

    with pytest.raises(ValueError, match="1 validation error"):
        _ = validate_region(b)
