# -*- coding: utf-8 -*-
"""Tests for the models.model_api and models.region of hydromt."""

import pytest
from unittest.mock import Mock, patch
import os
from os.path import join, isfile, isdir
import geopandas as gpd

import hydromt
from hydromt.models.region import parse_region


def test_region(tmpdir, geodf):
    # model
    region = {"region": [0.0, -1.0]}
    with pytest.raises(ValueError, match=r"Region key .* not understood.*"):
        parse_region(region)
    from hydromt.models import MODELS

    if len(MODELS) > 0:
        model = [x for x in MODELS][0]
        root = str(tmpdir.join(model)) + "_test_region"
        if not isdir(root):
            os.mkdir(root)
        region = {model: root}
        kind, region = parse_region(region)
        assert kind == "model"

    # geom
    region = {"geom": geodf}
    with pytest.raises(ValueError, match=r"Region value.*"):
        kind, region = parse_region(region)
    fn_gdf = str(tmpdir.join("test.geojson"))
    geodf.to_file(fn_gdf, driver="GeoJSON")
    region = {"geom": fn_gdf}
    with pytest.raises(ValueError, match=r"Region value.*"):
        kind, region = parse_region(region)

    # basid
    region = {"basin": [1001, 1002, 1003, 1004, 1005]}
    kind, region = parse_region(region)
    assert kind == "basin"
    assert region.get("basid") == [1001, 1002, 1003, 1004, 1005]
    region = {"basin": 101}
    kind, region = parse_region(region)

    # bbox
    region = {"outlet": [0.0, -5.0, 3.0, 0.0]}
    kind, region = parse_region(region)
    assert kind == "outlet"
    assert "bbox" in region

    # xy
    region = {"subbasin": [1.0, -1.0], "uparea": 5.0}
    kind, region = parse_region(region)
    assert kind == "subbasin"
    assert "xy" in region
    region = {"basin": [[1.0, 1.5], [0.0, -1.0]]}
    kind, region = parse_region(region)
    assert "xy" in region
    region = {"subbasin": geodf}
    kind, region = parse_region(region)
    assert "xy" in region
