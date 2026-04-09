"""Tests for vector_utils.nearest_merge."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from hydromt.gis import vector_utils


@pytest.fixture
def gdf_points() -> gpd.GeoDataFrame:
    """GeoDataFrame with 3 points and some attributes."""
    return gpd.GeoDataFrame(
        {
            "name": ["A", "B", "C"],
            "value": [1.0, np.nan, 3.0],
        },
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs=3857,
    )


@pytest.fixture
def gdf_targets() -> gpd.GeoDataFrame:
    """GeoDataFrame with 3 nearby target points and attributes to merge."""
    return gpd.GeoDataFrame(
        {
            "name": ["X", "Y", "Z"],
            "score": [10, 20, 30],
            "label": ["low", "mid", "high"],
        },
        geometry=[Point(0.1, 0.1), Point(1.1, 1.1), Point(2.1, 2.1)],
        crs=3857,
    )


@pytest.fixture
def gdf_far_targets() -> gpd.GeoDataFrame:
    """GeoDataFrame with one target far away from the third point."""
    return gpd.GeoDataFrame(
        {"val": [100, 200, 300]},
        geometry=[Point(0.01, 0.01), Point(1.01, 1.01), Point(999, 999)],
        crs=3857,
    )


@pytest.fixture
def gdf_with_nan() -> gpd.GeoDataFrame:
    """GeoDataFrame with NaN values in score column."""
    return gpd.GeoDataFrame(
        {"score": [np.nan, 5.0, np.nan]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs=3857,
    )


@pytest.fixture
def gdf_with_empty() -> gpd.GeoDataFrame:
    """GeoDataFrame with empty string values in label column."""
    return gpd.GeoDataFrame(
        {"label": ["", "keep", ""]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs=3857,
    )


@pytest.fixture
def gdf_lines() -> gpd.GeoDataFrame:
    """GeoDataFrame with two LineString geometries."""
    return gpd.GeoDataFrame(
        {"name": ["line_a", "line_b"]},
        geometry=[
            LineString([(0, 0), (2, 0)]),
            LineString([(10, 0), (12, 0)]),
        ],
        crs=3857,
    )


@pytest.fixture
def gdf_line_targets() -> gpd.GeoDataFrame:
    """Target points near midpoints of gdf_lines."""
    return gpd.GeoDataFrame(
        {"label": ["near_a", "near_b"]},
        geometry=[Point(1, 0.1), Point(11, 0.1)],
        crs=3857,
    )


@pytest.fixture
def gdf_polys() -> gpd.GeoDataFrame:
    """GeoDataFrame with two Polygon geometries."""
    return gpd.GeoDataFrame(
        {"name": ["poly_a", "poly_b"]},
        geometry=[
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]),
        ],
        crs=3857,
    )


@pytest.fixture
def gdf_poly_targets() -> gpd.GeoDataFrame:
    """Target points near centers of gdf_polys."""
    return gpd.GeoDataFrame(
        {"label": ["near_a", "near_b"]},
        geometry=[Point(1, 1), Point(11, 11)],
        crs=3857,
    )


@pytest.fixture
def gdf_far_target() -> gpd.GeoDataFrame:
    """Single target point far from the origin."""
    return gpd.GeoDataFrame(
        {"val": [99]},
        geometry=[Point(1000, 1000)],
        crs=3857,
    )


@pytest.fixture
def gdf_multipolygon() -> gpd.GeoDataFrame:
    """GeoDataFrame with a single MultiPolygon geometry."""
    mp = MultiPolygon(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ]
    )
    return gpd.GeoDataFrame(geometry=[mp], crs=3857)


@pytest.fixture
def gdf_mixed() -> gpd.GeoDataFrame:
    """GeoDataFrame with mixed geometry types (Point + LineString)."""
    return gpd.GeoDataFrame(
        geometry=[Point(0, 0), LineString([(1, 1), (2, 2)])],
        crs=3857,
    )


class TestNearestMergeBasic:
    """Basic merge behaviour."""

    def test_returns_geodataframe(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(gdf_points, gdf_targets)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_does_not_modify_original_by_default(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        original_cols = list(gdf_points.columns)
        vector_utils.nearest_merge(gdf_points, gdf_targets)
        assert list(gdf_points.columns) == original_cols

    def test_inplace_modifies_original(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        original_id = id(gdf_points)
        result = vector_utils.nearest_merge(gdf_points, gdf_targets, inplace=True)
        assert id(result) == original_id
        assert "score" in gdf_points.columns

    def test_adds_distance_and_index_columns(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(gdf_points, gdf_targets)
        assert "distance_right" in result.columns
        assert "index_right" in result.columns

    def test_all_columns_merged_by_default(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(gdf_points, gdf_targets)
        for col in ["score", "label"]:
            assert col in result.columns

    def test_geometry_column_not_merged(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(gdf_points, gdf_targets)
        # geometry should be from gdf1, not overwritten
        assert result.geometry.equals(gdf_points.geometry)


class TestNearestMergeColumns:
    """Selective column merging."""

    def test_specific_columns(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(gdf_points, gdf_targets, columns=["score"])
        assert "score" in result.columns
        assert "label" not in result.columns

    def test_missing_column_warns(
        self,
        gdf_points: gpd.GeoDataFrame,
        gdf_targets: gpd.GeoDataFrame,
        caplog: pytest.LogCaptureFixture,
    ):
        vector_utils.nearest_merge(gdf_points, gdf_targets, columns=["nonexistent"])
        assert "nonexistent" in caplog.text

    def test_geometry_in_columns_is_skipped(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(
            gdf_points, gdf_targets, columns=["geometry", "score"]
        )
        assert "score" in result.columns
        # geometry should remain from gdf1
        assert result.geometry.equals(gdf_points.geometry)


class TestNearestMergeMaxDist:
    """Distance-based filtering."""

    def test_no_max_dist_merges_all(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(gdf_points, gdf_targets)
        assert np.all(result["index_right"] != -1)

    def test_max_dist_filters_far_features(
        self, gdf_points: gpd.GeoDataFrame, gdf_far_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(gdf_points, gdf_far_targets, max_dist=1.0)
        # The first two points should match (close), but the third shouldn't
        close_mask = result["distance_right"] < 1.0
        assert close_mask.sum() == 2
        assert np.all(result.loc[~close_mask, "index_right"] == -1)

    def test_max_dist_zero_filters_everything(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(gdf_points, gdf_targets, max_dist=0.0)
        assert np.all(result["index_right"] == -1)


class TestNearestMergeOverwrite:
    """Overwrite vs fill-only behaviour."""

    def test_no_overwrite_preserves_existing_values(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        # gdf_points has name = ["A", "B", "C"], gdf_targets has name = ["X", "Y", "Z"]
        result = vector_utils.nearest_merge(
            gdf_points, gdf_targets, columns=["name"], overwrite=False
        )
        # "A" and "C" should be preserved; "B" is not NaN (it's a string), so preserved
        assert result.loc[0, "name"] == "A"
        assert result.loc[2, "name"] == "C"

    def test_no_overwrite_fills_nan(
        self, gdf_with_nan: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(
            gdf_with_nan, gdf_targets, columns=["score"], overwrite=False
        )
        # NaN positions should be filled from gdf_targets
        assert result.loc[0, "score"] == 10
        # Existing value should be preserved
        assert result.loc[1, "score"] == 5.0
        assert result.loc[2, "score"] == 30

    def test_no_overwrite_fills_empty_string(
        self, gdf_with_empty: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(
            gdf_with_empty, gdf_targets, columns=["label"], overwrite=False
        )
        assert result.loc[0, "label"] == "low"
        assert result.loc[1, "label"] == "keep"
        assert result.loc[2, "label"] == "high"

    def test_overwrite_replaces_existing_values(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(
            gdf_points, gdf_targets, columns=["name"], overwrite=True
        )
        # All values should come from gdf_targets
        assert result.loc[0, "name"] == "X"
        assert result.loc[1, "name"] == "Y"
        assert result.loc[2, "name"] == "Z"

    def test_new_column_always_added(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        # "score" doesn't exist in gdf_points, so it should always be added
        result = vector_utils.nearest_merge(
            gdf_points, gdf_targets, columns=["score"], overwrite=False
        )
        assert list(result["score"]) == [10, 20, 30]


class TestNearestMergeIndexRight:
    """Index mapping correctness."""

    def test_index_right_maps_to_gdf2_index(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(gdf_points, gdf_targets)
        valid = result["index_right"] != -1
        # Every index_right value should be a valid gdf2 index
        for idx in result.loc[valid, "index_right"]:
            assert idx in gdf_targets.index.values

    def test_distance_right_nonnegative(
        self, gdf_points: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(gdf_points, gdf_targets)
        assert np.all(result["distance_right"] >= 0)

    def test_self_merge_zero_distance(self, gdf_points: gpd.GeoDataFrame):
        result = vector_utils.nearest_merge(gdf_points, gdf_points)
        np.testing.assert_array_almost_equal(result["distance_right"], 0.0)


class TestNearestMergeGeometryTypes:
    """Tests for LineString, Polygon, and mixed geometry inputs."""

    def test_linestring_merge_uses_midpoint(
        self, gdf_lines: gpd.GeoDataFrame, gdf_line_targets: gpd.GeoDataFrame
    ):
        """LineStrings match based on their midpoint."""
        result = vector_utils.nearest_merge(gdf_lines, gdf_line_targets)
        assert result.loc[0, "label"] == "near_a"
        assert result.loc[1, "label"] == "near_b"

    def test_linestring_merge_columns_transferred(
        self, gdf_lines: gpd.GeoDataFrame, gdf_line_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(gdf_lines, gdf_line_targets)
        assert result.loc[0, "label"] == "near_a"
        assert "distance_right" in result.columns

    def test_polygon_merge_uses_representative_point(
        self, gdf_polys: gpd.GeoDataFrame, gdf_poly_targets: gpd.GeoDataFrame
    ):
        """Polygons match based on their representative point (inside polygon)."""
        result = vector_utils.nearest_merge(gdf_polys, gdf_poly_targets)
        assert result.loc[0, "label"] == "near_a"
        assert result.loc[1, "label"] == "near_b"

    def test_polygon_merge_with_max_dist(
        self, gdf_polys: gpd.GeoDataFrame, gdf_far_target: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(gdf_polys, gdf_far_target, max_dist=1.0)
        assert np.all(result["index_right"] == -1)

    def test_multipolygon_merge(
        self, gdf_multipolygon: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        result = vector_utils.nearest_merge(gdf_multipolygon, gdf_targets)
        assert "distance_right" in result.columns
        assert result.loc[0, "index_right"] != -1

    def test_mixed_geometry_raises(
        self, gdf_mixed: gpd.GeoDataFrame, gdf_targets: gpd.GeoDataFrame
    ):
        """Mixed geometry types (e.g. Point + LineString) are not supported."""
        with pytest.raises(NotImplementedError, match="Mixed geometry"):
            vector_utils.nearest_merge(gdf_mixed, gdf_targets)
