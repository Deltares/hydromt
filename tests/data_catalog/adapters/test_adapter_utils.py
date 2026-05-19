"""Tests for _create_time_slice utility function."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydromt.data_catalog.adapters.adapter_utils import _create_time_slice
from hydromt.error import NoDataException, NoDataStrategy


@pytest.fixture
def daily_xr_dataset():
    """xr.Dataset with daily timestamps from 2020-01-01 to 2020-01-10."""
    times = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    return xr.Dataset({"var": ("time", np.arange(len(times)))}, coords={"time": times})


@pytest.fixture
def daily_dataframe():
    """pd.DataFrame with daily DatetimeIndex from 2020-01-01 to 2020-01-10."""
    times = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    return pd.DataFrame({"var": np.arange(len(times))}, index=times)


@pytest.fixture
def single_step_xr_dataset():
    """xr.Dataset with a single time step."""
    times = pd.date_range("2020-01-05", periods=1, freq="D")
    return xr.Dataset({"var": ("time", [42])}, coords={"time": times})


@pytest.fixture
def single_step_dataframe():
    """pd.DataFrame with a single time step."""
    times = pd.date_range("2020-01-05", periods=1, freq="D")
    return pd.DataFrame({"var": [42]}, index=times)


class TestCreateTimeSliceXarray:
    """Tests for _create_time_slice with xr.Dataset input."""

    def test_exact_match_inclusive(self, daily_xr_dataset):
        """Exact timestamps that match data points."""
        result = _create_time_slice(
            daily_xr_dataset, "2020-01-03", "2020-01-07", inclusive=True
        )
        assert result.start == np.datetime64("2020-01-03")
        assert result.stop == np.datetime64("2020-01-07")

    def test_exact_match_exclusive(self, daily_xr_dataset):
        """Exact timestamps that match data points, exclusive mode."""
        result = _create_time_slice(
            daily_xr_dataset, "2020-01-03", "2020-01-07", inclusive=False
        )
        assert result.start == np.datetime64("2020-01-03")
        assert result.stop == np.datetime64("2020-01-07")

    def test_tstart_between_points_inclusive(self, daily_xr_dataset):
        """Tstart falls between data points with inclusive=True should pad (use earlier point)."""
        result = _create_time_slice(
            daily_xr_dataset, "2020-01-03T12:00:00", "2020-01-07", inclusive=True
        )
        # inclusive: pad -> last value <= tstart -> 2020-01-03
        assert result.start == np.datetime64("2020-01-03")
        assert result.stop == np.datetime64("2020-01-07")

    def test_tstart_between_points_exclusive(self, daily_xr_dataset):
        """Tstart falls between data points with inclusive=False should backfill (use later point)."""
        result = _create_time_slice(
            daily_xr_dataset, "2020-01-03T12:00:00", "2020-01-07", inclusive=False
        )
        # exclusive: backfill -> first value >= tstart -> 2020-01-04
        assert result.start == np.datetime64("2020-01-04")
        assert result.stop == np.datetime64("2020-01-07")

    def test_tstop_between_points_inclusive(self, daily_xr_dataset):
        """Tstop falls between data points with inclusive=True should backfill (use later point)."""
        result = _create_time_slice(
            daily_xr_dataset, "2020-01-03", "2020-01-07T12:00:00", inclusive=True
        )
        # inclusive: backfill -> first value >= tstop -> 2020-01-08
        assert result.start == np.datetime64("2020-01-03")
        assert result.stop == np.datetime64("2020-01-08")

    def test_tstop_between_points_exclusive(self, daily_xr_dataset):
        """Tstop falls between data points with inclusive=False should pad (use earlier point)."""
        result = _create_time_slice(
            daily_xr_dataset, "2020-01-03", "2020-01-07T12:00:00", inclusive=False
        )
        # exclusive: pad -> last value <= tstop -> 2020-01-07
        assert result.start == np.datetime64("2020-01-03")
        assert result.stop == np.datetime64("2020-01-07")

    def test_both_between_points_inclusive(self, daily_xr_dataset):
        """Both tstart and tstop between data points, inclusive."""
        result = _create_time_slice(
            daily_xr_dataset,
            "2020-01-03T12:00:00",
            "2020-01-07T12:00:00",
            inclusive=True,
        )
        assert result.start == np.datetime64("2020-01-03")
        assert result.stop == np.datetime64("2020-01-08")

    def test_both_between_points_exclusive(self, daily_xr_dataset):
        """Both tstart and tstop between data points, exclusive."""
        result = _create_time_slice(
            daily_xr_dataset,
            "2020-01-03T12:00:00",
            "2020-01-07T12:00:00",
            inclusive=False,
        )
        assert result.start == np.datetime64("2020-01-04")
        assert result.stop == np.datetime64("2020-01-07")

    def test_tstart_before_data_clamps(self, daily_xr_dataset):
        """Tstart before data extent should clamp to data start."""
        result = _create_time_slice(
            daily_xr_dataset, "2019-12-25", "2020-01-05", inclusive=True
        )
        assert result is not None
        assert result.start == np.datetime64("2020-01-01")
        assert result.stop == np.datetime64("2020-01-05")

    def test_tstop_after_data_clamps(self, daily_xr_dataset):
        """Tstop after data extent should clamp to data end."""
        result = _create_time_slice(
            daily_xr_dataset, "2020-01-05", "2020-02-01", inclusive=True
        )
        assert result is not None
        assert result.start == np.datetime64("2020-01-05")
        assert result.stop == np.datetime64("2020-01-10")

    def test_both_beyond_data_clamps(self, daily_xr_dataset):
        """Both tstart and tstop beyond data extent should clamp both."""
        result = _create_time_slice(
            daily_xr_dataset, "2019-12-01", "2020-02-01", inclusive=True
        )
        assert result is not None
        assert result.start == np.datetime64("2020-01-01")
        assert result.stop == np.datetime64("2020-01-10")

    def test_no_overlap_before_raises(self, daily_xr_dataset):
        """Completely out-of-range (before) with RAISE strategy."""
        with pytest.raises(NoDataException):
            _create_time_slice(
                daily_xr_dataset,
                "2019-01-01",
                "2019-06-01",
                handle_nodata=NoDataStrategy.RAISE,
            )

    def test_no_overlap_after_raises(self, daily_xr_dataset):
        """Completely out-of-range (after) with RAISE strategy."""
        with pytest.raises(NoDataException):
            _create_time_slice(
                daily_xr_dataset,
                "2021-01-01",
                "2021-06-01",
                handle_nodata=NoDataStrategy.RAISE,
            )

    def test_no_overlap_ignore_returns_none(self, daily_xr_dataset):
        """Completely out-of-range with IGNORE strategy returns None."""
        result = _create_time_slice(
            daily_xr_dataset,
            "2021-01-01",
            "2021-06-01",
            handle_nodata=NoDataStrategy.IGNORE,
        )
        assert result is None

    def test_no_overlap_warn_returns_none(self, daily_xr_dataset):
        """Completely out-of-range with WARN strategy returns None."""
        result = _create_time_slice(
            daily_xr_dataset,
            "2021-01-01",
            "2021-06-01",
            handle_nodata=NoDataStrategy.WARN,
        )
        assert result is None

    def test_single_time_step(self, single_step_xr_dataset):
        """Dataset with single time step, range encompassing it."""
        result = _create_time_slice(
            single_step_xr_dataset, "2020-01-01", "2020-01-10", inclusive=True
        )
        assert result is not None
        assert result.start == np.datetime64("2020-01-05")
        assert result.stop == np.datetime64("2020-01-05")

    def test_single_time_step_exact(self, single_step_xr_dataset):
        """Dataset with single time step, exact match."""
        result = _create_time_slice(
            single_step_xr_dataset, "2020-01-05", "2020-01-05", inclusive=True
        )
        assert result is not None
        assert result.start == np.datetime64("2020-01-05")
        assert result.stop == np.datetime64("2020-01-05")

    def test_single_time_step_no_overlap(self, single_step_xr_dataset):
        """Dataset with single time step, no overlap raises."""
        with pytest.raises(NoDataException):
            _create_time_slice(
                single_step_xr_dataset,
                "2020-01-06",
                "2020-01-10",
                handle_nodata=NoDataStrategy.RAISE,
            )


class TestCreateTimeSliceDataFrame:
    """Tests for _create_time_slice with pd.DataFrame input."""

    def test_exact_match_inclusive(self, daily_dataframe):
        """Exact timestamps that match data points."""
        result = _create_time_slice(
            daily_dataframe, "2020-01-03", "2020-01-07", inclusive=True
        )
        assert result.start == pd.Timestamp("2020-01-03")
        assert result.stop == pd.Timestamp("2020-01-07")

    def test_exact_match_exclusive(self, daily_dataframe):
        """Exact timestamps, exclusive mode."""
        result = _create_time_slice(
            daily_dataframe, "2020-01-03", "2020-01-07", inclusive=False
        )
        assert result.start == pd.Timestamp("2020-01-03")
        assert result.stop == pd.Timestamp("2020-01-07")

    def test_tstart_between_points_inclusive(self, daily_dataframe):
        """Tstart between data points with inclusive=True pads to earlier."""
        result = _create_time_slice(
            daily_dataframe, "2020-01-03T12:00:00", "2020-01-07", inclusive=True
        )
        # inclusive: pad -> last value <= tstart -> 2020-01-03
        assert result.start == pd.Timestamp("2020-01-03")
        assert result.stop == pd.Timestamp("2020-01-07")

    def test_tstart_between_points_exclusive(self, daily_dataframe):
        """Tstart between data points with inclusive=False backfills to later."""
        result = _create_time_slice(
            daily_dataframe, "2020-01-03T12:00:00", "2020-01-07", inclusive=False
        )
        # exclusive: backfill -> first value >= tstart -> 2020-01-04
        assert result.start == pd.Timestamp("2020-01-04")
        assert result.stop == pd.Timestamp("2020-01-07")

    def test_tstop_between_points_inclusive(self, daily_dataframe):
        """Tstop between data points with inclusive=True backfills to later."""
        result = _create_time_slice(
            daily_dataframe, "2020-01-03", "2020-01-07T12:00:00", inclusive=True
        )
        # inclusive: backfill -> first value >= tstop -> 2020-01-08
        assert result.start == pd.Timestamp("2020-01-03")
        assert result.stop == pd.Timestamp("2020-01-08")

    def test_tstop_between_points_exclusive(self, daily_dataframe):
        """Tstop between data points with inclusive=False pads to earlier."""
        result = _create_time_slice(
            daily_dataframe, "2020-01-03", "2020-01-07T12:00:00", inclusive=False
        )
        # exclusive: pad -> last value <= tstop -> 2020-01-07
        assert result.start == pd.Timestamp("2020-01-03")
        assert result.stop == pd.Timestamp("2020-01-07")

    def test_both_between_points_inclusive(self, daily_dataframe):
        """Both between data points, inclusive."""
        result = _create_time_slice(
            daily_dataframe,
            "2020-01-03T12:00:00",
            "2020-01-07T12:00:00",
            inclusive=True,
        )
        assert result.start == pd.Timestamp("2020-01-03")
        assert result.stop == pd.Timestamp("2020-01-08")

    def test_both_between_points_exclusive(self, daily_dataframe):
        """Both between data points, exclusive."""
        result = _create_time_slice(
            daily_dataframe,
            "2020-01-03T12:00:00",
            "2020-01-07T12:00:00",
            inclusive=False,
        )
        assert result.start == pd.Timestamp("2020-01-04")
        assert result.stop == pd.Timestamp("2020-01-07")

    def test_tstart_before_data_clamps(self, daily_dataframe):
        """Tstart before data extent clamps to data start."""
        result = _create_time_slice(
            daily_dataframe, "2019-12-25", "2020-01-05", inclusive=True
        )
        assert result is not None
        assert result.start == pd.Timestamp("2020-01-01")
        assert result.stop == pd.Timestamp("2020-01-05")

    def test_tstop_after_data_clamps(self, daily_dataframe):
        """Tstop after data extent clamps to data end."""
        result = _create_time_slice(
            daily_dataframe, "2020-01-05", "2020-02-01", inclusive=True
        )
        assert result is not None
        assert result.start == pd.Timestamp("2020-01-05")
        assert result.stop == pd.Timestamp("2020-01-10")

    def test_both_beyond_data_clamps(self, daily_dataframe):
        """Both beyond data extent clamps both."""
        result = _create_time_slice(
            daily_dataframe, "2019-12-01", "2020-02-01", inclusive=True
        )
        assert result is not None
        assert result.start == pd.Timestamp("2020-01-01")
        assert result.stop == pd.Timestamp("2020-01-10")

    def test_no_overlap_before_raises(self, daily_dataframe):
        """Completely out-of-range (before) with RAISE strategy."""
        with pytest.raises(NoDataException):
            _create_time_slice(
                daily_dataframe,
                "2019-01-01",
                "2019-06-01",
                handle_nodata=NoDataStrategy.RAISE,
            )

    def test_no_overlap_after_raises(self, daily_dataframe):
        """Completely out-of-range (after) with RAISE strategy."""
        with pytest.raises(NoDataException):
            _create_time_slice(
                daily_dataframe,
                "2021-01-01",
                "2021-06-01",
                handle_nodata=NoDataStrategy.RAISE,
            )

    def test_no_overlap_ignore_returns_none(self, daily_dataframe):
        """Completely out-of-range with IGNORE strategy returns None."""
        result = _create_time_slice(
            daily_dataframe,
            "2021-01-01",
            "2021-06-01",
            handle_nodata=NoDataStrategy.IGNORE,
        )
        assert result is None

    def test_no_overlap_warn_returns_none(self, daily_dataframe):
        """Completely out-of-range with WARN strategy returns None."""
        result = _create_time_slice(
            daily_dataframe,
            "2021-01-01",
            "2021-06-01",
            handle_nodata=NoDataStrategy.WARN,
        )
        assert result is None

    def test_single_time_step(self, single_step_dataframe):
        """DataFrame with single time step, range encompassing it."""
        result = _create_time_slice(
            single_step_dataframe, "2020-01-01", "2020-01-10", inclusive=True
        )
        assert result is not None
        assert result.start == pd.Timestamp("2020-01-05")
        assert result.stop == pd.Timestamp("2020-01-05")

    def test_single_time_step_exact(self, single_step_dataframe):
        """DataFrame with single time step, exact match."""
        result = _create_time_slice(
            single_step_dataframe, "2020-01-05", "2020-01-05", inclusive=True
        )
        assert result is not None
        assert result.start == pd.Timestamp("2020-01-05")
        assert result.stop == pd.Timestamp("2020-01-05")

    def test_single_time_step_no_overlap(self, single_step_dataframe):
        """DataFrame with single time step, no overlap raises."""
        with pytest.raises(NoDataException):
            _create_time_slice(
                single_step_dataframe,
                "2020-01-06",
                "2020-01-10",
                handle_nodata=NoDataStrategy.RAISE,
            )

    def test_non_datetime_index_raises(self):
        """DataFrame with non-DatetimeIndex raises ValueError."""
        df = pd.DataFrame({"var": [1, 2, 3]}, index=[0, 1, 2])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            _create_time_slice(df, "2020-01-01", "2020-01-05")
