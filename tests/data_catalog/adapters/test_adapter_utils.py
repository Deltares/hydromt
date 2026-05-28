"""Tests for _create_time_slice utility function."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydromt.data_catalog.adapters.adapter_utils import _create_time_slice
from hydromt.error import NoDataException, NoDataStrategy

# Default value of `inclusive` in _create_time_slice; update here if the default changes.
DEFAULT_INCLUSIVE = False


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

    @pytest.mark.parametrize(
        "tstart, tstop, inclusive, expected_start, expected_stop",
        [
            # exact match on data points, inclusive
            ("2020-01-03", "2020-01-07", True, "2020-01-03", "2020-01-07"),
            # exact match on data points, exclusive (same result)
            ("2020-01-03", "2020-01-07", False, "2020-01-03", "2020-01-07"),
            # tstart between points, inclusive pads to earlier point
            ("2020-01-03T12:00:00", "2020-01-07", True, "2020-01-03", "2020-01-07"),
            # tstart between points, exclusive backfills to later point
            ("2020-01-03T12:00:00", "2020-01-07", False, "2020-01-04", "2020-01-07"),
            # tstop between points, inclusive backfills to later point
            ("2020-01-03", "2020-01-07T12:00:00", True, "2020-01-03", "2020-01-08"),
            # tstop between points, exclusive pads to earlier point
            ("2020-01-03", "2020-01-07T12:00:00", False, "2020-01-03", "2020-01-07"),
            # both between points, inclusive widens range
            (
                "2020-01-03T12:00:00",
                "2020-01-07T12:00:00",
                True,
                "2020-01-03",
                "2020-01-08",
            ),
            # both between points, exclusive narrows range
            (
                "2020-01-03T12:00:00",
                "2020-01-07T12:00:00",
                False,
                "2020-01-04",
                "2020-01-07",
            ),
        ],
    )
    def test_time_alignment(
        self, daily_xr_dataset, tstart, tstop, inclusive, expected_start, expected_stop
    ):
        result = _create_time_slice(
            daily_xr_dataset, tstart, tstop, inclusive=inclusive
        )
        assert result.start == np.datetime64(expected_start)
        assert result.stop == np.datetime64(expected_stop)

    @pytest.mark.parametrize(
        "tstart, tstop, expected_start, expected_stop",
        [
            # tstart before data: clamps to data start
            ("2019-12-25", "2020-01-05", "2020-01-01", "2020-01-05"),
            # tstop after data: clamps to data end
            ("2020-01-05", "2020-02-01", "2020-01-05", "2020-01-10"),
            # both beyond data: clamps to full data extent
            ("2019-12-01", "2020-02-01", "2020-01-01", "2020-01-10"),
        ],
    )
    def test_clamping(
        self, daily_xr_dataset, tstart, tstop, expected_start, expected_stop
    ):
        result = _create_time_slice(
            daily_xr_dataset, tstart, tstop, inclusive=DEFAULT_INCLUSIVE
        )
        assert result is not None
        assert result.start == np.datetime64(expected_start)
        assert result.stop == np.datetime64(expected_stop)

    @pytest.mark.parametrize(
        "tstart, tstop, strategy, expect_raises",
        [
            # no overlap before data, RAISE raises
            ("2019-01-01", "2019-06-01", NoDataStrategy.RAISE, True),
            # no overlap after data, RAISE raises
            ("2021-01-01", "2021-06-01", NoDataStrategy.RAISE, True),
            # no overlap, IGNORE returns None
            ("2021-01-01", "2021-06-01", NoDataStrategy.IGNORE, False),
            # no overlap, WARN returns None
            ("2021-01-01", "2021-06-01", NoDataStrategy.WARN, False),
        ],
    )
    def test_no_overlap(self, daily_xr_dataset, tstart, tstop, strategy, expect_raises):
        if expect_raises:
            with pytest.raises(NoDataException):
                _create_time_slice(
                    daily_xr_dataset, tstart, tstop, handle_nodata=strategy
                )
        else:
            result = _create_time_slice(
                daily_xr_dataset, tstart, tstop, handle_nodata=strategy
            )
            assert result is None

    @pytest.mark.parametrize(
        "tstart, tstop, expected_start, expected_stop",
        [
            # range encompassing single time step
            ("2020-01-01", "2020-01-10", "2020-01-05", "2020-01-05"),
            # exact match on single time step
            ("2020-01-05", "2020-01-05", "2020-01-05", "2020-01-05"),
        ],
    )
    def test_single_time_step(
        self, single_step_xr_dataset, tstart, tstop, expected_start, expected_stop
    ):
        result = _create_time_slice(
            single_step_xr_dataset, tstart, tstop, inclusive=DEFAULT_INCLUSIVE
        )
        assert result is not None
        assert result.start == np.datetime64(expected_start)
        assert result.stop == np.datetime64(expected_stop)

    def test_single_time_step_no_overlap(self, single_step_xr_dataset):
        """No overlap with single time step raises."""
        with pytest.raises(NoDataException):
            _create_time_slice(
                single_step_xr_dataset,
                "2020-01-06",
                "2020-01-10",
                handle_nodata=NoDataStrategy.RAISE,
            )

    def test_unsorted_time_sorts_and_slices(self):
        """Descending time dimension is sorted internally."""
        times = pd.date_range("2020-01-01", "2020-01-10", freq="D")[::-1]
        ds = xr.Dataset(
            {"var": ("time", np.arange(len(times)))}, coords={"time": times}
        )
        result = _create_time_slice(ds, "2020-01-03", "2020-01-07")
        assert result is not None
        assert result.start == np.datetime64("2020-01-03")
        assert result.stop == np.datetime64("2020-01-07")


class TestCreateTimeSliceDataFrame:
    """Tests for _create_time_slice with pd.DataFrame input."""

    @pytest.mark.parametrize(
        "tstart, tstop, inclusive, expected_start, expected_stop",
        [
            # exact match on data points, inclusive
            ("2020-01-03", "2020-01-07", True, "2020-01-03", "2020-01-07"),
            # exact match on data points, exclusive (same result)
            ("2020-01-03", "2020-01-07", False, "2020-01-03", "2020-01-07"),
            # tstart between points, inclusive pads to earlier point
            ("2020-01-03T12:00:00", "2020-01-07", True, "2020-01-03", "2020-01-07"),
            # tstart between points, exclusive backfills to later point
            ("2020-01-03T12:00:00", "2020-01-07", False, "2020-01-04", "2020-01-07"),
            # tstop between points, inclusive backfills to later point
            ("2020-01-03", "2020-01-07T12:00:00", True, "2020-01-03", "2020-01-08"),
            # tstop between points, exclusive pads to earlier point
            ("2020-01-03", "2020-01-07T12:00:00", False, "2020-01-03", "2020-01-07"),
            # both between points, inclusive widens range
            (
                "2020-01-03T12:00:00",
                "2020-01-07T12:00:00",
                True,
                "2020-01-03",
                "2020-01-08",
            ),
            # both between points, exclusive narrows range
            (
                "2020-01-03T12:00:00",
                "2020-01-07T12:00:00",
                False,
                "2020-01-04",
                "2020-01-07",
            ),
        ],
    )
    def test_time_alignment(
        self, daily_dataframe, tstart, tstop, inclusive, expected_start, expected_stop
    ):
        result = _create_time_slice(daily_dataframe, tstart, tstop, inclusive=inclusive)
        assert result.start == pd.Timestamp(expected_start)
        assert result.stop == pd.Timestamp(expected_stop)

    @pytest.mark.parametrize(
        "tstart, tstop, expected_start, expected_stop",
        [
            # tstart before data: clamps to data start
            ("2019-12-25", "2020-01-05", "2020-01-01", "2020-01-05"),
            # tstop after data: clamps to data end
            ("2020-01-05", "2020-02-01", "2020-01-05", "2020-01-10"),
            # both beyond data: clamps to full data extent
            ("2019-12-01", "2020-02-01", "2020-01-01", "2020-01-10"),
        ],
    )
    def test_clamping(
        self, daily_dataframe, tstart, tstop, expected_start, expected_stop
    ):
        result = _create_time_slice(
            daily_dataframe, tstart, tstop, inclusive=DEFAULT_INCLUSIVE
        )
        assert result is not None
        assert result.start == pd.Timestamp(expected_start)
        assert result.stop == pd.Timestamp(expected_stop)

    @pytest.mark.parametrize(
        "tstart, tstop, strategy, expect_raises",
        [
            # no overlap before data, RAISE raises
            ("2019-01-01", "2019-06-01", NoDataStrategy.RAISE, True),
            # no overlap after data, RAISE raises
            ("2021-01-01", "2021-06-01", NoDataStrategy.RAISE, True),
            # no overlap, IGNORE returns None
            ("2021-01-01", "2021-06-01", NoDataStrategy.IGNORE, False),
            # no overlap, WARN returns None
            ("2021-01-01", "2021-06-01", NoDataStrategy.WARN, False),
        ],
    )
    def test_no_overlap(self, daily_dataframe, tstart, tstop, strategy, expect_raises):
        if expect_raises:
            with pytest.raises(NoDataException):
                _create_time_slice(
                    daily_dataframe, tstart, tstop, handle_nodata=strategy
                )
        else:
            result = _create_time_slice(
                daily_dataframe, tstart, tstop, handle_nodata=strategy
            )
            assert result is None

    @pytest.mark.parametrize(
        "tstart, tstop, expected_start, expected_stop",
        [
            # range encompassing single time step
            ("2020-01-01", "2020-01-10", "2020-01-05", "2020-01-05"),
            # exact match on single time step
            ("2020-01-05", "2020-01-05", "2020-01-05", "2020-01-05"),
        ],
    )
    def test_single_time_step(
        self, single_step_dataframe, tstart, tstop, expected_start, expected_stop
    ):
        result = _create_time_slice(
            single_step_dataframe, tstart, tstop, inclusive=DEFAULT_INCLUSIVE
        )
        assert result is not None
        assert result.start == pd.Timestamp(expected_start)
        assert result.stop == pd.Timestamp(expected_stop)

    def test_single_time_step_no_overlap(self, single_step_dataframe):
        """No overlap with single time step raises."""
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

    def test_unsorted_index_sorts_and_slices(self):
        """Descending DatetimeIndex is sorted internally."""
        times = pd.date_range("2020-01-01", "2020-01-10", freq="D")[::-1]
        df = pd.DataFrame({"var": np.arange(len(times))}, index=times)
        result = _create_time_slice(df, "2020-01-03", "2020-01-07")
        assert result is not None
        assert result.start == pd.Timestamp("2020-01-03")
        assert result.stop == pd.Timestamp("2020-01-07")
