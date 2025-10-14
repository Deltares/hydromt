from datetime import datetime

import pytest
from pydantic import ValidationError

from hydromt.typing import TimeRange
from hydromt.typing.type_def import DATETIME_FORMAT


def test_valid_datetimes():
    start = datetime(2020, 1, 1)
    end = datetime(2020, 12, 31)
    tr = TimeRange(start=start, end=end)
    assert tr.start == start
    assert tr.end == end


@pytest.mark.parametrize(
    ("start_str", "end_str", "expected_start", "expected_end"),
    [
        # Exact format
        (
            "2020-01-01_00:00:00",
            "2020-12-31_23:59:59",
            datetime(2020, 1, 1, 0, 0, 0),
            datetime(2020, 12, 31, 23, 59, 59),
        ),
        # ISO 8601 date-only (flexible fallback)
        ("2020-01-01", "2020-12-31", datetime(2020, 1, 1), datetime(2020, 12, 31)),
        # ISO 8601 full datetime (flexible fallback)
        (
            "2020-01-01T00:00:00",
            "2020-12-31T23:59:59",
            datetime(2020, 1, 1, 0, 0, 0),
            datetime(2020, 12, 31, 23, 59, 59),
        ),
        # Human-readable formats (dateutil.parse fallback)
        ("Jan 1, 2020", "Dec 31, 2020", datetime(2020, 1, 1), datetime(2020, 12, 31)),
        (
            "1 January 2020",
            "31 December 2020",
            datetime(2020, 1, 1),
            datetime(2020, 12, 31),
        ),
    ],
)
def test_string_parsing(start_str, end_str, expected_start, expected_end):
    """Ensure different datetime string formats are parsed correctly."""
    tr = TimeRange(start=start_str, end=end_str)
    assert tr.start == expected_start
    assert tr.end == expected_end
    assert isinstance(tr.start, datetime)
    assert isinstance(tr.end, datetime)


def test_start_after_end_raises():
    with pytest.raises(ValueError, match="should be less than end"):
        TimeRange(start="2020-12-31", end="2020-01-01")


def test_invalid_date_string():
    """Invalid date strings should trigger a parse error."""
    with pytest.raises(ValidationError):
        TimeRange(start="not-a-date", end="2020-01-01")


def test_equality():
    """Ensure the model behaves as expected when printed or compared."""
    tr1 = TimeRange(start=datetime(2020, 1, 1), end=datetime(2020, 12, 31))
    tr2 = TimeRange(start=datetime(2020, 1, 1), end=datetime(2020, 12, 31))
    assert tr1 == tr2


def test_serialize():
    tr = TimeRange(start="2020-01-01", end="2020-12-31")
    d = tr.model_dump()
    assert d == {
        "start": tr.start.strftime(DATETIME_FORMAT),
        "end": tr.end.strftime(DATETIME_FORMAT),
    }


def test_serialize_deserialize():
    tr = TimeRange(start="2020-01-01", end="2020-12-31")
    d = tr.model_dump()
    tr2 = TimeRange.model_validate(d)
    assert tr == tr2
