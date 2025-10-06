import logging

from hydromt.error import NoDataStrategy, exec_nodata_strat

_LOGGER_NAME = "test_logger"
logger = logging.getLogger(_LOGGER_NAME)

# This test is dependent on the caller and call stack.
# Don't let pytest optimize the test.
# pytest: disable_assert_rewriting


def test_logger_from_frame_in_nodata_strat(caplog):
    exec_nodata_strat("foo", NoDataStrategy.WARN)
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "foo"
    assert caplog.records[0].name == _LOGGER_NAME
