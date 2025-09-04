import logging

from hydromt._typing.error import NoDataStrategy, exec_nodata_strat

_LOGGER_NAME = "test_logger"
logger = logging.getLogger(_LOGGER_NAME)


def test_logger_from_frame_in_nodata_strat(caplog):
    exec_nodata_strat("foo", NoDataStrategy.WARN)
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "foo"
    assert caplog.records[0].name == _LOGGER_NAME
