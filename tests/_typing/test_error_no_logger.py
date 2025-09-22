from hydromt._typing.error import NoDataStrategy, exec_nodata_strat

# This test is dependent on the caller and call stack.
# Don't let pytest optimize the test.
# pytest: disable_assert_rewriting


def test_logger_from_frame_in_nodata_strat(caplog):
    exec_nodata_strat("foo", NoDataStrategy.WARN)
    assert caplog.records[-1].levelname == "WARNING"
    assert caplog.records[-1].message == "foo"
    # Test that the name of the logger is this current frame's module, not the error.py logger.
    assert caplog.records[-1].name == __name__
