import pytest

from hydromt.model.mode import ModelMode


@pytest.mark.parametrize("mode", ["r", "r+", ModelMode.READ, ModelMode.APPEND])
def test_assert_reading_modes(mode):
    # Simple check whether it is in reading mode
    assert ModelMode.from_str_or_mode(mode).is_reading_mode()


@pytest.mark.parametrize("mode", ["w", "w+", ModelMode.WRITE, ModelMode.FORCED_WRITE])
def test_assert_not_reading_modes(mode):
    # Simple check whether it is not in reading mode
    assert not ModelMode.from_str_or_mode(mode).is_reading_mode()


@pytest.mark.parametrize(
    "mode", ["w", "w+", "r+", ModelMode.APPEND, ModelMode.WRITE, ModelMode.FORCED_WRITE]
)
def test_assert_writing_modes(mode):
    # Simple check whether it is in writing mode
    assert ModelMode.from_str_or_mode(mode).is_writing_mode()


@pytest.mark.parametrize("mode", ["r", ModelMode.READ])
def test_assert_not_writing_modes(mode):
    # Simple check whether it is not in writing mode
    assert not ModelMode.from_str_or_mode(mode).is_writing_mode()


@pytest.mark.parametrize(
    "mode",
    [
        "a",
        "wr",
        "rw",
        "r++",
        "w2",
        "\\w",
        "ww",
        "",
        "+w",
        "lorum ipsum",
        1,
        None,
        -8,
        3.14,
        "⽀",
        "ðŸ˜Š",
    ],
)
def test_errors_on_unknown_modes(mode):
    with pytest.raises(ValueError, match="Unknown mode"):
        # Assert that these modes are invalid
        _ = ModelMode.from_str_or_mode(mode)
