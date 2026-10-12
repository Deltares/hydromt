import pytest

from hydromt._utils.warnings import HydroMTDeprecationWarning, deprecation_warning


def test_deprecation_warning_is_visible_by_default():
    assert issubclass(HydroMTDeprecationWarning, FutureWarning)
    assert not issubclass(HydroMTDeprecationWarning, DeprecationWarning)


def test_deprecation_warning_message():
    with pytest.warns(HydroMTDeprecationWarning) as rec:
        deprecation_warning(
            "`a` is deprecated.", alternative="Use `b`", removed_in_release="2.0"
        )
    assert (
        str(rec[0].message) == "`a` is deprecated. Use `b`. It will be removed in v2.0."
    )


def test_deprecation_warning_defaults():
    with pytest.warns(HydroMTDeprecationWarning) as rec:
        deprecation_warning("`a` is deprecated")
    assert str(rec[0].message).endswith("removed in a future release.")


def test_deprecation_warning_removed_in_release():
    with pytest.warns(HydroMTDeprecationWarning) as rec:
        deprecation_warning("`a` is deprecated", removed_in_release="v2.0")
    assert str(rec[0].message).endswith("removed in v2.0.")
