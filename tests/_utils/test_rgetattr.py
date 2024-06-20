import os

import pytest

from hydromt._utils.rgetattr import _rgetattr


def test_rgetattr_os_sep():
    assert _rgetattr(os, "path.sep") == os.path.sep


def test_rgetattr_os_fake_default():
    assert _rgetattr(os, "path.sepx", None) is None
    assert _rgetattr(os, "path.sepx", "--") == "--"


def test_rgetattr_os_fake_no_default_fails():
    with pytest.raises(AttributeError):
        _rgetattr(os, "path.sepx")
