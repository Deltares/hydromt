import os

import pytest

from hydromt._utils.rgetattr import rgetattr


def test_rgetattr_os_sep():
    assert rgetattr(os, "path.sep") == os.path.sep


def test_rgetattr_os_fake_default():
    assert rgetattr(os, "path.sepx", None) is None
    assert rgetattr(os, "path.sepx", "--") == "--"


def test_rgetattr_os_fake_no_default_fails():
    with pytest.raises(AttributeError):
        rgetattr(os, "path.sepx")
