"""Testing for uris utils."""
from hydromt._utils.uris import is_valid_url, strip_scheme


def test_is_valid_url():
    assert is_valid_url("https://example.com")
    assert is_valid_url("s3://example-bucket/file.html")
    assert not is_valid_url("/mnt/data")
    assert not is_valid_url(r"C:\\MyComputer\Downloads")


def test_strip_scheme():
    assert strip_scheme("https://example.com") == "example.com"
    assert strip_scheme("s3://example-bucket/file.html") == "example-bucket/file.html"
    assert strip_scheme("/mnt/data") == "/mnt/data"
