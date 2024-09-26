"""Testing for uris utils."""

from hydromt._utils.uris import _is_valid_url, _strip_scheme


def test_is_valid_url():
    assert _is_valid_url("https://example.com")
    assert _is_valid_url("s3://example-bucket/file.html")
    assert not _is_valid_url("/mnt/data")
    assert not _is_valid_url(r"C:\\MyComputer\Downloads")


def test_strip_scheme():
    assert _strip_scheme("https://example.com") == ("https://", "example.com")
    assert _strip_scheme("s3://example-bucket/file.html") == (
        "s3://",
        "example-bucket/file.html",
    )
    assert _strip_scheme("/mnt/data") == (None, "/mnt/data")
