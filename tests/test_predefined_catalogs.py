from hydromt.predefined_catalogs import _get_catalog_eps


def test_eps():
    # TODO mock actual entrypoints
    eps = _get_catalog_eps()
    assert "artifact_data" in eps
    assert "deltares_data" in eps
