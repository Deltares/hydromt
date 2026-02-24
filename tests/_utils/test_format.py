import io

import yaml

from hydromt._utils.format import CatalogDumper


def test_catalog_dumper(catalog_dummy: dict):
    # Setup a buffer
    buffer = io.StringIO()

    # Dump the dictionary with the dumper
    yaml.dump(catalog_dummy, buffer, Dumper=CatalogDumper)

    # Assert the output
    assert buffer.tell() == 342
    # Read the buffer
    buffer.seek(0)
    content = buffer.read()
    # Assert content specifics
    assert content.count("\n") == 23
    assert "baz.tif\n\nvector1:" in content  # Extra line between last and new entry


def test_catalog_dumper_without(catalog_dummy: dict):
    # Setup a buffer
    buffer = io.StringIO()

    # Dump the dictionary with the dumper
    yaml.dump(catalog_dummy, buffer)

    # Assert the output
    assert buffer.tell() == 340
    # Read the buffer
    buffer.seek(0)
    content = buffer.read()
    # Assert content specifics
    assert content.count("\n") == 21
    assert "baz.tif\nvector1:" in content  # No extra line between last and new entry
