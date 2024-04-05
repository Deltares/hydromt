from pathlib import Path

import pandas as pd
import pytest

from hydromt.components.tables import TablesComponent
from hydromt.models import Model


def test_model_tables_key_error(df, tmpdir: Path):
    m = Model(root=str(tmpdir), mode="r+")
    m.add_component("test_table", TablesComponent(m))
    component = m.get_component("test_table", TablesComponent)

    with pytest.raises(KeyError):
        component.data["1"]


def test_model_tables_merges_correctly(df, tmpdir: Path):
    m = Model(root=str(tmpdir), mode="r+")
    m.add_component("test_table", TablesComponent(m))
    component = m.get_component("test_table", TablesComponent)

    # make a couple copies of the dfs for testing
    dfs = {str(i): df.copy() * i for i in range(5)}

    component.set(tables=dfs)

    computed = component.get_tables_merged()
    expected = pd.concat([df.assign(table_origin=name) for name, df in dfs.items()])
    assert computed.equals(expected)


def test_model_tables_sets_correctly(df, tmpdir: Path):
    m = Model(root=str(tmpdir), mode="r+")
    m.add_component("test_table", TablesComponent(m))
    component = m.get_component("test_table", TablesComponent)

    # make a couple copies of the dfs for testing
    dfs = {str(i): df.copy() for i in range(5)}

    for i, d in dfs.items():
        component.set(tables=d, name=i)
        assert df.equals(component.data[i])

    assert list(component.data.keys()) == list(map(str, range(5)))


@pytest.mark.skip(reason="Needs raster dataset implementation")
def test_model_tables_reads_and_writes_correctly(df, tmpdir: Path):
    model = Model(root=str(tmpdir), mode="r+")
    model.add_component("test_table", TablesComponent(model))
    component = model.get_component("test_table", TablesComponent)

    component.set(tables=df, name="table")

    model.write()
    clean_model = Model(root=str(tmpdir), mode="r")
    clean_model.add_component("test_table", TablesComponent(model))
    clean_model.read()

    clean_component = clean_model.get_component("test_table", TablesComponent)

    assert component.data["table"].equals(clean_component.data["table"])
