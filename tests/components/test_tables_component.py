from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from hydromt.components.tables import TablesComponent
from hydromt.model import Model


def test_model_tables_key_error(df, tmpdir: Path):
    m = Model(root=str(tmpdir), mode="r+")
    m.add_component("test_table", TablesComponent(m))
    component = cast(TablesComponent, m.test_table)

    with pytest.raises(KeyError):
        component.data["1"]


def test_model_tables_merges_correctly(df, tmpdir: Path):
    m = Model(root=str(tmpdir), mode="r+")
    m.add_component("test_table", TablesComponent(m))
    component = cast(TablesComponent, m.test_table)

    # make a couple copies of the dfs for testing
    dfs = {str(i): df.copy() * i for i in range(5)}

    component.set(tables=dfs)

    computed = component.get_tables_merged()
    expected = pd.concat([df.assign(table_origin=name) for name, df in dfs.items()])
    assert computed.equals(expected)


def test_model_tables_sets_correctly(df, tmpdir: Path):
    m = Model(root=str(tmpdir), mode="r+")
    m.add_component("test_table", TablesComponent(m))
    component = cast(TablesComponent, m.test_table)

    # make a couple copies of the dfs for testing
    dfs = {str(i): df.copy() for i in range(5)}

    for i, d in dfs.items():
        component.set(tables=d, name=i)
        assert df.equals(component.data[i])

    assert list(component.data.keys()) == list(map(str, range(5)))


def test_model_tables_reads_and_writes_correctly(df, tmpdir: Path):
    model = Model(root=str(tmpdir), mode="r+")
    model.add_component("test_table", TablesComponent(model))
    component = cast(TablesComponent, model.test_table)

    component.set(tables=df, name="table")

    model.write()
    clean_model = Model(root=str(tmpdir), mode="r")
    clean_model.add_component("test_table", TablesComponent(model))
    clean_model.read()

    clean_component = cast(TablesComponent, clean_model.test_table)

    assert component.data["table"].equals(clean_component.data["table"])
