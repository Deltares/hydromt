from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from hydromt.model import Model
from hydromt.model.components.tables import TablesComponent


def test_model_tables_key_error(df, tmp_path: Path):
    m = Model(root=tmp_path, mode="r+")
    m.add_component("test_table", TablesComponent(m))
    component = cast(TablesComponent, m.test_table)

    with pytest.raises(KeyError):
        component.data["1"]


def test_model_tables_merges_correctly(df, tmp_path: Path):
    m = Model(root=tmp_path, mode="r+")
    m.add_component("test_table", TablesComponent(m))
    component = cast(TablesComponent, m.test_table)

    # make a couple copies of the dfs for testing
    dfs = {str(i): df.copy() * i for i in range(5)}

    component.set(tables=dfs)

    computed = component.get_tables_merged()
    expected = pd.concat([df.assign(table_origin=name) for name, df in dfs.items()])
    assert computed.equals(expected)


def test_model_tables_sets_correctly(df, tmp_path: Path):
    m = Model(root=tmp_path, mode="r+")
    m.add_component("test_table", TablesComponent(m))
    component = cast(TablesComponent, m.test_table)

    # make a couple copies of the dfs for testing
    dfs = {str(i): df.copy() for i in range(5)}

    for i, d in dfs.items():
        component.set(tables=d, name=i)
        assert df.equals(component.data[i])

    assert list(component.data.keys()) == list(map(str, range(5)))


def test_model_tables_reads_and_writes_correctly(df, tmp_path: Path):
    model = Model(root=tmp_path, mode="r+")
    model.add_component("test_table", TablesComponent(model))
    component = cast(TablesComponent, model.test_table)

    component.set(tables=df, name="table")

    model.write()
    clean_model = Model(root=tmp_path, mode="r")
    clean_model.add_component("test_table", TablesComponent(model))
    clean_model.read()

    clean_component = cast(TablesComponent, clean_model.test_table)

    assert component.data["table"].equals(clean_component.data["table"])
