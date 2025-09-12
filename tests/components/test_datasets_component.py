from pathlib import Path
from typing import cast

import pytest

from hydromt.model import Model
from hydromt.model.components.datasets import DatasetsComponent


def test_model_dataset_key_error(tmp_path: Path):
    m = Model(root=tmp_path, mode="r+")
    m.add_component("test_dataset", DatasetsComponent(m))
    component = cast(DatasetsComponent, m.get_component("test_dataset"))

    with pytest.raises(KeyError):
        component.data["1"]


def test_model_dataset_sets_correctly(obsda, tmp_path: Path):
    m = Model(root=tmp_path, mode="r+")
    component = DatasetsComponent(m)
    m.add_component("test_dataset", component)

    # make a couple copies of the da for testing
    das = {str(i): obsda.copy() for i in range(5)}

    for i, d in das.items():
        component.set(data=d, name=i)
        assert obsda.equals(component.data[i])

    assert list(component.data.keys()) == list(map(str, range(5)))


def test_model_dataset_reads_and_writes_correctly(obsda, tmp_path: Path):
    model = Model(root=tmp_path, mode="w+")
    component = DatasetsComponent(model)
    model.add_component("test_dataset", component)

    component.set(data=obsda, name="data")

    model.write()
    clean_model = Model(root=tmp_path, mode="r")
    clean_component = DatasetsComponent(clean_model)
    clean_model.add_component("test_dataset", clean_component)
    clean_model.read()

    # we'll know that these types will always be the same, which mypy doesn't know
    assert component.data["data"].equals(clean_component.data["data"])  # type: ignore


def test_model_read_dataset(obsda, tmp_path: Path):
    write_path = tmp_path / "datasets" / "forcing.nc"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    obsda.to_netcdf(write_path, engine="netcdf4")

    model = Model(root=tmp_path, mode="r")
    dataset_component = DatasetsComponent(model)
    model.add_component("forcing", dataset_component)

    component_data = dataset_component.data["forcing"]
    assert obsda.equals(component_data)
