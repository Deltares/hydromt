from pathlib import Path

import pytest

from hydromt.components.dataset import DatasetsComponent
from hydromt.models import Model


def test_model_dataset_key_error(tmpdir: Path):
    m = Model(root=str(tmpdir), mode="r+")
    m.add_component("test_dataset", DatasetsComponent(m))
    component = m.get_component("test_dataset", DatasetsComponent)

    with pytest.raises(KeyError):
        component.data["1"]


def test_model_dataset_sets_correctly(obsda, tmpdir: Path):
    m = Model(root=str(tmpdir), mode="r+")
    m.add_component("test_dataset", DatasetsComponent(m))
    component = m.get_component("test_dataset", DatasetsComponent)

    # make a couple copies of the da for testing
    das = {str(i): obsda.copy() for i in range(5)}

    for i, d in das.items():
        component.set(data=d, name=i)
        assert obsda.equals(component.data[i])

    assert list(component.data.keys()) == list(map(str, range(5)))


def test_model_dataset_reads_and_writes_correctly(obsda, tmpdir: Path):
    model = Model(root=str(tmpdir), mode="w+")
    model.add_component("test_dataset", DatasetsComponent(model))
    component = model.get_component("test_dataset", DatasetsComponent)

    component.set(data=obsda, name="data")

    # don't need region for this test
    _ = model._components.pop("region")
    model.write()
    clean_model = Model(root=str(tmpdir), mode="r")
    # don't need region for this test
    _ = clean_model._components.pop("region")
    clean_model.add_component("test_dataset", DatasetsComponent(clean_model))
    clean_model.read()

    clean_component = clean_model.get_component("test_dataset", DatasetsComponent)

    # we'll know that these types will always be the same, which mypy doesn't know
    assert component.data["data"].equals(clean_component.data["data"])  # type: ignore


def test_model_read_dataset(obsda, tmpdir):
    write_path = Path(tmpdir) / "forcing.nc"
    obsda.to_netcdf(write_path, engine="netcdf4")

    model = Model(root=tmpdir, mode="r")
    model.add_component("forcing", DatasetsComponent(model))

    dataset_component = model.get_component("forcing", DatasetsComponent)

    component_data = dataset_component.data["forcing"]
    assert obsda.equals(component_data)
