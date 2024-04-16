from os.path import join
from pathlib import Path

import pytest
from xarray import DataArray, Dataset, open_dataset

from hydromt.components.dataset import DatasetComponent
from hydromt.models import Model


def test_model_dataset_key_error(tmpdir: Path):
    m = Model(root=str(tmpdir), mode="r+")
    m.add_component("test_dataset", DatasetComponent(m))
    component = m.get_component("test_dataset", DatasetComponent)

    with pytest.raises(KeyError):
        component.data["1"]


def test_model_dataset_sets_correctly(obsda, tmpdir: Path):
    m = Model(root=str(tmpdir), mode="r+")
    m.add_component("test_dataset", DatasetComponent(m))
    component = m.get_component("test_dataset", DatasetComponent)

    # make a couple copies of the da for testing
    das = {str(i): obsda.copy() for i in range(5)}

    for i, d in das.items():
        component.set(data=d, name=i)
        assert obsda.equals(component.data[i])

    assert list(component.data.keys()) == list(map(str, range(5)))


def test_check_data(demda):
    data_dict = DatasetComponent._harmonise_data_names(demda.copy(), "elevtn")
    assert isinstance(data_dict["elevtn"], DataArray)
    assert data_dict["elevtn"].name == "elevtn"
    with pytest.raises(ValueError, match="Name required for DataArray"):
        DatasetComponent._harmonise_data_names(demda)
    demda.name = "dem"
    demds = demda.to_dataset()
    data_dict = DatasetComponent._harmonise_data_names(demds, "elevtn", False)
    assert isinstance(data_dict["elevtn"], Dataset)
    data_dict = DatasetComponent._harmonise_data_names(demds, split_dataset=True)
    assert isinstance(data_dict["dem"], DataArray)
    with pytest.raises(ValueError, match="Name required for Dataset"):
        DatasetComponent._harmonise_data_names(demds, split_dataset=False)

    # testing wrong type therefore type ignore
    with pytest.raises(ValueError, match='Data type "dict" not recognized'):
        DatasetComponent._harmonise_data_names({"wrong": "type"})  # type: ignore


def test_model_dataset_reads_and_writes_correctly(obsda, tmpdir: Path):
    model = Model(root=str(tmpdir), mode="w")
    model.add_component("test_dataset", DatasetComponent(model))
    component = model.get_component("test_dataset", DatasetComponent)

    component.set(data=obsda, name="data")

    model.write()
    clean_model = Model(root=str(tmpdir), mode="r")
    clean_model.add_component("test_dataset", DatasetComponent(model))
    clean_model.read()

    clean_component = clean_model.get_component("test_dataset", DatasetComponent)

    # we'll know that these types will always be the same, which mypy doesn't know
    assert component.data["data"].equals(clean_component.data["data"])  # type: ignore


def test_model_read_dataset(obsda, tmpdir):
    write_path = Path(tmpdir) / "forcing.nc"
    obsda.to_file(write_path)

    model = Model(root=tmpdir, mode="r")
    model.add_component("forcing", DatasetComponent(model))

    dataset_component = model.get_component("forcing", DatasetComponent)

    component_data = dataset_component.data["forcing"]
    assert obsda.equals(component_data)


def test_model_write_dataset_with_target_crs(obsda, tmpdir):
    model = Model(root=str(tmpdir), mode="w", target_model_crs=3857)
    model.add_component("forcing", DatasetComponent(model))
    dataset_component = model.get_component("forcing", DatasetComponent)

    write_path = join(tmpdir, "test_geom.geojson")
    dataset_component.set(obsda, "test_dataset")
    dataset_component.write()
    read_dataset = open_dataset(write_path)

    assert read_dataset.crs.to_epsg() == 3857
