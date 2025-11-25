import numpy as np

from hydromt.model import ExampleModel


def test_example_model(tmp_path_factory):
    example_model = ExampleModel(
        root=str(tmp_path_factory.mktemp("example_model")),
        data_libs=["artifact_data"],
    )

    assert "config" in example_model.components
    assert "grid" in example_model.components

    # Update config
    example_model.config.update(
        data={
            "parameter1": 10,
            "parameter2": "value2",
        }
    )

    assert example_model.config.data["parameter1"] == 10
    assert example_model.config.data["parameter2"] == "value2"

    # Create grid from region
    bbox = [12.05, 45.30, 12.85, 45.65]
    example_model.grid.create_from_region(
        region={"bbox": bbox},
        res=0.05,
        add_mask=False,
        align=True,
    )

    assert example_model.grid.data.raster.res[0] == 0.05
    assert np.all(np.round(example_model.grid.data.raster.bounds, 2) == bbox)
    assert example_model.grid.data.sizes["y"] == 7

    example_model.grid.add_data_from_rasterdataset(
        raster_data="merit_hydro",
        variables=["elevtn", "basins"],
        reproject_method=["average", "mode"],
        mask_name="mask",
    )

    assert "basins" in example_model.grid.data
    assert np.isclose(
        example_model.grid.data["elevtn"].raster.mask_nodata().mean().values, 3.9021976
    )
